from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
import requests
import uuid
from langgraph.prebuilt import InjectedState

from base_agents import DefaultAgent, make_supervisor_node, MessagesState
from typing import Annotated, Callable, Literal, TypedDict
import operator

from typer.cli import state

# Resolve the project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


class SiteResponseOutput(BaseModel):
    """Structure an output for sites suggested by LLM for input topic"""

    topic: str = Field(description="Input topic")
    sites: list[str] = Field(description="Sites suggested")


class GuessURLOutput(BaseModel):
    """Structured output for guess_url tool (full URLs)"""

    topic: str = Field(description="Input topic")
    sites: list[str] = Field(
        description="Full URLs prefixed as Google News search queries"
    )


class Endpoint(str, Enum):
    """Class to hold API endpoint information"""

    health = "/health"
    url = "/url"


class APIHealthEndpoint(BaseModel):
    """Structured output for API health endpoint check"""

    endpoint: str = Field(description="Endpoint for checking API health")
    status: str = Field(description="Status of the API health endpoint")
    message: str = Field(
        description="Message providing details from the endpoint about API health"
    )


class APISiteEndpoint(BaseModel):
    """Structured output for API health endpoint check"""

    endpoint: str = Field(description="Endpoint for providing site url data")
    site: str = Field(description="Site url data to be provided")


class Subtask(TypedDict):
    id: str
    team: Literal["request_team", "search_team"]
    instruction: str
    status: Literal["pending", "completed"]


class RoutingDecision(BaseModel):
    """Which pending subtask to dispatch next, or FINISH."""

    next_subtask_id: str = Field(
        description="id of the pending subtask to route to, or 'FINISH' if none remain"
    )


class SubtaskPlan(BaseModel):
    """Initial decomposition of the user's request into subtasks."""

    subtasks: list[Subtask] = Field(
        description="Each item: {'team': 'request_team'|'search_team', 'instruction': str}"
    )


class SupervisorState(MessagesState):
    current_instruction: str | None
    subtasks: list[Subtask]
    subtasks_planned: bool  # have we done the initial breakdown?
    _dispatched_id: str
    search_results: dict | None  # to hold search results for provide_site tool


search = GoogleSerperAPIWrapper()
api_host = "http://127.0.0.1:8000"


def build_api_url(endpoint: Endpoint) -> str:
    return f"{api_host}{endpoint.value}"


@tool
def relevant_site(topic: str) -> dict:
    """Provide a relevant link for input topic using google serper"""
    results = search.results(k=5, query=topic)
    results = results["organic"]
    sites_out = [item.get("link") for item in results]
    output = SiteResponseOutput(topic=topic, sites=sites_out)
    return output.model_dump()


@tool
def guess_url(topic: str, sites: list[str]) -> dict:
    """Prefix relevant site with google news search query url"""
    sites_res = relevant_site.invoke({"topic": topic})
    prefix = "https://news.google.com/search?q="
    full_sites = [f"{prefix}site:{s}" for s in sites_res.get("sites", [])]
    output = GuessURLOutput(topic=sites_res.get("topic", topic), sites=full_sites)
    return output.model_dump()


@tool
def api_health(endpoint: str) -> dict:
    """Check health of API by making a GET request to input parameter value and returning status"""
    endpoint = build_api_url(Endpoint.health)
    try:
        response = requests.get(endpoint, timeout=5)

        if response.status_code == 200:
            return APIHealthEndpoint(
                endpoint=endpoint, status="OK", message="API is healthy"
            ).model_dump()
        else:
            return APIHealthEndpoint(
                endpoint=endpoint,
                status="NOT OK",
                message=f"API returned status code {response.status_code}",
            ).model_dump()
    except Exception as e:
        return APIHealthEndpoint(
            endpoint=endpoint, status="ERROR", message=str(e)
        ).model_dump()


@tool
def provide_site(config: RunnableConfig) -> dict:
    """Provides the site url data captured from state to API endpoint"""
    results = config["configurable"].get("search_results",{})
    sites = results.get("sites",[])
    site = sites[0] if sites else None  # Get the first site from the list
    endpoint = build_api_url(Endpoint.url)
    try:
        response = requests.get(endpoint, params={"url_input": site}, timeout=5)
        response.raise_for_status()  # raises HTTPError for 4xx/5xx status codes
        return {"status": "success", "response": response.json()}
    except (
        Exception
    ) as e:  # covers connection errors, timeouts, and HTTP errors from raise_for_status
        print(f"Request to {endpoint} failed: {e}")
        return {"status": "error", "message": str(e)}


def request_team(
    state: SupervisorState, model: str, config: RunnableConfig | None
) -> Callable[[SupervisorState], Command[Literal["supervisor"]]]:
    def health_requester(state: SupervisorState) -> Command[Literal["supervisor"]]:
        tools = [api_health]
        system_prompt = """You are an agent tasked with checking the health of an API. Use the check_api_health tool to make a GET request to the API health endpoint and return the status and message. Ensure that you handle any errors gracefully and provide a clear response."""
        messages = state["messages"]
        last_message = messages[-1]
        api_agent = DefaultAgent(
            state=state,
            model=model,
            tools=tools,
            schema=APIHealthEndpoint,
            config=config,
            system_prompt=system_prompt,
        ).graph.invoke({"messages": [last_message]}, config=config)
        result = api_agent["messages"][-1]
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"API request completed successfully. Data: {result.model_dump_json()}",
                        name="health_requester",
                    )
                ]
            },
            goto="supervisor",
        )

    def site_requester(state: SupervisorState) -> Command[Literal["supervisor"]]:
        tools = [provide_site]
        system_prompt = """You are an agent tasked with providing site url data to API. Use the provide_site tool to make a GET request to the API url endpoint and return the status and message. Ensure that you handle any errors gracefully and provide a clear response."""
        messages = state["messages"]
        last_message = messages[-1]
        search_results = state.get("search_results", {})

        inner_config = {
            **(config or {}),
            "configurable": {
                **(config or {}).get("configurable", {}),
                "search_results": search_results,
            },
        }  # needs to include search_results for provide_site tool
        api_agent = DefaultAgent(
            state=state,
            model=model,
            tools=tools,
            schema=APISiteEndpoint,
            config=inner_config,
            system_prompt=system_prompt,
        ).graph.invoke({"messages": [last_message]}, config=inner_config)

        result = api_agent["messages"][-1]
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"API request completed successfully. Data: {result.model_dump_json()}",
                        name="site_requester",
                    )
                ]
            },
            goto="supervisor",
        )

    agent_roles = [
        "health_requester",
        "site_requester",
    ]  # used to direct supervisor to agent(s)
    supervisor_node = make_supervisor_node(
        init_chat_model(model), agent_roles, config=config
    )

    builder = StateGraph(SupervisorState)
    builder.add_node(
        "health_requester", health_requester
    )  # agent subgraph node, returns updates to supervisor
    builder.add_node(
        "site_requester", site_requester
    )  # agent subgraph node, returns updates to supervisor
    builder.add_node("supervisor", supervisor_node)

    builder.add_edge(START, "supervisor")

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    def call_request_team(state: SupervisorState) -> Command[Literal["supervisor"]]:
        inner_config = {
            **(config or {}),
            "configurable": {
                **(config or {}).get("configurable", {}),
                "thread_id": str(uuid.uuid4()),
            },
        }

        response = app.invoke(
            {
                "messages": [HumanMessage(content=state["current_instruction"])],
                "search_results": state.get("search_results",{}),
            },  # Pass instruction from supervisor to search team
            config=inner_config,
        )
        last = response["messages"][-1]
        updated_subtasks = [
            {**s, "status": "completed"} if s["id"] == state["_dispatched_id"] else s
            for s in state["subtasks"]
        ]

        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=last.content, name="request_team")],
                "subtasks": updated_subtasks,
            },
        )

    return call_request_team


def search_team(
    state: SupervisorState, model: str, config: RunnableConfig | None
) -> Callable[[SupervisorState], Command[Literal["supervisor"]]]:
    def search_agent(state: SupervisorState) -> Command[Literal["supervisor"]]:
        tools = [relevant_site, guess_url]
        system_prompt = """You are a search agent tasked with finding relevant sites for a given topic. Use the relevant_site and guess_url tools to provide full URLs prefixed as Google News search queries."""
        messages = state["messages"]
        last_message = messages[-1]
        search_agent = DefaultAgent(
            state=state,
            model=model,
            tools=tools,
            schema=GuessURLOutput,
            config=config,
            system_prompt=system_prompt,
        ).graph.invoke({"messages": [last_message]}, config=config)
        result = search_agent["messages"][-1]  # structured output only
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"Search completed successfully. Data: {result.model_dump()}",
                        name="search",
                    )
                ],
                "search_results": result.model_dump(),
            },
            goto="supervisor",
        )

    agent_roles = ["searcher"]  # used to direct supervisor to agent(s)
    supervisor_node = make_supervisor_node(
        init_chat_model(model), agent_roles, config=config
    )

    builder = StateGraph(SupervisorState)
    builder.add_node(
        "searcher", search_agent
    )  # agent subgraph node, returns updates to supervisor
    builder.add_node("supervisor", supervisor_node)

    builder.set_entry_point("supervisor")

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    def call_search_team(state: SupervisorState) -> Command[Literal["supervisor"]]:
        instruction = state["current_instruction"]
        inner_config = {
            **(config or {}),
            "configurable": {
                **(config or {}).get("configurable", {}),
                "thread_id": str(uuid.uuid4()),
            },
        }
        response = app.invoke(
            {
                "messages": [HumanMessage(content=instruction)]
            },  # Pass instruction from supervisor to search team
            config=inner_config,
        )
        last = response["messages"][-1]
        updated_subtasks = [
            {**s, "status": "completed"} if s["id"] == state["_dispatched_id"] else s
            for s in state["subtasks"]
        ]

        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=last.content, name="search_team")],
                "subtasks": updated_subtasks,
                "search_results": response.get("search_results"),
            },
        )

    return call_search_team


def call_teams(state: SupervisorState):
    thread_id = str(uuid.uuid4())
    model = "openai:gpt-5.4-mini"
    config = {"configurable": {"thread_id": thread_id}}
    lead_model = init_chat_model(model=model, temperature=0)
    PLAN_PROMPT = (
        "Break the user's request into independent subtasks. Each subtask goes to "
        "exactly one team:\n"
        "- 'request_team': checks API health and provides site urls as data to API.\n"
        "- 'search_team': finds relevant sites for a topic and returns full Google "
        "News search URLs.\n\n"
        "A single team may own multiple subtasks — list them separately, each with "
        "self-contained instruction text covering only that piece of the request."
    )

    ROUTE_PROMPT = (
        "Below are the remaining pending subtasks with their ids, teams, and "
        "instructions. Pick the id of the one to dispatch next. If the list is "
        "empty, respond with next_subtask_id='FINISH'."
    )

    FINISH_TOKEN = "FINISH"

    def supervisor_node(
        state: SupervisorState, config: RunnableConfig
    ) -> Command[Literal["request_team", "search_team", "__end__"]]:
        # Phase 1: plan subtasks once, on first entry
        if not state.get("subtasks_planned"):
            messages = [SystemMessage(content=PLAN_PROMPT)] + state["messages"]

            plan = lead_model.with_structured_output(SubtaskPlan).invoke(
                messages, config=config
            )
            subtasks: list[Subtask] = [
                {
                    "id": str(uuid.uuid4())[:8],
                    "team": s["team"],
                    "instruction": s["instruction"],
                    "status": "pending",
                }
                for s in plan.subtasks
            ]
            state = {**state, "subtasks": subtasks, "subtasks_planned": True}

        pending = [s for s in state["subtasks"] if s["status"] == "pending"]

        if not pending:
            return Command(goto="__end__", update={"next": FINISH_TOKEN})
            # Block any request_team subtask that posts sites until search_results is ready

        def _is_blocked(s: Subtask) -> bool:
            needs_search_results = (
                s["team"] == "request_team" and "site" in s["instruction"].lower()
            )
            if not needs_search_results:
                return False
            results = state.get("search_results")
            return not (results and len(results.get("sites", [])) > 0)

        dispatchable = [s for s in pending if not _is_blocked(s)]

        if not dispatchable:
            return Command(
                goto="__end__",
                update={
                    "next": FINISH_TOKEN,
                    "current_instruction": "BLOCKED: no dispatchable subtasks",
                },
            )

        if len(dispatchable) == 1:
            chosen = dispatchable[0]
        else:
            listing = "\n".join(
                f"- id={s['id']} team={s['team']} instruction={s['instruction']!r}"
                for s in dispatchable
            )
            decision = lead_model.with_structured_output(RoutingDecision).invoke(
                [SystemMessage(content=ROUTE_PROMPT + "\n\n" + listing)],
                config=config,
            )
            if decision.next_subtask_id == FINISH_TOKEN:
                return Command(goto="__end__", update={"next": FINISH_TOKEN})
            chosen = next(
                s for s in dispatchable if s["id"] == decision.next_subtask_id
            )

            # Deterministic fallback: if there's only one pending, skip the LLM call
        if len(pending) == 1:
            chosen = pending[0]
        else:
            messages = [SystemMessage(content=ROUTE_PROMPT)] + state["messages"]
            listing = "\n".join(
                f"- id={s['id']} team={s['team']} instruction={s['instruction']!r}"
                for s in pending
            )
            decision = lead_model.with_structured_output(RoutingDecision).invoke(
                [SystemMessage(content=ROUTE_PROMPT + "\n\n" + listing)],
                config=config,
            )
            if decision.next_subtask_id == FINISH_TOKEN:
                return Command(goto="__end__", update={"next": FINISH_TOKEN})
            chosen = next(s for s in pending if s["id"] == decision.next_subtask_id)

        return Command(
            goto=chosen["team"],
            update={
                "next": chosen["team"],
                "current_instruction": chosen["instruction"],
                "subtasks": state["subtasks"],
                "subtasks_planned": True,
                "_dispatched_id": chosen["id"],  # so the team node can mark it done
            },
        )

    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node(
        "request_team",
        request_team(state=state, model=model, config=config),
    )
    builder.add_node(
        "search_team",
        search_team(state=state, model=model, config=config),
    )

    builder.set_entry_point("supervisor")

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    messages = app.invoke(
        {"messages": state["messages"]},
        config=config,
    )

    for m in messages["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    topic = "latest reliable news source affecting stock market"
    call_teams(
        state=SupervisorState(
            messages=[
                HumanMessage(
                    content=f"Check the health of the API then provide full URLs prefixed as Google News search queries for relevant sites for {topic}. Finally provide those urls from previous step to the API via provide site tool."
                )
            ]
        )
    )
