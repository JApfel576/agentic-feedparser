from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from langgraph.types import Command, interrupt
from langchain_core.runnables import RunnableConfig
import requests
import uuid

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
    rss = "/rss"
    data = "/data"


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
    dispatched_agents_run: list[str]
    subtasks: list[Subtask]
    subtasks_planned: bool  # have we done the initial breakdown?
    _dispatched_id: str
    search_results: dict | None  # to hold search results for provide_site tool
    approval_attempts: int  # caps the human_approval -> searcher retry cycle


search = GoogleSerperAPIWrapper()
api_host = "http://127.0.0.1:8000"
MAX_APPROVAL_ATTEMPTS = 3  # how many times human_approval may bounce back to searcher


def build_api_url(endpoint: Endpoint) -> str:
    return f"{api_host}{endpoint.value}"


from urllib.parse import urlparse


def normalize_site_operator(raw_site: str) -> str:
    """Convert a full URL (or bare domain) into a valid Google site: operator value."""
    if "://" in raw_site:
        parsed = urlparse(raw_site)
        domain = parsed.netloc
        path = parsed.path.rstrip("/")
        return f"{domain}{path}" if path else domain
    return raw_site.rstrip("/")


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
    normalized_sites = [normalize_site_operator(s) for s in full_sites]
    output = GuessURLOutput(topic=sites_res.get("topic", topic), sites=normalized_sites)
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
def provide_site(site: str) -> dict:
    """Provides the given site url to the API endpoints (validate -> rss -> data).

    `site` must be a full URL; the caller supplies it from the earlier search step.
    """

    def make_request(endpoint: str, param_site: str, site: str) -> dict:
        try:
            response = requests.get(endpoint, params={param_site: site}, timeout=5)
            response.raise_for_status()  # raises HTTPError for 4xx/5xx status codes
            return {"status": "success", "response": response.json()}
        except Exception as e:  # covers connection errors, timeouts, and HTTP errors from raise_for_status
            print(f"Request to {endpoint} failed: {e}")
            return {"status": "error", "message": str(e)}

    def fetch_site_data(site: str) -> dict:
        validate_url = make_request(
            endpoint=build_api_url(Endpoint.url), param_site="url_input", site=site
        )
        if validate_url.get("status") == "error":
            return validate_url

        converted_url = make_request(
            endpoint=build_api_url(Endpoint.rss),
            param_site="url_input",
            site=validate_url.get("response"),
        )
        if converted_url.get("status") == "error":
            return converted_url

        # This will use converted_url instead of original site?
        request_data = make_request(
            endpoint=build_api_url(Endpoint.data),
            param_site="url_input",
            site=converted_url.get("response"),
        )
        return request_data

    return (
        fetch_site_data(site)
        if site
        else {"status": "error", "message": "No site provided"}
    )


def request_team(
    state: SupervisorState, model: str, config: RunnableConfig | None
) -> Callable[[SupervisorState], Command[Literal["supervisor"]]]:
    def health_requester(state: SupervisorState) -> Command[Literal["supervisor"]]:
        tools = [api_health]
        system_prompt = """You are an agent tasked with checking the health of an API. Use the check_api_health tool to make a GET request to the API health endpoint and return the status and message. Ensure that you handle any errors gracefully and provide a clear response."""
        instruction = state["current_instruction"]

        api_agent = DefaultAgent(
            state=state,
            model=model,
            tools=tools,
            schema=APIHealthEndpoint,
            config=config,
            system_prompt=system_prompt,
        ).graph.invoke({"messages": [instruction]}, config=config)
        result = api_agent["messages"][-1]
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"API request completed successfully. Data: {result.model_dump_json()}",
                        name="health_requester",
                    )
                ],
                "dispatched_agents_run": state.get("dispatched_agents_run", [])
                + ["health_requester"],
            },
            goto="supervisor",
        )

    def site_requester(state: SupervisorState) -> Command[Literal["supervisor"]]:
        tools = [provide_site]
        system_prompt = """You are an agent tasked with providing site url data to API. Use the provide_site tool to make a GET request to the API url endpoint and return the status and message. Ensure that you handle any errors gracefully and provide a clear response."""
        # The site lives in graph state, not in config — hand it to the agent in the
        # instruction so the provide_site tool receives it as an argument.
        sites = (state.get("search_results") or {}).get("sites", [])
        instruction = state["current_instruction"]
        if sites:
            instruction = f"{instruction}\n\nUse this site url: {sites[0]}"
        else:
            instruction = (
                f"{instruction}\n\nNo site url is available from the search step — "
                "report this as an error instead of calling provide_site."
            )

        api_agent = DefaultAgent(
            state=state,
            model=model,
            tools=tools,
            schema=APISiteEndpoint,
            config=config,
            system_prompt=system_prompt,
        ).graph.invoke({"messages": [instruction]}, config=config)

        result = api_agent["messages"][-1]
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"API request completed successfully. Data: {result.model_dump_json()}",
                        name="site_requester",
                    )
                ],
                "dispatched_agents_run": state.get("dispatched_agents_run", [])
                + ["site_requester"],
            },
            goto="supervisor",
        )

    agent_roles = [
        "health_requester",
        "site_requester",
    ]  # used to direct supervisor to agent(s)
    supervisor_node = make_supervisor_node(
        model, agent_roles, config=config, team_name="request_team"
    )

    builder = StateGraph(SupervisorState)
    builder.add_node(
        "health_requester", health_requester
    )  # agent subgraph node, returns updates to supervisor
    builder.add_node(
        "site_requester", site_requester
    )  # agent subgraph node, returns updates to supervisor
    builder.add_node("supervisor", supervisor_node)

    builder.set_entry_point("supervisor")
    return builder.compile()


def search_team(
    state: SupervisorState, model: str, config: RunnableConfig | None
) -> Callable[[SupervisorState], Command[Literal["supervisor"]]]:
    def search_agent(state: SupervisorState) -> Command[Literal["supervisor"]]:
        print("SEARCH_AGENT ENTERED, dispatched_agents_run:",
            state.get("dispatched_agents_run"))
        tools = [relevant_site, guess_url]
        system_prompt = """You are a search agent tasked with finding relevant sites for a given topic. Use the relevant_site and guess_url tools to provide full URLs prefixed as Google News search queries."""
        instruction = state["current_instruction"]
        search_agent = DefaultAgent(
            state=state,
            model=model,
            tools=tools,
            schema=GuessURLOutput,
            config=config,
            system_prompt=system_prompt,
        ).graph.invoke({"messages": [instruction]}, config=config)
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
                "dispatched_agents_run": state.get("dispatched_agents_run", [])
                + ["searcher"],
            },
            goto="human_approval",
        )

    def human_approval(
        state: SupervisorState,
    ) -> Command[Literal["supervisor", "searcher"]]:
        attempts = (state.get("approval_attempts") or 0) + 1
        print(
            f"HUMAN_APPROVAL ENTERED attempt={attempts}, dispatched_agents_run:",
            state.get("dispatched_agents_run"),
        )
        response = interrupt(
            {
                "question": "Do you approve the search results? (yes/no)",
                "search_results": state.get("search_results", {}),
                "attempt": attempts,
                "max_attempts": MAX_APPROVAL_ATTEMPTS,
            }
        )
        is_approved = str(response).strip().lower() in ("yes", "y", "true", "1")
        update = {
            "dispatched_agents_run": state.get("dispatched_agents_run", [])
            + ["human_approval"],
            "approval_attempts": attempts,
        }
        if is_approved:
            return Command(update=update, goto="supervisor")
        if attempts >= MAX_APPROVAL_ATTEMPTS:
            # Bound the reject cycle: without this, searcher <-> human_approval never settles.
            update["messages"] = [
                AIMessage(
                    content=(
                        f"Search results were not approved after {attempts} attempts; "
                        "giving up on re-running the search."
                    ),
                    name="human_approval",
                )
            ]
            return Command(update=update, goto="supervisor")
        return Command(update=update, goto="searcher")

    # human_approval is reached by an explicit goto from searcher, not by supervisor routing,
    # so it must not appear in the roster the supervisor picks from.
    agent_roles = ["searcher"]  # used to direct supervisor to agent(s)
    supervisor_node = make_supervisor_node(
        model,
        agent_roles,
        config=config,
        additional_instructions="After search results are returned, ask for human approval before proceeding to the next step.",
        team_name="search_team",
    )

    builder = StateGraph(SupervisorState)
    builder.add_node(
        "searcher", search_agent
    )  # agent subgraph node, returns updates to supervisor
    builder.add_node("human_approval", human_approval)
    builder.add_node("supervisor", supervisor_node)
    builder.set_entry_point("supervisor")
    return builder.compile()


def build_top_graph(model_str, config):
    model = init_chat_model(model_str, temperature=0, max_retries=3, timeout=30)
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

    def mark_subtask_complete(state: SupervisorState) -> Command[Literal["supervisor"]]:
        updated = [
            {**s, "status": "completed"} if s["id"] == state["_dispatched_id"] else s
            for s in state["subtasks"]
        ]
        return Command(goto="supervisor", update={"subtasks": updated})

    def supervisor_node(
        state: SupervisorState, config: RunnableConfig
    ) -> Command[Literal["request_team", "search_team", "__end__"]]:
        # Phase 1: plan subtasks once, on first entry
        if not state.get("subtasks_planned"):
            messages = [SystemMessage(content=PLAN_PROMPT)] + state["messages"]

            plan = model.with_structured_output(SubtaskPlan).invoke(
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
            decision = model.with_structured_output(RoutingDecision).invoke(
                [SystemMessage(content=ROUTE_PROMPT + "\n\n" + listing)],
                config=config,
            )
            if decision.next_subtask_id == FINISH_TOKEN:
                return Command(goto="__end__", update={"next": FINISH_TOKEN})
            chosen = next(
                s for s in dispatchable if s["id"] == decision.next_subtask_id
            )

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
        request_team(state=None, model=model, config=config),
    )
    builder.add_node(
        "search_team",
        search_team(state=None, model=model, config=config),
    )
    builder.add_node("mark_subtask_complete", mark_subtask_complete)

    builder.set_entry_point("supervisor")
    builder.add_edge("search_team", "mark_subtask_complete")
    builder.add_edge(
        "request_team", "mark_subtask_complete"
    )  # request_team needs this too
    builder.add_edge("mark_subtask_complete", "supervisor")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    topic = "latest reliable news source affecting stock market"
    # recursion_limit kept low so a routing regression surfaces immediately instead of
    # burning 25 supersteps of LLM calls.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 12}
    graph = build_top_graph(model_str="openai:gpt-5.4-mini", config=config)

    try:
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Check the health of the API then provide full URLs prefixed as Google News search queries for relevant sites for {topic}. Finally provide those urls from previous step to the API via provide site tool."
                    )
                ]
            },
            config=config,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(payload["question"], payload["search_results"])
        answer = input("yes/no: ")
        result = graph.invoke(Command(resume=answer), config=config)

    print(result)
