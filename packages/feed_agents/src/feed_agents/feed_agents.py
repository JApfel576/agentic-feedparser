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


from base_agents import DefaultAgent, make_supervisor_node, MessagesState
from typing import Callable, Literal

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


class APIHealthEndpoint(BaseModel):
    """Structured output for API health endpoint check"""

    endpoint: str = Field(description="Endpoint for checking API health")
    status: str = Field(description="Status of the API health endpoint")
    message: str = Field(
        description="Message providing details from the endpoint about API health"
    )


class RoutingDecision(BaseModel):
    """Structured output for lead model routing"""

    next: Literal["request_team", "search_team", "FINISH"]
    instruction: str | None = Field(
        default=None,
        description="The specific sub-task text to hand to the chosen team. "
        "Should contain only what that team needs — not the full original request. "
        "Omit or leave empty when next is FINISH.",
    )


class SupervisorState(MessagesState):
    next: Literal["request_team", "search_team", "FINISH"] | None
    current_instruction: str | None


search = GoogleSerperAPIWrapper()
api_host = "http://127.0.0.1:8000"


def build_api_url(endpoint: Endpoint) -> str:
    return f"{api_host}{endpoint.value}"


@tool()
def relevant_site(topic: str) -> dict:
    """Provide a relevant link for input topic using google serper"""
    results = search.results(k=5, query=topic)
    results = results["organic"]
    sites_out = [item.get("link") for item in results]
    output = SiteResponseOutput(topic=topic, sites=sites_out)
    return output.model_dump()


@tool()
def guess_url(topic: str, sites: list[str]) -> dict:
    """Prefix relevant site with google news search query url"""
    sites_res = relevant_site.invoke({"topic": topic})
    prefix = "https://news.google.com/search?q="
    full_sites = [f"{prefix}site:{s}" for s in sites_res.get("sites", [])]
    output = GuessURLOutput(topic=sites_res.get("topic", topic), sites=full_sites)
    return output.model_dump()


@tool()
def check_api_health(endpoint: str) -> dict:
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


def request_team(
    state: SupervisorState, model: str, config: RunnableConfig | None
) -> Callable[[SupervisorState], Command[Literal["supervisor"]]]:
    def api_request_agent(state: SupervisorState) -> Command[Literal["supervisor"]]:
        tools = [check_api_health]
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
        api_agent_last = api_agent["messages"][-1]
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"API request completed successfully. Data: {api_agent_last.model_dump_json()}",
                        name="api_requester",
                    )
                ]
            },
            goto="supervisor",
        )

    agent_roles = ["requester"]  # used to direct supervisor to agent(s)
    supervisor_node = make_supervisor_node(
        init_chat_model(model), agent_roles, config=config
    )

    builder = StateGraph(SupervisorState)
    builder.add_node(
        "requester", api_request_agent
    )  # agent subgraph node, returns updates to supervisor
    builder.add_node("supervisor", supervisor_node)

    builder.add_edge(START, "supervisor")

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    def call_request_team(state: SupervisorState) -> Command[Literal["supervisor"]]:
        instruction = state["current_instruction"]
        response = app.invoke(
            {
                "messages": [HumanMessage(content=instruction)]
            },  # Pass instruction from supervisor to search team
            config=config,
        )
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=response["messages"][-1].content,
                        name="request_team",
                    )
                ]
            },
            goto="supervisor",
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
        search_agent_last = search_agent["messages"][-1]  # structured output only
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"Search completed successfully. Data: {search_agent_last.model_dump_json()}",
                        name="search",
                    )
                ]
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

    builder.add_edge(START, "supervisor")

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    def call_search_team(state: SupervisorState) -> Command[Literal["supervisor"]]:
        instruction = state["current_instruction"]
        response = app.invoke(
            {
                "messages": [HumanMessage(content=instruction)]
            },  # Pass instruction from supervisor to search team
            config=config,
        )
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=response["messages"][-1].content,
                        name="search_team",
                    )
                ]
            },
            goto="supervisor",
        )

    return call_search_team


def call_teams(state: SupervisorState):
    thread_id = str(uuid.uuid4())
    model = "openai:gpt-5.4-mini"
    config = {"configurable": {"thread_id": thread_id}}
    lead_model = init_chat_model(model=model, temperature=0)
    system_prompt = (
        "You are the lead supervisor routing between two worker teams:\n"
        "- 'request_team': checks the health of the API.\n"
        "- 'search_team': finds relevant sites for a topic and returns full Google "
        "News search URLs.\n\n"
        "The user's request may contain multiple sub-tasks. Break it down and, for "
        "each unfinished sub-task, route to the team that owns it. When you route to "
        "a team, write a clear, self-contained 'instruction' string containing ONLY "
        "the sub-task text that team needs — do not include unrelated parts of the "
        "request. A team's result appears as a message named after that team once it "
        "has run; do not route to a team whose part is already done. When every "
        "sub-task has been completed, respond with next=FINISH."
    )
    FINISH_TOKEN = "FINISH"

    def supervisor_node(
        state: SupervisorState, config: RunnableConfig
    ) -> Command[Literal["request_team", "search_team", "__end__"]]:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        routing_decision = lead_model.with_structured_output(RoutingDecision).invoke(
            messages, config=config
        )
        choice = routing_decision.next
        goto = "__end__" if choice == FINISH_TOKEN else choice
        return Command(
            goto=goto,
            update={
                "next": choice,
                "current_instruction": routing_decision.instruction,
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

    builder.add_edge(START, "supervisor")

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
                    content=f"Check the health of the API.Provide full URLs prefixed as Google News search queries for relevant sites for {topic}"
                )
            ]
        )
    )
