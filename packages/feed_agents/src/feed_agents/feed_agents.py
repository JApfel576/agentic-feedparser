from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain import messages
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
import requests
import uuid
from typing import Annotated
import operator


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

    next: Literal["request_team", "search_team", "FINISH"] = Field(
        description="Next step to route to, or FINISH if the task is complete"
    )


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
    state: MessagesState, model: str, config: RunnableConfig | None
) -> Callable[[MessagesState], Command[Literal["supervisor"]]]:

    def api_request_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
        tools = [check_api_health]
        system_prompt = """You are an agent tasked with checking the health of an API. Use the check_api_health tool to make a GET request to the API health endpoint and return the status and message. Ensure that you handle any errors gracefully and provide a clear response."""
        api_agent = DefaultAgent(
            state=state,
            model=model,
            tools=tools,
            schema=APIHealthEndpoint,
            config=config,
            system_prompt=system_prompt,
        ).graph.invoke(state["messages"][-1])
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

    agent_roles = ["requester"]
    supervisor_node = make_supervisor_node(
        init_chat_model(model), agent_roles, config=config
    )

    builder = StateGraph(MessagesState)
    builder.add_node("requester", api_request_agent)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("FINISH", lambda state: state)  # Terminal node

    builder.add_edge(START, "supervisor")
    builder.add_edge("FINISH", END)

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    def call_request_team(state: MessagesState) -> Command[Literal["supervisor"]]:
        response = app.invoke(
            {
                "messages": state["messages"][-1:]
            },  # Pass only the last message to the request team
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
    state: MessagesState, model: str, config: RunnableConfig | None
) -> Callable[[MessagesState], Command[Literal["supervisor"]]]:

    def search_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
        tools = [relevant_site, guess_url]
        system_prompt = """You are a search agent tasked with finding relevant sites for a given topic. Use the relevant_site and guess_url tools to provide full URLs prefixed as Google News search queries."""
        search_agent = DefaultAgent(
            state=state,
            model=model,
            tools=tools,
            schema=GuessURLOutput,
            config=config,
            system_prompt=system_prompt,
        ).graph.invoke(state["messages"][-1])
        search_agent_last = search_agent["messages"][-1]
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

    agent_roles = ["searcher"]
    supervisor_node = make_supervisor_node(
        init_chat_model(model), agent_roles, config=config
    )

    builder = StateGraph(MessagesState)
    builder.add_node("searcher", search_agent)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("FINISH", lambda state: state)  # Terminal node

    builder.add_edge(START, "supervisor")
    builder.add_edge("FINISH", END)

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    def call_search_team(state: MessagesState) -> Command[Literal["supervisor"]]:
        response = app.invoke(
            {
                "messages": state["messages"][-1:]
            },  # Pass only the last message to the search team
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


def main(state: MessagesState):
    thread_id = str(uuid.uuid4())
    model = "openai:gpt-5.4-mini"
    config = {"configurable": {"thread_id": thread_id}}
    lead_model = init_chat_model(model=model, temperature=0)
    system_prompt = "You are an agent. Your only job is to route to a single next step. When the tasks are complete, respond with exactly the single word FINISH and nothing else."

    FINISH_TOKEN = "FINISH"

    def supervisor_node(state: MessagesState, config: RunnableConfig) -> MessagesState:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        routing_decision = lead_model.with_structured_output(RoutingDecision).invoke(
            messages, config=config
        )
        choice = routing_decision.next
        goto = "__end__" if choice == FINISH_TOKEN else choice
        return Command(goto=goto, update={"next": choice})

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor_node)
    # builder.add_node("route_next_step", route_next_step)
    builder.add_node(
        "request_team",
        request_team(state=state, model=model, config=config),
    )
    builder.add_node(
        "search_team",
        search_team(state=state, model=model, config=config),
    )
    builder.add_node("FINISH", lambda state: state)  # Terminal node

    builder.add_edge(START, "supervisor")

    # builder.add_edge("supervisor", "route_next_step")
    # builder.add_edge("request_team", "route_next_step")
    # builder.add_edge("search_team", "route_next_step")

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    messages = app.invoke(
        {"messages": state["messages"]},
        config=config,
    )

    for m in messages["messages"]:
        m.pretty_print()


topic = "latest reliable news source affecting stock market"


main(
    state=MessagesState(
        messages=[
            HumanMessage(
                content=f"Check the health of the API then provide full URLs prefixed as Google News search queries for relevant sites for {topic}."
            )
        ]
    )
)
