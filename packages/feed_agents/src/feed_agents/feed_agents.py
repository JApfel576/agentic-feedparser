from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from langgraph.types import Command
import requests
from base_agents import DefaultAgent, make_supervisor_node, MessagesState

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
    poll = "/poll"


class APIHealthEndpoint(BaseModel):
    """Structured output for API health endpoint check"""

    endpoint: str = Field(description="Endpoint for checking API health")
    status: str = Field(description="Status of the API health endpoint")
    message: str = Field(
        description="Message providing details from the endpoint about API health"
    )


search = GoogleSerperAPIWrapper()
api_host = "http://127.0.0.1:8000"


def build_api_url(endpoint: Endpoint) -> str:
    return f"{api_host}{endpoint.value}"


@tool(args_schema=SiteResponseOutput)
def relevant_site(topic: str, sites: list[str]) -> dict:
    """Provide a relevant link for input topic using google serper"""
    results = search.results(k=5, query=topic)
    results = results["organic"]
    sites_out = [item.get("link") for item in results]
    output = SiteResponseOutput(topic=topic, sites=sites_out)
    return output.model_dump()


@tool(args_schema=SiteResponseOutput)
def guess_url(topic: str, sites: list[str]) -> dict:
    """Prefix relevant site with google news search query url"""
    sites_res = relevant_site.invoke({"topic": topic, "sites": sites})
    prefix = "https://news.google.com/search?q="
    full_sites = [f"{prefix}site:{s}" for s in sites_res.get("sites", [])]
    output = GuessURLOutput(topic=sites_res.get("topic", topic), sites=full_sites)
    return output.model_dump()


@tool(args_schema=APIHealthEndpoint)
def check_api_health(endpoint: str, status: str, message: str) -> dict:
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


def search_node(state: MessagesState):
    tools = [relevant_site, guess_url]
    search_agent = DefaultAgent(
        state=state,
        model=model,
        tools=tools,
        tool_choice=None,
        schema=GuessURLOutput,
        config=config,
    ).graph.invoke(state)
    search_agent_last = search_agent["messages"][-1]
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=f"Research completed successfully. Data: {search_agent_last.model_dump_json()}",
                    name="search",
                )
            ]
        },
        goto="supervisor",
    )


def api_request_node(state: MessagesState):
    tools = [check_api_health]
    system_prompt = """You are an agent tasked with checking the health of an API. Use the check_api_health tool to make a GET request to the API health endpoint and return the status and message. Ensure that you handle any errors gracefully and provide a clear response."""
    api_agent = DefaultAgent(
        state=state,
        model=model,
        tools=tools,
        tool_choice=None,
        schema=APIHealthEndpoint,
        config=config,
        system_prompt=system_prompt,
    ).graph.invoke(state, config={"configurable": {"thread_id": "2"}})
    api_agent_last = api_agent["messages"][-1]
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=f"API request completed successfully. Data: {api_agent_last.model_dump_json()}",
                    name="api",
                )
            ]
        },
        goto="supervisor",
    )


config = {"configurable": {"thread_id": "1"}}
model = "openai:gpt-4o-mini"


def main():
    search_supervisor_node = make_supervisor_node(
        init_chat_model(model), ["search", "api"], config=config
    )

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", search_supervisor_node)
    builder.add_node("search", search_node)
    builder.add_node("api", api_request_node)
    builder.add_node("FINISH", lambda state: state)  # Terminal node

    builder.add_edge(START, "supervisor")
    builder.add_edge("search", "supervisor")
    builder.add_edge("api", "supervisor")
    builder.add_edge("FINISH", END)

    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    topic = "latest reliable news source affecting stock market"
    messages = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a GET request to check the health of the API and return the status and message."
                )
            ]
        },
        config=config,
    )

    for m in messages["messages"]:
        m.pretty_print()

    # for s in app.stream({
    #         "messages": [
    #             HumanMessage(
    #                 content=f"Make a GET request using check_api_health tool to check the health of the API and return the status and message."
    #             )
    #         ]
    #     }, config=config):
    #     print(s)
    #     print("---")


main()

# If it is healthy, then search for a reliable site for {topic} through TLD as .com and return the full URL after appending it as a search query to Google News URL. If it is not healthy, return the status and message from the API health check.
