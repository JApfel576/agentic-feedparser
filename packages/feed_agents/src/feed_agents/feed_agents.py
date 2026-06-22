from pathlib import Path
from urllib import response
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model, BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated, Literal
from collections.abc import Callable
import operator
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field, HttpUrl
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command


# Resolve the project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


SYSTEM_PROMPT = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully."""


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

class APIHealthUrl(BaseModel):
    """Structured output for API health check"""
    url: str = Field(description="URL to check API health")
    status: str = Field(
        description="Status of the API health check"
    )
    message: str = Field(description="Message providing details about the health check")


search = GoogleSerperAPIWrapper()


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

@tool(args_schema=APIHealthUrl)
def check_api_health(url: str, status: str, message: str) -> dict:
    """Check health of API by making a request and returning status"""
    import requests
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            return APIHealthUrl(url=url, status=response.text, message="API is healthy").model_dump()
        else:
            return APIHealthUrl(url=url, status=response.text, message=f"API returned status code {response.status_code}").model_dump()
    except Exception as e:
        return APIHealthUrl(url=url, status=response.text, message=str(e)).model_dump()


tools = [relevant_site, guess_url, check_api_health]
config = {"configurable": {"thread_id": "1"}}
api_health_url = "http://127.0.0.1:8000/health"


# Agent State
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class DefaultAgent:
    def __init__(
        self,
        state: MessagesState,
        model: str,
        tools: list,
        schema: BaseModel,
        config: dict,
    ):

        self.state = state
        self.model = model
        self.tools = tools
        self.schema = schema
        self.config = config

        self.workflow = StateGraph(MessagesState)
        self.setup_graph()
        self.checkpointer = MemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.checkpointer)

        self.llm = init_chat_model(model=model, temperature=0)
        self.llm_with_tools = self.llm.bind_tools(tools=tools)
        self.system_prompt = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully."""

    def setup_graph(self):
        self.workflow.add_node("agent", self.call_agent)
        self.workflow.add_node("tools", ToolNode(tools))
        self.workflow.add_node("structured_agent", self.call_structured_agent)

        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: "structured_agent"}
        )
        self.workflow.add_edge("tools", "agent")
        self.workflow.add_edge("structured_agent", END)

    def call_agent(self, state: MessagesState) -> dict:
        return {
            "messages": [
                self.llm_with_tools.invoke(
                    [SystemMessage(content=self.system_prompt)] + state["messages"],
                    config=self.config,
                )
            ]
        }

    def call_structured_agent(self, state: MessagesState) -> dict:
        messages = state["messages"]
        last_message = messages[-1]
        return {
            "messages": [
                self.llm.with_structured_output(
                    self.schema, strict=True, include_raw=False
                ).invoke(
                    last_message.content,
                    config=self.config,
                )
            ]
        }


def make_supervisor_node(
    llm: BaseChatModel, members: list[str], config: dict
) -> Callable[[MessagesState], Command[str]]:
    options = ",".join(["FINISH"] + members)
    system_prompt = f"You are a supervisor tasked with managing a conversation between the following workers: {options}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH."

    
    class Router(TypedDict):
        """Worker to route to next. If no workers needed route to FINISH"""

        next: str = Field(description=f"Next worker to route to. Options: {options}")


    def supervisor_node(state: MessagesState) -> Command[str]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages, config=config)
        # print(f"Supervisor response: {response}")
        choice = response["next"]
        if choice == "FINISH":
            goto = END
        else:
            goto = choice
        return Command(goto=goto, update={"next": choice})

    return supervisor_node


def search_node(state: MessagesState):
    search_agent = DefaultAgent(
        state=state, model=model, tools=tools, schema=GuessURLOutput, config=config
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

def api_node(state: MessagesState):
    api_agent = DefaultAgent(
        state=state, model=model, tools=tools, schema=APIHealthUrl, config={"configurable": {"thread_id": "2"}}
    ).graph.invoke(state)
    api_agent_last = api_agent["messages"][-1]
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=f"API call completed successfully. Data: {api_agent_last.model_dump_json()}",
                    name="api",
                )
            ]
        },
        goto="supervisor",
    )


model = "openai:gpt-4o-mini"
search_supervisor_node = make_supervisor_node(
    init_chat_model(model), ["search", "api"], config=config
)


builder = StateGraph(MessagesState)
builder.add_node("supervisor", search_supervisor_node)
builder.add_node("search", search_node)
builder.add_node("api", api_node)
builder.add_node("FINISH", lambda state: state)  # Terminal node

builder.add_edge(START, "supervisor")
builder.add_edge("search", "supervisor")
builder.add_edge("api", "supervisor")
builder.add_edge("FINISH", END)


checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)


topic = "latest reliable news source affecting stock market"
# messages = app.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content=f"What is a reliable site for {topic} through TLD as .com? What is the full URL after appending it as search query to Google News URL?"
#             )
#         ]
#     },
#     config=config,
# )

# for m in messages["messages"]:
#     m.pretty_print()

for s in app.stream({
        "messages": [
            HumanMessage(
                content=f"What is a reliable site for {topic} through TLD as .com? What is the full URL after appending it as search query to Google News URL?"
            )
        ]
    }, config=config):
    print(s)
    print("---")
