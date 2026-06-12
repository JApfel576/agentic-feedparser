from pathlib import Path
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model, BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import AnyMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict
from typing import Annotated
from collections.abc import Callable
import operator
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command


# Resolve the project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[0]
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


SYSTEM_PROMPT = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully."""


class SiteResponseOutput(BaseModel):
    """Structure an output for sites suggested by LLM for input topic"""

    topic: str = Field(description="Input topic")
    sites: list[str] = Field(description="Sites suggested")


search = GoogleSerperAPIWrapper()


@tool(args_schema=SiteResponseOutput)
def relevant_site(topic: str, sites: list[str]) -> dict:
    """Provide a relevant link for input topic using google serper"""
    results = search.results(k=5, query=topic)
    results = results["organic"]
    return {"topic": topic, "sites": [item.get("link") for item in results]}


@tool(args_schema=SiteResponseOutput, return_direct=True, response_format="content")
def guess_url(topic: str, sites: list[str]) -> dict:
    """Prefix relevant site with google news search query url"""
    sites = relevant_site.invoke({"topic": topic, "sites": sites})
    prefix = "https://news.google.com/search?q="
    full_urls = {
        "topic": sites.get("topic"),
        "sites": [f"{prefix}site:{s}" for s in sites.get("sites")],
    }
    return full_urls


tools = [relevant_site, guess_url]


# Agent State
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class DefaultAgent:
    def __init__(self, state: MessagesState, model: str, tools: list):
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_agent)
        workflow.add_node("tools", ToolNode(tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        checkpointer = MemorySaver()
        self.graph = workflow.compile(checkpointer=checkpointer)

        self.llm = init_chat_model(model=model, temperature=0)
        self.llm_with_tools = self.llm.bind_tools(tools=tools)
        self.system_prompt = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully."""

    def call_agent(self, state: MessagesState) -> dict:
        return {
            "messages": [
                self.llm_with_tools.invoke(
                    [SystemMessage(content=self.system_prompt)] + state["messages"],
                    config={"configurable": {"thread_id": "1"}},
                )
            ]
        }


def make_supervisor_node(
    llm: BaseChatModel, members: list[str]
) -> Callable[[MessagesState], Command[str]]:
    options = ["FINISH"] + members
    system_prompt = f"You are a supervisor tasked with managing a conversation between the following workers: {members}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH."

    class Router(TypedDict):
        """Worker to route to next. If no workers needed route to FINISH"""

        next: list[str] = options

    def supervisor_node(state: MessagesState) -> Command[str]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.with_structured_output(Router).invoke(
            messages, config={"configurable": {"thread_id": "1"}}
        )
        choice = response["next"]
        if choice == "FINISH":
            goto = END
        else:
            goto = choice
        return Command(goto=goto, update={"next": choice})

    return supervisor_node


# TO DO update to return search node agent to supervisor
def search_node(state: MessagesState):
    search_agent = DefaultAgent(state=state, model=model, tools=tools).graph.invoke(
        state
    )
    search_agent_message = search_agent["messages"][-1].content
    return Command(
        update={
            "messages": [HumanMessage(content=search_agent_message, name="search")]
        },
        goto="supervisor",
    )


model = "openai:gpt-4o-mini"
search_supervisor_node = make_supervisor_node(init_chat_model(model), ["search"])


builder = StateGraph(MessagesState)
builder.add_node("supervisor", search_supervisor_node)
builder.add_node("search", search_node)

builder.add_edge(START, "supervisor")
builder.add_edge("search", "supervisor")

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

# Topic for message
topic = "latest reliable news source affecting stock market"
messages = app.invoke(
    {
        "messages": [
            HumanMessage(
                content=f"What is a reliable site for {topic} through TLD as .com? What is the full URL after appending it as search query to Google News URL?"
            )
        ]
    },
    {"configurable": {"thread_id": "1"}},
)

for m in messages["messages"]:
    m.pretty_print()
