from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict
from typing import Annotated, Literal
import operator
from IPython.display import Image, display
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


SYSTEM_PROMPT = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully."""


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class RoutingState(TypedDict):
    messages: list[AnyMessage]
    tool_type: str


class SiteResponseOutput(BaseModel):
    """Structure an output for sites suggested by LLM for input topic"""

    topic: str = Field(description="Input topic")
    sites: list[str] = Field(description="Sites suggested")


search = GoogleSerperAPIWrapper()


@tool(args_schema=SiteResponseOutput)
def relevant_site(topic: str, sites: list[str]) -> list[str]:
    """Provide a relevant link for input topic using google serper"""
    results = search.results(k=5, query=topic)
    results = results["organic"]
    return {"topic": topic, "sites": [item.get("link") for item in results]}


@tool(args_schema=SiteResponseOutput)
def guess_url(topic: str, sites: list[str]) -> list[str]:
    """Prefix relevant site with google news search query url"""
    sites = relevant_site.invoke({"topic": topic, "sites": sites})
    prefix = "https://news.google.com/search?q="
    full_urls = {
        "topic": sites.get("topic"),
        "sites": [f"{prefix}site:{sites.get('sites')}"],
    }
    return full_urls


tools = [relevant_site, guess_url]
tools_by_name = {tool.name: tool for tool in tools}
model = init_chat_model("openai:gpt-4o-mini")
model_with_tools = model.bind_tools(tools)


def classifier(state: RoutingState) -> dict:
    """Classify messages to return tool type"""
    content = state["messages"][-1].content.lower()
    if "query" in content:
        return {"tool_type": "url_thru_query"}


def route_by_tool(
    state: RoutingState,
) -> Literal["url_thru_tld_node", "url_thru_query_node"]:
    return f"{state['tool_type']}_node"


def url_thru_query_handler(state: MessagesState) -> dict:
    result = guess_url.invoke(state["messages"][-1].content)
    return [{"messages": state["messages"]}] + [{"role": "tool", "content": result}]


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or not based on if LLM made tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    """If the LLM performs a tool call then perform an action"""
    if last_message.tool_calls:
        return "tool_node"

    """Otherwise we stop, (reply to the user)"""
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("classifier", classifier)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("url_thru_query_node", url_thru_query_handler)


# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, "classifier")
agent_builder.add_conditional_edges(
    "classifier", route_by_tool, ["url_thru_query_node"]
)
agent_builder.add_edge("tool_node", "llm_call")


# Compile agent
checkpointer = MemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

# Show agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Topic for message
topic = "latest reliable news source affecting stock market"


# Invoke with sequential messages instead of sending them all at once
config = {"configurable": {"thread_id": "1"}}

# List for agent invocation input
agent_invoke_list = [
    {
        "message_content": f"What is a reliable site for {topic} through TLD as .com? What is the full URL after appending it as search query to Google News URL",
        "config": config,
    },
]


def agent_invoke_message(item: list[dict]) -> MessagesState:
    """Invoke agent to execute messages synchronously"""
    messages = agent.invoke(
        {"messages": [HumanMessage(content=item["message_content"])]},
        item["config"],
    )
    return messages


for item in agent_invoke_list:
    messages = agent_invoke_message(item)

for m in messages["messages"]:
    m.pretty_print()
