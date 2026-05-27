from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict
from typing import Annotated, Literal
import operator
from IPython.display import Image, display

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


SYSTEM_PROMPT = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully."""


@tool
def relevant_site(topic: str) -> str:
    """Provide a relevant site for input topic."""
    return "A site relevant to that topic would be <site>"


@tool
def guess_url(site: str) -> str:
    """Guess the url"""
    return f"The url would be {site}"


tools = [relevant_site, guess_url]
tools_by_name = {tool.name: tool for tool in tools}
model = init_chat_model("openai:gpt-4o-mini")
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class RoutingState(TypedDict):
    messages: list[AnyMessage]
    tool_node_name: str


def classifier(state: MessagesState) -> Literal["url_thru_tld_node", "url_thru_query_node"]:
    """Classify messages for specific tool node use"""
    messages = state["messages"]
    last_message = messages[-1]
    if "TLD" in str(last_message.content):
        return "url_thru_tld_node"
    if "query" in str(last_message.content):
        return "url_thru_query_node"


def url_thru_tld_handler(state: MessagesState) -> dict:
    result = relevant_site.invoke(state["messages"][-1].content)
    return {"messages": [HumanMessage(content="TLD")]}
#[{"messages": state["messages"]}] + [{"role": "tool", "content": result}]


def url_thru_query_handler(state: MessagesState) -> dict:
    result = guess_url.invoke(state["messages"][-1].content)
    return {"messages": [HumanMessage(content="Query")]}
#[{"messages": state["messages"]}] + [{"role": "tool", "content": result}]


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
agent_builder.add_node("url_thru_tld_node", url_thru_tld_handler)
agent_builder.add_node("url_thru_query_node", url_thru_query_handler)


# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node","classifier"])
agent_builder.add_edge("tool_node", "classifier")
agent_builder.add_conditional_edges("classifier", lambda x:x, ["url_thru_tld_node","url_thru_query_node"])

# agent_builder.add_edge("url_thru_tld_node", END)
# agent_builder.add_edge("url_thru_query_node", END)
# agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
# agent_builder.add_edge("tool_node", "llm_call")

# Compile agent
agent = agent_builder.compile()

# Show agent
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


topic = (
    "unbiased reporting on current global events which would affect the stock market"
)


messages = [
    HumanMessage(content=f"What is a reliable site for {topic} with TLD as .com?"),
    HumanMessage(
        content="What is the full url after appending site: <ai_provided_site> to query in news.google.com/search"
    ),
]


# Invoke
config = {"configurable": {"thread_id": "1"}}
messages = agent.invoke({"messages": messages}, config)
# state = agent.get_state(config)
# for message in state.values.get("messages",[]):
#     print(f"Message ID: {message.id}")
#     print(f"Content: {message.content}")
