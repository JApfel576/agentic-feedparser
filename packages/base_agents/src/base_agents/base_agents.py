from langchain.chat_models import init_chat_model, BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import AnyMessage, SystemMessage
from typing_extensions import TypedDict
from typing import Annotated, Literal
from collections.abc import Callable
import operator
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command


# Agent State
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    next: str  # last-value-wins routing signal written by make_supervisor_node


class DefaultAgent:
    def __init__(
        self,
        state: MessagesState,
        model: BaseChatModel | None,
        tools: list,
        schema: BaseModel,
        config: dict,
        system_prompt: str = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.""",
    ):
        self.model = model
        self.tools = tools
        self.schema = schema
        self.config = config
        self.system_prompt = system_prompt

        self.workflow = StateGraph(MessagesState)
        self.setup_graph()
        self.checkpointer = MemorySaver()
        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        self.llm_with_tools = self.model.bind_tools(tools=self.tools)

    def setup_graph(self):
        self.workflow.add_node("agent", self.call_agent)
        self.workflow.add_node("tools", ToolNode(self.tools))
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
        return {
            "messages": [
                self.model.with_structured_output(
                    self.schema, strict=True, include_raw=False
                ).invoke(
                    messages,
                    config=self.config,
                )
            ]
        }


def make_supervisor_node(
    llm: BaseChatModel,
    members: list[str],
    config: dict,
    additional_instructions: str = "",
) -> Callable[[MessagesState], Command[str]]:
    options = ["FINISH"] + members
    options_str = ",".join(options)
    default_prompt = f"You are a supervisor tasked with managing a conversation between the following workers: {options_str}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH."
    if additional_instructions:
        system_prompt = f"{default_prompt}\n\nAdditional instructions: {additional_instructions.strip()}"
    else:
        system_prompt = default_prompt

    class Router(BaseModel):
        """Worker to route to next. If no workers needed route to FINISH"""

        next: Literal[tuple(options)] = Field(
            description=f"Next worker to route to. Options: {options_str}"
        )  # type: ignore

    def supervisor_node(state: MessagesState) -> Command[str]:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages, config=config)
        choice = response.next
        if choice == "FINISH":
            goto = "__end__"
        else:
            goto = choice
        return Command(goto=goto, update={"next": choice})

    return supervisor_node
