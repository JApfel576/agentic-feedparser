from langchain.chat_models import init_chat_model, BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
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
    current_instruction: str | None
    supervisor_turns: dict[str, int]  # per-team supervisor turn counter, keyed by team
    # supervisor_node is annotated `state: MessagesState`, and LangGraph uses that annotation
    # as the node's input schema — any key absent here is filtered out of the state the node
    # sees. dispatched_agents_run MUST be declared here or the supervisor reads it as empty
    # every turn and its "all members have run" exit condition can never fire.
    dispatched_agents_run: list[str]
    # Same reason: the supervisor namespaces its turn budget by the parent's dispatch id, so
    # that id has to be visible through this schema or every dispatch shares one budget.
    _dispatched_id: str


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
        # self.checkpointer = MemorySaver()
        self.graph = self.workflow.compile() #checkpointer=self.checkpointer
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
    team_name: str = "",
) -> Callable[[MessagesState], Command[str]]:
    options_str = ",".join(["FINISH"] + members)
    default_prompt = f"You are a supervisor tasked with managing a conversation between the following workers: {options_str}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH."
    if additional_instructions:
        system_prompt = f"{default_prompt}\n\nAdditional instructions: {additional_instructions.strip()}"
    else:
        system_prompt = default_prompt

    # `supervisor_turns` is a channel shared by every team in the parent graph, so the turn
    # counter has to be namespaced or one team's dispatches would exhaust another's budget.
    team_label = team_name or ",".join(members)
    max_turns = len(members) + 1

    def _router_for(choices: list[str]) -> type[BaseModel]:
        """Router restricted to the workers that have not run yet — re-picking a
        completed worker is then structurally impossible, not merely discouraged."""

        class Router(BaseModel):
            """Worker to route to next. If no workers needed route to FINISH"""

            next: Literal[tuple(choices)] = Field(
                description=f"Next worker to route to. Options: {','.join(choices)}"
            )  # type: ignore

        return Router

    def supervisor_node(state: MessagesState) -> Command[str]:
        instruction = state.get("current_instruction")
        if not instruction:
            raise ValueError(
            "supervisor_node requires current_instruction in state — "
            "got none. Check that the dispatching graph sets it before entering this subgraph."
        )
        # Filter to this team's roster: the parent-level list also carries other teams' workers.
        run_so_far = [
            m for m in (state.get("dispatched_agents_run") or []) if m in members
        ]
        remaining = [m for m in members if m not in run_so_far]

        # max_turns bounds ONE dispatch, so the counter must be scoped to one dispatch too.
        # Keyed by team alone it is cumulative for the whole run: a team dispatched twice by
        # the parent starts its second dispatch with the first dispatch's turns already spent,
        # trips `turn > max_turns` immediately, and FINISHes without dispatching anyone —
        # which silently skips that member's downstream nodes (e.g. a human_approval interrupt).
        turns = dict(state.get("supervisor_turns") or {})
        turns_key = f"{team_label}:{state.get('_dispatched_id') or ''}"
        turn = turns.get(turns_key, 0) + 1
        turns[turns_key] = turn

        print(
            f"INNER SUPERVISOR[{turns_key}] turn={turn} ran={run_so_far} "
            f"remaining={remaining}",
            flush=True,
        )

        if not remaining:
            return Command(
                goto="__end__", update={"next": "FINISH", "supervisor_turns": turns}
            )

        if turn > max_turns:
            return Command(
                goto="__end__",
                update={
                    "next": "FINISH",
                    "supervisor_turns": turns,
                    "messages": [
                        AIMessage(
                            content=(
                                f"Supervisor for {team_label} gave up after {turn} turns "
                                f"without dispatching: {remaining}."
                            ),
                            name="supervisor",
                        )
                    ],
                },
            )

        if len(remaining) == 1:
            choice = remaining[0]  # nothing to decide — skip the LLM call entirely
        else:
            choices = remaining + ["FINISH"]
            scoped_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"{instruction}\n\n"
                        f"Already completed: {run_so_far or 'none'}. "
                        f"Still to do: {remaining}."
                    )
                ),
            ]
            response = llm.with_structured_output(_router_for(choices)).invoke(
                scoped_messages, config=config
            )
            choice = response.next

        goto = "__end__" if choice == "FINISH" else choice
        return Command(goto=goto, update={"next": choice, "supervisor_turns": turns})

    return supervisor_node
