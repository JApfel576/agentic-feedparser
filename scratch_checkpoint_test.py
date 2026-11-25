import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from feed_agents.feed_agents import build_top_graph

config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 40}

graph = build_top_graph(
    model_str="openai:gpt-5.4-mini",
    config=config,
    checkpointer=InMemorySaver(),   # only works if build_top_graph accepts this
)

payload = {"messages": [HumanMessage(content="...same prompt as run()...")]}

# 1. run until the first interrupt
for _ in graph.stream(payload, config, stream_mode="updates"):
    pass
print("paused at:", graph.get_state(config).next)   # non-empty = paused at interrupt

# 2. resume the same thread in a separate stream call
for _ in graph.stream(Command(resume="yes"), config, stream_mode="updates"):
    pass
print("after resume:", graph.get_state(config).next)