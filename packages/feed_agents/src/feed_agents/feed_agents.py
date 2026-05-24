from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


@tool
def relevant_site(topic: str) -> str:
    """Provide a relevant site for a topic."""
    return "A site relevant to that topic would be <site>"


@tool
def guess_url(site: str) -> str:
    """Guess the url"""
    return f"The url would be {site}"


SYSTEM_PROMPT = """You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully."""


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[relevant_site, guess_url],
    system_prompt=SYSTEM_PROMPT,
    # interrupt_before=["guess_url"],
    checkpointer=InMemorySaver(),
)

# All decisions approve, reject, edit, and respond allowed

topic = (
    "unbiased reporting on current global events which would affect the stock market"
)

messages = [
    {
        "role": "user",
        "content": f"What is a reliable site for {topic} with TLD as .com?",
    },
    {
        "role": "user",
        "content": "What is the full url after appending site: <ai_provided_site> to query in news.google.com/search",
    },
]

# result = agent.invoke({"messages": messages},
#                        {"configurable":{"thread_id":"1"}})

# print(result["messages"][-1].content_blocks)

from langchain.messages import AIMessage, HumanMessage

for chunk in agent.stream(
    {"messages": messages}, stream_mode="values", config={"thread_id": "1"}
):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
