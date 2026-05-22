from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


@tool
def relevant_site(topic: str) -> str:
    (
        """Provide a relevant site for a topic."""
    )
    return "A site relevant to that topic would be <site>"


@tool
def guess_url(site: str) -> str:
    """Guess the url"""
    return f"The url would be {site}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[relevant_site, guess_url],
    system_prompt="You are a helpful assistant that keeps messages brief purely for answering user questions",
    checkpointer = InMemorySaver()
)

topic = "unbiased reporting on current global events"

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

result = agent.invoke({"messages": messages},
                       {"configurable":{"thread_id":"1"}})

print(result["messages"][-1].content_blocks)
