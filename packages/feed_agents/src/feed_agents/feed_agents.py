from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


@tool
def relevant_site(topic: str) -> str:
    (
        """Provide relevant site for topic."""
        """ Sites and topics must be appropriate for all audiences"""
    )
    return "A site relevant to that topic would be <site>"


@tool
def guess_url(site: str) -> str:
    """Guess the url"""
    return f"The url would be {site}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[relevant_site, guess_url],
    system_prompt="You are a helpful assistant",
)

topic = "unbiased reporting on current events"

messages = [
    {
        "role": "user",
        "content": f"What is a reliable site for {topic} with TLD as .com?",
    },
    {
        "role": "user",
        "content": "What is the full url after appending site: site to query in news.google.com/search",
    },
]

result = agent.invoke({"messages": messages[0]})

print(result["messages"][-1].content_blocks)
