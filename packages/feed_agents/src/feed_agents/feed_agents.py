from dotenv import load_dotenv
import os
from langchain.agents import create_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def guess_url(site: str) -> str:
    """Guess the url"""
    return f"The url would be {site}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[guess_url],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the full url after appending site: example.com to query in news.google.com/search",
            }
        ]
    }
)

print(result["messages"][-1].content_blocks)
