from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Resolve the project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[0]
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Same embedding model used during storage
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the specific Chroma vector store
vector_store = Chroma(
    persist_directory="var/data/Anasdaq_com/chroma_db_storage",
    embedding_function=embedding_model,
)

results = vector_store.similarity_search(
    "Where can I find the data feed for more info from Nasdaq?", k=3
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")
