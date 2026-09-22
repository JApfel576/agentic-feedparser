import json
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Resolve the project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[0]
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

filename = "20260918_001730.txt"
path = "var/data/Anasdaq_com/"

source_path = Path(path) / "document" / filename
target_path = Path(path) / "chroma_db_storage"

# data
document_path = Path(source_path)
with open(document_path, encoding="utf-8") as f:
    data = f.read()

# Convert text to LangChain Document
document = [Document(page_content=data, metadata={"source": filename})]


# chunk document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_documents(document)

# vectorize and store in Chroma
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma.from_documents(
    chunks, embedding_model, persist_directory=Path(target_path)
)

print(f"Successfully vectorized and stored {len(chunks)} chunks.")
