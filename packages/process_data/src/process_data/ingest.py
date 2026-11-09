import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import logging

# Resolve the project root
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_PERSIST_DIR = ROOT / "var" / "data" / "chroma_db_storage"
DEFAULT_COLLECTION = "langchain"  # must match the collection used at ingest
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # must match ingest

class Ingest:
    def __init__(self, source_path: str | Path, target_path: str | Path, logger_name: str | None = None, collection_name: str = DEFAULT_COLLECTION, logger: logging.Logger | None = None):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path) if target_path else DEFAULT_PERSIST_DIR

        self.logger_name = logger_name
        self.logger = logger or logging.getLogger(logger_name)

        self.target_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)

        self.collection_name = collection_name

    def read_file(self):
        try:
            with open(self.source_path, encoding="utf-8") as f:
                data = f.read()
            return data
        except FileNotFoundError:
            self.logger.info(f"File not found: {self.source_path}")
            return None
        
    def convert_to_document(self, data):
        try:
            # Convert text to LangChain Document
            documents = [Document(page_content=data, metadata={"source": self.source_path.parent.parent.name, "file_path": f"{self.source_path.as_posix()}"}),
            ]
            return documents
        except Exception as e:
            self.logger.info(f"Error converting file to document: {e}")
            return None

    def chunk_and_store(self, document, chunk_size=512, chunk_overlap=50):
        # chunk document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(document)

        doc_id = hashlib.sha256(self.source_path.as_posix().encode()).hexdigest()[:16]

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
        "doc_id": doc_id,
        "chunk_index": i,
        "chunk_count": len(chunks),
        })

        # vectorize and store in Chroma
        vector_db = Chroma.from_documents(
            chunks, self.embedding_model, persist_directory=Path(str(self.target_path)), collection_name=self.collection_name
        )

        return f"Successfully vectorized and stored {len(chunks)} chunks."


    def run(self):
        result = {"ok": None, "error": None}

        try:
            data = self.read_file()
            if data is None:
                result["error"] = f"File not found: {self.source_path}"
                return result

            document = self.convert_to_document(data)
            if document is None:
                result["error"] = f"Error converting file to document: {self.source_path}"
                return result

            stored = self.chunk_and_store(document)
            if stored is None:
                result["error"] = f"Error chunking and storing document: {self.source_path}"
                return result

        except Exception as exc:
            self.logger.exception("Failed processing %s", self.source_path)
            result["error"] = f"Unexpected error processing {self.source_path}: {exc}"
            return result

        result["ok"] = f"Successfully processed file: {self.source_path}"
        return result

if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    file_paths = ["C:/Users/japfe/Documents/genai-pipeline-project/var/data/cnn_com/document/20261105_004714.txt",
    "C:/Users/japfe/Documents/genai-pipeline-project/var/data/finance_yahoo_com/document/20261105_004716.txt"]
    for file_path in file_paths:
        target_dir = Path("var/data/chroma_db_storage")
        ingest = Ingest(source_path=file_path, target_path=target_dir, logger_name="IngestLogger")
        result = ingest.run()
        if result["ok"]:
            print(result["ok"])
        else:
            print(result["error"])

    import chromadb
    from collections import Counter

    client = chromadb.PersistentClient(path=str(target_dir))  # same str(target_path) as ingest
    for col in client.list_collections():
        name = getattr(col, "name", col)
        c = client.get_collection(name)
        metas = c.get(include=["metadatas"])["metadatas"]
        print(name, c.count(), Counter(m["file_path"] for m in metas))