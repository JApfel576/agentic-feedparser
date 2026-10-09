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
# ROOT = Path(__file__).resolve().parents[4]
# load_dotenv(ROOT / ".env")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ROOT))

# DB_DIR = os.path.join(PROJECT_ROOT, "var", "data", "chroma_db_storage")

class Ingest:
    def __init__(self, source_path: str, target_path: str, logger_name: str | None = None, logger: logging.Logger | None = None):
        self.filename = os.path.basename(source_path)
        self.source_path = source_path
        self.target_path = target_path

        self.logger_name = logger_name
        self.logger = logger or logging.getLogger(logger_name)

        self.source_path.parent.mkdir(parents=True, exist_ok=True)
        self.target_path.parent.mkdir(parents=True, exist_ok=True)

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
            document = [Document(page_content=data, metadata={"source": self.filename})]
            return document
        except Exception as e:
            self.logger.info(f"Error converting file to document: {e}")
            return None

    def chunk_and_store(self, document, chunk_size=512, chunk_overlap=50):
        # chunk document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(document)

        # vectorize and store in Chroma
        # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        vector_db = Chroma.from_documents(
            chunks, embedding_model, persist_directory=Path(self.target_path)
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
            logger.exception("Failed processing %s", self.source_path)
            result["error"] = f"Unexpected error processing {self.source_path}: {exc}"
            return result

        result["ok"] = f"Successfully processed file: {self.source_path}"
        return result