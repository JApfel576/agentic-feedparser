import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[4]  # repo root
DEFAULT_PERSIST_DIR = ROOT / "var" / "data" / "chroma_db_storage"
DEFAULT_COLLECTION = "langchain"  # must match the collection used at ingest
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # must match ingest


class VectorStoreQuery:
    def __init__(
        self,
        persist_directory: Path | str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

        load_dotenv(ROOT / ".env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.persist_directory = Path(persist_directory).resolve()
        if not self.persist_directory.exists():
            raise FileNotFoundError(
                f"Chroma directory not found: {self.persist_directory}"
            )

        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model=embedding_model, api_key=api_key),
            persist_directory=str(self.persist_directory),
        )

        count = self.store._collection.count()
        self.logger.info(
            "Opened collection=%s dir=%s count=%d",
            collection_name,
            self.persist_directory,
            count,
        )
        if count == 0:
            raise RuntimeError(
                f"Collection '{collection_name}' is empty at {self.persist_directory}; "
                "check collection name and path against ingest"
            )

    def search(
        self,
        query: str,
        k: int = 3,
        source: str | None = None,
        mmr: bool = False,
        fetch_k: int = 40,
    ) -> list[Document]:
        """Similarity or MMR search, optionally restricted to one source (e.g. 'cnn_com')."""
        flt = {"source": source} if source else None
        try:
            if mmr:
                return self.store.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, filter=flt
                )
            return self.store.similarity_search(query, k=k, filter=flt)
        except Exception:
            self.logger.exception("Search failed for query=%r", query)
            raise

    def search_per_source(
        self, query: str, sources: list[str], k_each: int = 2
    ) -> list[Document]:
        """Guarantee coverage across outlets: k_each hits from every source."""
        results: list[Document] = []
        for src in sources:
            results.extend(self.search(query, k=k_each, source=src))
        return results

    @staticmethod
    def print_results(results: list[Document]) -> None:
        for doc in results:
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vq = VectorStoreQuery()
    query = "What factors affected stocks the most today?"

    vq.print_results(vq.search(query, k=3))
