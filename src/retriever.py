"""Embeddings, ChromaDB storage, and hybrid retrieval."""

import logging
import os
from typing import Any

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv()
logger = logging.getLogger(__name__)


class SafeGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """Convert embedding inputs to strings before sending them to Google."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(str(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([str(text) for text in texts])


def get_embeddings() -> SafeGoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    return SafeGoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Replace the persistent collection with the supplied document chunks."""
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    print(f"Stored {vectorstore._collection.count()} vectors in {CHROMA_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Open the existing persistent collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings() if os.getenv("GOOGLE_API_KEY") else None,
    )


class RRFRetriever(BaseRetriever):
    """Reciprocal Rank Fusion retriever over two underlying retrievers."""

    retriever_a: Any
    retriever_b: Any
    k: int = 60
    semantic_weight: float = 0.5

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        try:
            results_a = list(self.retriever_a.invoke(query))
        except Exception as exc:
            logger.warning("Semantic retrieval failed; using BM25 results: %s", exc)
            results_a = []
        results_b = list(self.retriever_b.invoke(query))
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for results, weight in (
            (results_a, self.semantic_weight),
            (results_b, 1 - self.semantic_weight),
        ):
            for rank, doc in enumerate(results):
                key = doc.page_content
                scores[key] = scores.get(key, 0) + weight / (self.k + rank + 1)
                docs[key] = doc
        return [docs[key] for key in sorted(scores, key=scores.__getitem__, reverse=True)]


def create_hybrid_retriever(
    vectorstore: Chroma,
    k: int = 5,
    semantic_weight: float = 0.5,
) -> BaseRetriever:
    """Combine semantic ChromaDB retrieval with BM25 keyword retrieval via RRF."""
    stored = vectorstore._collection.get(include=["documents", "metadatas"])

    if not stored["documents"]:
        raise ValueError("Vector store is empty. Run `python scripts/build_vectorstore.py` first.")

    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(stored["documents"], stored["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # Offline fallback: if no API key or vector retriever fails, return BM25 alone
    if not os.getenv("GOOGLE_API_KEY"):
        return bm25_retriever

    try:
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
    except Exception:
        return bm25_retriever

    return RRFRetriever(
        retriever_a=vector_retriever,
        retriever_b=bm25_retriever,
        semantic_weight=semantic_weight,
    )
