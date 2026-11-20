"""Ingest generated PDFs into the persistent ChromaDB collection."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_and_split_documents
from src.retriever import build_vectorstore


def main() -> None:
    print("Loading and splitting PDF documents...")
    chunks = load_and_split_documents()
    print("Embedding and storing document chunks...")
    build_vectorstore(chunks)


if __name__ == "__main__":
    main()
