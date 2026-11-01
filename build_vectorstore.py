"""
Ingest PDF documents into the ChromaDB vector store.

Run this script *after* generate_employees.py and generate_documents.py:

    python build_vectorstore.py

It loads every PDF from data/documents/, splits them into chunks,
embeds them with Google Generative AI, and persists the vectors
in the chroma_db/ directory.
"""

from rag_engine import load_and_split_documents, build_vectorstore


def main():
    print("=" * 60)
    print(" Building ChromaDB Vector Store")
    print("=" * 60)
    print("\n📂 Loading & splitting documents...")
    chunks = load_and_split_documents()

    print("\n📦 Embedding & storing in ChromaDB...")
    vectorstore = build_vectorstore(chunks)

    count = vectorstore._collection.count()
    print(f"\n{'=' * 60}")
    print(f" ✅ Done — {count} vectors stored in chroma_db/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
