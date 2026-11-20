"""Load and split PDF policy documents."""

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DOCS_DIR
from .database import KNOWLEDGE_SEED


def load_and_split_documents(docs_dir: str = DOCS_DIR) -> list[Document]:
    """Load PDFs from the document directory and split them into chunks."""
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        try:
            pages = PyPDFLoader(os.path.join(docs_dir, filename)).load()
            for page in pages:
                page.metadata["source"] = filename
            documents.extend(pages)
            print(f"Loaded {filename} ({len(pages)} pages)")
        except Exception as exc:
            print(f"Failed to load {filename}: {exc}")

    if not documents:
        raise FileNotFoundError(f"No PDF documents found in {docs_dir}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    operational_facts = [
        Document(
            page_content=f"{title}: {content}",
            metadata={"source": f"company_guide:{title}", "key": key},
        )
        for key, title, content in KNOWLEDGE_SEED
    ]
    chunks.extend(operational_facts)
    print(
        f"Split {len(documents)} pages into {len(chunks) - len(operational_facts)} chunks "
        f"and added {len(operational_facts)} operational facts"
    )
    return chunks
