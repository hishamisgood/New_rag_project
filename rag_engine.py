"""
RAG Engine — Hybrid search (Semantic + BM25) with Gemini LLM.

Components
----------
- ChromaDB           : persistent vector store (semantic search)
- BM25Retriever      : keyword / lexical search
- EnsembleRetriever  : combines both retrievers with configurable weights
- ChatGoogleGenerativeAI : Gemini model for answer generation
- ConversationalRetrievalChain : manages chat history + retrieval + generation
"""

import os
from typing import Optional

from dotenv import load_dotenv

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

CHROMA_DIR = "chroma_db"
DOCS_DIR = os.path.join("data", "documents")
COLLECTION_NAME = "novacorp_docs"

# ── Embedding model ──────────────────────────────────────────────────────────

class SafeGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """Ensures input queries are always cast to plain str before sending to Google API."""
    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(str(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([str(t) for t in texts])


def get_embeddings() -> SafeGoogleGenerativeAIEmbeddings:
    return SafeGoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ── Document loading & chunking ─────────────────────────────────────────────

def load_and_split_documents(docs_dir: str = DOCS_DIR) -> list[Document]:
    """Load all PDFs from *docs_dir*, split into overlapping chunks."""
    documents: list[Document] = []

    for filename in sorted(os.listdir(docs_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(docs_dir, filename)
        try:
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = filename
            documents.extend(pages)
            print(f"  📄 Loaded {filename} ({len(pages)} pages)")
        except Exception as e:
            print(f"  ⚠️  Failed to load {filename}: {e}")

    if not documents:
        raise FileNotFoundError(
            f"No PDF documents found in {docs_dir}. "
            "Run generate_documents.py first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"\n  🔪 Split into {len(chunks)} chunks "
          f"(from {len(documents)} pages)")
    return chunks


# ── Build / load the vector store ────────────────────────────────────────────

def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Create (or overwrite) the ChromaDB vector store on disk."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"  💾 ChromaDB persisted → {CHROMA_DIR}/ "
          f"({vectorstore._collection.count()} vectors)")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB from disk."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


# ── Hybrid retriever ────────────────────────────────────────────────────────

def create_hybrid_retriever(
    vectorstore: Chroma,
    k: int = 5,
    semantic_weight: float = 0.5,
) -> EnsembleRetriever:
    """Combine ChromaDB (semantic) + BM25 (keyword) retrievers."""

    # Semantic retriever
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # BM25 retriever — needs the raw documents
    collection = vectorstore._collection
    result = collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(result["documents"], result["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # Ensemble
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[semantic_weight, 1 - semantic_weight],
    )
    print(f"  🔍 Hybrid retriever ready  "
          f"(semantic={semantic_weight:.0%}, keyword={1-semantic_weight:.0%}, k={k})")
    return ensemble


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm(
    model: str = "models/gemini-2.5-flash",
    temperature: float = 0.3,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=True,
    )


# ── Conversational RAG chain ────────────────────────────────────────────────

def build_rag_chain(
    retriever: EnsembleRetriever,
    employee_info: Optional[dict] = None,
    model: str = "models/gemini-2.5-flash",
) -> ConversationalRetrievalChain:
    """
    Create a conversational chain that answers using retrieved docs
    and (optionally) the logged-in employee's profile.
    """
    llm = get_llm(model=model)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=10,
    )

    # Build a system preamble with employee context
    preamble = (
        "You are NovaCorp AI Assistant, a helpful and friendly chatbot for "
        "NovaCorp Industries employees. Answer questions using ONLY the "
        "retrieved context and the employee's profile below. If the answer "
        "isn't in the context, say so honestly.\n\n"
    )
    if employee_info:
        months = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
        month_names = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        absence_str = ", ".join(
            f"{mn}: {employee_info.get(f'absent_days_{m}', 0)} days"
            for m, mn in zip(months, month_names)
        )
        preamble += (
            "── EMPLOYEE PROFILE ──\n"
            f"ID: {employee_info.get('id', 'N/A')}\n"
            f"Name: {employee_info.get('first_name', '')} {employee_info.get('last_name', '')}\n"
            f"Sex: {employee_info.get('sex', 'N/A')}\n"
            f"Department: {employee_info.get('department', 'N/A')}\n"
            f"Role: {employee_info.get('role', 'N/A')}\n"
            f"Years of Experience: {employee_info.get('years_of_expertise', 'N/A')}\n"
            f"Salary: ${employee_info.get('salary', 0):,.2f}\n"
            f"Sick Leave Days: {employee_info.get('sick_leave_days', 0)}\n"
            f"Maternity Leave Days: {employee_info.get('maternity_leave_days', 0)}\n"
            f"Paternity Leave Days: {employee_info.get('paternity_leave_days', 0)}\n"
            f"Paid Time Off Days: {employee_info.get('paid_time_off_days', 0)}\n"
            f"Mental Health Leave Days: {employee_info.get('mental_health_leave_days', 0)}\n"
            f"Monthly Absence: {absence_str}\n"
            "──────────────────────\n\n"
        )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt": _build_qa_prompt(preamble),
        },
    )
    return chain


def _build_qa_prompt(preamble: str):
    """Build the QA prompt template with the system preamble."""
    from langchain_core.prompts import PromptTemplate

    template = (
        preamble
        + "Context from company documents:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Provide a helpful, accurate answer. If the question is about the "
        "employee's personal information (salary, leave, department, etc.), "
        "use the Employee Profile above. For company policies and general "
        "information, use the retrieved context. Cite the source document "
        "when possible."
    )
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )


# ── Convenience wrapper ─────────────────────────────────────────────────────

def query_rag(
    chain: ConversationalRetrievalChain,
    question: str,
) -> dict:
    """
    Ask a question and return ``{"answer": str, "sources": list[str]}``.
    """
    result = chain.invoke({"question": question})
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in result.get("source_documents", [])
    })
    return {
        "answer": result.get("answer", ""),
        "sources": sorted(sources),
    }
