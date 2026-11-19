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
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

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
from langchain_ollama import ChatOllama

from auth import DB_PATH

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
    """Create (or overwrite) the ChromaDB vector store on disk.

    FIX D: Chroma.from_documents() appends to an existing collection, so
    running this script twice doubled all vectors. We now explicitly clear
    the collection before ingesting so re-runs are idempotent.
    """
    embeddings = get_embeddings()
    # Open (or create) the existing store and wipe it before re-ingesting.
    existing = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    existing.reset_collection()
    existing.add_documents(chunks)
    print(f"  💾 ChromaDB persisted → {CHROMA_DIR}/ "
          f"({existing._collection.count()} vectors)")
    return existing


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
) -> Any:
    """Prefer Gemini, with local Ollama llama3.2 as a transparent fallback."""
    ollama = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=temperature)
    if not os.getenv("GOOGLE_API_KEY"):
        return ollama
    gemini = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=True,
    )
    # Network/quota failures from Gemini then continue with a locally running
    # Ollama model.  The same chain API is preserved for the UI.
    return gemini.with_fallbacks([ollama])


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


@dataclass
class ToolResult:
    """A result returned by one of the RAG agent's explicit tools."""
    name: str
    documents: list[Document]
    detail: str | None = None


class LLMGeneratedSQLTool:
    """Generate and safely execute a read-only SQLite query for the RAG agent."""
    name = "llm_sql_database"
    employee_columns = {
        "id", "first_name", "last_name", "sex", "department", "role", "salary",
        "years_of_expertise", "sick_leave_days", "paid_time_off_days",
        "mental_health_leave_days", "maternity_leave_days", "paternity_leave_days",
        "absent_days_jan", "absent_days_feb", "absent_days_mar", "absent_days_apr",
        "absent_days_may", "absent_days_jun", "absent_days_jul", "absent_days_aug",
        "absent_days_sep", "absent_days_oct", "absent_days_nov", "absent_days_dec",
    }
    knowledge_columns = {"key", "title", "content"}

    def __init__(self, llm: Any):
        self.llm = llm

    def generate_sql(self, question: str) -> str:
        prompt = f"""Generate exactly one SQLite SELECT query for this employee-portal question.
Return SQL only: no Markdown, explanation, comments, or semicolon.

Allowed tables and columns:
- employees: {', '.join(sorted(self.employee_columns))}
- rag_knowledge: key, title, content

Rules:
- Use only one table, never JOIN, UNION, subqueries, or SELECT *.
- For every employees query, use exactly `WHERE id = :employee_id`.
- Never query password_hash or any table not listed.
- For rag_knowledge searches, use LIKE with the user-facing terms and LIMIT 3.
- Do not write data or use PRAGMA.

Question: {question}"""
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        sql = str(content).strip()
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
        return sql.rstrip(";").strip()

    def validate_sql(self, sql: str) -> None:
        normalized = " ".join(sql.lower().split())
        # FIX (Security): Use word-boundary regex instead of substring `in`
        # so that a legitimate query like "WHERE content LIKE '%updated%'" is
        # not falsely blocked because "update" appears inside "updated".
        write_ops = re.compile(
            r"\b(insert|update|delete|drop|alter|create|attach|pragma|union|join)\b"
        )
        if not normalized.startswith("select ") or write_ops.search(normalized):
            raise ValueError("Only a single read-only SELECT query is allowed.")
        if ";" in normalized or "--" in normalized or "/*" in normalized or "*/" in normalized:
            raise ValueError("Query contains forbidden punctuation.")
        if "password_hash" in normalized:
            raise ValueError("Access to password_hash is forbidden.")
        tables = re.findall(r"\bfrom\s+([a-z_][a-z0-9_]*)", normalized)
        if len(tables) != 1 or tables[0] not in {"employees", "rag_knowledge"}:
            raise ValueError("The query must use exactly one approved table.")
        selected = normalized.split(" from ", 1)[0].removeprefix("select ")
        columns = {part.strip().split(".")[-1] for part in selected.split(",")}
        allowed = self.employee_columns if tables[0] == "employees" else self.knowledge_columns
        if not columns or "*" in columns or not columns.issubset(allowed):
            raise ValueError("The query requested a disallowed column.")
        if tables[0] == "employees" and not re.fullmatch(
            r"select [a-z_,. ]+ from employees where id = :employee_id(?: limit [0-9]+)?",
            normalized,
        ):
            raise ValueError("Employee queries must be scoped to the logged-in employee.")

    def run(self, question: str, employee_id: str | None) -> ToolResult:
        try:
            sql = self.generate_sql(question)
            self.validate_sql(sql)
            if not os.path.exists(DB_PATH):
                return ToolResult(self.name, [], "Database not available")
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(sql, {"employee_id": employee_id}).fetchall()
            finally:
                conn.close()
            payload = [dict(row) for row in rows]

            # FIX E: When querying rag_knowledge, use "database:{title}" as the
            # source so live evaluation matches the same contract as --dry-run.
            # For employee queries (no "title" column), fall back to the SQL tag.
            documents = []
            for row_dict in payload:
                if "title" in row_dict:
                    source = f"database:{row_dict['title']}"
                else:
                    source = "database:LLM-generated SQL"
                documents.append(Document(
                    page_content=f"SQL query result: {json.dumps(row_dict, default=str)}",
                    metadata={"source": source, "sql": sql},
                ))

            return ToolResult(self.name, documents if payload else [], sql)
        except Exception as exc:
            # The agent can still answer from policy documents when the model
            # produces unsuitable SQL or the database is unavailable.
            return ToolResult(self.name, [], f"SQL not run: {exc}")


class RAGAgent:
    """Agent facade that calls retrieval and SQLite tools before generating."""

    def __init__(self, chain: ConversationalRetrievalChain, retriever: Any, employee_id: str | None,
                 sql_llm: Any):
        self.chain = chain
        self.retriever = retriever
        self.employee_id = employee_id
        self.database_tool = LLMGeneratedSQLTool(sql_llm)

    def invoke(self, question: str) -> dict:
        # FIX C: Previously self.retriever.invoke(question) was called here but
        # its results were discarded (only db_context was used), causing:
        #   • 2× retrieval cost (Gemini embedding API called twice per turn)
        #   • Degraded semantic search (augmented_question with raw JSON was embedded)
        #   • Bloated memory (SQL payload was stored in ConversationBufferWindowMemory)
        # Now the ConversationalRetrievalChain handles its own retrieval internally.
        # Database facts are appended only to the question string seen by the LLM,
        # keeping the stored memory key clean.
        db_result = self.database_tool.run(question, self.employee_id)
        db_context = "\n".join(
            doc.page_content
            for doc in db_result.documents
            if doc.metadata.get("source", "").startswith("database:")
        )
        augmented_question = (
            question if not db_context
            else f"{question}\n\nDatabase tool results:\n{db_context}"
        )
        result = self.chain.invoke({"question": augmented_question})
        source_documents = list(result.get("source_documents", [])) + db_result.documents
        return {
            "answer": result.get("answer", ""),
            "source_documents": source_documents,
            "tools_used": ["hybrid_retriever", db_result.name],
            "sql": db_result.detail,
        }


def build_rag_agent(retriever: EnsembleRetriever, employee_info: Optional[dict] = None,
                    model: str = "models/gemini-2.5-flash") -> RAGAgent:
    """Build the conversational chain and expose it as a tool-using RAG agent."""
    chain = build_rag_chain(retriever, employee_info, model)
    return RAGAgent(chain, retriever, (employee_info or {}).get("id"), get_llm(model, temperature=0))


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
    chain: Any,
    question: str,
) -> dict:
    """
    Ask a question and return ``{"answer": str, "sources": list[str]}``.
    """
    result = chain.invoke(question) if isinstance(chain, RAGAgent) else chain.invoke({"question": question})
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in result.get("source_documents", [])
    })
    return {
        "answer": result.get("answer", ""),
        "sources": sorted(sources),
        "tools_used": result.get("tools_used", []),
        "sql": result.get("sql"),
    }
