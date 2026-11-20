"""Conversational RAG chain with a dynamic SQL or policy router."""

import logging
from typing import Any, Optional

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from .config import GEMINI_MODEL
from .llm import extract_message_text, get_llm
from .sql_tool import LLMGeneratedSQLTool, ToolResult

logger = logging.getLogger(__name__)


class _DatabaseContextRetriever(BaseRetriever):
    """Append the current turn's SQL documents to policy retrieval results."""

    base_retriever: Any
    _database_documents: list[Document] = PrivateAttr(default_factory=list)

    def set_database_documents(self, documents: list[Document]) -> None:
        self._database_documents = list(documents)

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return list(self.base_retriever.invoke(query)) + list(self._database_documents)


def _build_employee_preamble(employee_info: Optional[dict]) -> str:
    """Build the employee profile system preamble string."""
    preamble = (
        "You are NovaCorp AI Assistant, a helpful and friendly chatbot for "
        "NovaCorp Industries employees. Answer questions using ONLY the "
        "retrieved context and the employee's profile below. If the answer "
        "isn't in the context, say so honestly.\n\n"
    )
    if employee_info:
        months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        absences = ", ".join(
            f"{name}: {employee_info.get(f'absent_days_{month}', 0)} days"
            for month, name in zip(months, month_names)
        )
        preamble += (
            "── EMPLOYEE PROFILE ──\n"
            f"ID: {employee_info.get('id', 'N/A')}\n"
            f"Name: {employee_info.get('first_name', '')} {employee_info.get('last_name', '')}\n"
            f"Sex: {employee_info.get('sex', 'N/A')}\n"
            f"Department: {employee_info.get('department', 'N/A')}\n"
            f"Role: {employee_info.get('role', 'N/A')}\n"
            f"Years of Experience: {employee_info.get('years_of_expertise', 'N/A')}\n"
            f"Hire Date: {employee_info.get('hire_date', 'N/A')}\n"
            f"Salary: ${employee_info.get('salary', 0):,.2f}\n"
            f"Work Mode: {employee_info.get('work_mode', 'N/A')}\n"
            f"Office Location: {employee_info.get('office_location', 'N/A')}\n"
            f"Health Plan: {employee_info.get('health_plan', 'N/A')}\n"
            f"Performance Rating: {employee_info.get('performance_rating', 'N/A')}/5\n"
            f"Sick Leave Days: {employee_info.get('sick_leave_days', 0)}\n"
            f"Maternity Leave Days: {employee_info.get('maternity_leave_days', 0)}\n"
            f"Paternity Leave Days: {employee_info.get('paternity_leave_days', 0)}\n"
            f"Paid Time Off Days: {employee_info.get('paid_time_off_days', 0)}\n"
            f"Mental Health Leave Days: {employee_info.get('mental_health_leave_days', 0)}\n"
            f"Monthly Absence: {absences}\n"
            "──────────────────────\n\n"
        )
    return preamble


def build_rag_chain(
    retriever: Any,
    employee_info: Optional[dict] = None,
    model: str = GEMINI_MODEL,
) -> Any:
    """Build an LCEL retrieval chain with employee profile context."""
    llm = get_llm(model=model)
    preamble = _build_employee_preamble(employee_info)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", preamble + "\nContext from company documents:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, qa_chain)


class RAGAgent:
    """Route each standalone question to SQL or policy retrieval, then answer."""

    router_table_descriptions = """- expense_claims(employee_id, category, amount, submitted_date, status, description): employee expense claims, including Travel, Meals, Equipment, Training, and Books.
- leave_requests(employee_id, leave_type, start_date, end_date, days, status, approver): the employee's leave requests and approval state.
- payroll_history(employee_id, pay_month, gross_pay, tax_deducted, benefits_deducted, bonus, net_pay, paid_date): monthly pay and bonus records.
- training_enrollments(employee_id, course_name, category, cost, status, completion_date): employee learning and development enrollment costs and progress."""

    def __init__(self, chain: Any, retriever: Any, employee_id: str | None, sql_llm: Any):
        self.chain = chain
        self.context_retriever = (
            retriever if isinstance(retriever, _DatabaseContextRetriever)
            else _DatabaseContextRetriever(base_retriever=retriever)
        )
        self.employee_id = employee_id
        self.router_llm = sql_llm
        self.database_tool = LLMGeneratedSQLTool(sql_llm)
        self._chat_history: list[BaseMessage] = []

    def _history_text(self) -> str:
        lines = []
        for message in self._chat_history:
            speaker = "Employee" if isinstance(message, HumanMessage) else "Assistant"
            lines.append(f"{speaker}: {message.content}")
        return "\n".join(lines)

    def _condense_question(self, question: str) -> str:
        if not self._chat_history:
            return question
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the latest employee question as a standalone question using "
                    "the conversation. Preserve its intent and details. Return only the "
                    "rewritten question; do not answer it.",
                ),
                ("human", "Conversation:\n{history}\n\nLatest question: {question}"),
            ]
        )
        try:
            response = self.router_llm.invoke(
                prompt.invoke({"history": self._history_text(), "question": question})
            )
            standalone = extract_message_text(response)
            return standalone or question
        except Exception as exc:
            logger.warning("Could not condense follow-up question; using original: %s", exc)
            return question

    def _route_question(self, standalone_question: str) -> str:
        prompt = f"""Choose the one data source needed to answer this employee question.
Return exactly one word: SQL or POLICY.

Choose SQL when the answer needs an employee's own transaction records (expense claims,
leave requests, payroll history, or training enrollments) or a personal employee record.
Choose POLICY when the answer is in company policies, PDFs, the company guide, or is
outside the available employee data. If uncertain, return POLICY.

Available transaction tables:
{self.router_table_descriptions}

Standalone question: {standalone_question}"""
        try:
            decision = extract_message_text(self.router_llm.invoke(prompt)).strip().strip("`").upper()
            return "SQL" if decision == "SQL" else "POLICY"
        except Exception as exc:
            logger.warning("Question router failed; falling back to policy retrieval: %s", exc)
            return "POLICY"

    def invoke(self, question: str) -> dict:
        standalone_question = self._condense_question(question)
        route = self._route_question(standalone_question)
        db_result = (
            self.database_tool.run(standalone_question, self.employee_id)
            if route == "SQL"
            else ToolResult(self.database_tool.name, [], None)
        )
        self.context_retriever.set_database_documents(db_result.documents)
        try:
            result = self.chain.invoke(
                {"input": standalone_question, "chat_history": list(self._chat_history)}
            )
            answer = result.get("answer", "")
            source_documents = list(result.get("context", result.get("source_documents", [])))
        finally:
            self.context_retriever.set_database_documents([])

        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))
        if len(self._chat_history) > 20:
            self._chat_history = self._chat_history[-20:]

        seen = {(doc.page_content, doc.metadata.get("source")) for doc in source_documents}
        for document in db_result.documents:
            key = (document.page_content, document.metadata.get("source"))
            if key not in seen:
                source_documents.append(document)
                seen.add(key)

        tools_used = ["hybrid_retriever"]
        if route == "SQL":
            tools_used.append(db_result.name)
        return {
            "answer": answer,
            "source_documents": source_documents,
            "tools_used": tools_used,
            "sql": db_result.detail,
        }


def build_rag_agent(
    retriever: Any,
    employee_info: Optional[dict] = None,
    model: str = GEMINI_MODEL,
) -> RAGAgent:
    """Build an agent over the supplied hybrid retriever."""
    context_retriever = _DatabaseContextRetriever(base_retriever=retriever)
    chain = build_rag_chain(context_retriever, employee_info, model)
    return RAGAgent(
        chain,
        context_retriever,
        (employee_info or {}).get("id"),
        get_llm(model, temperature=0),
    )


def _format_source(document: Document) -> str:
    source = str(document.metadata.get("source", "unknown"))
    if source.startswith(("database:", "company_guide:")):
        return source
    if source.lower().endswith(".pdf") and document.metadata.get("page") is not None:
        try:
            page = int(document.metadata["page"]) + 1
            return f"{source} (p. {page})"
        except (TypeError, ValueError):
            pass
    return source


def query_rag(chain: Any, question: str) -> dict:
    """Return the answer, clean source citations, tools used, and generated SQL."""
    if isinstance(chain, RAGAgent):
        result = chain.invoke(question)
        source_documents = result.get("source_documents", [])
    else:
        result = chain.invoke({"input": question, "chat_history": []})
        source_documents = result.get("context", result.get("source_documents", []))
    sources = {_format_source(document) for document in source_documents}
    return {
        "answer": result.get("answer", ""),
        "sources": sorted(sources),
        "tools_used": result.get("tools_used", []),
        "sql": result.get("sql"),
    }
