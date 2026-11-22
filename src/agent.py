"""Conversational RAG agent with policy, SQL, action, and web routing."""

import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from .action_tool import HRActionTool, PendingAction
from .auth import get_access_level
from .config import GEMINI_MODEL
from .llm import extract_message_text, get_llm
from .sql_tool import LLMGeneratedSQLTool, ToolResult
from .web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class _ToolContextRetriever(BaseRetriever):
    """Append current-turn SQL or web results to policy retrieval results."""

    base_retriever: Any
    _tool_documents: list[Document] = PrivateAttr(default_factory=list)

    def set_tool_documents(self, documents: list[Document]) -> None:
        self._tool_documents = list(documents)

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return list(self.base_retriever.invoke(query)) + list(self._tool_documents)


def _build_employee_preamble(employee_info: Optional[dict]) -> str:
    """Build the employee profile system preamble string."""
    preamble = (
        "You are NovaCorp AI Assistant, a helpful and friendly chatbot for "
        "NovaCorp Industries employees. Answer company questions using the "
        "retrieved context and the employee's profile below. For external "
        "questions, use the supplied public web results and identify them as "
        "external sources. If the answer isn't in the context, say so honestly.\n\n"
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


def _format_documents(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


class _RAGChain:
    """Plain langchain_core LCEL retrieval and answer composition."""

    def __init__(self, retriever: Any, llm: Any, prompt: ChatPromptTemplate):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    def invoke(self, values: dict[str, Any]) -> dict[str, Any]:
        docs = list(self.retriever.invoke(values["input"]))
        prompt_value = self.prompt.invoke({**values, "context": _format_documents(docs)})
        answer = extract_message_text(self.llm.invoke(prompt_value))
        return {**values, "context": docs, "answer": answer}


def build_rag_chain(
    retriever: Any,
    employee_info: Optional[dict] = None,
    model: str = GEMINI_MODEL,
) -> Any:
    """Build a plain langchain_core retrieval chain with profile context."""
    llm = get_llm(model=model)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _build_employee_preamble(employee_info)
                + "\nRetrieved context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return _RAGChain(retriever, llm, prompt)


class RAGAgent:
    """Route each standalone question to company data, actions, or search."""

    router_table_descriptions = """- employees(id, department, role, salary, hire_date): employee profile records.
- expense_claims(employee_id, category, amount, submitted_date, status, description): expense claims.
- leave_requests(employee_id, leave_type, start_date, end_date, days, status, approver): leave requests.
- payroll_history(employee_id, pay_month, gross_pay, tax_deducted, benefits_deducted, bonus, net_pay, paid_date): pay records.
- training_enrollments(employee_id, course_name, category, cost, status, completion_date): learning records."""

    def __init__(
        self,
        chain: Any,
        retriever: Any,
        employee_id: str | None,
        sql_llm: Any,
        access_level: str = "employee",
        action_tool: HRActionTool | None = None,
        web_search_tool: WebSearchTool | None = None,
    ):
        self.chain = chain
        self.context_retriever = (
            retriever if isinstance(retriever, _ToolContextRetriever)
            else _ToolContextRetriever(base_retriever=retriever)
        )
        self.employee_id = employee_id
        self.access_level = access_level
        self.router_llm = sql_llm
        self.database_tool = LLMGeneratedSQLTool(sql_llm)
        self.action_tool = action_tool or HRActionTool(sql_llm)
        self.web_search_tool = web_search_tool or WebSearchTool()
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
        prompt = f"""Choose the one operation needed to answer this employee question.
Return exactly one word: POLICY, SQL, ACTION, or WEB_SEARCH.

Choose ACTION for submitting a leave request or expense claim, or for approving
or rejecting a leave request or expense claim. ACTION only prepares a proposal;
the user must confirm it before any write occurs.
Choose SQL for personal or HR employee and transaction records. HR users may
query company-wide records; employee queries are scoped to the signed-in user.
Choose WEB_SEARCH only for public information outside NovaCorp documents, such
as public labor laws, IRS limits, or industry benchmarks.
Choose POLICY for NovaCorp policies, benefits, procedures, and company guidance.
If uncertain or unable to decide, return POLICY.

Available database tables:\n{self.router_table_descriptions}

Standalone question: {standalone_question}"""
        try:
            decision = extract_message_text(self.router_llm.invoke(prompt)).strip()
            route = decision.strip("` \t\r\n").upper()
            return route if route in {"POLICY", "SQL", "ACTION", "WEB_SEARCH"} else "POLICY"
        except Exception as exc:
            logger.warning("Question router failed; falling back to policy retrieval: %s", exc)
            return "POLICY"

    def _record_turn(self, question: str, answer: str) -> None:
        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))
        if len(self._chat_history) > 20:
            self._chat_history = self._chat_history[-20:]

    def invoke(self, question: str) -> dict:
        standalone_question = self._condense_question(question)
        route = self._route_question(standalone_question)

        if route == "ACTION":
            pending_action, message = self.action_tool.propose(
                standalone_question, self.employee_id, self.access_level
            )
            answer = (
                f"Please review and confirm: {pending_action.summary}"
                if pending_action else f"I couldn't prepare that HR action: {message}"
            )
            self._record_turn(question, answer)
            return {
                "answer": answer,
                "source_documents": [],
                "tools_used": [self.action_tool.name],
                "sql": None,
                "pending_action": pending_action,
            }

        tool_result = ToolResult("", [], None)
        if route == "SQL":
            tool_result = self.database_tool.run(
                standalone_question,
                self.employee_id,
                access_level=self.access_level,
            )
        elif route == "WEB_SEARCH":
            tool_result = self.web_search_tool.run(standalone_question)

        self.context_retriever.set_tool_documents(tool_result.documents)
        try:
            result = self.chain.invoke(
                {"input": standalone_question, "chat_history": list(self._chat_history)}
            )
            answer = result.get("answer", "")
            source_documents = list(result.get("context", result.get("source_documents", [])))
        finally:
            self.context_retriever.set_tool_documents([])

        self._record_turn(question, answer)
        seen = {(doc.page_content, doc.metadata.get("source")) for doc in source_documents}
        for document in tool_result.documents:
            key = (document.page_content, document.metadata.get("source"))
            if key not in seen:
                source_documents.append(document)
                seen.add(key)

        tools_used = ["hybrid_retriever"]
        if route in {"SQL", "WEB_SEARCH"}:
            tools_used.append(tool_result.name)
        return {
            "answer": answer,
            "source_documents": source_documents,
            "tools_used": tools_used,
            "sql": tool_result.detail if route == "SQL" else None,
            "pending_action": None,
        }


def build_rag_agent(
    retriever: Any,
    employee_info: Optional[dict] = None,
    model: str = GEMINI_MODEL,
) -> RAGAgent:
    """Build an agent over a role-filtered hybrid retriever."""
    context_retriever = _ToolContextRetriever(base_retriever=retriever)
    chain = build_rag_chain(context_retriever, employee_info, model)
    router_llm = get_llm(model, temperature=0)
    return RAGAgent(
        chain,
        context_retriever,
        (employee_info or {}).get("id"),
        router_llm,
        access_level=get_access_level(employee_info),
        action_tool=HRActionTool(router_llm),
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
    """Return answer, citations, tools used, and any pending action."""
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
        "pending_action": result.get("pending_action"),
    }
