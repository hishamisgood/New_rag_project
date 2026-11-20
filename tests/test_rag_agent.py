import logging

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import RAGAgent, query_rag
from src.sql_tool import ToolResult


def _fake_vectorstore(documents, metadatas):
    class FakeCollection:
        def get(self, include):
            return {"documents": documents, "metadatas": metadatas}

    class FakeVectorstore:
        _collection = FakeCollection()

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    return FakeVectorstore


def _stub_sql_tool(monkeypatch, agent, documents, detail):
    calls = []

    def run(question, employee_id):
        calls.append((question, employee_id))
        return ToolResult(agent.database_tool.name, documents, detail)

    monkeypatch.setattr(agent.database_tool, "run", run)
    return calls


class FakeRetriever:
    def __init__(self, documents=None):
        self.documents = documents or [
            Document(
                page_content="Leave policy context",
                metadata={"source": "leave_policy_report.pdf", "page": 0},
            )
        ]
        self.last_question = None

    def invoke(self, question):
        self.last_question = question
        return list(self.documents)


class FakeChain:
    """Small LCEL-shaped chain: input/chat_history in, answer/context out."""

    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.received_values = None

    def invoke(self, values):
        assert set(values) == {"input", "chat_history"}
        self.received_values = values
        documents = self.base_retriever.invoke(values["input"])
        return {"answer": "Use the service desk.", "context": documents}


class FakeLLM:
    def __init__(self, route="POLICY", standalone=None):
        self.route = route
        self.standalone = standalone
        self.prompts = []

    def invoke(self, prompt):
        prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        self.prompts.append(prompt_text)
        if "Rewrite the latest employee question" in prompt_text:
            return type("Response", (), {"content": self.standalone or "Which leave request is pending?"})()
        if "Choose the one data source" in prompt_text:
            return type("Response", (), {"content": self.route})()
        return type("Response", (), {"content": "SELECT SUM(amount) FROM expense_claims "
                                              "WHERE employee_id = :employee_id"})()


def test_dynamic_router_runs_sql_for_transactional_question(monkeypatch):
    retriever = FakeRetriever()
    chain = FakeChain(retriever)
    llm = FakeLLM(route="SQL")
    agent = RAGAgent(chain, retriever, "EMP00001", llm)
    database_document = Document(
        page_content='SQL query result: {"sum": 35.0}',
        metadata={"source": "database:expense_claims"},
    )
    calls = _stub_sql_tool(
        monkeypatch,
        agent,
        [database_document],
        "SELECT SUM(amount) FROM expense_claims WHERE employee_id = :employee_id",
    )

    result = query_rag(agent, "How much did I spend on travel?")

    assert calls == [("How much did I spend on travel?", "EMP00001")]
    assert "How much did I spend on travel?" in llm.prompts[0]
    assert chain.received_values["input"] == "How much did I spend on travel?"
    assert retriever.last_question == "How much did I spend on travel?"
    assert "database:expense_claims" in result["sources"]
    assert "leave_policy_report.pdf (p. 1)" in result["sources"]
    assert result["tools_used"] == ["hybrid_retriever", "llm_sql_database"]


def test_dynamic_router_skips_sql_for_pure_policy_question(monkeypatch):
    retriever = FakeRetriever()
    chain = FakeChain(retriever)
    agent = RAGAgent(chain, retriever, "EMP00001", FakeLLM(route="POLICY"))

    def unexpected_sql_call(*_args, **_kwargs):
        raise AssertionError("SQL should not run for a policy question")

    monkeypatch.setattr(agent.database_tool, "run", unexpected_sql_call)
    result = query_rag(agent, "What is the leave policy?")

    assert chain.received_values["input"] == "What is the leave policy?"
    assert result["sources"] == ["leave_policy_report.pdf (p. 1)"]
    assert result["tools_used"] == ["hybrid_retriever"]
    assert result["sql"] is None


def test_followup_is_condensed_before_router_and_sql_tool(monkeypatch):
    retriever = FakeRetriever()
    chain = FakeChain(retriever)
    llm = FakeLLM(route="SQL", standalone="Which of my expense claims are still pending approval?")
    agent = RAGAgent(chain, retriever, "EMP00001", llm)
    agent._chat_history = [
        HumanMessage(content="Show me my expense claims."),
        AIMessage(content="You have several expense claims this year."),
    ]
    calls = _stub_sql_tool(
        monkeypatch, agent, [], "SELECT ..."
    )

    agent.invoke("And which are still pending?")

    standalone = "Which of my expense claims are still pending approval?"
    assert calls == [(standalone, "EMP00001")]
    assert standalone in llm.prompts[-1]
    assert "And which are still pending?" not in llm.prompts[-1]
    assert chain.received_values["input"] == standalone
    assert len(chain.received_values["chat_history"]) == 2
    assert agent._chat_history[-2].content == "And which are still pending?"


def test_query_rag_lcel_fallback_uses_context_and_page_citations():
    documents = [
        Document(page_content="Policy", metadata={"source": "benefits_guide.pdf", "page": 1}),
        Document(page_content="Fact", metadata={"source": "company_guide:Payroll schedule", "page": 0}),
        Document(page_content="SQL", metadata={"source": "database:payroll_history"}),
    ]

    class LCELChain:
        def invoke(self, values):
            assert values == {"input": "Question?", "chat_history": []}
            return {"answer": "Answer", "context": documents}

    result = query_rag(LCELChain(), "Question?")
    assert result["sources"] == [
        "benefits_guide.pdf (p. 2)",
        "company_guide:Payroll schedule",
        "database:payroll_history",
    ]


def test_load_vectorstore_and_hybrid_retriever_work_without_google_key(monkeypatch):
    from src import retriever as retriever_module
    from langchain_community.retrievers import BM25Retriever

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    fake_vectorstore = _fake_vectorstore(["Policy text here"], [{"source": "policy.pdf"}])
    monkeypatch.setattr(retriever_module, "Chroma", fake_vectorstore)
    vectorstore = retriever_module.load_vectorstore()
    assert vectorstore.kwargs["embedding_function"] is None
    retriever = retriever_module.create_hybrid_retriever(vectorstore)
    assert isinstance(retriever, BM25Retriever)


def test_rrf_falls_back_to_bm25_when_semantic_retrieval_raises(caplog):
    from src.retriever import RRFRetriever

    document = Document(page_content="BM25 result", metadata={"source": "policy.pdf"})

    class RaisingRetriever:
        def invoke(self, _query):
            raise RuntimeError("quota exhausted")

    class KeywordRetriever:
        def invoke(self, _query):
            return [document]

    retriever = RRFRetriever(
        retriever_a=RaisingRetriever(), retriever_b=KeywordRetriever(),
    )
    with caplog.at_level(logging.WARNING):
        assert retriever.invoke("policy") == [document]
    assert "Semantic retrieval failed" in caplog.text


def test_rrf_applies_semantic_weight():
    from src.retriever import RRFRetriever

    semantic_doc = Document(page_content="semantic", metadata={})
    keyword_doc = Document(page_content="keyword", metadata={})

    class StaticRetriever:
        def __init__(self, documents):
            self.documents = documents

        def invoke(self, _query):
            return self.documents

    retriever = RRFRetriever(
        retriever_a=StaticRetriever([semantic_doc]),
        retriever_b=StaticRetriever([keyword_doc]),
        semantic_weight=0.8,
    )
    assert [doc.page_content for doc in retriever.invoke("query")] == ["semantic", "keyword"]


def test_create_hybrid_retriever_raises_on_empty_store():
    from src import retriever as retriever_module

    fake_vectorstore = _fake_vectorstore([], [])
    with pytest.raises(ValueError, match="Vector store is empty"):
        retriever_module.create_hybrid_retriever(fake_vectorstore())
