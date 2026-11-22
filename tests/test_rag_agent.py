import logging

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import RAGAgent, build_rag_chain, query_rag
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

    def run(question, employee_id, access_level="employee"):
        calls.append((question, employee_id, access_level))
        return ToolResult(agent.database_tool.name, documents, detail)

    monkeypatch.setattr(agent.database_tool, "run", run)
    return calls


def _stub_post(monkeypatch, payload):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("src.web_search_tool.requests.post", fake_post)
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
        self.retrieved_documents = None

    def invoke(self, values):
        assert set(values) == {"input", "chat_history"}
        self.received_values = values
        documents = self.base_retriever.invoke(values["input"])
        self.retrieved_documents = documents
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
        if "Choose the one operation" in prompt_text:
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

    assert calls == [("How much did I spend on travel?", "EMP00001", "employee")]
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
    assert calls == [(standalone, "EMP00001", "employee")]
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


def test_build_rag_chain_uses_plain_langchain_core(monkeypatch):
    from langchain_core.messages import AIMessage
    from src import agent as agent_module

    document = Document(page_content="Company policy context", metadata={"source": "policy.pdf"})

    class ContextCheckingLLM:
        def invoke(self, prompt):
            assert "Company policy context" in prompt.to_string()
            return AIMessage(content="Answer from context")

    monkeypatch.setattr(agent_module, "get_llm", lambda **_kwargs: ContextCheckingLLM())
    chain = build_rag_chain(FakeRetriever([document]), model="fake-model")
    result = chain.invoke({"input": "What is the policy?", "chat_history": []})

    assert result["answer"] == "Answer from context"
    assert result["context"] == [document]


def test_role_filtered_hybrid_retriever_filters_bm25_documents(monkeypatch):
    from src import retriever as retriever_module

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    documents = [
        "Public office holiday schedule",
        "Confidential HR compensation investigation",
    ]
    metadatas = [
        {"source": "holidays.pdf", "access_level": "public"},
        {"source": "hr_notes.pdf", "access_level": "hr_only"},
    ]
    employee_retriever = retriever_module.create_hybrid_retriever(
        _fake_vectorstore(documents, metadatas)(), access_level="employee"
    )
    hr_retriever = retriever_module.create_hybrid_retriever(
        _fake_vectorstore(documents, metadatas)(), access_level="hr"
    )

    assert all("Confidential" not in doc.page_content for doc in employee_retriever.invoke("compensation"))
    assert any("Confidential" in doc.page_content for doc in hr_retriever.invoke("compensation"))


def test_web_search_tool_uses_tavily(monkeypatch):
    from src.web_search_tool import WebSearchTool

    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    calls = _stub_post(monkeypatch, {
        "results": [{"title": "IRS", "url": "https://irs.gov/401k", "content": "Public limit"}]
    })
    result = WebSearchTool().run("IRS 401(k) limits")

    assert calls[0][0] == "https://api.tavily.com/search"
    assert calls[0][1]["json"]["api_key"] == "test-tavily"
    assert result.documents[0].page_content == "Public limit"
    assert result.documents[0].metadata == {
        "source": "web:https://irs.gov/401k", "title": "IRS"
    }


def test_web_search_tool_uses_serper_when_tavily_key_is_missing(monkeypatch):
    from src.web_search_tool import WebSearchTool

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "test-serper")
    calls = _stub_post(monkeypatch, {
        "organic": [{"title": "Labor law", "link": "https://dol.gov/rule", "snippet": "Public rule"}]
    })
    result = WebSearchTool().run("public labor law")

    assert calls[0][0] == "https://google.serper.dev/search"
    assert calls[0][1]["headers"]["X-API-KEY"] == "test-serper"
    assert result.documents[0].metadata == {
        "source": "web:https://dol.gov/rule", "title": "Labor law"
    }


def test_web_search_tool_is_empty_when_no_provider_key_is_set(monkeypatch, caplog):
    from src.web_search_tool import WebSearchTool

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        result = WebSearchTool().run("public question")

    assert result.documents == []
    assert "no Tavily or Serper API key" in caplog.text


def test_web_search_route_adds_web_documents_to_retrieval_context(monkeypatch):
    retriever = FakeRetriever()
    agent = RAGAgent(None, retriever, "EMP00001", FakeLLM(route="WEB_SEARCH"))
    chain = FakeChain(agent.context_retriever)
    agent.chain = chain
    web_document = Document(
        page_content="Public IRS limit information",
        metadata={"source": "web:https://irs.gov/limits", "title": "IRS limits"},
    )
    monkeypatch.setattr(
        agent.web_search_tool,
        "run",
        lambda _question: ToolResult("web_search", [web_document], None),
    )

    result = query_rag(agent, "What are the public IRS contribution limits?")

    assert any(
        doc.page_content == "Public IRS limit information"
        for doc in chain.retrieved_documents
    )
    assert "web:https://irs.gov/limits" in result["sources"]
    assert result["tools_used"] == ["hybrid_retriever", "web_search"]
