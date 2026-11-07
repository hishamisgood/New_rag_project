from langchain_core.documents import Document

from rag_engine import RAGAgent, query_rag


class FakeRetriever:
    def invoke(self, question):
        return [Document(page_content="Leave policy context", metadata={"source": "leave.pdf"})]


class FakeChain:
    def invoke(self, values):
        assert "Database tool results" in values["question"]
        return {"answer": "Use the service desk.", "source_documents": []}


def test_rag_agent_calls_retrieval_and_database_tools(monkeypatch):
    agent = RAGAgent(FakeChain(), FakeRetriever(), "EMP1", object())
    monkeypatch.setattr(
        agent.database_tool, "run",
        lambda question, employee_id: type("Result", (), {"name": "llm_sql_database", "documents": [
            Document(page_content="Use the service desk.", metadata={"source": "database:IT support"})
        ], "detail": "SELECT content FROM rag_knowledge LIMIT 1"})(),
    )

    result = query_rag(agent, "Where do I get IT help?")

    assert result["answer"] == "Use the service desk."
    assert result["sources"] == ["database:IT support"]
    assert result["tools_used"] == ["hybrid_retriever", "llm_sql_database"]
    assert result["sql"] == "SELECT content FROM rag_knowledge LIMIT 1"
