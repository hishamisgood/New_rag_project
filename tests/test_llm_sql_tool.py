import pytest

from rag_engine import LLMGeneratedSQLTool


def test_employee_sql_must_be_scoped_to_current_employee():
    tool = LLMGeneratedSQLTool(object())
    tool.validate_sql("SELECT salary, department FROM employees WHERE id = :employee_id")
    with pytest.raises(ValueError):
        tool.validate_sql("SELECT salary FROM employees")


@pytest.mark.parametrize("sql", [
    "DELETE FROM employees WHERE id = :employee_id",
    "SELECT password_hash FROM employees WHERE id = :employee_id",
    "SELECT salary FROM employees WHERE id = :employee_id OR 1 = 1",
    "SELECT * FROM rag_knowledge",
    "SELECT content FROM rag_knowledge; DROP TABLE employees",
])
def test_llm_sql_guard_rejects_unsafe_queries(sql):
    with pytest.raises(ValueError):
        LLMGeneratedSQLTool(object()).validate_sql(sql)
