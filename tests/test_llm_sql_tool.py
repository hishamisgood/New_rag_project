import sqlite3
from contextlib import closing

import pytest

from src.sql_tool import LLMGeneratedSQLTool


@pytest.mark.parametrize("sql", [
    "SELECT salary, department FROM employees WHERE id = :employee_id",
    "SELECT category, SUM(amount) FROM expense_claims "
    "WHERE employee_id = :employee_id GROUP BY category ORDER BY SUM(amount) DESC LIMIT 3",
    "SELECT ROUND(AVG(net_pay), 2) FROM payroll_history WHERE employee_id = :employee_id",
    "SELECT SUM(amount) FROM expense_claims WHERE category = 'Travel' "
    "AND employee_id = :employee_id",
    "SELECT COUNT(*) FROM leave_requests WHERE employee_id = :employee_id",
    "SELECT course_name, MAX(cost) FROM training_enrollments "
    "WHERE employee_id = :employee_id GROUP BY course_name ORDER BY MAX(cost) DESC LIMIT 2",
])
def test_valid_employee_scoped_analytical_queries(sql):
    LLMGeneratedSQLTool(object()).validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "SELECT SUM(amount) AS total_spend FROM expense_claims WHERE employee_id = :employee_id",
    "SELECT category FROM expense_claims WHERE employee_id = :employee_id AND category LIKE 'T%'",
    "SELECT category FROM expense_claims WHERE employee_id = :employee_id AND category IN ('Travel', 'Meals')",
    "SELECT amount FROM expense_claims WHERE employee_id = :employee_id AND amount BETWEEN 10 AND 500",
    "SELECT DISTINCT category FROM expense_claims WHERE employee_id = :employee_id",
    "SELECT category, SUM(amount) FROM expense_claims WHERE employee_id = :employee_id "
    "GROUP BY category HAVING SUM(amount) > 100",
    "SELECT id FROM employees WHERE id = :employee_id AND health_plan IS NULL",
    "SELECT strftime('%Y', paid_date) FROM payroll_history WHERE employee_id = :employee_id",
])
def test_flexible_analytical_sql_is_allowed(sql):
    LLMGeneratedSQLTool(object()).validate_sql(sql)


def test_alias_does_not_allow_unknown_real_columns():
    tool = LLMGeneratedSQLTool(object())
    tool.validate_sql(
        "SELECT SUM(amount) AS total_spend FROM expense_claims "
        "WHERE employee_id = :employee_id"
    )
    with pytest.raises(ValueError):
        tool.validate_sql(
            "SELECT unapproved_amount AS total_spend FROM expense_claims "
            "WHERE employee_id = :employee_id"
        )


TABLE_SCOPES = {
    "employees": ("salary", "id"),
    "expense_claims": ("amount", "employee_id"),
    "leave_requests": ("days", "employee_id"),
    "payroll_history": ("net_pay", "employee_id"),
    "training_enrollments": ("cost", "employee_id"),
}


@pytest.mark.parametrize("table", TABLE_SCOPES)
@pytest.mark.parametrize(
    "attack",
    ["cross_employee", "or_bypass", "subquery", "join", "select_star", "password_hash"],
)
def test_unsafe_queries_are_rejected_for_every_table(table, attack):
    value_column, scope_column = TABLE_SCOPES[table]
    if attack == "cross_employee":
        sql = f"SELECT {value_column} FROM {table} WHERE {scope_column} = 'EMP00002'"
    elif attack == "or_bypass":
        sql = (
            f"SELECT {value_column} FROM {table} WHERE "
            f"{scope_column} = :employee_id OR 1 = 1"
        )
    elif attack == "subquery":
        sql = (
            f"SELECT {value_column} FROM {table} WHERE {scope_column} = :employee_id "
            "AND id IN (SELECT id FROM employees)"
        )
    elif attack == "join":
        sql = (
            f"SELECT {value_column} FROM {table} JOIN employees ON employees.id = "
            f"{table}.{scope_column} WHERE {table}.{scope_column} = :employee_id"
        )
    elif attack == "select_star":
        sql = f"SELECT * FROM {table} WHERE {scope_column} = :employee_id"
    else:
        sql = f"SELECT password_hash FROM {table} WHERE {scope_column} = :employee_id"

    with pytest.raises(ValueError):
        LLMGeneratedSQLTool(object()).validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM employees WHERE id = :employee_id",
    "SELECT salary FROM employees WHERE id = :employee_id; DROP TABLE employees",
    "SELECT salary FROM employees WHERE id = :employee_id -- comment",
    "SELECT salary FROM employees WHERE id = :employee_id /* comment */",
    "SELECT salary FROM unknown_table WHERE id = :employee_id",
    "SELECT salary FROM employees WHERE id = :employee_id UNION SELECT salary FROM employees",
])
def test_read_only_sql_guard_rejects_write_ddl_and_statement_tokens(sql):
    with pytest.raises(ValueError):
        LLMGeneratedSQLTool(object()).validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "SELECT department, COUNT(id) FROM employees GROUP BY department",
    "SELECT category, SUM(amount) FROM expense_claims GROUP BY category",
])
def test_hr_can_run_company_wide_analytical_queries(sql):
    LLMGeneratedSQLTool(object()).validate_sql(sql, access_level="hr")


@pytest.mark.parametrize("sql", [
    "SELECT password_hash FROM employees",
    "DROP TABLE employees",
    "SELECT salary FROM employees JOIN leave_requests ON employees.id = leave_requests.employee_id",
    "SELECT * FROM employees",
    "SELECT salary FROM employees WHERE id IN (SELECT id FROM employees)",
])
def test_hr_still_cannot_run_unsafe_queries(sql):
    with pytest.raises(ValueError):
        LLMGeneratedSQLTool(object()).validate_sql(sql, access_level="hr")


def test_run_uses_space_safe_read_only_uri_and_table_source(tmp_path, monkeypatch):
    from src import sql_tool as sql_tool_module

    db_dir = tmp_path / "database with spaces"
    db_dir.mkdir()
    db_path = db_dir / "test.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            "CREATE TABLE expense_claims "
            "(id INTEGER PRIMARY KEY, employee_id TEXT, category TEXT, amount REAL)"
        )
        conn.execute(
            "INSERT INTO expense_claims (employee_id, category, amount) "
            "VALUES ('EMP00001', 'Travel', 125.50)"
        )
        conn.commit()
    monkeypatch.setattr(sql_tool_module, "DB_PATH", str(db_path))

    class FakeLLM:
        def invoke(self, _prompt):
            class Response:
                content = (
                    "SELECT SUM(amount) FROM expense_claims "
                    "WHERE employee_id = :employee_id AND category = 'Travel'"
                )
            return Response()

    result = LLMGeneratedSQLTool(FakeLLM()).run("Travel reimbursement total", "EMP00001")
    assert result.documents
    assert result.documents[0].metadata["source"] == "database:expense_claims"

    read_only_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(read_only_uri, uri=True) as conn:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO expense_claims (employee_id, category, amount) "
                "VALUES ('EMP00001', 'Travel', 1)"
            )


def test_run_retries_once_after_validation_failure(tmp_path, monkeypatch):
    from src import sql_tool as sql_tool_module

    db_path = tmp_path / "retry.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            "CREATE TABLE expense_claims "
            "(id INTEGER PRIMARY KEY, employee_id TEXT, amount REAL)"
        )
        conn.execute(
            "INSERT INTO expense_claims (employee_id, amount) VALUES ('EMP00001', 125.50)"
        )
        conn.commit()
    monkeypatch.setattr(sql_tool_module, "DB_PATH", str(db_path))

    class FakeLLM:
        def __init__(self):
            self.prompts = []
            self.responses = [
                "SELECT unknown_column FROM expense_claims WHERE employee_id = :employee_id",
                "SELECT SUM(amount) AS total FROM expense_claims WHERE employee_id = :employee_id",
            ]

        def invoke(self, prompt):
            self.prompts.append(prompt)

            class Response:
                pass

            response = Response()
            response.content = self.responses.pop(0)
            return response

    llm = FakeLLM()
    tool = LLMGeneratedSQLTool(llm)
    result = tool.run("Total expenses", "EMP00001")

    assert len(llm.prompts) == 2
    assert "previous query failed validation or execution" in llm.prompts[1]
    assert tool.validation_failures == 1
    assert result.documents
    assert result.detail.startswith("SELECT SUM(amount) AS total")
