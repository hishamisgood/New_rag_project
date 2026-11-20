"""Read-only SQLite tool for employee profile and transaction lookups."""

import json
import logging
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .config import DB_PATH
from .llm import extract_message_text

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    name: str
    documents: list[Document]
    detail: str | None = None


class LLMGeneratedSQLTool:
    """Generate and safely execute one employee-scoped SQLite query."""

    name = "llm_sql_database"
    table_columns = {
        "employees": {
            "id", "first_name", "last_name", "sex", "department", "role", "salary",
            "years_of_expertise", "sick_leave_days", "paid_time_off_days",
            "mental_health_leave_days", "maternity_leave_days", "paternity_leave_days",
            "absent_days_jan", "absent_days_feb", "absent_days_mar", "absent_days_apr",
            "absent_days_may", "absent_days_jun", "absent_days_jul", "absent_days_aug",
            "absent_days_sep", "absent_days_oct", "absent_days_nov", "absent_days_dec",
            "hire_date", "work_mode", "office_location", "health_plan", "performance_rating",
        },
        "expense_claims": {
            "id", "employee_id", "category", "amount", "submitted_date", "status", "description",
        },
        "leave_requests": {
            "id", "employee_id", "leave_type", "start_date", "end_date", "days", "status", "approver",
        },
        "payroll_history": {
            "id", "employee_id", "pay_month", "gross_pay", "tax_deducted",
            "benefits_deducted", "bonus", "net_pay", "paid_date",
        },
        "training_enrollments": {
            "id", "employee_id", "course_name", "category", "cost", "status", "completion_date",
        },
    }
    sql_keywords = {
        "select", "from", "where", "and", "group", "by", "order", "asc", "desc", "limit",
    }
    sql_functions = {"sum", "count", "avg", "min", "max", "round", "coalesce"}
    forbidden_words = re.compile(
        r"\b(insert|update|delete|drop|alter|create|attach|pragma|union|join)\b",
        re.IGNORECASE,
    )
    string_literal = re.compile(r"'(?:''|[^'])*'")

    def __init__(self, llm: Any):
        self.llm = llm

    def generate_sql(self, question: str) -> str:
        prompt = f"""Generate exactly one SQLite SELECT query to answer this employee-portal question.
Return SQL only, with no Markdown, explanation, comments, or semicolon.

Allowed tables and columns:
- employees: {', '.join(sorted(self.table_columns['employees']))}
- expense_claims: {', '.join(sorted(self.table_columns['expense_claims']))}
- leave_requests: {', '.join(sorted(self.table_columns['leave_requests']))}
- payroll_history: {', '.join(sorted(self.table_columns['payroll_history']))}
- training_enrollments: {', '.join(sorted(self.table_columns['training_enrollments']))}

Use exactly one table. Allowed analytical functions are SUM, COUNT, AVG, MIN, MAX, ROUND,
and COALESCE. You may use WHERE with AND filters, GROUP BY, ORDER BY, and LIMIT.
Never use JOIN, UNION, subqueries, SELECT *, or any unlisted table or column.
Every employees query must include `WHERE id = :employee_id`.
Every other table query must include `employee_id = :employee_id` in WHERE.
Use :employee_id as a bound parameter, and do not add a semicolon.

Question: {question}"""
        response = self.llm.invoke(prompt)
        sql = extract_message_text(response)
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
        return sql

    def validate_sql(self, sql: str) -> None:
        """Reject anything beyond one scoped, read-only analytical SELECT."""
        raw = sql.strip()
        lowered = " ".join(raw.lower().split())
        if not lowered.startswith("select "):
            raise ValueError("Only a single read-only SELECT query is allowed.")
        if any(marker in lowered for marker in (";", "--", "/*", "*/")):
            raise ValueError("Query contains forbidden statement or comment syntax.")
        if "password_hash" in lowered:
            raise ValueError("Access to password_hash is forbidden.")

        sanitized = " ".join(self.string_literal.sub(" STRING ", lowered).split())
        if "'" in sanitized:
            raise ValueError("Query contains an invalid string literal.")
        if self.forbidden_words.search(sanitized):
            raise ValueError("Only read-only analytical SQL is allowed.")
        if re.search(r"\bor\b", sanitized):
            raise ValueError("OR filters are not allowed in employee-scoped queries.")
        if len(re.findall(r"\bselect\b", sanitized)) != 1:
            raise ValueError("Subqueries are not allowed.")
        if len(re.findall(r"\bfrom\b", sanitized)) != 1:
            raise ValueError("The query must use exactly one FROM table.")

        table_match = re.search(r"\bfrom\s+([a-z_][a-z0-9_]*)\b", sanitized)
        if not table_match or table_match.group(1) not in self.table_columns:
            raise ValueError("The query must use exactly one approved table.")
        table = table_match.group(1)
        from_clause = sanitized[table_match.start():]
        if not re.match(
            rf"\bfrom\s+{re.escape(table)}(?:\s+(?:where|group\s+by|order\s+by|limit)\b|\s*$)",
            from_clause,
        ):
            raise ValueError("Aliases and additional FROM tables are not allowed.")

        select_clause = sanitized.split(" from ", 1)[0]
        if re.match(r"^select\s+(?:[a-z_][a-z0-9_]*\.)?\*", select_clause):
            raise ValueError("SELECT * is forbidden.")
        # COUNT(*) is useful for analytical queries; no other wildcard is accepted.
        without_count_star = re.sub(r"\bcount\s*\(\s*\*\s*\)", "count ( id )", sanitized)
        if "*" in without_count_star:
            raise ValueError("Wildcard expressions are not allowed.")

        parameters = re.findall(r":[a-z_][a-z0-9_]*", sanitized)
        if any(parameter != ":employee_id" for parameter in parameters):
            raise ValueError("Only the :employee_id bound parameter is allowed.")
        token_sql = re.sub(r":employee_id\b", " ", sanitized)

        allowed_punctuation = set("_,:().=<>!+-*'")
        if not all(
            character.isalnum() or character.isspace() or character in allowed_punctuation
            for character in token_sql
        ):
            raise ValueError("Query contains unsupported SQL syntax.")

        identifiers = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", token_sql))
        allowed_identifiers = (
            self.table_columns[table] | self.sql_keywords | self.sql_functions | {table, "string"}
        )
        if not identifiers <= allowed_identifiers:
            raise ValueError("The query requested an unapproved column or SQL keyword.")

        qualified_columns = re.findall(
            r"\b([a-z_][a-z0-9_]*)\s*\.\s*([a-z_][a-z0-9_]*)\b", token_sql
        )
        if any(prefix != table or column not in self.table_columns[table]
               for prefix, column in qualified_columns):
            raise ValueError("Qualified columns must belong to the selected table.")

        where_match = re.search(
            r"\bwhere\s+(.+?)(?=\s+group\s+by\b|\s+order\s+by\b|\s+limit\b|$)",
            sanitized,
        )
        if where_match is None:
            raise ValueError("Employee-scoped queries must include a WHERE clause.")
        scope_column = "id" if table == "employees" else "employee_id"
        scope_pattern = rf"\b{scope_column}\s*=\s*:employee_id\b"
        if not re.search(scope_pattern, where_match.group(1)):
            raise ValueError("Query must be scoped to the logged-in employee.")

    def run(self, question: str, employee_id: str | None) -> ToolResult:
        try:
            sql = self.generate_sql(question)
            self.validate_sql(sql)
            if not os.path.exists(DB_PATH):
                return ToolResult(self.name, [], "Database not available")

            db_uri = f"{Path(DB_PATH).resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(db_uri, uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, {"employee_id": employee_id}).fetchall()

            table_match = re.search(r"\bfrom\s+([a-z_][a-z0-9_]*)\b", sql, re.IGNORECASE)
            source = f"database:{table_match.group(1)}" if table_match else "database:unknown"
            documents = [
                Document(
                    page_content=f"SQL query result: {json.dumps(dict(row), default=str)}",
                    metadata={"source": source, "sql": sql},
                )
                for row in rows
            ]
            return ToolResult(self.name, documents, sql)
        except Exception as exc:
            logger.warning("SQL tool could not run the requested query: %s", exc)
            return ToolResult(self.name, [], f"SQL not run: {exc}")
