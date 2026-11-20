import sqlite3
from contextlib import closing

from src import database as database_module


def test_knowledge_seed_is_searchable(tmp_path, monkeypatch):
    database = tmp_path / "employees.db"
    monkeypatch.setattr(database_module, "DB_PATH", str(database))

    database_module.ensure_knowledge_base()
    results = database_module.search_knowledge("When is payroll processed?")

    assert results
    assert any(row["key"] == "payroll" for row in results)
    with closing(sqlite3.connect(str(database))) as conn:
        assert conn.execute("SELECT count(*) FROM rag_knowledge").fetchone()[0] >= 12


def test_ensure_transactional_tables_seeds_employee_rows_once(tmp_path, monkeypatch):
    database = tmp_path / "employees.db"
    monkeypatch.setattr(database_module, "DB_PATH", str(database))
    with closing(sqlite3.connect(str(database))) as conn:
        conn.execute(
            "CREATE TABLE employees (id TEXT PRIMARY KEY, salary REAL, "
            "health_plan TEXT, performance_rating INTEGER)"
        )
        conn.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?)",
            [
                ("EMP00001", 72000, "Standard Plan", 3),
                ("EMP00002", 120000, "Premium Plan", 5),
            ],
        )
        conn.commit()

    database_module.ensure_transactional_tables()
    database_module.ensure_transactional_tables()

    expected_tables = {
        "expense_claims", "leave_requests", "payroll_history", "training_enrollments",
    }
    with closing(sqlite3.connect(str(database))) as conn:
        actual_tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert expected_tables <= actual_tables
        for table in expected_tables:
            rows = conn.execute(
                f"SELECT employee_id, count(*) FROM {table} GROUP BY employee_id"
            ).fetchall()
            assert {row[0] for row in rows} == {"EMP00001", "EMP00002"}
            assert all(row[1] > 0 for row in rows)

        payroll = conn.execute(
            "SELECT pay_month, bonus FROM payroll_history "
            "WHERE employee_id = 'EMP00002' AND pay_month IN ('2026-07', '2026-08') "
            "ORDER BY pay_month"
        ).fetchall()
        assert payroll[0][0] == "2026-07" and payroll[0][1] == 30000
        assert payroll[1][0] == "2026-08" and payroll[1][1] == 0

        book_amounts = conn.execute(
            "SELECT amount FROM expense_claims WHERE category = 'Books'"
        ).fetchall()
        assert all(40 <= amount[0] <= 280 for amount in book_amounts)

        training_totals = conn.execute(
            "SELECT employee_id, sum(cost) FROM training_enrollments GROUP BY employee_id"
        ).fetchall()
        assert all(200 <= total <= 2800 for _employee_id, total in training_totals)

        indexes = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert {f"idx_{table}_employee_id" for table in expected_tables} <= indexes


def test_ingestion_appends_operational_facts_to_pdf_chunks(tmp_path, monkeypatch):
    from langchain_core.documents import Document
    from src import ingestion as ingestion_module

    (tmp_path / "policy.pdf").touch()

    class FakePDFLoader:
        def __init__(self, _path):
            pass

        def load(self):
            return [Document(page_content="Policy page", metadata={"page": 0})]

    monkeypatch.setattr(ingestion_module, "PyPDFLoader", FakePDFLoader)
    chunks = ingestion_module.load_and_split_documents(str(tmp_path))

    guide_chunks = [doc for doc in chunks if doc.metadata.get("source", "").startswith("company_guide:")]
    assert len(guide_chunks) == len(database_module.KNOWLEDGE_SEED)
    assert all(doc.page_content.startswith(doc.metadata["source"].split(":", 1)[1] + ":")
               for doc in guide_chunks)
