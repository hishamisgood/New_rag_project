import sqlite3
from contextlib import closing

import bcrypt

from src import auth as auth_module
from src import database as database_module


def test_auth_hashes_passwords_and_authenticates_valid_credentials(tmp_path, monkeypatch):
    database = tmp_path / "employees.db"
    monkeypatch.setattr(auth_module, "DB_PATH", str(database))
    password_hash = auth_module.hash_password("correct-horse")
    assert bcrypt.checkpw(b"correct-horse", password_hash.encode())

    with closing(sqlite3.connect(str(database))) as conn:
        conn.execute(
            "CREATE TABLE employees (id TEXT PRIMARY KEY, department TEXT, password_hash TEXT)"
        )
        conn.execute(
            "INSERT INTO employees VALUES (?, ?, ?)",
            ("EMP00001", "Engineering", password_hash),
        )
        conn.commit()

    employee = auth_module.authenticate("EMP00001", "correct-horse")
    assert employee == {"id": "EMP00001", "department": "Engineering"}
    assert auth_module.authenticate("EMP00001", "wrong-password") is None
    assert auth_module.authenticate("EMP99999", "correct-horse") is None


def test_get_access_level_uses_department():
    assert auth_module.get_access_level({"department": "Human Resources"}) == "hr"
    assert auth_module.get_access_level({"department": "Engineering"}) == "employee"
    assert auth_module.get_access_level(None) == "employee"


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

    expected_tables = set(database_module.TRANSACTIONAL_TABLES)
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


def test_ingestion_appends_operational_facts_to_pdf_chunks(tmp_path):
    from src import ingestion as ingestion_module
    from reportlab.pdfgen import canvas

    policy_path = tmp_path / "policy.pdf"
    pdf = canvas.Canvas(str(policy_path))
    pdf.drawString(72, 720, "A policy page with enough selectable text for ingestion.")
    pdf.save()
    chunks = ingestion_module.load_and_split_documents(str(tmp_path))

    guide_chunks = [doc for doc in chunks if doc.metadata.get("source", "").startswith("company_guide:")]
    assert len(guide_chunks) == len(database_module.KNOWLEDGE_SEED)
    assert all(doc.page_content.startswith(doc.metadata["source"].split(":", 1)[1] + ":")
               for doc in guide_chunks)
    assert all(doc.metadata["access_level"] == "public"
               and doc.metadata["extraction_method"] == "seed" for doc in guide_chunks)
