import sqlite3

import auth


def test_knowledge_seed_is_searchable(tmp_path, monkeypatch):
    database = tmp_path / "employees.db"
    monkeypatch.setattr(auth, "DB_PATH", str(database))

    auth.ensure_knowledge_base()
    results = auth.search_knowledge("When is payroll processed?")

    assert results
    assert any(row["key"] == "payroll" for row in results)
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT count(*) FROM rag_knowledge").fetchone()[0] >= 4
