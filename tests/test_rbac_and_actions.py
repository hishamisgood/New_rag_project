import json
import re
import sqlite3

import pytest

from src.action_tool import HRActionTool, PendingAction


class FakeActionLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _prompt):
        return type("Response", (), {"content": json.dumps(self.payload)})()


def _prepare_action_db(monkeypatch, tmp_path):
    import src.action_tool as action_module

    db_path = tmp_path / "actions.db"
    monkeypatch.setattr(action_module, "DB_PATH", str(db_path))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE leave_requests ("
            "id INTEGER PRIMARY KEY, employee_id TEXT, leave_type TEXT, "
            "start_date TEXT, end_date TEXT, days INTEGER, status TEXT, approver TEXT)"
        )
        conn.execute(
            "CREATE TABLE expense_claims ("
            "id INTEGER PRIMARY KEY, employee_id TEXT, category TEXT, amount REAL, "
            "submitted_date TEXT, status TEXT, description TEXT)"
        )
    return db_path


def test_submit_leave_request_is_not_written_until_execute(monkeypatch, tmp_path):
    db_path = _prepare_action_db(monkeypatch, tmp_path)
    tool = HRActionTool(FakeActionLLM({
        "action_type": "submit_leave_request",
        "params": {
            "leave_type": "PTO",
            "start_date": "2026-12-01",
            "end_date": "2026-12-03",
            "days": 3,
        },
    }))

    action, summary = tool.propose("Submit three days of PTO", "EMP00001", "employee")

    assert action is not None
    assert "3 day(s) of PTO" in summary
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM leave_requests").fetchone()[0] == 0

    confirmation = tool.execute(action)

    assert "Pending status" in confirmation
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT employee_id, leave_type, days, status, approver FROM leave_requests"
        ).fetchone()
    assert row == ("EMP00001", "PTO", 3, "Pending", "People Manager")


def test_submit_expense_claim_is_not_written_until_execute(monkeypatch, tmp_path):
    db_path = _prepare_action_db(monkeypatch, tmp_path)
    tool = HRActionTool(FakeActionLLM({
        "action_type": "submit_expense_claim",
        "params": {"category": "Meals", "amount": 45.5, "description": "Client lunch"},
    }))

    action, summary = tool.propose("Submit my client lunch expense", "EMP00002", "employee")

    assert action is not None
    assert "Meals expense claim" in summary
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM expense_claims").fetchone()[0] == 0

    confirmation = tool.execute(action)

    assert "Pending status" in confirmation
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT employee_id, category, amount, submitted_date, status, description "
            "FROM expense_claims"
        ).fetchone()
    assert row == ("EMP00002", "Meals", 45.5, row[3], "Pending", "Client lunch")
    assert re.fullmatch(r"\d{4}-\d{2}", row[3])


@pytest.mark.parametrize(
    ("action_type", "params", "table", "record_id"),
    [
        ("review_leave_request", {"request_id": 1, "decision": "Approved"}, "leave_requests", 1),
        ("review_expense_claim", {"claim_id": 1, "decision": "Rejected"}, "expense_claims", 1),
    ],
)
def test_review_actions_require_hr_and_update_status(
    monkeypatch, tmp_path, action_type, params, table, record_id
):
    db_path = _prepare_action_db(monkeypatch, tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO leave_requests "
            "(id, employee_id, leave_type, start_date, end_date, days, status, approver) "
            "VALUES (1, 'EMP00001', 'PTO', '2026-12-01', '2026-12-01', 1, 'Pending', 'People Manager')"
        )
        conn.execute(
            "INSERT INTO expense_claims "
            "(id, employee_id, category, amount, submitted_date, status, description) "
            "VALUES (1, 'EMP00001', 'Travel', 10, '2026-11', 'Pending', 'Taxi')"
        )

    payload = {"action_type": action_type, "params": params}
    employee_tool = HRActionTool(FakeActionLLM(payload))
    denied_action, denial = employee_tool.propose(
        f"Review record {record_id}", "EMP00002", "employee"
    )
    assert denied_action is None
    assert "Only HR Admins" in denial
    forged_action = PendingAction(
        action_type, "Review", params, "EMP00002", "employee"
    )
    with pytest.raises(PermissionError, match="Only HR Admins"):
        employee_tool.execute(forged_action)

    hr_tool = HRActionTool(FakeActionLLM(payload))
    pending_action, _summary = hr_tool.propose(
        f"Review record {record_id}", "HR00001", "hr"
    )
    assert pending_action is not None
    assert f"#{record_id}" in hr_tool.execute(pending_action)

    with sqlite3.connect(db_path) as conn:
        status = conn.execute(f"SELECT status FROM {table} WHERE id = ?", (record_id,)).fetchone()[0]
    assert status == params["decision"]
