"""Validated, confirmation-gated writes for employee HR actions."""

import json
import logging
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from typing import Any

from .config import DB_PATH
from .llm import extract_message_text, get_llm

logger = logging.getLogger(__name__)

LEAVE_TYPES = {"PTO", "Sick", "Mental Health", "Volunteer"}
EXPENSE_CATEGORIES = {"Travel", "Meals", "Equipment", "Training", "Books"}
REVIEW_ACTIONS = {"review_leave_request", "review_expense_claim"}


def _positive_int(params: dict[str, Any], key: str) -> int:
    value = params.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        label = {"days": "Days", "request_id": "Request ID", "claim_id": "Claim ID"}.get(
            key, key.replace("_", " ").capitalize()
        )
        raise ValueError(f"{label} must be a positive whole number.")
    return value


@dataclass
class PendingAction:
    """A validated action waiting for an explicit user confirmation."""

    action_type: str
    summary: str
    params: dict[str, Any]
    employee_id: str
    access_level: str


class HRActionTool:
    """Extract action fields with an LLM and execute fixed SQL after confirmation."""

    name = "hr_action"

    def __init__(self, llm: Any | None = None):
        self.llm = llm

    def _parse_payload(self, question: str) -> tuple[str, dict[str, Any]]:
        llm = self.llm or get_llm(temperature=0)
        prompt = f"""Extract one NovaCorp HR action from the user's request.
Return only a JSON object with `action_type` and `params`. Do not return SQL.
Allowed action_type values and params:
- submit_leave_request: leave_type, start_date, end_date, days
- submit_expense_claim: category, amount, description
- review_leave_request: request_id, decision
- review_expense_claim: claim_id, decision
Leave types: PTO, Sick, Mental Health, Volunteer.
Expense categories: Travel, Meals, Equipment, Training, Books.
Review decisions: Approved or Rejected.
Use null for any missing field. User request: {question}"""
        raw = extract_message_text(llm.invoke(prompt)).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("I couldn't identify the requested action details.")
        try:
            payload = json.loads(raw[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("I couldn't read the requested action details.") from exc
        if not isinstance(payload, dict):
            raise ValueError("The requested action details must be a JSON object.")
        action_type = payload.get("action_type")
        if not isinstance(action_type, str):
            raise ValueError("I couldn't identify the requested HR action.")
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("The requested action parameters are invalid.")
        return action_type, params

    @staticmethod
    def _required_text(params: dict[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Please provide a valid {key.replace('_', ' ')}.")
        return value.strip()

    @classmethod
    def _validate(cls, action_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if action_type == "submit_leave_request":
            leave_type = cls._required_text(params, "leave_type")
            if leave_type not in LEAVE_TYPES:
                raise ValueError("Leave type must be PTO, Sick, Mental Health, or Volunteer.")
            dates = {}
            for key in ("start_date", "end_date"):
                value = cls._required_text(params, key)
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    raise ValueError(f"{key.replace('_', ' ').capitalize()} must use YYYY-MM-DD.")
                try:
                    parsed = date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(f"{key.replace('_', ' ').capitalize()} is not a valid date.") from exc
                dates[key] = parsed
            if dates["end_date"] < dates["start_date"]:
                raise ValueError("End date must be on or after start date.")
            days = _positive_int(params, "days")
            return {
                "leave_type": leave_type,
                "start_date": dates["start_date"].isoformat(),
                "end_date": dates["end_date"].isoformat(),
                "days": days,
            }

        if action_type == "submit_expense_claim":
            category = cls._required_text(params, "category")
            if category not in EXPENSE_CATEGORIES:
                raise ValueError("Expense category must be Travel, Meals, Equipment, Training, or Books.")
            amount = params.get("amount")
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                raise ValueError("Amount must be a positive number.")
            amount = float(amount)
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError("Amount must be a positive number.")
            return {
                "category": category,
                "amount": amount,
                "description": cls._required_text(params, "description"),
            }

        if action_type in REVIEW_ACTIONS:
            id_key = "request_id" if action_type == "review_leave_request" else "claim_id"
            record_id = _positive_int(params, id_key)
            decision = cls._required_text(params, "decision")
            if decision not in {"Approved", "Rejected"}:
                raise ValueError("Decision must be Approved or Rejected.")
            return {id_key: record_id, "decision": decision}

        raise ValueError("That HR action is not supported.")

    @staticmethod
    def _summary(action_type: str, params: dict[str, Any]) -> str:
        if action_type == "submit_leave_request":
            return (
                f"Submit {params['days']} day(s) of {params['leave_type']} leave "
                f"from {params['start_date']} to {params['end_date']}."
            )
        if action_type == "submit_expense_claim":
            return (
                f"Submit a {params['category']} expense claim for ${params['amount']:.2f}: "
                f"{params['description']}"
            )
        if action_type == "review_leave_request":
            return f"{params['decision']} leave request #{params['request_id']}."
        return f"{params['decision']} expense claim #{params['claim_id']}."

    def propose(
        self, question: str, employee_id: str | None, access_level: str
    ) -> tuple[PendingAction | None, str]:
        """Parse and validate a proposed action without touching the database."""
        if not employee_id:
            return None, "Sign in before submitting or reviewing an HR action."
        try:
            action_type, params = self._parse_payload(question)
            if action_type in REVIEW_ACTIONS and access_level != "hr":
                raise PermissionError("Only HR Admins can approve or reject requests.")
            params = self._validate(action_type, params)
            summary = self._summary(action_type, params)
            return PendingAction(action_type, summary, params, employee_id, access_level), summary
        except (ValueError, PermissionError) as exc:
            return None, str(exc)
        except Exception:
            logger.exception("HR action proposal failed")
            return None, "I couldn't prepare that HR action. Please check the details and try again."

    def execute(self, action: PendingAction) -> str:
        """Execute a validated action using fixed parameterized SQLite statements."""
        if action.action_type in REVIEW_ACTIONS and action.access_level != "hr":
            raise PermissionError("Only HR Admins can approve or reject requests.")
        params = self._validate(action.action_type, action.params)

        with closing(sqlite3.connect(DB_PATH)) as conn, conn:
            if action.action_type == "submit_leave_request":
                if not action.employee_id:
                    raise ValueError("An employee ID is required to submit a leave request.")
                cursor = conn.execute(
                    "INSERT INTO leave_requests "
                    "(employee_id, leave_type, start_date, end_date, days, status, approver) "
                    "VALUES (?, ?, ?, ?, ?, 'Pending', 'People Manager')",
                    (
                        action.employee_id, params["leave_type"], params["start_date"],
                        params["end_date"], params["days"],
                    ),
                )
                return f"Leave request #{cursor.lastrowid} submitted with Pending status."

            if action.action_type == "submit_expense_claim":
                if not action.employee_id:
                    raise ValueError("An employee ID is required to submit an expense claim.")
                cursor = conn.execute(
                    "INSERT INTO expense_claims "
                    "(employee_id, category, amount, submitted_date, status, description) "
                    "VALUES (?, ?, ?, ?, 'Pending', ?)",
                    (
                        action.employee_id, params["category"], params["amount"],
                        date.today().strftime("%Y-%m"), params["description"],
                    ),
                )
                return f"Expense claim #{cursor.lastrowid} submitted with Pending status."

            if action.action_type == "review_leave_request":
                record_id = params["request_id"]
                cursor = conn.execute(
                    "UPDATE leave_requests SET status = ? WHERE id = ?",
                    (params["decision"], record_id),
                )
            else:
                record_id = params["claim_id"]
                cursor = conn.execute(
                    "UPDATE expense_claims SET status = ? WHERE id = ?",
                    (params["decision"], record_id),
                )
            if cursor.rowcount == 0:
                raise ValueError(f"No record found with ID {record_id}.")
            return f"Record #{record_id} updated to {params['decision']}."
