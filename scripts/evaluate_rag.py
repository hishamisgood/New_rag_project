"""Evaluate policy retrieval, operational facts, and employee SQL questions."""

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_PATH
from src.database import (
    KNOWLEDGE_SEED,
    ensure_knowledge_base,
    ensure_transactional_tables,
    get_employee,
    TRANSACTIONAL_TABLES,
)

TEST_EMPLOYEE = "EMP00001"
EVALUATION_SET = [
    {"kind": "pdf", "question": "What are the core conduct expectations?", "expected_source": "employee_handbook.pdf"},
    {"kind": "pdf", "question": "What should I complete during my first 30 days?", "expected_source": "onboarding_policy.pdf"},
    {"kind": "pdf", "question": "How often am I expected to work from the office?", "expected_source": "return_to_office_policy.pdf"},
    {"kind": "pdf", "question": "What salary band applies to a senior engineer?", "expected_source": "salary_bands_report.pdf"},
    {"kind": "pdf", "question": "What health plan benefits and copays are available?", "expected_source": "benefits_guide.pdf"},
    {"kind": "pdf", "question": "How much paid leave can employees take?", "expected_source": "leave_policy_report.pdf"},
    {"kind": "guide", "question": "When is payroll processed?", "guide_key": "payroll", "expected_source": "company_guide:Payroll schedule"},
    {"kind": "guide", "question": "Where can I get IT support?", "guide_key": "it_support", "expected_source": "company_guide:IT support"},
    {"kind": "guide", "question": "What is the HR service-level agreement?", "guide_key": "hr_sla", "expected_source": "company_guide:HR service-level agreement"},
    {"kind": "guide", "question": "What is the remote-work stipend?", "guide_key": "remote_stipend", "expected_source": "company_guide:Remote-work stipend"},
    {"kind": "guide", "question": "How many company holidays do we get?", "guide_key": "holidays", "expected_source": "company_guide:Company holidays"},
    {"kind": "sql", "question": "How much have I been reimbursed for travel this year?", "table": "expense_claims", "query_key": "travel"},
    {"kind": "sql", "question": "Which of my expense claims are still pending approval?", "table": "expense_claims", "query_key": "pending_expenses"},
    {"kind": "sql", "question": "What was my net pay in September and did I receive a bonus in 2026?", "table": "payroll_history", "query_key": "september_pay"},
    {"kind": "sql", "question": "How much of my $3,000 annual Learning & Development budget do I have left?", "table": "training_enrollments", "query_key": "training_spend"},
    {"kind": "sql", "question": "What is the status of my recent leave requests?", "table": "leave_requests", "query_key": "recent_leave"},
    {"kind": "sql", "question": "How much have I claimed for Books, and how much of the $300 book allowance remains?", "table": "expense_claims", "query_key": "book_spend"},
    {"kind": "refusal", "question": "What is NovaCorp's stock ticker?"},
    {"kind": "refusal", "question": "What was NovaCorp's revenue last quarter?"},
    {"kind": "refusal", "question": "Who will win next year's national election?"},
]

DRY_RUN_QUERIES = {
    "travel": (
        "SELECT SUM(amount) FROM expense_claims "
        "WHERE employee_id = ? AND category = 'Travel'"
    ),
    "pending_expenses": (
        "SELECT COUNT(id) FROM expense_claims "
        "WHERE employee_id = ? AND status = 'Pending'"
    ),
    "september_pay": (
        "SELECT net_pay FROM payroll_history "
        "WHERE employee_id = ? AND pay_month = '2026-09'"
    ),
    "training_spend": (
        "SELECT SUM(cost) FROM training_enrollments WHERE employee_id = ?"
    ),
    "recent_leave": (
        "SELECT COUNT(id) FROM leave_requests WHERE employee_id = ?"
    ),
    "book_spend": (
        "SELECT SUM(amount) FROM expense_claims "
        "WHERE employee_id = ? AND category = 'Books'"
    ),
}
REFUSAL_MARKERS = (
    "not available", "not in the context", "not in the available", "not covered",
    "don't have information", "do not have information", "cannot verify",
    "can't verify", "outside the available", "don't have access", "do not have access",
)


def _dry_run() -> dict:
    ensure_knowledge_base()
    ensure_transactional_tables()
    seeded_facts = {key: title for key, title, _content in KNOWLEDGE_SEED}
    guide_cases = [case for case in EVALUATION_SET if case["kind"] == "guide"]
    results = []
    for case in guide_cases:
        present = case["guide_key"] in seeded_facts
        results.append({
            **case,
            "sources": [case["expected_source"]] if present else [],
            "passed": present,
        })
    all_seed_content_present = all(
        key and title and content for key, title, content in KNOWLEDGE_SEED
    )
    results.append({
        "kind": "guide_coverage",
        "question": "Are all operational facts represented in KNOWLEDGE_SEED?",
        "sources": [f"company_guide:{title}" for _key, title, _content in KNOWLEDGE_SEED],
        "passed": len(seeded_facts) == len(KNOWLEDGE_SEED) and all_seed_content_present,
    })

    table_checks = {}
    with closing(sqlite3.connect(DB_PATH)) as conn:
        for table in TRANSACTIONAL_TABLES:
            try:
                row_count = conn.execute(
                    f"SELECT count(*) FROM {table} WHERE employee_id = ?", (TEST_EMPLOYEE,)
                ).fetchone()[0]
                present = row_count > 0
            except sqlite3.OperationalError:
                row_count = 0
                present = False
            table_checks[table] = {"present": present, "test_employee_rows": row_count}

        for case in [item for item in EVALUATION_SET if item["kind"] == "sql"]:
            try:
                row = conn.execute(DRY_RUN_QUERIES[case["query_key"]], (TEST_EMPLOYEE,)).fetchone()
                passed = row is not None and table_checks[case["table"]]["present"]
            except sqlite3.OperationalError:
                passed = False
            results.append({
                **case,
                "sources": [f"database:{case['table']}"] if passed else [],
                "passed": passed,
            })

    skipped = sum(case["kind"] in {"pdf", "refusal"} for case in EVALUATION_SET)
    return {
        "mode": "dry-run",
        "total": len(results),
        "passed": sum(result["passed"] for result in results),
        "skipped_live_only": skipped,
        "transactional_tables": table_checks,
        "cases": results,
    }


def _live_evaluate() -> dict:
    ensure_knowledge_base()
    ensure_transactional_tables()
    from src.agent import build_rag_agent, query_rag
    from src.retriever import create_hybrid_retriever, load_vectorstore

    agent = build_rag_agent(
        create_hybrid_retriever(load_vectorstore()),
        employee_info=get_employee(TEST_EMPLOYEE),
    )
    results = []
    for case in EVALUATION_SET:
        answer = query_rag(agent, case["question"])
        sources = answer["sources"]
        if case["kind"] == "refusal":
            answer_text = answer["answer"].lower()
            passed = any(marker in answer_text for marker in REFUSAL_MARKERS)
        else:
            expected = case.get("expected_source", f"database:{case.get('table', '')}")
            passed = any(source == expected or source.startswith(expected + " (p. ") for source in sources)
        results.append({
            **case,
            "sources": sources,
            "answer": answer["answer"],
            "passed": passed,
        })
    return {
        "mode": "live",
        "total": len(results),
        "passed": sum(item["passed"] for item in results),
        "cases": results,
    }


def evaluate(dry_run: bool = False) -> dict:
    return _dry_run() if dry_run else _live_evaluate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.dry_run)
    (PROJECT_ROOT / "rag_evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
