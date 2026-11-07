"""Evaluate representative NovaCorp RAG questions and save a JSON report.

Run after building the vector store: ``python evaluate_rag.py``.
Use ``--dry-run`` to validate SQLite-backed RAG knowledge without an LLM.
"""
import argparse
import json
from pathlib import Path

from auth import ensure_knowledge_base, search_knowledge

EVALUATION_SET = [
    {"question": "When is payroll processed?", "expected_source": "database:Payroll schedule"},
    {"question": "How do I submit an expense reimbursement?", "expected_source": "database:Expense reimbursement"},
    {"question": "Where can I get IT support?", "expected_source": "database:IT support"},
]


def evaluate(dry_run: bool = False) -> dict:
    ensure_knowledge_base()
    results = []
    if dry_run:
        for case in EVALUATION_SET:
            rows = search_knowledge(case["question"])
            sources = [f"database:{row['title']}" for row in rows]
            results.append({**case, "sources": sources, "passed": case["expected_source"] in sources})
    else:
        from rag_engine import build_rag_agent, create_hybrid_retriever, load_vectorstore, query_rag
        retriever = create_hybrid_retriever(load_vectorstore())
        agent = build_rag_agent(retriever)
        for case in EVALUATION_SET:
            answer = query_rag(agent, case["question"])
            results.append({**case, "sources": answer["sources"], "passed": case["expected_source"] in answer["sources"]})
    return {"total": len(results), "passed": sum(item["passed"] for item in results), "cases": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.dry_run)
    Path("rag_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)
