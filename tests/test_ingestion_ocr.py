"""Integration checks for scanned-page OCR and multi-page table extraction."""

import pymupdf as fitz
import pytest

from src.config import DOCS_DIR
from src.ingestion import load_and_split_documents


@pytest.fixture(scope="module")
def chunks():
    return load_and_split_documents()


def _source_chunks(chunks, source):
    return [chunk for chunk in chunks if chunk.metadata.get("source") == source]


def _page_texts(filename):
    with fitz.open(f"{DOCS_DIR}/{filename}") as pdf:
        return [page.get_text("text") for page in pdf]


def test_scanned_pages_have_no_selectable_text_and_ocr_extracts_facts(chunks):
    remote_pages = _page_texts("scanned_remote_work_addendum_2026.pdf")
    leave_pages = _page_texts("leave_policy_report.pdf")
    assert all(not page.strip() for page in remote_pages)
    assert not leave_pages[-1].strip()

    remote_chunks = _source_chunks(chunks, "scanned_remote_work_addendum_2026.pdf")
    leave_appendix = [
        chunk for chunk in _source_chunks(chunks, "leave_policy_report.pdf")
        if chunk.metadata.get("page") == len(leave_pages) - 1
    ]
    remote_text = " ".join(chunk.page_content.lower() for chunk in remote_chunks)
    leave_text = " ".join(chunk.page_content.lower() for chunk in leave_appendix)
    assert "30 calendar days" in remote_text
    assert "$150" in remote_text
    assert "10" in leave_text and "immediate family" in leave_text
    assert remote_chunks and all(chunk.metadata["extraction_method"] == "ocr"
                                 for chunk in remote_chunks)
    assert leave_appendix and all(chunk.metadata["extraction_method"] == "ocr"
                                  for chunk in leave_appendix)


def test_hr_documents_and_public_scanned_addendum_have_access_tags(chunks):
    for filename in ("scanned_hr_audit_memo_2026.pdf", "salary_bands_report.pdf"):
        documents = _source_chunks(chunks, filename)
        assert documents
        assert all(chunk.metadata["access_level"] == "hr_only" for chunk in documents)

    audit_chunks = _source_chunks(chunks, "scanned_hr_audit_memo_2026.pdf")
    audit_text = " ".join(chunk.page_content for chunk in audit_chunks)
    assert "4.5%" in audit_text and "$180,000" in audit_text
    assert all(chunk.metadata["extraction_method"] == "ocr" for chunk in audit_chunks)

    public_documents = _source_chunks(chunks, "scanned_remote_work_addendum_2026.pdf")
    assert public_documents
    assert all(chunk.metadata["access_level"] == "public" for chunk in public_documents)
    assert all(chunk.metadata["extraction_method"] == "seed"
               and chunk.metadata["access_level"] == "public"
               for chunk in chunks if chunk.metadata.get("source", "").startswith("company_guide:"))


@pytest.mark.parametrize(
    ("filename", "header", "last_page_minimum"),
    [
        ("benefits_guide.pdf", "Matrix ID", 2),
        ("employee_handbook.pdf", "Schedule ID", 2),
        ("salary_bands_report.pdf", "Role Code", 3),
    ],
)
def test_long_tables_span_pages_and_repeat_markdown_headers(chunks, filename, header,
                                                            last_page_minimum):
    page_texts = _page_texts(filename)
    table_pages = [index for index, text in enumerate(page_texts)
                   if header in text]
    assert len(table_pages) >= last_page_minimum + 1

    table_chunks = [
        chunk for chunk in _source_chunks(chunks, filename)
        if chunk.metadata.get("page") in table_pages and header in chunk.page_content
    ]
    assert len(table_chunks) >= last_page_minimum + 1
    later_chunks = [chunk for chunk in table_chunks if chunk.metadata["page"] > table_pages[0]]
    assert later_chunks
    assert all(chunk.page_content.startswith("|") and header in chunk.page_content
               for chunk in later_chunks)


def test_table_narrative_and_buried_rows_are_retained(chunks):
    benefits = _source_chunks(chunks, "benefits_guide.pdf")
    handbook = _source_chunks(chunks, "employee_handbook.pdf")
    salary = _source_chunks(chunks, "salary_bands_report.pdf")

    benefits_text = " ".join(chunk.page_content.lower() for chunk in benefits)
    handbook_text = " ".join(chunk.page_content.lower() for chunk in handbook)
    assert "48 hours" in benefits_text
    assert "pre-authorization requirement is waived" in benefits_text
    assert "$25 or more" in handbook_text
    assert "foreign-currency purchases" in handbook_text
    assert any("ben-063" in chunk.page_content.lower() for chunk in benefits)
    assert any("trv-068" in chunk.page_content.lower() for chunk in handbook)
    assert any("nc-087" in chunk.page_content.lower() for chunk in salary)
