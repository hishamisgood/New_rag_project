"""Load policy PDFs with table-aware extraction and page-level OCR."""

import os
from io import BytesIO
from typing import Any

import numpy as np
import pdfplumber
import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from .config import DOCS_DIR
from .database import KNOWLEDGE_SEED

HR_ONLY_DOCUMENTS = {"salary_bands_report.pdf", "scanned_hr_audit_memo_2026.pdf"}
TABLE_CHUNK_SIZE = 1000
MIN_SELECTABLE_TEXT = 40


def _metadata(filename: str, page_number: int, extraction_method: str) -> dict[str, Any]:
    access_level = "hr_only" if filename in HR_ONLY_DOCUMENTS else "public"
    return {
        "source": filename,
        "page": page_number,
        "access_level": access_level,
        "extraction_method": extraction_method,
    }


def _inside_table(char: dict, bounds: list[tuple[float, float, float, float]]) -> bool:
    center_x = (char["x0"] + char["x1"]) / 2
    center_y = (char["top"] + char["bottom"]) / 2
    return any(x0 <= center_x <= x1 and top <= center_y <= bottom
               for x0, top, x1, bottom in bounds)


def _clean_row(row: list[Any]) -> list[str]:
    return [" ".join(str(cell or "").split()).replace("|", "\\|") for cell in row]


def _markdown_header(header: list[str]) -> str:
    return "| " + " | ".join(header) + " |\n| " + " | ".join("---" for _ in header) + " |"


def _table_row_chunks(header: list[str], rows: list[list[str]]) -> list[str]:
    header_text = _markdown_header(header)
    chunks = []
    row_group = []
    current_length = len(header_text)
    for row in rows:
        row_text = "| " + " | ".join(row) + " |"
        if row_group and current_length + len(row_text) + 1 > TABLE_CHUNK_SIZE:
            chunks.append(header_text + "\n" + "\n".join(row_group))
            row_group = []
            current_length = len(header_text)
        row_group.append(row_text)
        current_length += len(row_text) + 1
    if row_group:
        chunks.append(header_text + "\n" + "\n".join(row_group))
    return chunks


def _table_parts(page: Any, active_headers: dict[int, list[str]]) -> tuple[list[tuple[list[str], list[list[str]]]], str]:
    tables = []
    bounds = []
    found_tables = page.find_tables()
    for index, found_table in enumerate(found_tables):
        rows = [_clean_row(row) for row in found_table.extract() if row]
        if not rows:
            continue
        bounds.append(found_table.bbox)
        previous_header = active_headers.get(index)
        first_row = rows[0]
        header = previous_header if previous_header and first_row == previous_header else first_row
        body = rows[1:]
        if not header or not any(header):
            continue
        body = [row for row in body if row != header and any(row)]
        active_headers[index] = header
        tables.append((header, body))

    prose_page = page.filter(
        lambda item: item.get("object_type") != "char" or not _inside_table(item, bounds)
    )
    return tables, prose_page.extract_text() or ""


def _run_ocr(page: Any) -> str:
    from rapidocr_onnxruntime import RapidOCR

    pixmap = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
    result, _elapsed = RapidOCR()(np.asarray(image))
    if not result:
        return ""
    ordered = sorted(result, key=lambda item: (min(point[1] for point in item[0]),
                                               min(point[0] for point in item[0])))
    return "\n".join(str(item[1]).strip() for item in ordered if str(item[1]).strip())


def _load_pdf(filename: str, path: str, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    documents = []
    active_headers: dict[int, list[str]] = {}
    with fitz.open(path) as pdf, pdfplumber.open(path) as parsed_pdf:
        for page_number, (pdf_page, text_page) in enumerate(zip(pdf, parsed_pdf.pages)):
            selectable_text = pdf_page.get_text("text").strip()
            extraction_method = "text"
            if len(selectable_text) < MIN_SELECTABLE_TEXT:
                page_text = _run_ocr(pdf_page)
                extraction_method = "ocr"
                active_headers.clear()
                if page_text:
                    documents.extend(splitter.split_documents([
                        Document(
                            page_content=page_text,
                            metadata=_metadata(filename, page_number, extraction_method),
                        )
                    ]))
                continue

            tables, prose = _table_parts(text_page, active_headers)
            for header, rows in tables:
                for table_chunk in _table_row_chunks(header, rows):
                    documents.append(Document(
                        page_content=table_chunk,
                        metadata=_metadata(filename, page_number, extraction_method),
                    ))
            if prose.strip():
                documents.extend(splitter.split_documents([
                    Document(
                        page_content=prose.strip(),
                        metadata=_metadata(filename, page_number, extraction_method),
                    )
                ]))
    return documents


def load_and_split_documents(docs_dir: str = DOCS_DIR) -> list[Document]:
    """Extract prose, stitched table rows, and OCR text from policy PDFs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        try:
            documents = _load_pdf(filename, os.path.join(docs_dir, filename), splitter)
            chunks.extend(documents)
            print(f"Loaded {filename} ({len(documents)} chunks)")
        except Exception as exc:
            print(f"Failed to load {filename}: {exc}")

    if not chunks:
        raise FileNotFoundError(f"No readable PDF documents found in {docs_dir}.")

    operational_facts = [
        Document(
            page_content=f"{title}: {content}",
            metadata={
                "source": f"company_guide:{title}",
                "key": key,
                "access_level": "public",
                "extraction_method": "seed",
            },
        )
        for key, title, content in KNOWLEDGE_SEED
    ]
    chunks.extend(operational_facts)
    print(f"Created {len(chunks) - len(operational_facts)} PDF chunks and added "
          f"{len(operational_facts)} operational facts")
    return chunks
