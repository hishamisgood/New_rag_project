"""Shared application paths and model defaults."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT_DIR / "data")
DB_PATH = str(ROOT_DIR / "data" / "employees.db")
EXCEL_PATH = str(ROOT_DIR / "data" / "employees.xlsx")
DOCS_DIR = str(ROOT_DIR / "data" / "documents")
CHROMA_DIR = str(ROOT_DIR / "chroma_db")
COLLECTION_NAME = "novacorp_docs"

GEMINI_MODEL = "models/gemini-2.5-flash"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "models/gemini-embedding-001"
