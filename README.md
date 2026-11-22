# NovaCorp Employee RAG Portal

An employee-facing retrieval-augmented assistant built with Streamlit, LangChain Core,
ChromaDB, and Google Gemini. It combines company policy documents, operational facts,
role-scoped employee records, confirmation-gated HR actions, and optional public web search.

## Features

- SQLite authentication with bcrypt password hashes and an employee profile dashboard.
- A router for policy retrieval, read-only SQL, HR actions, and external web search.
- Hybrid policy retrieval using ChromaDB and the project's custom `RRFRetriever`.
- Role-based access control for documents, SQL data, and action permissions.
- Proposals for leave requests and expense claims that require explicit confirmation before writing.
- Optional Tavily or Serper search for public information outside NovaCorp's policies.
- Table-aware extraction for long, multi-page PDFs and RapidOCR fallback for scanned pages.

## Architecture

The RAG chain is plain `langchain_core`; the project has zero `langchain_classic` dependency.
The custom `RRFRetriever` combines semantic and BM25 results. When Gemini embeddings are
unavailable, BM25 retrieval remains available offline.

The assistant routes each question to one of four paths:

1. **Policy search:** hybrid retrieval over generated PDFs and 12 operational seed facts.
2. **SQL:** generated read-only SQLite queries validated against approved tables and columns.
3. **HR actions:** a structured proposal for a leave request or expense claim, followed by
   a user confirmation card before any database write.
4. **Public web search:** optional Tavily or Serper results for external questions.

### Role-based access

Standard employees retrieve public documents and can query only their own employee and
transaction records. Human Resources users can retrieve HR-only compensation documents
and run company-wide SQL queries. Only HR users can approve or reject leave requests and
expense claims; employees may submit their own requests. SQL is read-only and HR writes use
fixed, parameterized statements.

Document access is stored in each chunk's `access_level` metadata. The salary bands report
and scanned HR audit memo are tagged `hr_only`; other policy documents are public. SQL
scoping and action permissions are enforced independently of the document retriever.

### PDF extraction

`src/ingestion.py` uses PyMuPDF and pdfplumber to separate table regions from surrounding
prose. Large tables are chunked by row group with their Markdown header repeated on every
chunk, including later pages. Narrative text beneath a table is extracted separately.
Pages with fewer than 40 characters of selectable text are rendered at 200 DPI and read
with RapidOCR (`rapidocr-onnxruntime`), so both image-only PDFs and scanned pages inside
otherwise native-text PDFs enter the same retrieval index. Each chunk records its source,
page, access level, and extraction method.

Generated material includes 3-page benefits and travel schedules, a 4-page compensation
directory, two standalone scanned addenda/memos, and a scanned bereavement appendix page
in the leave policy report.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set `GOOGLE_API_KEY` for Gemini embeddings and answers.
Generate the demo data and documents, then build the vector store:

```bash
python scripts/generate_employees.py
python scripts/init_db.py
python scripts/generate_documents.py
python scripts/build_vectorstore.py
streamlit run app.py
```

`data/employees.xlsx` contains synthetic plaintext demo credentials for local sign-in.
It is demo data, not a secure credential store. Authentication uses bcrypt hashes in SQLite.

The 12 static operational facts are defined in `src/database.py` as `KNOWLEDGE_SEED`.
After editing those facts, rebuild ChromaDB with:

```bash
python scripts/build_vectorstore.py
```

## Generated documents

The six native-text policy PDFs cover the employee handbook, onboarding, return-to-office,
salary bands, benefits, and leave. The generator also creates:

- `scanned_remote_work_addendum_2026.pdf` — two image-only public pages.
- `scanned_hr_audit_memo_2026.pdf` — one image-only HR-only page.
- A final image-only sign-off and bereavement addendum page in `leave_policy_report.pdf`.

## Evaluation and checks

```bash
pytest
python scripts/verify_upgrade.py
python scripts/evaluate_rag.py --dry-run
```

The dry run checks seed coverage and transaction tables without calling an LLM. Live
evaluation also checks expected facts and source retrieval, including later-page table
rows, scanned OCR facts, and HR-scoped cases.

## Project structure

```text
app.py
src/
  agent.py                 # Router, plain langchain_core chain, and tool context
  action_tool.py           # Confirmation-gated leave and expense writes
  auth.py                  # Authentication and access-level mapping
  database.py              # Employee data, transactions, and KNOWLEDGE_SEED
  ingestion.py             # Table-aware PDF extraction and RapidOCR fallback
  retriever.py             # Custom RRFRetriever and role filters
  sql_tool.py              # Validated, read-only SQLite queries
  web_search_tool.py       # Tavily and Serper adapters
scripts/
  generate_documents.py    # Native policy PDFs and scanned-image PDFs
  build_vectorstore.py
  evaluate_rag.py
  verify_upgrade.py
tests/
```
