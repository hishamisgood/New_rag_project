# NovaCorp Employee RAG Portal 🏢

A Retrieval-Augmented Generation (RAG) application for company employees, built with **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini**.

## Features

- 🔐 **Secure Authentication** — SQLite-backed login with bcrypt-hashed passwords
- 👤 **Employee Dashboard** — Personal profile with salary, leave stats, and monthly absence charts
- 🤖 **Dynamic RAG Agent** — An LLM router selects policy retrieval or employee-scoped analytical SQL for each question
- 🧠 **Resilient LLMs** — Gemini is preferred, with local Ollama `llama3.2` as the automatic fallback; BM25 keeps retrieval available offline
- 🔍 **Hybrid Search** — ChromaDB + BM25 with reciprocal-rank fusion indexes policies and 12 operational facts
- 📊 **Employee Analytics** — Read-only SQL over expense claims, leave requests, payroll history, and training enrollments
- 📄 **Rich Documents** — PDFs with charts, tables, and complex layouts

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                             │
│  Login + Profile ── SQLite employees table                       │
│                                                                  │
│  Chat ── Condense follow-up ── Dynamic LLM Router                │
│                                  ├─ POLICY ─ Hybrid RRF Search    │
│                                  │           ├─ ChromaDB          │
│                                  │           │  6 PDFs + 12 facts │
│                                  │           └─ BM25              │
│                                  └─ SQL ─ Read-only scoped SQLite │
│                                            ├─ expense_claims      │
│                                            ├─ leave_requests      │
│                                            ├─ payroll_history     │
│                                            └─ training_enrollments│
│                                                                  │
│  Gemini ── automatic fallback to Ollama llama3.2                 │
│  No Google key ── BM25 policy retrieval remains available         │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
cp .env.example .env
# Edit .env and add your Google API key from https://aistudio.google.com/app/apikey
```

### 3. Generate Data & Build Knowledge Base

```bash
# Step 1: Generate 2,000 synthetic employees
python scripts/generate_employees.py

# Step 2: Load employees into SQLite (with hashed passwords)
python scripts/init_db.py

# Step 3: Generate PDF documents (policies, reports with charts/tables)
python scripts/generate_documents.py

# Step 4: Ingest documents into ChromaDB vector store
python scripts/build_vectorstore.py
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Log In

`data/employees.xlsx` contains synthetic test credentials for demo login (ID + password columns).

Example: Log in with `EMP00001` and the corresponding password from the Excel file.

## Project Structure

```
NEW_RAG_Project/
├── app.py                      # Streamlit UI (streamlit run app.py)
├── src/
│   ├── __init__.py
│   ├── config.py               # Shared paths and model names
│   ├── database.py             # SQLite employees, knowledge, and transactions
│   ├── auth.py                 # Password hashing and login verification
│   ├── ingestion.py            # PDF chunks and operational facts
│   ├── retriever.py            # Embeddings, ChromaDB, and hybrid search
│   ├── llm.py                  # Gemini with Ollama fallback
│   ├── sql_tool.py             # Read-only generated SQL
│   └── agent.py                # RAG chain, router, and agent
├── scripts/
│   ├── generate_employees.py
│   ├── generate_documents.py
│   ├── init_db.py
│   ├── build_vectorstore.py
│   ├── evaluate_rag.py
│   └── verify_upgrade.py
├── tests/
│   ├── conftest.py
│   ├── test_database_knowledge.py
│   ├── test_llm_sql_tool.py
│   └── test_rag_agent.py
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── data/
│   ├── employees.xlsx          # Generated employee data
│   ├── employees.db            # SQLite database
│   └── documents/              # Generated PDF documents
│       ├── employee_handbook.pdf
│       ├── onboarding_policy.pdf
│       ├── return_to_office_policy.pdf
│       ├── salary_bands_report.pdf
│       ├── benefits_guide.pdf
│       └── leave_policy_report.pdf
└── chroma_db/                  # ChromaDB persistent vector store
```

## Documents Generated

| Document | Content | Visual Elements |
|----------|---------|-----------------|
| Employee Handbook | Company overview, code of conduct, performance review | Dept pie chart, disciplinary action table, performance rating table |
| Onboarding Policy | 30/60/90 day plan, checklists, training modules | Multi-column tables, IT setup table |
| Return to Office Policy | Hybrid schedule, safety guidelines | Office utilisation bar chart, rotation table |
| Salary Bands Report | Compensation analysis across departments | Salary histogram, heatmap, bar chart |
| Benefits Guide | Health insurance, retirement, wellness | Plan comparison tables, benefits value bar chart |
| Leave Policy Report | Leave types, absence analytics | Pie chart, bar chart, monthly trends |

## How It Works

### Authentication
1. Employee enters ID + password in the login form
2. SQLite lookup with bcrypt hash verification
3. Employee profile loaded into session state

### RAG Pipeline
1. **Document Loading**: PDFs loaded with PyPDFLoader
2. **Chunking**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
3. **Embeddings**: Google Generative AI Embeddings (`models/gemini-embedding-001`)
4. **Vector Store**: ChromaDB (persistent on disk)
5. **Hybrid Search**: RRFRetriever (Reciprocal Rank Fusion) combines:
   - ChromaDB semantic search (50% weight)
   - BM25 keyword search (50% weight)
6. **LLM**: Gemini generates answers from retrieved context + employee profile
7. **History**: Bounded list (last 20 messages = 10 exchanges) stored on RAGAgent

### Operational facts and transactional analytics

The six policy PDFs and 12 static operational facts (including payroll timing,
IT support, HR response SLAs, the remote stipend, and holidays) are indexed in
ChromaDB and BM25. The LLM router sends employee transaction questions to
read-only, employee-scoped SQL over `expense_claims`, `leave_requests`,
`payroll_history`, and `training_enrollments`. It can combine SQL results with
retrieved policies, such as comparing training spend with the annual budget.
The app creates and seeds these tables at startup. Run the offline checks with:

```bash
python scripts/evaluate_rag.py --dry-run
```

For end-to-end evaluation, build ChromaDB and ensure Gemini or Ollama is
available, then run `python scripts/evaluate_rag.py`.

### Employee Context
When a user logs in, their full profile (salary, department, leave days, etc.)
is injected into the LLM system prompt. This allows questions like:
- "What is my salary?"
- "How many sick days do I have?"
- "Am I eligible for paternity leave?"

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Auth | SQLite + bcrypt |
| LLM | Google Gemini (via langchain-google-genai) |
| Embeddings | Google Generative AI Embeddings |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Hybrid Search | Custom RRFRetriever (Reciprocal Rank Fusion, zero langchain_classic) |
| Data Generation | Faker |
| PDF Generation | ReportLab + Matplotlib |
| PDF Parsing | PyPDF |
