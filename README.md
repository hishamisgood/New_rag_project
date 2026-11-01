# NovaCorp Employee RAG Portal 🏢

A Retrieval-Augmented Generation (RAG) application for company employees, built with **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini**.

## Features

- 🔐 **Secure Authentication** — SQLite-backed login with bcrypt-hashed passwords
- 👤 **Employee Dashboard** — Personal profile with salary, leave stats, and monthly absence charts
- 🤖 **RAG Chat Interface** — Ask questions about company policies, benefits, and your own profile
- 🔍 **Hybrid Search** — Combines semantic search (ChromaDB) with keyword search (BM25) for best retrieval
- 📄 **Rich Documents** — PDFs with charts, tables, and complex layouts

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  ┌──────────┐  ┌───────────────────────────────────┐    │
│  │  Login   │  │         RAG Chat Interface        │    │
│  │  Screen  │  │  ┌─────────┐    ┌──────────────┐  │    │
│  └────┬─────┘  │  │ Hybrid  │───>│   Gemini     │  │    │
│       │        │  │ Search  │    │   LLM        │  │    │
│       ▼        │  └────┬────┘    └──────────────┘  │    │
│  ┌──────────┐  │  ┌────┴────────────────┐          │    │
│  │ SQLite   │  │  │  ┌───────┐ ┌─────┐  │          │    │
│  │ Auth DB  │  │  │  │ChromaDB││BM25 │  │          │    │
│  └──────────┘  │  │  │Semantic││Key- │  │          │    │
│                │  │  │Search  ││word │  │          │    │
│                │  │  └───────┘ └─────┘  │          │    │
│                │  └─────────────────────┘          │    │
│                └───────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
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
python generate_employees.py

# Step 2: Load employees into SQLite (with hashed passwords)
python auth.py

# Step 3: Generate PDF documents (policies, reports with charts/tables)
python generate_documents.py

# Step 4: Ingest documents into ChromaDB vector store
python build_vectorstore.py
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Log In

Open `data/employees.xlsx` to find test credentials (ID + password columns).

Example: Log in with `EMP00001` and the corresponding password from the Excel file.

## Project Structure

```
NEW_RAG_Project/
├── app.py                      # Streamlit app (login + dashboard + RAG chat)
├── auth.py                     # Authentication (SQLite + bcrypt)
├── rag_engine.py               # RAG engine (LangChain + Gemini + Hybrid Search)
├── build_vectorstore.py        # Ingest documents → ChromaDB
├── generate_employees.py       # Generate 2,000 employees with Faker → Excel
├── generate_documents.py       # Generate 6 PDFs with charts, tables, policies
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
3. **Embeddings**: Google Generative AI Embeddings (`embedding-001`)
4. **Vector Store**: ChromaDB (persistent on disk)
5. **Hybrid Search**: EnsembleRetriever combines:
   - ChromaDB semantic search (50% weight)
   - BM25 keyword search (50% weight)
6. **LLM**: Gemini generates answers from retrieved context + employee profile
7. **Memory**: ConversationBufferWindowMemory (last 10 exchanges)

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
| Hybrid Search | LangChain EnsembleRetriever |
| Data Generation | Faker |
| PDF Generation | ReportLab + Matplotlib |
| PDF Parsing | PyPDF |
