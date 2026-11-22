"""
NovaCorp Employee Portal — Streamlit Application

Features
--------
• Login screen authenticated via SQLite (ID + password)
• Employee profile dashboard in the sidebar
• RAG-powered chat interface (hybrid search + Gemini)

Run with:  streamlit run app.py
"""

import html
import logging
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="NovaCorp Employee Portal",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.action_tool import PendingAction
from src.auth import authenticate, get_access_level
from src.config import DB_PATH
from src.database import ensure_transactional_tables
from src.agent import build_rag_agent, query_rag
from src.retriever import (
    load_vectorstore,
    create_hybrid_retriever,
)

logger = logging.getLogger(__name__)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Chat message styling */
    .source-badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        background-color: #e3f2fd;
        color: #1565c0;
        font-size: 0.75rem;
    }

    /* Sidebar profile */
    .profile-card {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin-bottom: 1rem;
    }
    .profile-card h3 { margin-bottom: 0.3rem; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────
def init_session_state(reset: bool = False) -> None:
    defaults = {
        "authenticated": False,
        "employee": None,
        "chat_history": [],
        "rag_chain": None,
        "pending_prompt": None,
        "pending_action": None,
    }
    for key, val in defaults.items():
        if reset or key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def show_login():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; margin-bottom: 1rem;">
            <h1 style="color: #1a237e;">🏢 NovaCorp</h1>
            <h3 style="color: #455a64;">Employee Portal</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("##### 🔐 Sign In")
            emp_id = st.text_input(
                "Employee ID",
                placeholder="e.g., EMP00001",
                help="Your unique employee ID (found in your welcome email)",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )
            submitted = st.form_submit_button(
                "Sign In", use_container_width=True, type="primary"
            )

            if submitted:
                if not emp_id or not password:
                    st.error("Please enter both Employee ID and Password.")
                else:
                    emp_id = emp_id.strip().upper()
                    with st.spinner("Authenticating..."):
                        employee = authenticate(emp_id, password)

                    if employee:
                        init_session_state(reset=True)
                        st.session_state.authenticated = True
                        st.session_state.employee = employee
                        st.success(
                            f"Welcome, {employee['first_name']}! Redirecting..."
                        )
                        st.rerun()
                    else:
                        st.error(
                            "❌ Invalid Employee ID or Password. Please try again."
                        )

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#9e9e9e; font-size:0.8rem;'>"
            "💡 Hint: Open <code>data/employees.xlsx</code> to find test credentials."
            "</p>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – Employee Profile
# ═══════════════════════════════════════════════════════════════════════════════

def show_sidebar_profile():
    emp = st.session_state.employee
    if emp is None:
        return

    with st.sidebar:
        fn = html.escape(str(emp['first_name']))
        ln = html.escape(str(emp['last_name']))
        role = html.escape(str(emp['role']))
        dept = html.escape(str(emp['department']))
        access_level = get_access_level(emp)
        access_badge = "🔑 HR Admin Access" if access_level == "hr" else "👤 Employee Access"
        st.markdown(f"""
        <div class="profile-card">
            <h3>👤 {fn} {ln}</h3>
            <p style="color:#616161; margin:0;">
                {role}<br/>
                <b>{dept}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**{access_badge}**")

        st.markdown("##### 📋 Profile Details")
        st.markdown(f"**ID:** `{emp['id']}`")
        st.markdown(f"**Sex:** {emp['sex']}")
        st.markdown(f"**Salary:** ${emp['salary']:,.2f}")
        st.markdown(f"**Experience:** {int(emp['years_of_expertise'])} years")
        st.markdown(f"**Hire Date:** {emp.get('hire_date', 'N/A')}")
        st.markdown(f"**Work Mode:** {emp.get('work_mode', 'N/A')}")
        st.markdown(f"**Office Location:** {emp.get('office_location', 'N/A')}")
        st.markdown(f"**Health Plan:** {emp.get('health_plan', 'N/A')}")
        st.markdown(f"**Performance Rating:** {emp.get('performance_rating', 'N/A')}/5")

        st.markdown("---")
        st.markdown("##### 📊 Leave Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sick Days", int(emp.get("sick_leave_days", 0)))
            st.metric("PTO", int(emp.get("paid_time_off_days", 0)))
            if emp.get("sex") == "Female":
                st.metric("Maternity", int(emp.get("maternity_leave_days", 0)))
        with col2:
            st.metric("Mental Health", int(emp.get("mental_health_leave_days", 0)))
            if emp.get("sex") == "Male":
                st.metric("Paternity", int(emp.get("paternity_leave_days", 0)))

        # Monthly absence
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_keys = ["jan", "feb", "mar", "apr", "may", "jun",
                      "jul", "aug", "sep", "oct", "nov", "dec"]
        absence_data = {
            m: int(emp.get(f"absent_days_{k}", 0))
            for m, k in zip(months, month_keys)
        }
        st.markdown("---")
        st.markdown("##### 📅 Monthly Absence")
        st.bar_chart(absence_data, height=150)

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            init_session_state(reset=True)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# RAG INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Preparing employee records...")
def _ensure_transactional_tables_cached():
    """Ensure employee transaction tables are ready once per server process."""
    ensure_transactional_tables()


@st.cache_resource(show_spinner="Loading knowledge base...")
def init_rag(access_level: str):
    """Load vector store and create hybrid retriever (cached)."""
    vectorstore = load_vectorstore()
    retriever = create_hybrid_retriever(
        vectorstore, k=5, semantic_weight=0.5, access_level=access_level
    )
    return retriever


def get_rag_chain():
    """Get or create the tool-using RAG agent for the current session."""
    if st.session_state.rag_chain is None:
        employee = st.session_state.employee
        access_level = get_access_level(employee)
        retriever = init_rag(access_level)
        st.session_state.rag_chain = build_rag_agent(
            retriever=retriever,
            employee_info=employee,
        )
    return st.session_state.rag_chain


def _render_pending_action() -> None:
    """Show a confirmation card and execute an action only after confirmation."""
    pending_action: PendingAction | None = st.session_state.pending_action
    if pending_action is None:
        return

    with st.container(border=True):
        st.markdown("#### Review this HR action")
        st.write(pending_action.summary)
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("✅ Confirm", key="confirm_hr_action", type="primary"):
                try:
                    confirmation = get_rag_chain().action_tool.execute(pending_action)
                except Exception:
                    logger.exception("HR action execution failed")
                    confirmation = "I couldn't complete that action. Please try again later."
                st.session_state.chat_history.append({
                    "role": "assistant", "content": confirmation,
                })
                st.session_state.pending_action = None
                st.rerun()
        with cancel_col:
            if st.button("❌ Cancel", key="cancel_hr_action"):
                st.session_state.chat_history.append({
                    "role": "assistant", "content": "The proposed action was cancelled.",
                })
                st.session_state.pending_action = None
                st.rerun()


def _render_message_metadata(sources, tools_used, sql):
    if sources:
        sources_html = " ".join(
            f'<span class="source-badge">📄 {html.escape(source)}</span>'
            for source in sources
        )
        st.markdown(
            f"<div style='margin-top:8px;'>{sources_html}</div>",
            unsafe_allow_html=True,
        )
    if tools_used:
        st.caption("Tools used: " + ", ".join(tools_used))
    if sql:
        with st.expander("LLM-generated SQL", expanded=False):
            st.code(sql, language="sql")


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def show_chat():
    emp = st.session_state.employee

    st.markdown(
        f"### 🤖 NovaCorp AI Assistant &nbsp; | &nbsp; "
        f"Welcome, {emp['first_name']}!"
    )
    st.markdown(
        "Ask me about **company policies**, **your profile**, "
        "**benefits**, **leave**, and the company guide. I can also look up your "
        "expense claims, leave requests, pay history, and training costs, then "
        "combine those records with relevant policy documents."
    )
    with st.expander("How the assistant answers", expanded=False):
        st.caption("Questions route to hybrid policy search, role-scoped SQL, confirmation-gated HR actions, or public web search. Gemini is preferred; Ollama llama3.2 is used automatically if Gemini is unavailable.")
    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_message_metadata(
                msg.get("sources", []), msg.get("tools_used", []), msg.get("sql")
            )

    if st.session_state.pending_action is not None:
        _render_pending_action()
        return

    # Pop a staged suggestion so it is handled only once after rerun.
    prompt = st.chat_input("Ask a question...") or st.session_state.pop("pending_prompt", None)
    if prompt:
        # Display user message
        st.session_state.chat_history.append({
            "role": "user", "content": prompt
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    chain = get_rag_chain()
                    result = query_rag(chain, prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                    tools_used = result.get("tools_used", [])
                    sql = result.get("sql")
                    pending_action = result.get("pending_action")
                    if pending_action is not None:
                        st.session_state.pending_action = pending_action
                except Exception:
                    logger.exception("Chat query failed")
                    answer = "I couldn't complete that request right now. Please try again later."
                    sources = []
                    tools_used = []
                    sql = None

            st.markdown(answer)
            _render_message_metadata(sources, tools_used, sql)

        # Save assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "tools_used": tools_used,
            "sql": sql,
        })

        if st.session_state.pending_action is not None:
            st.rerun()

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        common_suggestions = [
            "What is my salary and department?",
            "What is the return to office policy?",
            "How many sick leave days do I have?",
            "What health insurance plans are available?",
            "What is the onboarding process for new hires?",
            "Am I eligible for maternity/paternity leave?",
            "What is the company's performance review process?",
            "How do I request time off?",
            "How much have I been reimbursed for travel this year?",
            "Which of my expense claims are still pending approval?",
            "What was my net pay in September and did I receive a bonus in 2026?",
        ]
        access_level = get_access_level(emp)
        if access_level == "hr":
            suggestions = common_suggestions + [
                "Show pending leave requests across the company.",
                "Show expense claim totals by category across the company.",
                "Approve leave request 123.",
                "Reject expense claim 456.",
            ]
        else:
            suggestions = common_suggestions + [
                "Submit PTO leave from 2026-12-01 to 2026-12-03 for 3 days.",
                "Submit a Meals expense claim for $45.50: lunch with a client.",
                "What are the IRS 401(k) contribution limits for 2026?",
                "How much of my $3,000 annual Learning & Development budget do I have left?",
                "What is the status of my recent leave requests?",
                "How much have I claimed for Books, and how much of the $300 book allowance remains?",
            ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💬 {suggestion}", key=f"sug_{i}",
                            use_container_width=True):
                    # Process the selected suggestion on the next rerun.
                    st.session_state.pending_prompt = suggestion
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Check prerequisites
    if not os.path.exists(DB_PATH):
        st.error(
            "⚠️ Database not found! Run these commands first:\n"
            "```\n"
            "python scripts/generate_employees.py\n"
            "python scripts/init_db.py\n"
            "python scripts/generate_documents.py\n"
            "python scripts/build_vectorstore.py\n"
            "```"
        )
        return

    _ensure_transactional_tables_cached()

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning(
            "⚠️ `GOOGLE_API_KEY` not found. Copy `.env.example` to `.env` "
            "and add your API key. The app will use local Ollama llama3.2 if it is running."
        )

    if not st.session_state.authenticated:
        show_login()
    else:
        show_sidebar_profile()
        show_chat()


if __name__ == "__main__":
    main()
