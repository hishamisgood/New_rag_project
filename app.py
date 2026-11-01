"""
NovaCorp Employee Portal — Streamlit Application

Features
--------
• Login screen authenticated via SQLite (ID + password)
• Employee profile dashboard in the sidebar
• RAG-powered chat interface (hybrid search + Gemini)

Run with:  streamlit run app.py
"""

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

from auth import authenticate, get_employee
from rag_engine import (
    load_vectorstore,
    create_hybrid_retriever,
    build_rag_chain,
    query_rag,
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Login card */
    .login-container {
        max-width: 420px;
        margin: 0 auto;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .login-container h1 { color: white; text-align: center; }
    .login-container p  { color: #e0e0e0; text-align: center; }

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
DEFAULTS = {
    "authenticated": False,
    "employee": None,
    "chat_history": [],
    "rag_chain": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


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
                        st.session_state.authenticated = True
                        st.session_state.employee = employee
                        st.session_state.chat_history = []
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
        st.markdown(f"""
        <div class="profile-card">
            <h3>👤 {emp['first_name']} {emp['last_name']}</h3>
            <p style="color:#616161; margin:0;">
                {emp['role']}<br/>
                <b>{emp['department']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📋 Profile Details")
        st.markdown(f"**ID:** `{emp['id']}`")
        st.markdown(f"**Sex:** {emp['sex']}")
        st.markdown(f"**Salary:** ${emp['salary']:,.2f}")
        st.markdown(f"**Experience:** {int(emp['years_of_expertise'])} years")

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
            for key in DEFAULTS:
                st.session_state[key] = DEFAULTS[key]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# RAG INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading knowledge base...")
def init_rag():
    """Load vector store and create hybrid retriever (cached)."""
    vectorstore = load_vectorstore()
    retriever = create_hybrid_retriever(vectorstore, k=5, semantic_weight=0.5)
    return retriever


def get_rag_chain():
    """Get or create the RAG chain for the current session."""
    if st.session_state.rag_chain is None:
        retriever = init_rag()
        st.session_state.rag_chain = build_rag_chain(
            retriever=retriever,
            employee_info=st.session_state.employee,
        )
    return st.session_state.rag_chain


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
        "**benefits**, **leave policies**, or anything from our knowledge base."
    )
    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                sources_html = " ".join(
                    f'<span class="source-badge">📄 {s}</span>'
                    for s in msg["sources"]
                )
                st.markdown(
                    f"<div style='margin-top:8px;'>{sources_html}</div>",
                    unsafe_allow_html=True,
                )

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
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
                except Exception as e:
                    answer = (
                        f"⚠️ I encountered an error: {str(e)}\n\n"
                        "Please make sure your `GOOGLE_API_KEY` is set in the "
                        "`.env` file and that the vector store has been built "
                        "(`python build_vectorstore.py`)."
                    )
                    sources = []

            st.markdown(answer)
            if sources:
                sources_html = " ".join(
                    f'<span class="source-badge">📄 {s}</span>'
                    for s in sources
                )
                st.markdown(
                    f"<div style='margin-top:8px;'>{sources_html}</div>",
                    unsafe_allow_html=True,
                )

        # Save assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "What is my salary and department?",
            "What is the return to office policy?",
            "How many sick leave days do I have?",
            "What health insurance plans are available?",
            "What is the onboarding process for new hires?",
            "Am I eligible for maternity/paternity leave?",
            "What is the company's performance review process?",
            "How do I request time off?",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💬 {suggestion}", key=f"sug_{i}",
                            use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user", "content": suggestion
                    })
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Check prerequisites
    if not os.path.exists("data/employees.db"):
        st.error(
            "⚠️ Database not found! Run these commands first:\n"
            "```\n"
            "python generate_employees.py\n"
            "python auth.py\n"
            "python generate_documents.py\n"
            "python build_vectorstore.py\n"
            "```"
        )
        return

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning(
            "⚠️ `GOOGLE_API_KEY` not found. Copy `.env.example` to `.env` "
            "and add your API key. The chat feature requires it."
        )

    if not st.session_state.authenticated:
        show_login()
    else:
        show_sidebar_profile()
        show_chat()


if __name__ == "__main__":
    main()
