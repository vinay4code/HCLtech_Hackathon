import os
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="NovaTel Customer Loyalty & Engagement Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and vector store if needed
from database.db_manager import init_db
from rag.vector_store import VectorStoreManager
from frontend.styles import get_custom_css
from database.repositories import ApprovalRepository
from backend.profile_builder import ProfileBuilder

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Bootstrap database on initial run
@st.cache_resource
def bootstrap_system():
    init_db()
    if VectorStoreManager.count_chunks() == 0:
        VectorStoreManager.reindex_all_offers()
    return True

bootstrap_system()

# Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #334155; margin-bottom: 1rem;">
    <h2 style="color: #60a5fa; margin: 0; font-size: 1.4rem;">📡 NovaTel AI</h2>
    <p style="color: #94a3b8; margin: 0.2rem 0 0 0; font-size: 0.8rem; font-weight: 500;">
        Customer Loyalty & Engagement Assistant
    </p>
    <div style="margin-top: 0.5rem;">
        <span style="background: #1e3a8a; color: #93c5fd; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600;">HCLTech Hackathon</span>
    </div>
</div>
""", unsafe_allow_html=True)

pages = [
    "Executive Dashboard",
    "Segmentation Engine",
    "Offer Knowledge Base",
    "Email Generation",
    "Approval Center"
]

# Support page switching via session state
if "active_page" not in st.session_state or st.session_state.active_page not in pages:
    st.session_state.active_page = "Executive Dashboard"

selected_nav = st.sidebar.radio(
    "Navigation Menu",
    pages,
    index=pages.index(st.session_state.active_page) if st.session_state.active_page in pages else 0,
    label_visibility="collapsed"
)
st.session_state.active_page = selected_nav

# Sidebar Status & Quick Metrics
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚦 System Status")

# Pending approvals badge
approvals = ApprovalRepository.get_approvals()
pending_reviews = sum(1 for a in approvals if a.get("reviewer_status") == "PENDING")
if pending_reviews > 0:
    st.sidebar.warning(f"⚠️ **{pending_reviews} Pending Approvals**")
else:
    st.sidebar.success("✅ **Queue Cleared** (0 pending)")

active_llm = os.getenv("LLM_PROVIDER", "MOCK").upper()
st.sidebar.info(f"🤖 **Provider:** `{active_llm}`")
st.sidebar.caption("Deterministic eligibility rules & safety guardrails enabled.")

st.sidebar.markdown("---")
st.sidebar.caption("HCLTech Future Ready AI Engineer Hackathon\nVersion 2.0.0 | Enterprise SaaS Edition")

# Route to active page
from frontend.pages.page_1_dashboard import render_dashboard_page
from frontend.pages.page_3_segmentation import render_segmentation_page
from frontend.pages.page_4_rag_knowledge import render_rag_page
from frontend.pages.page_6_email_generation import render_email_generation_page
from frontend.pages.page_7_approval_center import render_approval_center_page

if st.session_state.active_page == "Executive Dashboard":
    render_dashboard_page()
elif st.session_state.active_page == "Segmentation Engine":
    render_segmentation_page()
elif st.session_state.active_page == "Offer Knowledge Base":
    render_rag_page()
elif st.session_state.active_page == "Email Generation":
    render_email_generation_page()
elif st.session_state.active_page == "Approval Center":
    render_approval_center_page()
