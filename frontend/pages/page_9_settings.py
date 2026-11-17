import os
import streamlit as st
from config.settings import (
    APP_NAME,
    APP_VERSION,
    DB_PATH,
    TELECOM_CSV_PATH,
    CHROMA_PERSIST_DIR,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    EMBEDDING_PROVIDER
)
from database.db_manager import init_db
from rag.vector_store import VectorStoreManager
from frontend.components import render_header

def render_settings_page():
    render_header(
        "Settings & Configuration",
        "Manage AI model providers, API credentials, vector embedding engines, and system diagnostics."
    )

    tab_llm, tab_embed, tab_diag, tab_reset = st.tabs([
        "🤖 LLM Model Configuration",
        "🧠 Vector Embedding Engine",
        "🛠️ System Diagnostics",
        "⚠️ Reset & Maintenance"
    ])

    with tab_llm:
        st.markdown("### Generative AI Model Provider")
        st.info("💡 **Enterprise Resilience:** If no external API key is provided, the system operates in **Deterministic Mock Mode** out-of-the-box, ensuring 100% offline functionality and zero ungrounded hallucinations.")

        current_provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).upper()
        provider_options = ["MOCK", "GEMINI", "OPENAI"]
        default_idx = provider_options.index(current_provider) if current_provider in provider_options else 0

        selected_provider = st.selectbox(
            "Select Active LLM Provider:",
            provider_options,
            index=default_idx,
            help="Choose between offline deterministic mock or live cloud LLM APIs."
        )

        gemini_key = st.text_input(
            "Google Gemini API Key:",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Used for Google Gemini 1.5 Flash / Pro model generation."
        )

        openai_key = st.text_input(
            "OpenAI API Key:",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Used for GPT-4o / GPT-4o-mini generation."
        )

        col_m, col_t = st.columns(2)
        with col_m:
            model_name = st.text_input("Model Identifier:", value=os.getenv("LLM_MODEL", DEFAULT_MODEL_NAME))
        with col_t:
            temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)

        if st.button("💾 Save Model Settings", type="primary"):
            os.environ["LLM_PROVIDER"] = selected_provider
            os.environ["GEMINI_API_KEY"] = gemini_key
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["LLM_MODEL"] = model_name
            os.environ["LLM_TEMPERATURE"] = str(temperature)
            st.success("LLM configuration updated successfully!")

    with tab_embed:
        st.markdown("### Vector Embeddings Configuration")
        st.caption("Select how campaign documentation is embedded into ChromaDB vectors.")

        embed_options = [
            "FAST_HASH (Fast normalized 384-dim Hashing - Instant startup)",
            "LOCAL_HF (SentenceTransformers all-MiniLM-L6-v2 - Local PyTorch)",
            "GEMINI (Google text-embedding-004 REST API)"
        ]
        chosen_embed = st.selectbox("Embedding Engine:", embed_options, index=0)

        st.markdown("#### Vector Database Status")
        st.markdown(f"- Storage Engine: **ChromaDB (Persistent)**")
        st.markdown(f"- Storage Location: `{CHROMA_PERSIST_DIR}`")
        st.markdown(f"- Current Stored Vectors: **{VectorStoreManager.count_chunks()} chunks**")

    with tab_diag:
        st.markdown("### Application Diagnostics & Environment")
        st.markdown(f"- **Application Name:** {APP_NAME} v{APP_VERSION}")
        st.markdown(f"- **Operating System:** Windows")
        st.markdown(f"- **SQLite Database:** `{DB_PATH}` (Exists: {DB_PATH.exists()})")
        st.markdown(f"- **Telecom Dataset:** `{TELECOM_CSV_PATH}` (Exists: {TELECOM_CSV_PATH.exists()})")

    with tab_reset:
        st.markdown("### Database & Vector Store Reset")
        st.warning("⚠️ Re-initializing will reset all approvals and audit logs, and reload default seed offers.")
        if st.button("🔄 Reset Database & Seed Offers", type="secondary"):
            init_db(force=True)
            VectorStoreManager.reindex_all_offers()
            st.success("Database and Vector Store re-initialized successfully!")
            st.rerun()
