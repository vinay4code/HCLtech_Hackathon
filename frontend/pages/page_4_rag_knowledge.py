import os
import hashlib
import streamlit as st
import pandas as pd
from pathlib import Path

from config.settings import OFFERS_DOCS_DIR
from rag.document_loader import DocumentLoader
from rag.chunking import SectionChunker
from rag.vector_store import VectorStoreManager
from rag.retriever import OfferRetriever
from database.repositories import DocumentRepository
from frontend.components import render_header, render_kpi

def render_rag_page():
    render_header(
        "Offer Knowledge Base",
        "Secure document ingestion for promotional offers with cryptographic SHA-256 duplicate prevention."
    )

    # 1. High-Level Ingestion Metrics
    all_docs = DocumentRepository.get_all_documents()
    total_chunks = VectorStoreManager.count_chunks()
    total_docs = len(all_docs)

    k1, k2, k3 = st.columns(3)
    render_kpi(k1, "Unique Ingested Documents", f"{total_docs}", "SHA-256 Verified", "normal")
    render_kpi(k2, "Total Active Chunks", f"{total_chunks:,}", "ChromaDB & BM25", "success")
    render_kpi(k3, "Storage Engine", "ChromaDB + BM25", "Hybrid Indexed", "normal")

    st.markdown("---")

    # 2. Document Upload & Ingestion with Checksum Verification
    st.markdown("### 📤 Document Ingestion")
    st.caption("Upload approved campaign files (.pdf, .docx, .txt, .csv). Redundant uploads are cryptographically blocked via SHA-256 checksums.")

    uploaded_files = st.file_uploader(
        "Choose campaign documents to ingest:",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        help="Files are parsed, validated against existing checksums, chunked, and dual-indexed in ChromaDB and BM25."
    )

    if uploaded_files:
        if st.button("🚀 Process & Ingest Documents", type="primary", use_container_width=True):
            success_count = 0
            skipped_count = 0

            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getbuffer().tobytes()
                checksum = hashlib.sha256(file_bytes).hexdigest()
                file_size = len(file_bytes)

                # Checksum redundancy check
                existing = DocumentRepository.get_by_checksum(checksum)
                if existing:
                    st.warning(
                        f"⚠️ **Duplicate Document Blocked:** `{uploaded_file.name}` has identical content to "
                        f"already indexed document `{existing['filename']}` (SHA-256: `{checksum[:16]}...`). "
                        f"Skipped to prevent redundant vector chunks."
                    )
                    skipped_count += 1
                    continue

                # Save file to disk
                save_path = OFFERS_DOCS_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(file_bytes)

                # Parse, chunk, and index
                try:
                    pages = DocumentLoader.load_document(save_path)
                    chunks = SectionChunker.chunk_document(pages)
                    added_chunks = VectorStoreManager.add_chunks(chunks)

                    # Update BM25 index
                    OfferRetriever.get_bm25_retriever(force_rebuild=True)

                    # Record document with checksum
                    DocumentRepository.record_document(
                        doc_id=f"DOC-{checksum[:8].upper()}",
                        filename=uploaded_file.name,
                        file_type=uploaded_file.name.split(".")[-1].upper(),
                        file_size_bytes=file_size,
                        checksum=checksum,
                        num_chunks=added_chunks
                    )

                    st.success(
                        f"✅ **Ingested `{uploaded_file.name}`:** {len(pages)} page(s), "
                        f"{added_chunks} chunk(s) indexed. (SHA-256: `{checksum[:16]}...`)"
                    )
                    success_count += 1
                except Exception as e:
                    st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")

            if success_count > 0:
                st.rerun()

    st.markdown("---")

    # 3. Ingested Documents Registry
    st.markdown("### 📚 Ingested Documents Registry")
    st.caption("Cryptographically verified documents currently powering the offer recommendation pipeline:")

    if not all_docs:
        st.info("No documents currently ingested. Upload campaign files above to populate the knowledge base.")
    else:
        table_rows = []
        for d in all_docs:
            chk = d.get("checksum_sha256", "N/A")
            table_rows.append({
                "Document Name": d["filename"],
                "Format": d["file_type"],
                "File Size": f"{d['file_size_bytes'] / 1024:.1f} KB",
                "SHA-256 Checksum": f"{chk[:16]}..." if chk != "N/A" else "N/A",
                "Chunks Indexed": d["num_chunks"],
                "Status": "✅ INDEXED",
                "Ingestion Date": d["upload_timestamp"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.caption("ℹ️ Redundant document protection: Ingested documents are identified by SHA-256 cryptographic hashes. Uploading identical content is automatically rejected.")
