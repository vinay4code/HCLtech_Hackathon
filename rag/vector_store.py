import os
import shutil
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import CHROMA_PERSIST_DIR, OFFERS_DOCS_DIR
from rag.embeddings import RobustEmbeddingFunction
from rag.document_loader import DocumentLoader
from rag.chunking import SectionChunker

COLLECTION_NAME = "telecom_approved_offers"

class VectorStoreManager:
    """
    Manages ChromaDB vector database lifecycle, document indexing, similarity queries,
    and metadata filtering.
    """
    _client = None
    _collection = None
    _embedding_function = None

    @classmethod
    def get_embedding_function(cls) -> RobustEmbeddingFunction:
        if cls._embedding_function is None:
            # Use local Hashing by default for instant speed, or HF if specified
            provider = os.getenv("EMBEDDING_PROVIDER", "FAST_HASH")
            cls._embedding_function = RobustEmbeddingFunction(provider=provider)
        return cls._embedding_function

    @classmethod
    def get_client(cls) -> chromadb.PersistentClient:
        if cls._client is None:
            cls._client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        return cls._client

    @classmethod
    def get_collection(cls):
        client = cls.get_client()
        ef = cls.get_embedding_function()
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"description": "Approved telecom promotional offers and campaigns"}
        )

    @classmethod
    def add_chunks(cls, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        coll = cls.get_collection()
        
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = []
        for c in chunks:
            m = {
                "chunk_id": str(c.get("chunk_id", "")),
                "document_id": str(c.get("document_id", "")),
                "document_name": str(c.get("document_name", "")),
                "page_number": int(c.get("page_number", 1)),
                "offer_id": str(c.get("offer_id", "")),
                "offer_category": str(c.get("offer_category", "")),
                "target_segment": str(c.get("target_segment", "")),
                "validity": str(c.get("validity", "")),
                "effective_from": str(c.get("effective_from", "2026-01-01")),
                "effective_until": str(c.get("effective_until", "2026-12-31")),
                "approval_status": str(c.get("approval_status", "APPROVED")),
                "document_version": str(c.get("document_version", "2026.1")),
                "source_type": str(c.get("source_type", "TXT"))
            }
            metadatas.append(m)

        coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    @classmethod
    def get_all_chunks(cls) -> List[Dict[str, Any]]:
        """Returns all chunks currently stored in ChromaDB with metadata and text content."""
        coll = cls.get_collection()
        count = coll.count()
        if count == 0:
            return []
        data = coll.get(include=["documents", "metadatas"])
        chunks = []
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        for cid, doc, meta in zip(ids, docs, metas):
            chunk = dict(meta)
            chunk["chunk_id"] = cid
            chunk["content"] = doc
            chunks.append(chunk)
        return chunks

    @classmethod
    def query_similar(
        cls,
        query_text: str,
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs semantic similarity query with optional metadata filtering.
        Defaults to top 10 relevant chunks.
        """
        coll = cls.get_collection()
        count = coll.count()
        if count == 0:
            return []
            
        kwargs = {
            "query_texts": [query_text],
            "n_results": min(n_results, count)
        }
        if where_filter and len(where_filter) > 0:
            kwargs["where"] = where_filter

        try:
            res = coll.query(**kwargs)
        except Exception as e:
            # If where filter was too restrictive, try without filter
            if "where" in kwargs:
                kwargs.pop("where")
                res = coll.query(**kwargs)
            else:
                return []

        results = []
        if res and res.get("documents") and len(res["documents"]) > 0:
            docs = res["documents"][0]
            metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
            ids = res["ids"][0] if res.get("ids") else [""] * len(docs)
            distances = res["distances"][0] if res.get("distances") else [0.0] * len(docs)

            for rank, (doc, meta, cid, dist) in enumerate(zip(docs, metas, ids, distances), start=1):
                # Calculate normalized similarity score (1.0 = identical)
                sim_score = round(max(0.0, 1.0 - (dist / 2.0)), 4)
                results.append({
                    "chunk_id": cid,
                    "content": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity_score": sim_score,
                    "relevance_score": sim_score,
                    "vector_rank": rank
                })
        return results


    @classmethod
    def list_indexed_documents(cls) -> List[Dict[str, Any]]:
        coll = cls.get_collection()
        all_data = coll.get(include=["metadatas"])
        docs_summary = {}
        for m in all_data.get("metadatas", []):
            name = m.get("document_name", "unknown")
            if name not in docs_summary:
                docs_summary[name] = {"document_name": name, "chunks_count": 0, "categories": set()}
            docs_summary[name]["chunks_count"] += 1
            if m.get("offer_category"):
                docs_summary[name]["categories"].add(m.get("offer_category"))

        for v in docs_summary.values():
            v["categories"] = list(v["categories"])
        return list(docs_summary.values())

    @classmethod
    def delete_document_chunks(cls, document_name: str) -> int:
        coll = cls.get_collection()
        coll.delete(where={"document_name": document_name})
        return True

    @classmethod
    def reindex_all_offers(cls) -> int:
        """
        Clears and reindexes all documents in OFFERS_DOCS_DIR.
        """
        client = cls.get_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        cls._collection = None
        
        total_chunks = 0
        if not OFFERS_DOCS_DIR.exists():
            return 0

        for f in OFFERS_DOCS_DIR.iterdir():
            if f.suffix.lower() in [".pdf", ".docx", ".txt", ".md", ".csv"]:
                try:
                    pages = DocumentLoader.load_document(f)
                    chunks = SectionChunker.chunk_document(pages)
                    if chunks:
                        cls.add_chunks(chunks)
                        total_chunks += len(chunks)
                except Exception as e:
                    print(f"Error indexing {f.name}: {e}")

        return total_chunks

    @classmethod
    def count_chunks(cls) -> int:
        try:
            return cls.get_collection().count()
        except Exception:
            return 0
