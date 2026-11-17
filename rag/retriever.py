import re
from typing import List, Dict, Any, Optional
from rag.vector_store import VectorStoreManager
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_fusion import HybridFusion
from rag.reranker import CrossEncoderReranker
from database.repositories import OfferRepository, AuditRepository
from backend.profile_builder import CustomerProfile
from backend.offer_validator import OfferValidator

class OfferRetriever:
    """
    Advanced Multi-Stage RAG Offer Retrieval Pipeline:
    1. Rich customer context query construction.
    2. Parallel BM25 keyword and vector semantic retrieval.
    3. Reciprocal Rank Fusion (RRF) result merging & deduplication.
    4. Cross-Encoder neural reranking with controlled fallback.
    5. Deterministic offer validation & document-registry conflict checking.
    6. Complete observability tracing.
    """
    _bm25_retriever: Optional[BM25Retriever] = None
    _reranker: Optional[CrossEncoderReranker] = None

    @classmethod
    def get_bm25_retriever(cls, force_rebuild: bool = False) -> BM25Retriever:
        if cls._bm25_retriever is None or force_rebuild:
            cls._bm25_retriever = BM25Retriever()
            chunks = VectorStoreManager.get_all_chunks()
            if chunks:
                cls._bm25_retriever.build_index(chunks)
        return cls._bm25_retriever

    @classmethod
    def get_reranker(cls) -> CrossEncoderReranker:
        if cls._reranker is None:
            cls._reranker = CrossEncoderReranker()
        return cls._reranker

    @classmethod
    def build_customer_context_query(
        cls,
        profile: CustomerProfile,
        strategy_info: Dict[str, Any]
    ) -> str:
        """
        Constructs a complete, customer-specific contextual retrieval query.
        """
        gaps_str = ", ".join(profile.service_gaps[:3]) if profile.service_gaps else "None"
        objective = strategy_info.get("objective", "customer retention and satisfaction")
        base_query = strategy_info.get("retrieval_query", "")

        query = (
            f"{base_query} | "
            f"Customer Segment: {strategy_info.get('target_segment', 'GENERAL')} | "
            f"Tenure: {profile.tenure_months} months | "
            f"Contract: {profile.contract} | "
            f"Internet: {profile.internet_service} | "
            f"Missing Services: {gaps_str} | "
            f"Campaign Objective: {objective}"
        )
        return query.strip()

    @classmethod
    def retrieve_offers_for_customer(
        cls,
        profile: CustomerProfile,
        strategy_info: Dict[str, Any],
        top_k: int = 3,
        bm25_top_k: int = 10,
        vector_top_k: int = 10,
        fused_top_k: int = 20,
        rerank_top_k: int = 5,
        rrf_k: int = 60,
        enforce_date_filtering: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the complete Hybrid RAG pipeline and returns verified offers,
        selected evidence chunks, and an end-to-end observability trace.
        """
        # 1. Query Construction
        context_query = cls.build_customer_context_query(profile, strategy_info)
        keyword_query = strategy_info.get("retrieval_query", context_query)

        # 2. Metadata filtering constraints
        # Only approved documents are eligible for production retrieval
        where_filter = {"approval_status": "APPROVED"} if enforce_date_filtering else {}

        # 3. Parallel Retrieval: A. BM25 Keyword Search
        bm25 = cls.get_bm25_retriever()
        bm25_results = bm25.query(
            query_text=keyword_query,
            top_k=bm25_top_k,
            metadata_filter=where_filter
        )

        # 4. Parallel Retrieval: B. Vector Semantic Search
        vector_results = VectorStoreManager.query_similar(
            query_text=context_query,
            n_results=vector_top_k,
            where_filter=where_filter
        )

        # 5. Hybrid Result Fusion via Reciprocal Rank Fusion (RRF)
        fused_candidates = HybridFusion.reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            k=rrf_k,
            weight_bm25=1.0,
            weight_vector=1.0,
            top_k=fused_top_k
        )

        # 6. Cross-Encoder Reranking Stage
        reranker = cls.get_reranker()
        rerank_res = reranker.rerank(
            query=context_query,
            candidates=fused_candidates,
            top_k=rerank_top_k
        )
        reranked_chunks = rerank_res.get("reranked_candidates", [])

        # 7. Extract candidate offer IDs from reranked evidence chunks
        candidate_offer_ids = []
        for c in reranked_chunks:
            meta = c.get("metadata", {})
            oid = meta.get("offer_id") or c.get("offer_id")
            if oid and oid != "GENERAL_OFFER" and oid not in candidate_offer_ids:
                candidate_offer_ids.append(oid)

        # 8. Deterministic Validation & Conflict Checking against SQLite Registry
        all_active_offers = OfferRepository.get_all_offers(active_only=True)
        active_offers_by_id = {o["offer_id"]: o for o in all_active_offers}

        verified_offers = []
        validation_results = []
        conflicts = []

        for oid in candidate_offer_ids:
            if oid in active_offers_by_id:
                reg_offer = active_offers_by_id[oid]
                val_result = OfferValidator.validate_offer(reg_offer, profile, strategy_info.get("target_segment"))
                validation_results.append(val_result.model_dump())

                # Document vs. Registry Conflict Check
                conflict = cls._check_document_registry_conflict(oid, reranked_chunks, reg_offer)
                if conflict:
                    conflicts.append(conflict)
                    # Block offer if conflict exists
                    continue

                if val_result.is_eligible:
                    verified_offers.append(reg_offer)

        # Fallback if no candidate from top chunks passed validation
        if not verified_offers and not conflicts:
            target_seg = strategy_info.get("target_segment", "")
            for o in all_active_offers:
                if o.get("target_segment") == target_seg or o.get("target_segment") == "ALL":
                    v_res = OfferValidator.validate_offer(o, profile, target_seg)
                    if v_res.is_eligible:
                        verified_offers.append(o)
                        break

        # 9. Determine Top RRF Confidence & Auto-Approval Eligibility
        top_rrf_confidence = 0.0
        if verified_offers and reranked_chunks:
            primary_oid = verified_offers[0].get("offer_id")
            matching_chunks = [
                c for c in reranked_chunks
                if c.get("metadata", {}).get("offer_id") == primary_oid or c.get("offer_id") == primary_oid
            ]
            if matching_chunks:
                top_rrf_confidence = matching_chunks[0].get("rrf_confidence", 0.0)
            else:
                top_rrf_confidence = reranked_chunks[0].get("rrf_confidence", 0.0)
        elif reranked_chunks:
            top_rrf_confidence = reranked_chunks[0].get("rrf_confidence", 0.0)
        elif fused_candidates:
            top_rrf_confidence = fused_candidates[0].get("rrf_confidence", 0.0)

        # Higher than 70% (> 0.70) auto-approval eligibility
        is_auto_approvable = bool(top_rrf_confidence > 0.70 and verified_offers)

        # Attach RRF confidence metrics to verified offers
        for vo in verified_offers:
            vo["rrf_confidence"] = round(float(top_rrf_confidence), 4)
            vo["rrf_confidence_pct"] = round(float(top_rrf_confidence * 100.0), 1)
            vo["is_auto_approvable"] = is_auto_approvable

        # 10. Format Source Citations
        source_citations = []
        for c in reranked_chunks[:top_k]:
            meta = c.get("metadata", {})
            source_citations.append({
                "chunk_id": c.get("chunk_id"),
                "document_name": meta.get("document_name", "campaign_doc"),
                "page_number": meta.get("page_number", 1),
                "offer_id": meta.get("offer_id", "GENERAL"),
                "excerpt": c.get("content", "")[:180] + "...",
                "rerank_score": c.get("rerank_score", 0.0),
                "rrf_score": c.get("rrf_score", 0.0),
                "rrf_confidence": c.get("rrf_confidence", 0.0),
                "rrf_confidence_pct": c.get("rrf_confidence_pct", 0.0),
                "source_retrievers": c.get("source_retrievers", ["HYBRID"])
            })

        # 11. Audit Logging
        status = "OFFERS_RETRIEVED" if verified_offers else "NO_VERIFIED_OFFER_FOUND"
        AuditRepository.log_action(
            action_type="RETRIEVAL",
            customer_id=profile.customer_id,
            details={
                "context_query": context_query,
                "strategy_id": strategy_info.get("strategy_id"),
                "bm25_matches_count": len(bm25_results),
                "vector_matches_count": len(vector_results),
                "fused_count": len(fused_candidates),
                "reranked_count": len(reranked_chunks),
                "reranker_status": rerank_res.get("status"),
                "conflicts_count": len(conflicts),
                "verified_offers_count": len(verified_offers),
                "offer_ids": [o["offer_id"] for o in verified_offers],
                "rrf_confidence": round(float(top_rrf_confidence), 4),
                "auto_approval_eligible": is_auto_approvable
            },
            compliance_status="PASSED" if not conflicts else "WARNING"
        )

        return {
            "status": status,
            "message": f"Successfully retrieved {len(verified_offers)} verified offer(s) via hybrid search." if verified_offers else "No verified active promotional offer found matching customer criteria.",
            "verified_offers": verified_offers[:top_k],
            "final_selected_chunks": reranked_chunks[:top_k],
            "source_citations": source_citations,
            "rrf_confidence": round(float(top_rrf_confidence), 4),
            "rrf_confidence_pct": round(float(top_rrf_confidence * 100.0), 1),
            "auto_approval_eligible": is_auto_approvable,
            "observability": {
                "context_query": context_query,
                "keyword_query": keyword_query,
                "bm25_results": bm25_results,
                "vector_results": vector_results,
                "fused_results": fused_candidates,
                "reranked_results": reranked_chunks,
                "rerank_metadata": rerank_res,
                "conflicts_detected": conflicts,
                "validation_verdicts": validation_results
            },
            "raw_context_chunks": [c.get("content", "") for c in reranked_chunks[:top_k]]
        }

    @staticmethod
    def _check_document_registry_conflict(
        offer_id: str,
        chunks: List[Dict[str, Any]],
        registry_offer: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Detects conflicts between extracted document chunk text and structured SQLite offer registry.
        Flags discrepancy if discount percentage in document contradicts registry value.
        """
        for c in chunks:
            meta = c.get("metadata", {})
            if meta.get("offer_id") == offer_id or c.get("offer_id") == offer_id:
                content = c.get("content", "")
                reg_discount = float(registry_offer.get("discount_value", 0.0))
                reg_type = registry_offer.get("discount_type")

                # If percentage discount
                if reg_type == "PERCENTAGE":
                    pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
                    if pct_matches:
                        found_vals = [float(p) for p in pct_matches]
                        # If numbers are mentioned but registry discount is completely absent
                        if not any(abs(v - reg_discount) < 0.1 for v in found_vals):
                            return {
                                "offer_id": offer_id,
                                "conflict_type": "DISCOUNT_MISMATCH",
                                "document_values": found_vals,
                                "registry_value": reg_discount,
                                "document_name": meta.get("document_name"),
                                "reason": f"Document mentions {found_vals}% but registry specifies {reg_discount}%."
                            }
        return None
