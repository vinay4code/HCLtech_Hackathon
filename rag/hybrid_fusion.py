from typing import List, Dict, Any, Optional

class HybridFusion:
    """
    Reciprocal Rank Fusion (RRF) module for combining BM25 keyword retrieval
    and vector embedding semantic retrieval.
    Formula:
        RRF(d) = Σ [ w_i / (k + rank_i(d)) ]
    """

    @classmethod
    def reciprocal_rank_fusion(
        cls,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        k: int = 60,
        weight_bm25: float = 1.0,
        weight_vector: float = 1.0,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Merges BM25 and vector retrieval candidate lists using RRF.
        Deduplicates by chunk_id, preserves original scores & ranks, and retains full metadata.
        """
        fused_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}
        bm25_info: Dict[str, Dict[str, Any]] = {}
        vector_info: Dict[str, Dict[str, Any]] = {}

        # 1. Process BM25 Rankings
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            rrf_contrib = weight_bm25 / (k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib
            chunk_data[cid] = item
            bm25_info[cid] = {
                "bm25_rank": rank,
                "bm25_score": item.get("bm25_score", 0.0)
            }

        # 2. Process Vector Semantic Rankings
        for rank, item in enumerate(vector_results, start=1):
            cid = item["chunk_id"]
            rrf_contrib = weight_vector / (k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib
            if cid not in chunk_data:
                chunk_data[cid] = item
            vector_info[cid] = {
                "vector_rank": rank,
                "vector_score": item.get("similarity_score", item.get("relevance_score", 0.0))
            }

        # 3. Sort by fused RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        # Theoretical maximum RRF score when ranked #1 across all retrieval streams
        max_possible_rrf = (weight_bm25 / (k + 1.0)) + (weight_vector / (k + 1.0))

        # 4. Assemble fused candidate list
        fused_results = []
        for fused_rank, cid in enumerate(sorted_cids[:top_k], start=1):
            base_item = chunk_data[cid]
            b_info = bm25_info.get(cid, {})
            v_info = vector_info.get(cid, {})

            sources = []
            if cid in bm25_info:
                sources.append("BM25")
            if cid in vector_info:
                sources.append("VECTOR")

            fused_score = float(fused_scores[cid])
            confidence = min(1.0, fused_score / max_possible_rrf) if max_possible_rrf > 0 else 0.0

            fused_results.append({
                "chunk_id": cid,
                "content": base_item.get("content", ""),
                "metadata": base_item.get("metadata", {}),
                "rrf_score": round(fused_score, 5),
                "rrf_confidence": round(confidence, 4),
                "rrf_confidence_pct": round(confidence * 100.0, 1),
                "is_auto_approvable": bool(confidence > 0.70),
                "fused_rank": fused_rank,
                "source_retrievers": sources,
                "bm25_rank": b_info.get("bm25_rank"),
                "bm25_score": b_info.get("bm25_score"),
                "vector_rank": v_info.get("vector_rank"),
                "vector_score": v_info.get("vector_score")
            })

        return fused_results
