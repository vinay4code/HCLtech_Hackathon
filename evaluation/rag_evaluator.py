import time
import math
import re
from typing import Dict, Any, List, Set, Optional

from rag.vector_store import VectorStoreManager
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_fusion import HybridFusion
from rag.reranker import CrossEncoderReranker
from database.repositories import BenchmarkRepository, OfferRepository

class RAGEvaluator:
    """
    Retrieval Evaluation Suite comparing 4 distinct RAG retrieval approaches:
    1. BM25 Only
    2. Vector Retrieval Only
    3. Hybrid Retrieval without Reranking (RRF)
    4. Hybrid Retrieval with Reranking (Hybrid + Cross-Encoder)

    Evaluates:
    - Precision@K
    - Recall@K
    - Mean Reciprocal Rank (MRR)
    - nDCG@K
    - Offer ID Retrieval Accuracy
    - Exact Numerical-Condition Retrieval Accuracy
    - Source Citation Accuracy
    - Query Latency (ms)
    """

    @classmethod
    def run_comparative_benchmark(cls, top_k: int = 3) -> Dict[str, Any]:
        """
        Runs identical evaluation queries against all four retrieval approaches
        and produces a rigorous comparative benchmark report.
        """
        benchmarks = BenchmarkRepository.get_all_benchmarks()
        if not benchmarks:
            return {"approaches": {}, "summary_table": []}

        # Initialize shared components
        chunks = VectorStoreManager.get_all_chunks()
        bm25 = BM25Retriever()
        if chunks:
            bm25.build_index(chunks)
        reranker = CrossEncoderReranker()
        active_offers = {o["offer_id"]: o for o in OfferRepository.get_all_offers(active_only=False)}

        approaches = [
            ("BM25 Only", "BM25"),
            ("Vector Retrieval Only", "VECTOR"),
            ("Hybrid (RRF Fusion)", "HYBRID_NO_RERANK"),
            ("Hybrid + Cross-Encoder", "HYBRID_RERANKED")
        ]

        approach_results = {}
        summary_rows = []

        for display_name, mode in approaches:
            metrics = cls._evaluate_single_approach(
                mode=mode,
                benchmarks=benchmarks,
                chunks=chunks,
                bm25=bm25,
                reranker=reranker,
                active_offers=active_offers,
                top_k=top_k
            )
            approach_results[mode] = metrics
            summary_rows.append({
                "Approach": display_name,
                "Mode": mode,
                f"Precision@{top_k}": f"{metrics['precision_at_k'] * 100:.1f}%",
                f"Recall@{top_k}": f"{metrics['recall_at_k'] * 100:.1f}%",
                "MRR": f"{metrics['mrr']:.3f}",
                f"nDCG@{top_k}": f"{metrics['ndcg_at_k']:.3f}",
                "Offer ID Acc": f"{metrics['offer_id_accuracy'] * 100:.1f}%",
                "Numerical Acc": f"{metrics['numerical_condition_accuracy'] * 100:.1f}%",
                "Citation Acc": f"{metrics['citation_accuracy'] * 100:.1f}%",
                "Latency (ms)": f"{metrics['avg_latency_ms']:.1f} ms"
            })

        return {
            "total_benchmark_queries": len(benchmarks),
            "top_k_evaluated": top_k,
            "approach_metrics": approach_results,
            "summary_table": summary_rows
        }

    @classmethod
    def _evaluate_single_approach(
        cls,
        mode: str,
        benchmarks: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        bm25: BM25Retriever,
        reranker: CrossEncoderReranker,
        active_offers: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        precisions = []
        recalls = []
        reciprocal_ranks = []
        ndcgs = []
        offer_id_hits = []
        numerical_hits = []
        citation_hits = []
        latencies = []
        per_query_details = []

        for b in benchmarks:
            query = b["query_text"]
            expected_ids = set(b.get("expected_offer_ids", []))

            # Expected numerical numbers for this benchmark
            expected_numbers = set()
            for oid in expected_ids:
                if oid in active_offers:
                    disc = str(active_offers[oid].get("discount_value", ""))
                    if disc:
                        expected_numbers.add(disc)

            t0 = time.time()
            retrieved_chunks = cls._retrieve_by_mode(
                mode=mode,
                query=query,
                top_k=top_k,
                chunks=chunks,
                bm25=bm25,
                reranker=reranker
            )
            lat_ms = (time.time() - t0) * 1000.0
            latencies.append(lat_ms)

            # Extract retrieved offer IDs
            retrieved_ids = []
            valid_citations_count = 0
            retrieved_text_combined = ""

            for c in retrieved_chunks:
                meta = c.get("metadata", {})
                oid = meta.get("offer_id") or c.get("offer_id")
                if oid and oid != "GENERAL_OFFER" and oid not in retrieved_ids:
                    retrieved_ids.append(oid)
                # Check citation validity
                doc_name = meta.get("document_name") or c.get("document_name")
                page_no = meta.get("page_number") or c.get("page_number")
                if doc_name and page_no is not None:
                    valid_citations_count += 1
                retrieved_text_combined += " " + c.get("content", "")

            # 1. Precision & Recall
            hits = [oid for oid in retrieved_ids if oid in expected_ids]
            p = len(hits) / max(1, len(retrieved_chunks))
            r = len(hits) / max(1, len(expected_ids))
            precisions.append(p)
            recalls.append(r)

            # 2. Reciprocal Rank (MRR)
            rr = 0.0
            for rank, oid in enumerate(retrieved_ids, start=1):
                if oid in expected_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # 3. nDCG@K
            dcg = 0.0
            for i, c in enumerate(retrieved_chunks[:top_k], start=1):
                meta = c.get("metadata", {})
                oid = meta.get("offer_id") or c.get("offer_id")
                rel = 1.0 if oid in expected_ids else 0.0
                dcg += rel / math.log2(i + 1)
            
            ideal_hits = min(len(expected_ids), top_k)
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
            ndcg = (dcg / idcg) if idcg > 0.0 else 0.0
            ndcgs.append(ndcg)

            # 4. Offer ID retrieval accuracy (Binary: at least 1 hit)
            offer_id_hits.append(1.0 if len(hits) > 0 else 0.0)

            # 5. Exact Numerical-condition Retrieval Accuracy
            # Verify if expected discount number appears in retrieved text
            num_matched = 0
            if expected_numbers:
                for num_str in expected_numbers:
                    clean_num = num_str.rstrip("0").rstrip(".") if "." in num_str else num_str
                    if re.search(r'\b' + re.escape(clean_num) + r'\b', retrieved_text_combined):
                        num_matched += 1
                numerical_hits.append(num_matched / len(expected_numbers))
            else:
                numerical_hits.append(1.0)

            # 6. Source Citation Accuracy
            citation_ratio = valid_citations_count / max(1, len(retrieved_chunks))
            citation_hits.append(citation_ratio)

            per_query_details.append({
                "benchmark_id": b.get("benchmark_id"),
                "query": query,
                "expected_offer_ids": list(expected_ids),
                "retrieved_offer_ids": retrieved_ids,
                "precision": round(p, 3),
                "recall": round(r, 3),
                "rr": round(rr, 3),
                "ndcg": round(ndcg, 3),
                "latency_ms": round(lat_ms, 1)
            })

        n = max(1, len(benchmarks))
        return {
            "mode": mode,
            "precision_at_k": round(sum(precisions) / n, 4),
            "recall_at_k": round(sum(recalls) / n, 4),
            "mrr": round(sum(reciprocal_ranks) / n, 4),
            "ndcg_at_k": round(sum(ndcgs) / n, 4),
            "offer_id_accuracy": round(sum(offer_id_hits) / n, 4),
            "numerical_condition_accuracy": round(sum(numerical_hits) / n, 4),
            "citation_accuracy": round(sum(citation_hits) / n, 4),
            "avg_latency_ms": round(sum(latencies) / n, 2),
            "per_query_details": per_query_details
        }

    @classmethod
    def _retrieve_by_mode(
        cls,
        mode: str,
        query: str,
        top_k: int,
        chunks: List[Dict[str, Any]],
        bm25: BM25Retriever,
        reranker: CrossEncoderReranker
    ) -> List[Dict[str, Any]]:
        """Executes retrieval according to the designated mode."""
        if mode == "BM25":
            return bm25.query(query, top_k=top_k)

        elif mode == "VECTOR":
            return VectorStoreManager.query_similar(query, n_results=top_k)

        elif mode == "HYBRID_NO_RERANK":
            bm25_res = bm25.query(query, top_k=10)
            vec_res = VectorStoreManager.query_similar(query, n_results=10)
            return HybridFusion.reciprocal_rank_fusion(
                bm25_results=bm25_res,
                vector_results=vec_res,
                k=60,
                top_k=top_k
            )

        elif mode == "HYBRID_RERANKED":
            bm25_res = bm25.query(query, top_k=10)
            vec_res = VectorStoreManager.query_similar(query, n_results=10)
            fused = HybridFusion.reciprocal_rank_fusion(
                bm25_results=bm25_res,
                vector_results=vec_res,
                k=60,
                top_k=20
            )
            rerank_res = reranker.rerank(query, fused, top_k=top_k)
            return rerank_res.get("reranked_candidates", fused[:top_k])

        return []
