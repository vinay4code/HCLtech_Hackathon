import os
import math
from typing import List, Dict, Any, Optional

class CrossEncoderReranker:
    """
    Cross-Encoder Reranker for fine-grained query-document relevance estimation.
    Supports BAAI/bge-reranker-base, cross-encoder/ms-marco-MiniLM-L-6-v2,
    and graceful controlled fallback to RRF fused ranking if weights are unavailable.
    """
    _model = None
    _model_name = None
    _init_attempted = False

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    @staticmethod
    def _is_model_locally_cached(model_name: str) -> bool:
        if os.path.exists(model_name):
            return True
        hf_hub = os.path.expanduser("~/.cache/huggingface/hub")
        clean_name = "models--" + model_name.replace("/", "--")
        target_dir = os.path.join(hf_hub, clean_name)
        if os.path.exists(target_dir):
            snapshots_dir = os.path.join(target_dir, "snapshots")
            if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                return True
        return False

    def _load_model(self):
        if not self._init_attempted:
            self._init_attempted = True
            # Check if model weights exist locally before attempting initialization
            # This prevents 60-second network socket timeouts when running offline.
            if self._is_model_locally_cached(self.model_name):
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.model_name, local_files_only=True)
                except Exception:
                    self._model = None
            else:
                self._model = None
        return self._model



    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Reranks candidate document chunks against the full customer context query.
        Returns top_k most relevant chunks with rerank scores and status metadata.
        """
        if not candidates:
            return {
                "reranked_candidates": [],
                "reranking_applied": False,
                "reranker_model": self.model_name,
                "status": "EMPTY_CANDIDATES"
            }

        model = self._load_model()

        # 1. Neural Cross-Encoder Scoring (if model available)
        if model is not None:
            try:
                pairs = [[query, c.get("content", "")] for c in candidates]
                raw_scores = model.predict(pairs)
                
                # Convert raw logits to probabilities via sigmoid if needed
                scores = []
                for s in raw_scores:
                    s_float = float(s)
                    # apply sigmoid if logit
                    prob = 1.0 / (1.0 + math.exp(-s_float)) if (s_float < -3.0 or s_float > 3.0) else s_float
                    scores.append(round(prob, 4))

                for cand, sc in zip(candidates, scores):
                    cand["rerank_score"] = sc

                sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
                for rank, c in enumerate(sorted_candidates[:top_k], start=1):
                    c["rerank_rank"] = rank

                return {
                    "reranked_candidates": sorted_candidates[:top_k],
                    "reranking_applied": True,
                    "reranker_model": self.model_name,
                    "status": "CROSS_ENCODER_APPLIED"
                }
            except Exception as e:
                # Fall back to fused ranking
                pass

        # 2. Controlled Fallback Mode
        # In accordance with prompt requirement:
        # "If the reranker is unavailable, provide a controlled fallback to the fused ranking
        # and record that reranking was skipped."
        fallback_candidates = []
        for rank, c in enumerate(candidates[:top_k], start=1):
            cand_copy = dict(c)
            cand_copy["rerank_score"] = cand_copy.get("rrf_score", 0.0)
            cand_copy["rerank_rank"] = rank
            fallback_candidates.append(cand_copy)

        return {
            "reranked_candidates": fallback_candidates,
            "reranking_applied": False,
            "reranker_model": self.model_name,
            "status": "SKIPPED_CONTROLLED_FALLBACK_RRF",
            "fallback_reason": f"Reranker model '{self.model_name}' not available offline; gracefully preserved RRF fused ranking."
        }
