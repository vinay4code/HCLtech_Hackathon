import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import unittest
from rag.bm25_retriever import BM25Tokenizer, BM25Retriever
from rag.vector_store import VectorStoreManager
from rag.hybrid_fusion import HybridFusion
from rag.reranker import CrossEncoderReranker
from rag.retriever import OfferRetriever
from backend.profile_builder import CustomerProfile
from backend.strategy_engine import StrategyEngine
from backend.segmentation_engine import SegmentationEngine

class TestHybridRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        VectorStoreManager.get_client()
        VectorStoreManager.get_collection()

    def test_bm25_tokenization_preserves_qualifiers_and_numbers(self):
        text = "OFFER-LOYALTY-CREDIT: $15.00/month discount. Not eligible if cancelled. Minimum 48 months tenure. Up to 500Mbps."
        tokens = BM25Tokenizer.tokenize(text)
        
        self.assertIn("$15.00", tokens)
        self.assertIn("offer-loyalty-credit", tokens)
        self.assertIn("not_eligible", tokens)
        self.assertIn("minimum", tokens)
        self.assertIn("48", tokens)
        self.assertIn("500mbps", tokens)

    def test_bm25_retrieval_scoring(self):
        chunks = [
            {"chunk_id": "c1", "content": "OFFER-LOYALTY-CREDIT gives $15.00 bill credit for long-term loyal customers with 48 months tenure.", "metadata": {"offer_id": "OFFER-LOYALTY-CREDIT"}},
            {"chunk_id": "c2", "content": "Fiber Speed Boost 500Mbps 20% discount on broadband.", "metadata": {"offer_id": "OFFER-FIBER-BOOST-20"}},
            {"chunk_id": "c3", "content": "Welcome streampass 25% discount for early stage onboarding.", "metadata": {"offer_id": "OFFER-WELCOME-STREAM-25"}}
        ]
        retriever = BM25Retriever()
        retriever.build_index(chunks)
        results = retriever.query("OFFER-LOYALTY-CREDIT $15.00 48 months", top_k=2)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertGreater(results[0]["bm25_score"], 0.0)

    def test_rrf_fusion_merging_and_deduplication(self):
        bm25_res = [
            {"chunk_id": "c1", "content": "Offer 1", "bm25_score": 3.5, "bm25_rank": 1},
            {"chunk_id": "c2", "content": "Offer 2", "bm25_score": 2.1, "bm25_rank": 2}
        ]
        vector_res = [
            {"chunk_id": "c2", "content": "Offer 2", "similarity_score": 0.95, "vector_rank": 1},
            {"chunk_id": "c3", "content": "Offer 3", "similarity_score": 0.80, "vector_rank": 2}
        ]
        fused = HybridFusion.reciprocal_rank_fusion(bm25_res, vector_res, k=60)

        # There are 3 unique chunks: c1, c2, c3
        self.assertEqual(len(fused), 3)
        # c2 appeared in both (rank 2 in BM25, rank 1 in Vector):
        # RRF(c2) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
        # RRF(c1) = 1/(60+1) = 0.01639
        # RRF(c3) = 1/(60+2) = 0.01613
        # Therefore, c2 MUST be ranked #1!
        self.assertEqual(fused[0]["chunk_id"], "c2")
        self.assertEqual(fused[0]["fused_rank"], 1)
        self.assertIn("BM25", fused[0]["source_retrievers"])
        self.assertIn("VECTOR", fused[0]["source_retrievers"])

    def test_cross_encoder_reranker_and_fallback(self):
        candidates = [
            {"chunk_id": "c1", "content": "Broadband fiber promo with 20% discount", "rrf_score": 0.03},
            {"chunk_id": "c2", "content": "Loyalty appreciation bill credit", "rrf_score": 0.02}
        ]
        reranker = CrossEncoderReranker()
        res = reranker.rerank(
            query="Find fiber optic internet discount",
            candidates=candidates,
            top_k=2
        )
        self.assertIn("reranked_candidates", res)
        self.assertEqual(len(res["reranked_candidates"]), 2)
        # Status should either be CROSS_ENCODER_APPLIED or SKIPPED_CONTROLLED_FALLBACK_RRF
        self.assertIn(res["status"], ["CROSS_ENCODER_APPLIED", "SKIPPED_CONTROLLED_FALLBACK_RRF"])

    def test_end_to_end_hybrid_pipeline_with_observability(self):
        # Test on loyal customer
        profile = CustomerProfile(
            customer_id="6388-TABGU",
            tenure_months=62,
            tenure_category="48-72",
            contract="One year",
            monthly_charges=56.15,
            total_charges=3487.95,
            payment_method="Bank transfer (automatic)",
            paperless_billing="No",
            phone_service="No",
            multiple_lines="No phone service",
            internet_service="DSL",
            subscribed_services={"InternetService": "DSL"},
            service_gaps=["OnlineSecurity", "DeviceProtection"],
            subscribed_addons_count=0,
            has_active_service=True,
            data_quality_warning=False,
            data_quality_notes=""
        )
        seg_eng = SegmentationEngine()
        cls_res = seg_eng.classify_customer(profile)
        strat = StrategyEngine.select_strategy(profile, cls_res)

        result = OfferRetriever.retrieve_offers_for_customer(profile, strat)
        self.assertIn("observability", result)
        obs = result["observability"]
        self.assertIn("context_query", obs)
        self.assertIn("bm25_results", obs)
        self.assertIn("vector_results", obs)
        self.assertIn("fused_results", obs)
        self.assertIn("reranked_results", obs)
        self.assertIn("source_citations", result)

    def test_document_registry_conflict_detection(self):
        conflict_chunk = [
            {"chunk_id": "c_bad", "content": "Fiber Speed Boost OFFER-FIBER-BOOST-20 has a massive 75% discount off retail rates.", "metadata": {"offer_id": "OFFER-FIBER-BOOST-20"}}
        ]
        reg_offer = {
            "offer_id": "OFFER-FIBER-BOOST-20",
            "discount_type": "PERCENTAGE",
            "discount_value": 20.0
        }
        conflict = OfferRetriever._check_document_registry_conflict("OFFER-FIBER-BOOST-20", conflict_chunk, reg_offer)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["conflict_type"], "DISCOUNT_MISMATCH")
        self.assertEqual(conflict["registry_value"], 20.0)

    def test_rrf_confidence_calculation_and_auto_approval_threshold(self):
        # High confidence item: ranked #1 in BM25 and #2 in Vector
        bm25_high = [{"chunk_id": "c_high", "content": "High agreement", "bm25_score": 5.0, "bm25_rank": 1}]
        vector_high = [{"chunk_id": "c_high", "content": "High agreement", "similarity_score": 0.98, "vector_rank": 2}]
        fused_high = HybridFusion.reciprocal_rank_fusion(bm25_high, vector_high, k=60)

        self.assertGreater(fused_high[0]["rrf_confidence"], 0.70)
        self.assertTrue(fused_high[0]["is_auto_approvable"])
        self.assertGreater(fused_high[0]["rrf_confidence_pct"], 70.0)

        # Low confidence item: ranked only in BM25 at rank 1, missing in Vector
        bm25_low = [{"chunk_id": "c_low", "content": "Only BM25", "bm25_score": 1.0, "bm25_rank": 1}]
        fused_low = HybridFusion.reciprocal_rank_fusion(bm25_low, [], k=60)

        # 1/61 / (2/61) = 0.50 <= 0.70
        self.assertEqual(round(fused_low[0]["rrf_confidence"], 2), 0.50)
        self.assertFalse(fused_low[0]["is_auto_approvable"])
        self.assertLessEqual(fused_low[0]["rrf_confidence_pct"], 70.0)

    def test_auto_approval_repository_persistence_and_batch_actions(self):
        from database.repositories import ApprovalRepository
        import uuid

        # Create a pending approval with high confidence (> 0.70)
        high_id = f"TEST-APPR-{uuid.uuid4().hex[:6].upper()}"
        ApprovalRepository.create_approval(
            approval_id=high_id,
            customer_id="TEST-CUST-1",
            offer_id="OFFER-LOYALTY-CREDIT",
            segment="LONG_TERM_LOYAL",
            subject="High Conf Auto Test",
            body="Test body",
            confidence_score=0.92,
            status="PENDING"
        )

        # Create a pending approval with low confidence (<= 0.70)
        low_id = f"TEST-APPR-{uuid.uuid4().hex[:6].upper()}"
        ApprovalRepository.create_approval(
            approval_id=low_id,
            customer_id="TEST-CUST-2",
            offer_id="OFFER-LOYALTY-CREDIT",
            segment="LONG_TERM_LOYAL",
            subject="Low Conf Pending Test",
            body="Test body",
            confidence_score=0.45,
            status="PENDING"
        )

        # Execute auto-approve for items > 70% threshold
        auto_approved_count = ApprovalRepository.auto_approve_high_confidence(threshold=0.70)
        self.assertGreaterEqual(auto_approved_count, 1)

        # High confidence record must now be APPROVED
        rec_high = ApprovalRepository.get_approval_by_id(high_id)
        self.assertEqual(rec_high["reviewer_status"], "APPROVED")
        self.assertIn("Auto-approved", rec_high["reviewer_notes"])

        # Low confidence record must remain PENDING
        rec_low = ApprovalRepository.get_approval_by_id(low_id)
        self.assertEqual(rec_low["reviewer_status"], "PENDING")

if __name__ == "__main__":
    unittest.main()
