import sys
from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import unittest
from rag.vector_store import VectorStoreManager
from rag.retriever import OfferRetriever
from backend.profile_builder import CustomerProfile
from backend.strategy_engine import StrategyEngine
from backend.segmentation_engine import SegmentationEngine

class TestRAGRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        VectorStoreManager.get_client()
        VectorStoreManager.get_collection()

    def test_vector_store_chunks_indexed(self):
        chunks_count = VectorStoreManager.count_chunks()
        self.assertGreater(chunks_count, 0)

    def test_retrieval_returns_grounded_offers(self):
        profile = CustomerProfile(
            customer_id="TEST-CUST-01",
            tenure_months=60,
            tenure_category="48-72",
            contract="Two year",
            monthly_charges=89.50,
            total_charges=5370.00,
            payment_method="Bank transfer (automatic)",
            paperless_billing="Yes",
            phone_service="Yes",
            multiple_lines="Yes",
            internet_service="Fiber optic",
            subscribed_services={"PhoneService": "Yes", "InternetService": "Fiber optic"},
            service_gaps=["OnlineSecurity"],
            subscribed_addons_count=3,
            has_active_service=True,
            data_quality_warning=False,
            data_quality_notes=""
        )
        seg_eng = SegmentationEngine()
        cls_res = seg_eng.classify_customer(profile)
        strat = StrategyEngine.select_strategy(profile, cls_res)
        
        result = OfferRetriever.retrieve_offers_for_customer(profile, strat, top_k=2)
        self.assertIn(result["status"], ["OFFERS_RETRIEVED", "NO_VERIFIED_OFFER_FOUND"])
        if result["status"] == "OFFERS_RETRIEVED":
            for o in result["verified_offers"]:
                self.assertIn("offer_id", o)
                self.assertIn("discount_value", o)
                self.assertIn("terms_and_conditions", o)

    def test_document_checksum_redundancy_prevention(self):
        import hashlib
        from database.repositories import DocumentRepository
        
        test_content = b"PROMOTIONAL CAMPAIGN TEST OFFER: 50% discount for annual lock-in."
        test_chk = hashlib.sha256(test_content).hexdigest()
        
        DocumentRepository.record_document(
            doc_id="TEST-DOC-01",
            filename="test_campaign.txt",
            file_type="TXT",
            file_size_bytes=len(test_content),
            checksum=test_chk,
            num_chunks=2
        )
        
        existing = DocumentRepository.get_by_checksum(test_chk)
        self.assertIsNotNone(existing)
        self.assertEqual(existing["filename"], "test_campaign.txt")
        self.assertEqual(existing["checksum_sha256"], test_chk)

if __name__ == "__main__":
    unittest.main()
