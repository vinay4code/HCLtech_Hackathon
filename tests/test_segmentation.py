import unittest
from backend.profile_builder import ProfileBuilder, CustomerProfile
from backend.segmentation_engine import SegmentationEngine

class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SegmentationEngine.get_instance()

    def test_random_forest_metrics_and_feature_importances(self):
        metrics_info = self.engine.get_model_metrics()
        self.assertIn("metrics", metrics_info)
        metrics = metrics_info["metrics"]
        self.assertGreater(metrics["accuracy"], 0.85)
        self.assertIn("feature_importances", metrics_info)
        importances = metrics_info["feature_importances"]
        self.assertIn("tenure", importances)
        self.assertIn("TotalCharges", importances)

    def test_long_term_loyal_ml_classification(self):
        # Tenure >= 48
        profile = CustomerProfile(
            customer_id="TEST-LTL",
            tenure_months=55,
            tenure_category="49-72m (Loyal)",
            contract="Two year",
            monthly_charges=75.0,
            total_charges=4125.0,
            payment_method="Bank transfer",
            paperless_billing="Yes",
            phone_service="Yes",
            multiple_lines="Yes",
            internet_service="Fiber optic",
            subscribed_services={},
            service_gaps=[],
            subscribed_addons_count=6,
            has_active_service=True,
            data_quality_warning=False,
            data_quality_notes=""
        )
        res = self.engine.classify_customer(profile)
        self.assertEqual(res["primary_segment"], "LONG_TERM_LOYAL")
        self.assertIn("LONG_TERM_LOYAL", res["segments"])
        self.assertGreater(res["confidence_score"], 0.70)
        self.assertIn("Random Forest", res["model_provenance"])
        # Probability distribution checks
        self.assertIn("probabilities", res)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=2)

    def test_early_stage_customer_ml_classification(self):
        # Tenure <= 6
        profile = CustomerProfile(
            customer_id="TEST-ESC",
            tenure_months=2,
            tenure_category="0-6m (New)",
            contract="Month-to-month",
            monthly_charges=45.0,
            total_charges=90.0,
            payment_method="Electronic check",
            paperless_billing="Yes",
            phone_service="Yes",
            multiple_lines="No",
            internet_service="DSL",
            subscribed_services={},
            service_gaps=["OnlineSecurity", "OnlineBackup"],
            subscribed_addons_count=1,
            has_active_service=True,
            data_quality_warning=False,
            data_quality_notes=""
        )
        res = self.engine.classify_customer(profile)
        self.assertEqual(res["primary_segment"], "EARLY_STAGE_CUSTOMER")
        self.assertIn("EARLY_STAGE_CUSTOMER", res["segments"])
        self.assertGreater(res["confidence_score"], 0.70)

    def test_contract_renewal_candidate_ml_classification(self):
        # Month-to-month, tenure >= 12
        profile = CustomerProfile(
            customer_id="TEST-CRC",
            tenure_months=20,
            tenure_category="13-24m (Established)",
            contract="Month-to-month",
            monthly_charges=60.0,
            total_charges=1200.0,
            payment_method="Credit card",
            paperless_billing="Yes",
            phone_service="Yes",
            multiple_lines="No",
            internet_service="Fiber optic",
            subscribed_services={},
            service_gaps=[],
            subscribed_addons_count=4,
            has_active_service=True,
            data_quality_warning=False,
            data_quality_notes=""
        )
        res = self.engine.classify_customer(profile)
        self.assertEqual(res["primary_segment"], "CONTRACT_RENEWAL_CANDIDATE")
        self.assertIn("CONTRACT_RENEWAL_CANDIDATE", res["segments"])
        self.assertGreater(res["confidence_score"], 0.50)

if __name__ == "__main__":
    unittest.main()
