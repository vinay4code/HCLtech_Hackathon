import unittest
import pandas as pd
from backend.preprocessing import PreprocessingPipeline
from backend.feature_engineering import FeatureEngineering

class TestPreprocessing(unittest.TestCase):
    def test_clean_telecom_data(self):
        df = PreprocessingPipeline.process_telecom_data(reload=True)
        self.assertEqual(len(df), 7043)
        self.assertTrue(pd.api.types.is_float_dtype(df["TotalCharges"]))
        self.assertTrue(pd.api.types.is_integer_dtype(df["tenure"]))
        self.assertFalse(df["TotalCharges"].isnull().any())

    def test_total_charges_imputation_and_warning(self):
        df = PreprocessingPipeline.process_telecom_data()
        # Verify rows where tenure == 0 have data quality warnings
        zero_tenure = df[df["tenure"] == 0]
        self.assertEqual(len(zero_tenure), 11)
        self.assertTrue(zero_tenure["data_quality_warning"].all())
        self.assertTrue((zero_tenure["TotalCharges"] == 0.0).all())

    def test_feature_engineering_enrichment(self):
        df = PreprocessingPipeline.process_telecom_data()
        enriched = FeatureEngineering.enrich_dataframe(df)
        self.assertIn("service_gaps", enriched.columns)
        self.assertIn("subscribed_addons_count", enriched.columns)
        self.assertIn("has_active_service", enriched.columns)
        self.assertIn("tenure_category", enriched.columns)

if __name__ == "__main__":
    unittest.main()
