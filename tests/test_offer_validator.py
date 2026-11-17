import unittest
from backend.profile_builder import CustomerProfile
from backend.offer_validator import OfferValidator

class TestOfferValidator(unittest.TestCase):
    def setUp(self):
        self.mock_offer = {
            "offer_id": "OFFER-TEST-20",
            "offer_name": "Test Fiber Promo",
            "offer_category": "UPGRADE",
            "discount_type": "PERCENTAGE",
            "discount_value": 20.0,
            "min_tenure": 12,
            "max_tenure": 999,
            "eligible_contract_types": ["Month-to-month", "One year"],
            "eligible_internet_services": ["Fiber optic"],
            "required_missing_services": ["OnlineSecurity"],
            "validity_start": "2026-01-01",
            "validity_end": "2026-12-31",
            "terms_and_conditions": "Valid for 12 months.",
            "source_document": "test.txt",
            "source_page": 1,
            "is_active": 1
        }

    def test_eligible_customer_passes(self):
        profile = CustomerProfile(
            customer_id="PASS-01",
            tenure_months=18,
            tenure_category="13-24m",
            contract="Month-to-month",
            monthly_charges=70.0,
            total_charges=1260.0,
            payment_method="Bank transfer",
            paperless_billing="Yes",
            phone_service="Yes",
            multiple_lines="No",
            internet_service="Fiber optic",
            subscribed_services={},
            service_gaps=["OnlineSecurity", "DeviceProtection"],
            subscribed_addons_count=1,
            has_active_service=True,
            data_quality_warning=False,
            data_quality_notes=""
        )
        res = OfferValidator.validate_offer(self.mock_offer, profile)
        self.assertTrue(res.is_eligible)
        self.assertEqual(len(res.failed_checks), 0)
        self.assertIsNotNone(res.verified_offer_details)

    def test_ineligible_tenure_fails(self):
        profile = CustomerProfile(
            customer_id="FAIL-TENURE",
            tenure_months=5, # Requires 12
            tenure_category="0-6m",
            contract="Month-to-month",
            monthly_charges=70.0,
            total_charges=350.0,
            payment_method="Bank transfer",
            paperless_billing="Yes",
            phone_service="Yes",
            multiple_lines="No",
            internet_service="Fiber optic",
            subscribed_services={},
            service_gaps=["OnlineSecurity"],
            subscribed_addons_count=1,
            has_active_service=True,
            data_quality_warning=False,
            data_quality_notes=""
        )
        res = OfferValidator.validate_offer(self.mock_offer, profile)
        self.assertFalse(res.is_eligible)
        self.assertIn("RULE_TENURE_REQUIREMENT_FAILED", res.reason_codes)

    def test_ineligible_contract_fails(self):
        profile = CustomerProfile(
            customer_id="FAIL-CONTRACT",
            tenure_months=24,
            tenure_category="13-24m",
            contract="Two year", # Offer requires Month-to-month or One year
            monthly_charges=70.0,
            total_charges=1680.0,
            payment_method="Bank transfer",
            paperless_billing="Yes",
            phone_service="Yes",
            multiple_lines="No",
            internet_service="Fiber optic",
            subscribed_services={},
            service_gaps=["OnlineSecurity"],
            subscribed_addons_count=1,
            has_active_service=True,
            data_quality_warning=False,
            data_quality_notes=""
        )
        res = OfferValidator.validate_offer(self.mock_offer, profile)
        self.assertFalse(res.is_eligible)
        self.assertIn("RULE_CONTRACT_INELIGIBLE", res.reason_codes)

if __name__ == "__main__":
    unittest.main()
