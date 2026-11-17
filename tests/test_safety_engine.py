import unittest
from backend.profile_builder import CustomerProfile
from backend.email_generator import EmailDraft
from backend.safety_engine import SafetyEngine

class TestSafetyEngine(unittest.TestCase):
    def setUp(self):
        self.profile = CustomerProfile(
            customer_id="SAFE-TEST-01",
            tenure_months=24,
            tenure_category="13-24m",
            contract="One year",
            monthly_charges=50.0,
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
            data_quality_notes="",
            simulated_email_consent=True,
            simulated_opt_out_all=False
        )
        self.verified_offer = {
            "offer_id": "OFFER-FIBER-BOOST-20",
            "offer_name": "Fiber Speed Boost",
            "discount_type": "PERCENTAGE",
            "discount_value": 20.0,
            "terms_and_conditions": "Valid for 6 months."
        }

    def test_compliant_draft_passes(self):
        draft = EmailDraft(
            customer_id="SAFE-TEST-01",
            recipient_greeting="Valued Customer",
            subject_line="Special 20% discount on your fiber speed upgrade",
            email_body="Enjoy an exclusive 20% discount on your fiber broadband from NovaTel. Visit us to opt-out or review terms.",
            channel="EMAIL",
            offer_id="OFFER-FIBER-BOOST-20",
            discount_mentioned="20%",
            key_benefits=["20% discount"],
            call_to_action="Activate today",
            compliance_disclaimer="NovaTel Telecom. To opt-out, visit preferences. Terms apply."
        )
        verdict = SafetyEngine.audit_draft(draft, self.profile, self.verified_offer)
        self.assertTrue(verdict.passed_safety)
        self.assertEqual(len(verdict.violations), 0)

    def test_hallucinated_discount_fails(self):
        # Claiming 50% discount when offer is 20%
        draft = EmailDraft(
            customer_id="SAFE-TEST-01",
            recipient_greeting="Valued Customer",
            subject_line="Massive 50% discount on broadband!",
            email_body="Get 50% off everything at NovaTel. Terms apply. Visit to opt-out.",
            channel="EMAIL",
            offer_id="OFFER-FIBER-BOOST-20",
            discount_mentioned="50%",
            key_benefits=["50% discount"],
            call_to_action="Activate now",
            compliance_disclaimer="NovaTel. Terms and opt-out apply."
        )
        verdict = SafetyEngine.audit_draft(draft, self.profile, self.verified_offer)
        self.assertFalse(verdict.passed_safety)
        self.assertFalse(verdict.grounding_passed)

    def test_prohibited_aggressive_phrase_fails(self):
        draft = EmailDraft(
            customer_id="SAFE-TEST-01",
            recipient_greeting="Valued Customer",
            subject_line="Pay immediately",
            email_body="Pay your bill now or face a lawsuit immediately from NovaTel. Terms apply. Opt-out link.",
            channel="EMAIL",
            offer_id=None,
            discount_mentioned=None,
            key_benefits=[],
            call_to_action="Pay now",
            compliance_disclaimer="NovaTel terms and opt-out."
        )
        verdict = SafetyEngine.audit_draft(draft, self.profile, None)
        self.assertFalse(verdict.passed_safety)
        self.assertFalse(verdict.prohibited_terms_passed)

    def test_opt_out_consent_violation_fails(self):
        opted_out_profile = self.profile.model_copy(update={"simulated_opt_out_all": True})
        draft = EmailDraft(
            customer_id="SAFE-TEST-01",
            recipient_greeting="Valued Customer",
            subject_line="Special 20% discount on fiber",
            email_body="Enjoy 20% off from NovaTel. Terms and opt-out link.",
            channel="EMAIL",
            offer_id="OFFER-FIBER-BOOST-20",
            discount_mentioned="20%",
            key_benefits=[],
            call_to_action="Activate",
            compliance_disclaimer="NovaTel terms and opt-out."
        )
        verdict = SafetyEngine.audit_draft(draft, opted_out_profile, self.verified_offer)
        self.assertFalse(verdict.passed_safety)
        self.assertFalse(verdict.consent_passed)

if __name__ == "__main__":
    unittest.main()
