import re
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from config.settings import BASE_DIR
from backend.profile_builder import CustomerProfile
from backend.email_generator import EmailDraft
from database.repositories import OfferRepository, AuditRepository

POLICY_CONFIG_PATH = BASE_DIR / "config" / "policy_rules.json"

class SafetyVerdict(BaseModel):
    passed_safety: bool
    grounding_passed: bool
    consent_passed: bool
    prohibited_terms_passed: bool
    protected_class_passed: bool
    disclaimers_passed: bool
    violations: List[str] = Field(default_factory=list)
    detected_figures: List[str] = Field(default_factory=list)
    safety_score: float = 1.0 # 0.0 to 1.0

class SafetyEngine:
    """
    Independent deterministic safety and compliance evaluation engine.
    Audits generated email drafts against policy rules, ground-truth records,
    hallucination detection, and customer consent preferences.
    """
    _policy_rules = None

    @classmethod
    def get_policy_rules(cls) -> Dict[str, Any]:
        if cls._policy_rules is None:
            if POLICY_CONFIG_PATH.exists():
                with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
                    cls._policy_rules = json.load(f)
            else:
                cls._policy_rules = {}
        return cls._policy_rules

    @classmethod
    def audit_draft(
        cls,
        draft: EmailDraft,
        profile: CustomerProfile,
        verified_offer: Optional[Dict[str, Any]] = None
    ) -> SafetyVerdict:
        policy = cls.get_policy_rules()
        violations = []

        # 1. Consent Verification
        consent_passed = True
        if policy.get("consent_enforced", True):
            if profile.simulated_opt_out_all:
                violations.append("Customer has globally opted out of all promotional communications.")
                consent_passed = False
            elif draft.channel == "EMAIL" and not profile.simulated_email_consent:
                violations.append("Customer has not granted consent for email marketing communications.")
                consent_passed = False
            elif draft.channel == "SMS" and not profile.simulated_sms_consent:
                violations.append("Customer has not granted consent for SMS marketing communications.")
                consent_passed = False

        # 2. Protected Class Protection
        # Ensure protected attributes (gender, SeniorCitizen, Partner, Dependents) are not mentioned
        protected_attrs = ["gender", "senior citizen", "seniorcitizen", "partner", "dependents", "female", "male"]
        full_text_lower = f"{draft.subject_line} {draft.email_body}".lower()
        protected_class_passed = True
        for attr in protected_attrs:
            if re.search(r'\b' + re.escape(attr) + r'\b', full_text_lower):
                violations.append(f"Protected characteristic '{attr}' detected in generated text.")
                protected_class_passed = False

        # 3. Prohibited Terms Check (Harassment, debt collection threats, deceptive promises)
        prohibited_phrases = policy.get("prohibited_phrases", [])
        prohibited_terms_passed = True
        for phrase in prohibited_phrases:
            if phrase.lower() in full_text_lower:
                violations.append(f"Prohibited aggressive or misleading term detected: '{phrase}'.")
                prohibited_terms_passed = False

        # 4. Claim Grounding and Discount Verification
        grounding_passed = True
        # Extract all numbers, percentages, and dollar amounts in subject and body
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', full_text_lower)
        dollar_matches = re.findall(r'\$\s*(\d+(?:\.\d+)?)', full_text_lower)
        all_detected_figures = [f"{p}%" for p in pct_matches] + [f"${d}" for d in dollar_matches]

        # Valid allowed numbers
        allowed_figures = set()
        # Monthly charges & total charges & balance
        allowed_figures.add(f"{profile.monthly_charges:.0f}")
        allowed_figures.add(f"{profile.monthly_charges:.2f}")
        allowed_figures.add(f"{profile.total_charges:.0f}")
        allowed_figures.add(f"{profile.total_charges:.2f}")
        allowed_figures.add(f"{profile.simulated_outstanding_balance:.0f}")
        allowed_figures.add(f"{profile.simulated_outstanding_balance:.2f}")
        allowed_figures.add(f"{profile.tenure_months}")
        # Standard durations (e.g. 3, 6, 12, 24 months)
        allowed_figures.update(["3", "6", "12", "24", "48", "60", "250", "100", "500"])

        # If verified offer is provided, verify discount value
        if verified_offer:
            offer_val = float(verified_offer.get("discount_value", 0.0))
            offer_type = verified_offer.get("discount_type")
            if offer_type == "PERCENTAGE":
                allowed_figures.add(f"{offer_val:.0f}")
                allowed_figures.add(f"{offer_val:.1f}")
                # Check percentage in text
                if pct_matches:
                    found_vals = [float(p) for p in pct_matches]
                    if not any(abs(v - offer_val) < 0.1 for v in found_vals):
                        violations.append(f"Discount percentage in text does not match verified offer ({offer_val}%).")
                        grounding_passed = False
            elif offer_type in ["FIXED_AMOUNT", "MONTHLY_CREDIT"]:
                allowed_figures.add(f"{offer_val:.0f}")
                allowed_figures.add(f"{offer_val:.2f}")
                if dollar_matches:
                    found_dollars = [float(d) for d in dollar_matches]
                    # Make sure the offer value or customer balance is among the dollar values
                    valid_dollars = [offer_val, profile.monthly_charges, profile.simulated_outstanding_balance]
                    for fd in found_dollars:
                        if not any(abs(fd - vd) < 0.1 for vd in valid_dollars):
                            violations.append(f"Unverified dollar amount ${fd:.2f} detected in draft.")
                            grounding_passed = False
        else:
            # If NO offer was verified, any promotional discount claim is an ungrounded hallucination!
            if pct_matches:
                violations.append(f"Promotional percentage {pct_matches} claimed when no offer was verified.")
                grounding_passed = False

        # 5. Mandatory Disclaimers Check
        disclaimers_passed = True
        mandatory_clauses = policy.get("mandatory_clauses", ["NovaTel", "opt-out", "terms"])
        disclaimer_text = f"{draft.compliance_disclaimer} {draft.email_body}".lower()
        for clause in mandatory_clauses:
            if clause.lower() not in disclaimer_text:
                violations.append(f"Mandatory disclosure clause '{clause}' missing from disclaimer.")
                disclaimers_passed = False

        # Overall Verdict
        passed_safety = (
            consent_passed and
            protected_class_passed and
            prohibited_terms_passed and
            grounding_passed and
            disclaimers_passed
        )

        # Compute continuous safety score (1.0 = flawless, drops with violations)
        penalty = len(violations) * 0.25
        score = max(0.0, round(1.0 - penalty, 2))

        # Log safety check to audit
        AuditRepository.log_action(
            action_type="POLICY_FAILURE" if not passed_safety else "GENERATION",
            customer_id=profile.customer_id,
            details={
                "customer_id": profile.customer_id,
                "passed_safety": passed_safety,
                "safety_score": score,
                "violations": violations,
                "detected_figures": all_detected_figures
            },
            compliance_status="PASSED" if passed_safety else "FAILED"
        )

        return SafetyVerdict(
            passed_safety=passed_safety,
            grounding_passed=grounding_passed,
            consent_passed=consent_passed,
            prohibited_terms_passed=prohibited_terms_passed,
            protected_class_passed=protected_class_passed,
            disclaimers_passed=disclaimers_passed,
            violations=violations,
            detected_figures=all_detected_figures,
            safety_score=score
        )
