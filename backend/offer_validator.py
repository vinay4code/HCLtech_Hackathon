import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from backend.profile_builder import CustomerProfile
from database.repositories import OfferRepository, AuditRepository

class ValidationResult(BaseModel):
    is_eligible: bool
    offer_id: str
    offer_name: str
    passed_checks: List[str] = Field(default_factory=list)
    failed_checks: List[str] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    verified_offer_details: Optional[Dict[str, Any]] = None
    validation_timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class OfferValidator:
    """
    Deterministic, rule-based verification engine for telecom promotional offers.
    Evaluates tenure, contract type, active services, validity dates, and active status.
    Guarantees no invalid or unverified offer is ever passed to the LLM.
    """

    @classmethod
    def validate_offer(
        cls,
        offer_data: Dict[str, Any],
        profile: CustomerProfile,
        target_segment: Optional[str] = None
    ) -> ValidationResult:
        offer_id = offer_data.get("offer_id", "UNKNOWN")
        offer_name = offer_data.get("offer_name", "Unknown Offer")

        passed_checks = []
        failed_checks = []
        reason_codes = []

        # 1. Active Status Check
        if bool(offer_data.get("is_active", 1)):
            passed_checks.append("Offer is active in registry.")
        else:
            failed_checks.append("Offer is marked INACTIVE in the offer registry.")
            reason_codes.append("RULE_OFFER_INACTIVE")

        # 2. Date Validity Check
        now_str = datetime.now().strftime("%Y-%m-%d")
        v_start = offer_data.get("validity_start", "2026-01-01")
        v_end = offer_data.get("validity_end", "2026-12-31")
        if v_start <= now_str <= v_end:
            passed_checks.append(f"Current date ({now_str}) is within validity window ({v_start} to {v_end}).")
        else:
            failed_checks.append(f"Offer is expired or not yet valid (valid: {v_start} to {v_end}).")
            reason_codes.append("RULE_DATE_OUT_OF_RANGE")

        # 3. Minimum Tenure Check
        min_tenure = int(offer_data.get("min_tenure", 0))
        max_tenure = int(offer_data.get("max_tenure", 999))
        if min_tenure <= profile.tenure_months <= max_tenure:
            passed_checks.append(f"Customer tenure of {profile.tenure_months}m satisfies [{min_tenure}m - {max_tenure}m] requirement.")
        else:
            failed_checks.append(f"Tenure requirement not met: required {min_tenure}m-{max_tenure}m, customer has {profile.tenure_months}m.")
            reason_codes.append("RULE_TENURE_REQUIREMENT_FAILED")

        # 4. Contract Type Eligibility Check
        eligible_contracts = offer_data.get("eligible_contract_types", [])
        if isinstance(eligible_contracts, str):
            try:
                eligible_contracts = json.loads(eligible_contracts)
            except:
                eligible_contracts = [eligible_contracts]
        
        if not eligible_contracts or profile.contract in eligible_contracts:
            passed_checks.append(f"Customer contract '{profile.contract}' is eligible ({eligible_contracts}).")
        else:
            failed_checks.append(f"Ineligible contract: customer has '{profile.contract}', offer requires one of {eligible_contracts}.")
            reason_codes.append("RULE_CONTRACT_INELIGIBLE")

        # 5. Service Prerequisites & Missing Services Check
        req_missing = offer_data.get("required_missing_services", [])
        if isinstance(req_missing, str):
            try:
                req_missing = json.loads(req_missing)
            except:
                req_missing = []

        if req_missing:
            # For upsell offers, customer must NOT already have the service
            unmet_missing = [s for s in req_missing if s not in profile.service_gaps]
            if not unmet_missing:
                passed_checks.append(f"Customer lacks required service(s) for upsell: {req_missing}.")
            else:
                failed_checks.append(f"Customer already subscribes to target upsell service: {unmet_missing}.")
                reason_codes.append("RULE_ALREADY_SUBSCRIBED")

        # 6. Internet Service Compatibility Check
        eligible_internet = offer_data.get("eligible_internet_services", [])
        if isinstance(eligible_internet, str):
            try:
                eligible_internet = json.loads(eligible_internet)
            except:
                eligible_internet = []

        if not eligible_internet or profile.internet_service in eligible_internet:
            passed_checks.append(f"Internet service '{profile.internet_service}' is compatible.")
        else:
            failed_checks.append(f"Incompatible internet type: customer has '{profile.internet_service}', offer requires {eligible_internet}.")
            reason_codes.append("RULE_INTERNET_SERVICE_INCOMPATIBLE")

        # Overall verdict
        is_eligible = len(failed_checks) == 0

        verified_details = None
        if is_eligible:
            verified_details = {
                "offer_id": offer_id,
                "offer_name": offer_name,
                "offer_category": offer_data.get("offer_category"),
                "discount_type": offer_data.get("discount_type"),
                "discount_value": float(offer_data.get("discount_value", 0.0)),
                "terms_and_conditions": offer_data.get("terms_and_conditions"),
                "source_document": offer_data.get("source_document"),
                "source_page": offer_data.get("source_page", 1),
                "validity_period": f"{v_start} to {v_end}"
            }

        # Log audit record
        AuditRepository.log_action(
            action_type="VALIDATION",
            customer_id=profile.customer_id,
            details={
                "offer_id": offer_id,
                "is_eligible": is_eligible,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "reason_codes": reason_codes
            },
            compliance_status="PASSED" if is_eligible else "FAILED"
        )

        return ValidationResult(
            is_eligible=is_eligible,
            offer_id=offer_id,
            offer_name=offer_name,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            reason_codes=reason_codes,
            verified_offer_details=verified_details
        )
