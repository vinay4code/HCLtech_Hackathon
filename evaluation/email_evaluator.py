from typing import Dict, Any, List
from database.repositories import AuditRepository, ApprovalRepository

class EmailQualityEvaluator:
    """
    Computes enterprise GenAI governance metrics from live audit logs and human decisions:
    - Groundedness Score
    - Hallucination Rate
    - Policy Compliance Rate
    - Human Reviewer Acceptance Rate
    - Refusal / Fallback Rate
    """

    @classmethod
    def compute_governance_metrics(cls) -> Dict[str, Any]:
        logs = AuditRepository.get_recent_logs(limit=500)
        approvals = ApprovalRepository.get_approvals()

        total_generations = 0
        policy_failures = 0
        grounding_failures = 0
        refusals = 0

        for l in logs:
            action = l.get("action_type")
            comp = l.get("compliance_status")
            details = l.get("details", {})

            if action == "GENERATION":
                total_generations += 1
            elif action == "POLICY_FAILURE":
                total_generations += 1
                policy_failures += 1
                if any("Unverified" in v or "percentage" in v for v in details.get("violations", [])):
                    grounding_failures += 1

            if action == "RETRIEVAL" and details.get("candidates_retrieved_count", 0) == 0:
                refusals += 1

        total_eval = max(1, total_generations)
        compliance_rate = round(1.0 - (policy_failures / total_eval), 3)
        hallucination_rate = round(grounding_failures / total_eval, 3)
        groundedness_score = round(1.0 - hallucination_rate, 3)

        # Human Approval metrics
        total_human_reviews = len(approvals)
        approved_count = sum(1 for a in approvals if a.get("reviewer_status") == "APPROVED")
        edited_count = sum(1 for a in approvals if a.get("reviewer_status") == "EDITED")
        rejected_count = sum(1 for a in approvals if a.get("reviewer_status") == "REJECTED")
        pending_count = sum(1 for a in approvals if a.get("reviewer_status") == "PENDING")

        human_acceptance_rate = round((approved_count + edited_count) / max(1, total_human_reviews), 3)

        return {
            "total_generations_audited": total_generations,
            "compliance_rate": compliance_rate,
            "hallucination_rate": hallucination_rate,
            "groundedness_score": groundedness_score,
            "policy_failures_count": policy_failures,
            "refusal_fallback_count": refusals,
            "total_human_reviews": total_human_reviews,
            "approved_count": approved_count,
            "edited_count": edited_count,
            "rejected_count": rejected_count,
            "pending_count": pending_count,
            "human_acceptance_rate": human_acceptance_rate
        }
