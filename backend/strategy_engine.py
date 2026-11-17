from typing import Dict, Any, List
from backend.profile_builder import CustomerProfile

class StrategyEngine:
    """
    Determines the optimal personalized engagement strategy, channel, and RAG retrieval query
    based on customer profile, segments, and service gaps.
    """

    @staticmethod
    def select_strategy(profile: CustomerProfile, classification: Dict[str, Any]) -> Dict[str, Any]:
        segments = classification.get("segments", [])
        primary = classification.get("primary_segment", "STANDARD_SUBSCRIBER")

        # 1. Check if simulated payment status is Overdue (demonstration of operational routing)
        if profile.simulated_billing_status == "Overdue":
            return {
                "strategy_id": "PAYMENT_COURTESY_REMINDER",
                "strategy_title": "Courteous Payment & Account Continuity Notice",
                "objective": "Gently notify customer of past due balance with flexible payment options while preserving trust.",
                "target_segment": primary,
                "recommended_channel": "Email" if profile.simulated_email_consent else "Direct Mail",
                "retrieval_query": "billing payment support flexible extension credit relief",
                "metadata_filter": {},
                "rationale": f"Simulated billing record indicates overdue balance of ${profile.simulated_outstanding_balance:.2f} due on {profile.simulated_due_date}.",
                "requires_promotional_offer": False
            }

        # 2. LONG_TERM_LOYAL
        if "LONG_TERM_LOYAL" in segments:
            return {
                "strategy_id": "LOYALTY_APPRECIATION",
                "strategy_title": "VIP Loyalty Milestone & Bill Credit Reward",
                "objective": "Express sincere appreciation for extended loyalty and reward customer with bill credit or premium upgrade.",
                "target_segment": "LONG_TERM_LOYAL",
                "recommended_channel": "Email",
                "retrieval_query": "long-term loyalty appreciation discount bill credit reward tenured customer",
                "metadata_filter": {"target_segment": "LONG_TERM_LOYAL"},
                "rationale": f"Customer has maintained continuous active service for {profile.tenure_months} months.",
                "requires_promotional_offer": True
            }

        # 3. CONTRACT_RENEWAL_CANDIDATE
        if "CONTRACT_RENEWAL_CANDIDATE" in segments:
            return {
                "strategy_id": "ANNUAL_CONTRACT_LOCKIN",
                "strategy_title": "Annual Commitment Price Lock Incentive",
                "objective": "Encourage customer to transition from Month-to-month to an annual agreement with guaranteed rate lock.",
                "target_segment": "CONTRACT_RENEWAL_CANDIDATE",
                "recommended_channel": "Email",
                "retrieval_query": "contract renewal switch month-to-month annual commitment discount rate lock",
                "metadata_filter": {"target_segment": "CONTRACT_RENEWAL_CANDIDATE"},
                "rationale": f"Customer has been on Month-to-month terms for {profile.tenure_months} months; high probability of churn without commitment.",
                "requires_promotional_offer": True
            }

        # 4. UPSELL_OPPORTUNITY
        if "UPSELL_OPPORTUNITY" in segments:
            missing_services = profile.service_gaps
            if "OnlineSecurity" in missing_services or "OnlineBackup" in missing_services:
                query_focus = "cyber security shield antivirus cloud backup bundle discount"
            elif "TechSupport" in missing_services:
                query_focus = "premium 24/7 technical support device care trial"
            elif "StreamingTV" in missing_services or "StreamingMovies" in missing_services:
                query_focus = "entertainment streaming movies tv trial discount"
            else:
                query_focus = "fiber high speed internet boost bandwidth upgrade"

            return {
                "strategy_id": "VALUE_ADD_UPSELL",
                "strategy_title": "Targeted Service Adoption & Protection Addon",
                "objective": "Introduce complementary security and device protection services tailored to customer's active internet connection.",
                "target_segment": "UPSELL_OPPORTUNITY",
                "recommended_channel": "Email",
                "retrieval_query": query_focus,
                "metadata_filter": {"target_segment": "UPSELL_OPPORTUNITY"},
                "rationale": f"Customer currently lacks {len(missing_services)} optional services ({', '.join(missing_services[:2])}).",
                "requires_promotional_offer": True
            }

        # 5. EARLY_STAGE_CUSTOMER
        if "EARLY_STAGE_CUSTOMER" in segments:
            return {
                "strategy_id": "NEW_SUBSCRIBER_WELCOME",
                "strategy_title": "New Subscriber Onboarding & Welcome Benefits",
                "objective": "Welcome customer to NovaTel, ensure smooth service setup, and offer introductory entertainment pass.",
                "target_segment": "EARLY_STAGE_CUSTOMER",
                "recommended_channel": "Email",
                "retrieval_query": "new subscriber welcome introductory discount entertainment streampass onboarding",
                "metadata_filter": {"target_segment": "EARLY_STAGE_CUSTOMER"},
                "rationale": f"Customer joined within the last {profile.tenure_months} month(s); critical window for onboarding satisfaction.",
                "requires_promotional_offer": True
            }

        # 6. Default / Consistent Subscriber
        return {
            "strategy_id": "ENGAGEMENT_CHECKIN",
            "strategy_title": "Valued Subscriber Check-In & Service Review",
            "objective": "Proactively verify customer satisfaction and showcase relevant features.",
            "target_segment": primary,
            "recommended_channel": "Email",
            "retrieval_query": "customer loyalty broadband phone service benefits",
            "metadata_filter": {},
            "rationale": f"Active subscriber with tenure {profile.tenure_months}m on {profile.contract} contract.",
            "requires_promotional_offer": True
        }
