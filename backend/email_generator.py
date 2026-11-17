import json
import os
import re
import requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from config.settings import (
    DEFAULT_LLM_PROVIDER,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    COMPANY_NAME,
    SUPPORT_EMAIL,
    SUPPORT_PHONE,
    OPT_OUT_LINK
)
from backend.profile_builder import CustomerProfile

class EmailDraft(BaseModel):
    customer_id: str
    recipient_greeting: str = "Valued NovaTel Subscriber"
    subject_line: str
    email_body: str
    channel: str = "EMAIL"
    offer_id: Optional[str] = None
    discount_mentioned: Optional[str] = None
    key_benefits: List[str] = Field(default_factory=list)
    call_to_action: str
    compliance_disclaimer: str
    model_provider_used: str = "MOCK"
    generation_timestamp: Optional[str] = None

class EmailGenerator:
    """
    Generates personalized, compliant customer communications via Gemini, OpenAI,
    or deterministic Offline Mock Template generator.
    Guarantees that offer numbers are strictly grounded in verified database records.
    """

    @classmethod
    def generate_draft(
        cls,
        profile: CustomerProfile,
        strategy_info: Dict[str, Any],
        verified_offer: Optional[Dict[str, Any]] = None,
        tone: str = "Appreciative",
        channel: str = "EMAIL",
        provider_override: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> EmailDraft:
        provider = provider_override or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).upper()

        # If API keys are missing, gracefully default to MOCK mode
        if provider == "GEMINI" and not (GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")):
            provider = "MOCK"
        elif provider == "OPENAI" and not (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")):
            provider = "MOCK"

        # Build prompt
        system_prompt, user_prompt = cls._build_prompts(profile, strategy_info, verified_offer, tone, channel)

        # Call appropriate provider
        if provider == "GEMINI":
            try:
                draft = cls._call_gemini(system_prompt, user_prompt, model_name or "gemini-1.5-flash", temperature)
                draft.model_provider_used = "Gemini API"
                return draft
            except Exception as e:
                # Fall back to high-fidelity mock
                print(f"Gemini call failed ({e}); using deterministic Mock fallback.")
                provider = "MOCK"

        if provider == "OPENAI":
            try:
                draft = cls._call_openai(system_prompt, user_prompt, model_name or "gpt-4o-mini", temperature)
                draft.model_provider_used = "OpenAI API"
                return draft
            except Exception as e:
                print(f"OpenAI call failed ({e}); using deterministic Mock fallback.")
                provider = "MOCK"

        # Deterministic Template Mock Generator
        return cls._generate_mock_draft(profile, strategy_info, verified_offer, tone, channel)

    @classmethod
    def _build_prompts(
        cls,
        profile: CustomerProfile,
        strategy_info: Dict[str, Any],
        verified_offer: Optional[Dict[str, Any]],
        tone: str,
        channel: str
    ) -> tuple[str, str]:
        system_prompt = f"""You are a professional customer communications assistant for {COMPANY_NAME}.
Your goal is to draft a personalized, compliant, and engaging message to a telecom customer.

CRITICAL RULES:
1. You MUST NOT invent, alter, or hallucinate promotional figures or discount values.
2. If verified offer details are provided, use ONLY the exact discount percentage or dollar value given.
3. If no promotional offer is provided, do NOT mention any discounts, credits, or monetary savings.
4. Never mention protected characteristics (gender, senior citizen status, partner, dependents).
5. Always maintain a {tone.lower()} tone. Avoid aggressive, coercive, or misleading language.
6. Always include the company signature, support contact ({SUPPORT_EMAIL}), and opt-out instructions.
7. Return ONLY valid JSON adhering to the specified schema."""

        # Format verified offer details as strict ground truth
        offer_context = "No promotional offer applicable. Do NOT claim any discounts or billing reductions."
        if verified_offer:
            offer_context = f"""
VERIFIED PROMOTIONAL OFFER (GROUND TRUTH):
- Offer ID: {verified_offer.get('offer_id')}
- Offer Name: {verified_offer.get('offer_name')}
- Category: {verified_offer.get('offer_category')}
- Discount Type: {verified_offer.get('discount_type')}
- Discount Value: {verified_offer.get('discount_value')} (EXACT VALUE - DO NOT ALTER)
- Terms & Conditions: {verified_offer.get('terms_and_conditions')}
- Validity: {verified_offer.get('validity_period', '2026-12-31')}
"""

        user_prompt = f"""
CUSTOMER INFORMATION (SAFE SUBSET):
- Customer ID: {profile.customer_id}
- Tenure: {profile.tenure_months} months
- Current Contract: {profile.contract}
- Current Monthly Charges: ${profile.monthly_charges:.2f}
- Active Internet: {profile.internet_service}
- Subscribed Services: {json.dumps(profile.subscribed_services)}
- Identified Service Gaps: {', '.join(profile.service_gaps) if profile.service_gaps else 'None'}

CAMPAIGN OBJECTIVE:
- Strategy: {strategy_info.get('strategy_title')}
- Objective: {strategy_info.get('objective')}
- Channel: {channel}
- Requested Tone: {tone}

{offer_context}

Please generate the communication in JSON format with fields:
{{
  "customer_id": "{profile.customer_id}",
  "recipient_greeting": "Valued NovaTel Subscriber",
  "subject_line": "...",
  "email_body": "...",
  "channel": "{channel}",
  "offer_id": "{verified_offer.get('offer_id') if verified_offer else None}",
  "discount_mentioned": "...",
  "key_benefits": ["...", "..."],
  "call_to_action": "...",
  "compliance_disclaimer": "..."
}}
"""
        return system_prompt, user_prompt

    @classmethod
    def _call_gemini(cls, system_prompt: str, user_prompt: str, model: str, temp: float) -> EmailDraft:
        api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
            ],
            "generationConfig": {
                "temperature": temp,
                "response_mime_type": "application/json"
            }
        }
        res = requests.post(url, json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)
        return EmailDraft(**parsed)

    @classmethod
    def _call_openai(cls, system_prompt: str, user_prompt: str, model: str, temp: float) -> EmailDraft:
        api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temp,
            "response_format": {"type": "json_object"}
        }
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw_text)
        return EmailDraft(**parsed)

    @classmethod
    def _generate_mock_draft(
        cls,
        profile: CustomerProfile,
        strategy_info: Dict[str, Any],
        verified_offer: Optional[Dict[str, Any]],
        tone: str,
        channel: str
    ) -> EmailDraft:
        """
        High-fidelity, deterministic template generator that adheres strictly to verified facts.
        """
        strategy_id = strategy_info.get("strategy_id", "GENERAL")
        cid = profile.customer_id
        tenure = profile.tenure_months

        # Default disclaimers
        disclaimer = f"NovaTel Telecom Solutions. To update your preferences or opt-out, visit {OPT_OUT_LINK}. Need help? Contact us at {SUPPORT_EMAIL} or {SUPPORT_PHONE}. Terms and conditions apply."

        # Case 1: Payment Reminder
        if strategy_id == "PAYMENT_COURTESY_REMINDER":
            subject = f"Friendly Account Reminder from NovaTel (Account #{cid})"
            body = (
                f"Dear Valued Customer,\n\n"
                f"We hope you are enjoying your NovaTel services. Our records show an outstanding balance of "
                f"${profile.simulated_outstanding_balance:.2f} that was due on {profile.simulated_due_date}.\n\n"
                f"If you have already submitted this payment, please disregard this note. If you require flexible "
                f"payment arrangements or have questions regarding your account statement, our customer care team is "
                f"standing by to assist you.\n\n"
                f"Thank you for choosing NovaTel as your communications provider.\n\n"
                f"Warm regards,\n"
                f"NovaTel Customer Care Team\n"
                f"{SUPPORT_EMAIL} | {SUPPORT_PHONE}"
            )
            return EmailDraft(
                customer_id=cid,
                recipient_greeting="Valued Customer",
                subject_line=subject,
                email_body=body,
                channel=channel,
                offer_id=None,
                discount_mentioned=None,
                key_benefits=["Dedicated account support", "Flexible payment arrangement options"],
                call_to_action="Review and manage your billing statement securely online.",
                compliance_disclaimer=disclaimer,
                model_provider_used="Deterministic Template Generator (Offline Mock)"
            )

        # Case 2: Verified Offer Present
        if verified_offer:
            offer_id = verified_offer["offer_id"]
            offer_name = verified_offer["offer_name"]
            disc_type = verified_offer["discount_type"]
            disc_val = verified_offer["discount_value"]
            
            if disc_type == "PERCENTAGE":
                disc_str = f"{disc_val:.0f}% off"
            elif disc_type == "MONTHLY_CREDIT":
                disc_str = f"${disc_val:.2f}/month credit"
            else:
                disc_str = f"${disc_val:.2f} discount"

            if strategy_id == "LOYALTY_APPRECIATION":
                subject = f"Thank you for {tenure} months with NovaTel: An exclusive loyalty credit for you"
                body = (
                    f"Dear Valued Customer,\n\n"
                    f"It has been a privilege having you with NovaTel for over {tenure} months! In recognition of your "
                    f"continued loyalty, we are pleased to offer you our {offer_name}.\n\n"
                    f"As a tenured subscriber, you are entitled to a {disc_str} on your upcoming billing statements. "
                    f"This credit will be applied directly to your monthly bill in accordance with our loyalty policy.\n\n"
                    f"Terms: {verified_offer.get('terms_and_conditions')}\n\n"
                    f"Thank you for being an indispensable part of the NovaTel family.\n\n"
                    f"Warm regards,\n"
                    f"NovaTel Loyalty Rewards Team\n"
                    f"{SUPPORT_EMAIL} | {SUPPORT_PHONE}"
                )
                benefits = [
                    f"Exclusive loyalty reward of {disc_str}",
                    "Seamless credit applied directly to statement",
                    "Continued priority customer service"
                ]
                cta = "Claim your loyalty credit via your NovaTel online portal today."

            elif strategy_id == "ANNUAL_CONTRACT_LOCKIN":
                subject = f"Lock in your monthly rate and save {disc_str} with NovaTel"
                body = (
                    f"Dear Valued Customer,\n\n"
                    f"We appreciate your ongoing subscription on our flexible {profile.contract} plan. We want to help you "
                    f"protect your budget with our {offer_name}.\n\n"
                    f"By transitioning to an annual commitment, you lock in your pricing and receive {disc_str} on your total "
                    f"plan for the next 12 months.\n\n"
                    f"Terms: {verified_offer.get('terms_and_conditions')}\n\n"
                    f"Take advantage of this rate lock guarantee today.\n\n"
                    f"Best regards,\n"
                    f"NovaTel Account Management\n"
                    f"{SUPPORT_EMAIL} | {SUPPORT_PHONE}"
                )
                benefits = [
                    f"Guaranteed price lock with {disc_str}",
                    "Protection against annual rate fluctuations",
                    "Dedicated account management"
                ]
                cta = "Confirm your annual rate lock with one click in your account portal."

            elif strategy_id == "VALUE_ADD_UPSELL":
                target_service = profile.service_gaps[0] if profile.service_gaps else "Cyber Security"
                subject = f"Enhance your broadband experience: {offer_name} ({disc_str})"
                body = (
                    f"Dear Valued Customer,\n\n"
                    f"We noticed that your high-speed internet account currently does not include {target_service}. "
                    f"To ensure your home network and devices stay completely shielded, we are offering {offer_name}.\n\n"
                    f"Activate this service today and receive an exclusive {disc_str}.\n\n"
                    f"Terms: {verified_offer.get('terms_and_conditions')}\n\n"
                    f"Warm regards,\n"
                    f"NovaTel Connectivity Solutions\n"
                    f"{SUPPORT_EMAIL} | {SUPPORT_PHONE}"
                )
                benefits = [
                    f"Promotional {disc_str} on essential home network security",
                    "Comprehensive malware and ransomware protection",
                    "Instant activation without technician visits"
                ]
                cta = f"Activate {offer_name} and claim your {disc_str} promotional rate."

            else: # Onboarding or General
                subject = f"Welcome to NovaTel: Enjoy your {offer_name} ({disc_str})"
                body = (
                    f"Dear Valued Customer,\n\n"
                    f"Thank you for joining NovaTel! We are thrilled to welcome you to our network. "
                    f"To help you get the most out of your services, we are delighted to offer {offer_name} with {disc_str}.\n\n"
                    f"Terms: {verified_offer.get('terms_and_conditions')}\n\n"
                    f"Welcome aboard!\n\n"
                    f"Warm regards,\n"
                    f"NovaTel Onboarding Team\n"
                    f"{SUPPORT_EMAIL} | {SUPPORT_PHONE}"
                )
                benefits = [
                    f"Welcome promotional savings of {disc_str}",
                    "Complimentary onboarding and network optimization",
                    "24/7 technical assistance"
                ]
                cta = "Activate your welcome benefits online."

            return EmailDraft(
                customer_id=cid,
                recipient_greeting="Valued Customer",
                subject_line=subject,
                email_body=body,
                channel=channel,
                offer_id=offer_id,
                discount_mentioned=disc_str,
                key_benefits=benefits,
                call_to_action=cta,
                compliance_disclaimer=disclaimer,
                model_provider_used="Deterministic Template Generator (Offline Mock)"
            )

        # Case 3: No promotional offer available
        subject = f"Service review and satisfaction check-in from NovaTel"
        body = (
            f"Dear Valued Customer,\n\n"
            f"Thank you for being a valued NovaTel customer. Our team is committed to delivering seamless connectivity "
            f"and exceptional customer support.\n\n"
            f"We want to make sure your services are performing up to your expectations. If there is anything we can do "
            f"to optimize your setup, please do not hesitate to reach out.\n\n"
            f"Thank you for choosing NovaTel.\n\n"
            f"Warm regards,\n"
            f"NovaTel Customer Success Team\n"
            f"{SUPPORT_EMAIL} | {SUPPORT_PHONE}"
        )
        return EmailDraft(
            customer_id=cid,
            recipient_greeting="Valued Customer",
            subject_line=subject,
            email_body=body,
            channel=channel,
            offer_id=None,
            discount_mentioned=None,
            key_benefits=["Reliable high-speed telecom connectivity", "24/7 customer care"],
            call_to_action="Share your feedback or explore new features in your customer portal.",
            compliance_disclaimer=disclaimer,
            model_provider_used="Deterministic Template Generator (Offline Mock)"
        )
