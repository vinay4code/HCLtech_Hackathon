import uuid
import streamlit as st
from backend.profile_builder import ProfileBuilder
from backend.segmentation_engine import SegmentationEngine
from backend.strategy_engine import StrategyEngine
from rag.retriever import OfferRetriever
from backend.offer_validator import OfferValidator
from backend.email_generator import EmailGenerator
from backend.safety_engine import SafetyEngine
from database.repositories import ApprovalRepository
from frontend.components import render_header, segment_badge

def render_email_generation_page():
    render_header(
        "Personalization & Email Generation",
        "Generate compliant, grounded customer communications with deterministic offer constraints."
    )

    df = ProfileBuilder.get_enriched_dataframe()
    seg_engine = SegmentationEngine()

    # Customer Selection
    default_cid = st.session_state.get("selected_customer_id", df["customerID"].iloc[0])
    all_cids = df["customerID"].head(100).tolist()
    if default_cid not in all_cids:
        all_cids.insert(0, default_cid)
    
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    with col_c1:
        selected_cid = st.selectbox("Select Target Customer:", all_cids, index=all_cids.index(default_cid))
        st.session_state.selected_customer_id = selected_cid
    with col_c2:
        tone = st.selectbox("Communication Tone:", ["Appreciative", "Professional", "Friendly", "Concise", "Urgent"])
    with col_c3:
        channel = st.selectbox("Delivery Channel:", ["EMAIL", "SMS", "IN_APP"])

    # Load customer profile & strategy
    profile = ProfileBuilder.build_profile(selected_cid)
    classification = seg_engine.classify_customer(profile)
    strategy = StrategyEngine.select_strategy(profile, classification)

    st.markdown("---")
    c_info, c_rag = st.columns([1, 1])

    with c_info:
        st.markdown(f"#### Customer Profile: `{profile.customer_id}`")
        st.markdown(f"**Tenure:** {profile.tenure_months}m | **Contract:** `{profile.contract}` | **Monthly:** `${profile.monthly_charges:.2f}`")
        st.markdown(f"**Internet:** `{profile.internet_service}` | **Phone:** `{profile.phone_service}`")
        st.markdown(f"**Service Gaps:** `{', '.join(profile.service_gaps) if profile.service_gaps else 'None'}`")
        st.markdown(f"**Segment:** {segment_badge(classification['primary_segment'])}", unsafe_allow_html=True)
        st.markdown(f"**Campaign Strategy:** `{strategy['strategy_title']}`")

    with c_rag:
        st.markdown("#### RAG Retrieval & Deterministic Validation")
        retrieval = OfferRetriever.retrieve_offers_for_customer(profile, strategy)
        verified_offer = None

        if retrieval["status"] == "OFFERS_RETRIEVED" and retrieval["verified_offers"]:
            candidate = retrieval["verified_offers"][0]
            val_res = OfferValidator.validate_offer(candidate, profile)
            if val_res.is_eligible:
                verified_offer = val_res.verified_offer_details
                st.success(f"✅ **Verified Offer:** `{candidate['offer_name']}` ({candidate['discount_value']} {candidate['discount_type']})")
                st.caption(f"Source: {candidate['source_document']} | ID: {candidate['offer_id']}")
            else:
                st.warning(f"⚠️ Top retrieved offer `{candidate['offer_name']}` failed eligibility: {val_res.failed_checks}")
        else:
            st.info("ℹ️ No promotional offer retrieved. Will generate non-promotional service check-in.")

        rrf_conf = retrieval.get("rrf_confidence", 0.0)
        rrf_pct = retrieval.get("rrf_confidence_pct", 0.0)
        is_auto_eligible = retrieval.get("auto_approval_eligible", False)

        if is_auto_eligible:
            st.markdown(
                f"""
                <div style="background-color: #ECFDF5; border: 1px solid #10B981; border-radius: 8px; padding: 8px 12px; margin-top: 6px; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #065F46; font-size: 13px;">🎯 RRF Retrieval Confidence: {rrf_pct:.1f}%</span>
                    <span style="background-color: #10B981; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-left: 8px;">⚡ Auto-Approval Eligible (&gt;70%)</span>
                    <div style="font-size: 11px; color: #047857; margin-top: 2px;">Score exceeds 70% threshold. Qualifies for instant automated approval upon generation.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #FFFBEB; border: 1px solid #F59E0B; border-radius: 8px; padding: 8px 12px; margin-top: 6px; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #92400E; font-size: 13px;">🎯 RRF Retrieval Confidence: {rrf_pct:.1f}%</span>
                    <span style="background-color: #F59E0B; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-left: 8px;">⏳ Manual Review Required (&le;70%)</span>
                    <div style="font-size: 11px; color: #B45309; margin-top: 2px;">Score &le; 70% threshold. Human oversight required before delivery.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if retrieval.get("source_citations"):
            with st.expander("📚 Supporting Hybrid RAG Evidence Citations", expanded=False):
                for cit in retrieval["source_citations"]:
                    st.markdown(f"- **{cit['document_name']}** (p. {cit['page_number']}) | `{cit['chunk_id']}` (RRF Conf: {cit.get('rrf_confidence_pct', 0.0)}%): {cit['excerpt']}")


    # Model Parameters & Generation
    st.markdown("---")
    st.markdown("### LLM Generation Controls")

    gen_col1, gen_col2 = st.columns([3, 1])
    with gen_col1:
        st.caption("Grounded Generation Mode: The LLM is strictly constrained to verified database facts. Sensitive customer demographics are excluded.")
    with gen_col2:
        generate_clicked = st.button("✨ Generate AI Email Draft", type="primary", use_container_width=True)

    if generate_clicked or "active_draft" in st.session_state:
        if generate_clicked:
            with st.spinner("Executing RAG-grounded email generation..."):
                draft = EmailGenerator.generate_draft(
                    profile=profile,
                    strategy_info=strategy,
                    verified_offer=verified_offer,
                    tone=tone,
                    channel=channel
                )
                safety = SafetyEngine.audit_draft(draft, profile, verified_offer)
                st.session_state.active_draft = draft
                st.session_state.active_safety = safety
                st.session_state.active_verified_offer = verified_offer

        draft = st.session_state.active_draft
        safety = st.session_state.active_safety
        verified_offer = st.session_state.get("active_verified_offer")

        st.markdown("### Generated Communication Preview")
        
        auto_approved = bool(is_auto_eligible and safety.passed_safety)

        # Safety & Auto-Approval Banner
        if safety.passed_safety:
            if auto_approved:
                st.success(
                    f"🛡️ **Compliance Audit PASSED** (Safety Score: {safety.safety_score * 100:.0f}%) | "
                    f"⚡ **AUTO-APPROVAL TRIGGERED** (RRF Confidence: **{rrf_pct:.1f}%** &gt; 70%)"
                )
            else:
                st.success(f"🛡️ **Compliance Audit PASSED** (Safety Score: {safety.safety_score * 100:.0f}%) | Generator: `{draft.model_provider_used}`")
        else:
            st.error(f"🚨 **Compliance Audit VIOLATIONS DETECTED:** {safety.violations}")

        prev_col1, prev_col2 = st.columns([2, 1])

        with prev_col1:
            st.markdown(f"**Subject:** `{draft.subject_line}`")
            st.markdown(f"**Recipient:** `{draft.recipient_greeting}` (Customer ID: {draft.customer_id})")
            st.text_area("Message Body", value=draft.email_body, height=260, disabled=True)
            st.caption(f"**Mandatory Disclaimer:** {draft.compliance_disclaimer}")

        with prev_col2:
            st.markdown("#### Message Metadata")
            st.markdown(f"**Channel:** `{draft.channel}`")
            st.markdown(f"**Offer Attached:** `{draft.offer_id or 'None'}`")
            st.markdown(f"**Discount Mentioned:** `{draft.discount_mentioned or 'None'}`")
            st.markdown(f"**RRF Retrieval Confidence:** `{rrf_pct:.1f}%`")
            st.markdown(f"**Governance Routing:** `{'⚡ Auto-Approved (>70%)' if auto_approved else '⏳ Pending Review (<=70%)'}`")
            st.markdown(f"**Key Benefits:**")
            for b in draft.key_benefits:
                st.markdown(f"- {b}")
            st.markdown(f"**Call To Action:** {draft.call_to_action}")

            st.markdown("---")
            # Push to Approval Center with Auto-Approval logic
            if auto_approved:
                btn_label = f"⚡ Submit & Auto-Approve (RRF: {rrf_pct:.1f}% > 70%)"
                help_txt = "RRF confidence exceeds 70% and compliance audit passed. Recorded directly as APPROVED."
            else:
                btn_label = "📤 Submit Draft to Human Approval Center"
                help_txt = "Placed into PENDING queue for human verification."

            if st.button(btn_label, type="primary", use_container_width=True, help=help_txt):
                appr_id = f"APPR-{uuid.uuid4().hex[:8].upper()}"
                final_status = "APPROVED" if auto_approved else "PENDING"
                notes = (
                    f"Auto-approved by System: RRF Retrieval Confidence ({rrf_pct:.1f}%) exceeds 70.0% threshold "
                    f"with 0 policy violations."
                    if auto_approved else "Submitted for human reviewer verification."
                )
                ApprovalRepository.create_approval(
                    approval_id=appr_id,
                    customer_id=profile.customer_id,
                    offer_id=draft.offer_id,
                    segment=classification["primary_segment"],
                    subject=draft.subject_line,
                    body=draft.email_body,
                    channel=draft.channel,
                    status=final_status,
                    reviewer_notes=notes,
                    confidence_score=rrf_conf
                )
                from database.repositories import AuditRepository
                AuditRepository.log_action(
                    action_type="APPROVAL" if auto_approved else "GENERATION",
                    customer_id=profile.customer_id,
                    details={
                        "approval_id": appr_id,
                        "auto_approved": auto_approved,
                        "rrf_confidence": rrf_conf,
                        "rrf_confidence_pct": rrf_pct,
                        "threshold": 0.70,
                        "status": final_status
                    },
                    compliance_status="PASSED",
                    user_id="system_auto_approval" if auto_approved else "system_operator"
                )
                if auto_approved:
                    st.success(f"🎉 Communication `{appr_id}` was **AUTO-APPROVED** (RRF: {rrf_pct:.1f}%)!")
                else:
                    st.success(f"Draft `{appr_id}` submitted to Human Approval Center!")
                st.session_state.selected_approval_id = appr_id
                st.session_state.active_page = "Approval Center"
                st.rerun()
