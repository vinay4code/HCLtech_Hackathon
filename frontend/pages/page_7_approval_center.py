import uuid
import difflib
import streamlit as st
from datetime import datetime

from backend.profile_builder import ProfileBuilder
from backend.segmentation_engine import SegmentationEngine
from backend.strategy_engine import StrategyEngine
from rag.retriever import OfferRetriever
from backend.offer_validator import OfferValidator
from backend.email_generator import EmailGenerator
from backend.safety_engine import SafetyEngine
from database.repositories import ApprovalRepository, OfferRepository, AuditRepository
from frontend.components import render_header, segment_badge

def render_approval_center_page():
    render_header(
        "Human Approval Center",
        "Enterprise 3-panel review console for verifying customer context, ground-truth offers, and editable drafts."
    )

    # 1. Check if triggered from Customer Explorer
    if st.session_state.get("trigger_pipeline") and st.session_state.get("trigger_customer_id"):
        cid = st.session_state.trigger_customer_id
        st.session_state.trigger_pipeline = False
        
        # Check if an existing approval exists for this customer or create new
        existing = [a for a in ApprovalRepository.get_approvals() if a["customer_id"] == cid and a["reviewer_status"] == "PENDING"]
        if not existing:
            # Auto-generate draft
            prof = ProfileBuilder.build_profile(cid)
            seg_eng = SegmentationEngine()
            cls_res = seg_eng.classify_customer(prof)
            strat = StrategyEngine.select_strategy(prof, cls_res)
            ret = OfferRetriever.retrieve_offers_for_customer(prof, strat)
            ver_off = None
            if ret["status"] == "OFFERS_RETRIEVED" and ret["verified_offers"]:
                v_res = OfferValidator.validate_offer(ret["verified_offers"][0], prof)
                if v_res.is_eligible:
                    ver_off = v_res.verified_offer_details
            
            dft = EmailGenerator.generate_draft(prof, strat, ver_off)
            sft = SafetyEngine.audit_draft(dft, prof, ver_off)
            rrf_conf = float(ret.get("rrf_confidence", 0.0))
            auto_appr = bool(rrf_conf > 0.70 and sft.passed_safety)

            new_id = f"APPR-{uuid.uuid4().hex[:8].upper()}"
            final_st = "APPROVED" if auto_appr else "PENDING"
            notes = (
                f"Auto-approved by System: RRF Retrieval Confidence ({rrf_conf * 100:.1f}%) exceeds 70.0% threshold."
                if auto_appr else "Pending human reviewer verification."
            )
            ApprovalRepository.create_approval(
                approval_id=new_id,
                customer_id=cid,
                offer_id=dft.offer_id,
                segment=cls_res["primary_segment"],
                subject=dft.subject_line,
                body=dft.email_body,
                channel=dft.channel,
                status=final_st,
                reviewer_notes=notes,
                confidence_score=rrf_conf
            )
            st.session_state.selected_approval_id = new_id

    # Filter Approvals List & Batch Actions
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        status_filter = st.selectbox("Filter Queue by Status:", ["PENDING", "APPROVED", "EDITED", "REJECTED", "ALL"])
    with f_col2:
        st.write("")
        st.write("")
        if st.button("⚡ Auto-Approve Eligible (>70% RRF)", use_container_width=True, help="Scans pending queue and auto-approves drafts with retrieval RRF confidence > 70%"):
            batch_cnt = ApprovalRepository.auto_approve_high_confidence(threshold=0.70)
            if batch_cnt > 0:
                st.success(f"Successfully auto-approved {batch_cnt} draft(s) with RRF confidence > 70%!")
                st.rerun()
            else:
                st.info("No pending drafts currently meet the >70% RRF auto-approval criteria.")
    
    approvals_list = ApprovalRepository.get_approvals(status_filter=status_filter)

    if not approvals_list:
        st.info("No approval records found in the selected status queue. Generate a draft from the Email Generation page.")
        return

    # Select Approval Item
    appr_options = {
        f"[{a['reviewer_status']}] {a['approval_id']} - Customer {a['customer_id']} ({a['segment']})": a["approval_id"]
        for a in approvals_list
    }
    
    selected_label = st.selectbox("Select Record to Review:", list(appr_options.keys()))
    appr_id = appr_options[selected_label]
    approval = ApprovalRepository.get_approval_by_id(appr_id)

    if not approval:
        st.error("Selected record could not be loaded.")
        return

    # Load associated Customer Profile & Offer Details
    cid = approval["customer_id"]
    profile = ProfileBuilder.build_profile(cid)
    seg_engine = SegmentationEngine()
    classification = seg_engine.classify_customer(profile)
    strategy = StrategyEngine.select_strategy(profile, classification)

    offer_data = None
    if approval.get("offer_id"):
        offer_data = OfferRepository.get_offer_by_id(approval["offer_id"])

    st.markdown("---")
    st.markdown(f"### Review Console: `{approval['approval_id']}` (Current Status: `{approval['reviewer_status']}`)")

    # 3-PANEL SIDE-BY-SIDE LAYOUT
    p1, p2, p3 = st.columns([1, 1, 1.3])

    # PANEL 1: Customer Profile
    with p1:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-box-header">👤 Panel 1: Customer Profile</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Customer ID:** `{profile.customer_id}`")
        st.markdown(f"**Primary Segment:** {segment_badge(approval['segment'])}", unsafe_allow_html=True)
        st.markdown(f"**Tenure:** {profile.tenure_months} months ({profile.tenure_category})")
        st.markdown(f"**Contract:** `{profile.contract}`")
        st.markdown(f"**Monthly Charges:** `${profile.monthly_charges:.2f}`")
        st.markdown(f"**Total Charges:** `${profile.total_charges:.2f}`")
        st.markdown(f"**Broadband Service:** `{profile.internet_service}`")
        st.markdown(f"**Active Addons Count:** `{profile.subscribed_addons_count} / 6`")

        if profile.service_gaps:
            st.markdown(f"**Service Gaps:** `{', '.join(profile.service_gaps)}`")

        st.markdown("**Simulated Billing Context:**")
        b_color = "red" if profile.simulated_billing_status == "Overdue" else "green"
        st.markdown(f"- Status: <span style='color:{b_color};font-weight:bold;'>{profile.simulated_billing_status}</span>", unsafe_allow_html=True)
        st.markdown(f"- Outstanding: `${profile.simulated_outstanding_balance:.2f}`")

        st.markdown("**Communication Consent:**")
        st.markdown(f"- Email Consent: `{'YES' if profile.simulated_email_consent else 'NO'}`")
        st.markdown(f"- Opt-Out All: `{'YES' if profile.simulated_opt_out_all else 'NO'}`")

    # PANEL 2: Verified Offer & Context
    with p2:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-box-header">🏷️ Panel 2: Verified Offer & RAG</div>
        </div>
        """, unsafe_allow_html=True)

        if offer_data:
            st.markdown(f"**Offer Name:** `{offer_data['offer_name']}`")
            st.markdown(f"**Offer ID:** `{offer_data['offer_id']}`")
            st.markdown(f"**Category:** `{offer_data['offer_category']}`")
            st.markdown(f"**Verified Discount:** `{offer_data['discount_value']} ({offer_data['discount_type']})`")
            st.markdown(f"**Source Document:** `{offer_data['source_document']}` (Page {offer_data['source_page']})")
            st.markdown(f"**Validity Window:** `{offer_data['validity_start']} to {offer_data['validity_end']}`")
            st.markdown(f"**Terms & Conditions:**\n_{offer_data['terms_and_conditions']}_")
        else:
            st.info("No promotional offer attached to this communication (Service check-in / account update).")

        st.markdown("---")
        st.markdown(f"**Campaign Strategy:** `{strategy['strategy_title']}`")
        st.caption(f"Objective: {strategy['objective']}")
        st.caption(f"Rationale: {strategy['rationale']}")

        conf_val = float(approval.get("confidence_score") or 0.0)
        conf_pct = conf_val * 100.0
        st.markdown("---")
        st.markdown("##### 🎯 Retrieval Confidence & Policy")
        if conf_val > 0.70:
            st.markdown(
                f"""
                <div style="background-color: #ECFDF5; border: 1px solid #10B981; border-radius: 6px; padding: 6px 10px; margin-top: 4px;">
                    <span style="font-weight: 700; color: #065F46; font-size: 13px;">RRF Confidence: {conf_pct:.1f}%</span>
                    <span style="background-color: #10B981; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px; font-weight: 600; margin-left: 6px;">⚡ Auto-Approved (&gt;70%)</span>
                    <div style="font-size: 11px; color: #047857; margin-top: 2px;">Meets system auto-approval policy threshold.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif conf_val > 0.0:
            st.markdown(
                f"""
                <div style="background-color: #FFFBEB; border: 1px solid #F59E0B; border-radius: 6px; padding: 6px 10px; margin-top: 4px;">
                    <span style="font-weight: 700; color: #92400E; font-size: 13px;">RRF Confidence: {conf_pct:.1f}%</span>
                    <span style="background-color: #F59E0B; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px; font-weight: 600; margin-left: 6px;">⏳ Manual Review (&le;70%)</span>
                    <div style="font-size: 11px; color: #B45309; margin-top: 2px;">Below 70% threshold. Human sign-off required.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.caption("Standard manual verification workflow.")

    # PANEL 3: Editable Draft & Review Actions
    with p3:
        st.markdown("""
        <div class="panel-box">
            <div class="panel-box-header">✍️ Panel 3: Human Review & Draft Editor</div>
        </div>
        """, unsafe_allow_html=True)

        editable_subject = st.text_input("Subject Line:", value=approval["final_subject"])
        editable_body = st.text_area("Message Body:", value=approval["final_body"], height=240)
        reviewer_notes = st.text_input("Reviewer Notes / Justification:", value=approval.get("reviewer_notes") or "")

        # Check if modified
        is_modified = (editable_subject != approval["original_subject"]) or (editable_body != approval["original_body"])

        if is_modified:
            st.markdown("##### 📝 Text Modification Detected")
            st.caption("Differences between original AI generation and reviewer edits:")
            # Display inline diff summary
            diff_lines = list(difflib.unified_diff(
                approval["original_body"].splitlines(),
                editable_body.splitlines(),
                fromfile="AI Original",
                tofile="Reviewer Modified",
                lineterm=""
            ))
            if diff_lines:
                st.code("\n".join(diff_lines[:15]), language="diff")

        st.markdown("---")
        st.markdown("#### Review Actions")
        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                new_st = "EDITED" if is_modified else "APPROVED"
                ApprovalRepository.update_decision(
                    approval_id=appr_id,
                    status=new_st,
                    final_subject=editable_subject,
                    final_body=editable_body,
                    notes=reviewer_notes or ("Approved with edits" if is_modified else "Approved as drafted"),
                    delivery_status="NOT_SENT"
                )
                AuditRepository.log_action(
                    action_type="APPROVAL" if not is_modified else "EDIT",
                    customer_id=cid,
                    details={"approval_id": appr_id, "is_modified": is_modified, "notes": reviewer_notes},
                    compliance_status="PASSED"
                )
                st.success(f"Record {appr_id} marked as {new_st}!")
                st.rerun()

        with b2:
            if st.button("🚫 Reject", type="secondary", use_container_width=True):
                if not reviewer_notes:
                    st.error("Please provide a rejection reason in the notes field.")
                else:
                    ApprovalRepository.update_decision(
                        approval_id=appr_id,
                        status="REJECTED",
                        final_subject=editable_subject,
                        final_body=editable_body,
                        notes=reviewer_notes,
                        delivery_status="NOT_SENT"
                    )
                    AuditRepository.log_action(
                        action_type="REJECTION",
                        customer_id=cid,
                        details={"approval_id": appr_id, "rejection_reason": reviewer_notes},
                        compliance_status="FAILED"
                    )
                    st.warning(f"Record {appr_id} has been REJECTED.")
                    st.rerun()

        with b3:
            if st.button("⚡ Mock Deliver", use_container_width=True):
                if approval["reviewer_status"] not in ["APPROVED", "EDITED"]:
                    st.error("Only approved drafts can be delivered.")
                else:
                    ApprovalRepository.update_decision(
                        approval_id=appr_id,
                        status=approval["reviewer_status"],
                        delivery_status="SENT_MOCK"
                    )
                    AuditRepository.log_action(
                        action_type="GENERATION",
                        customer_id=cid,
                        details={"approval_id": appr_id, "delivery_event": "SIMULATED_EMAIL_SENT", "recipient_id": cid},
                        compliance_status="PASSED"
                    )
                    st.balloons()
                    st.success(f"Mock email successfully dispatched to customer `{cid}`!")
                    st.rerun()
