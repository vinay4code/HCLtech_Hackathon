import streamlit as st
import pandas as pd
from backend.profile_builder import ProfileBuilder
from backend.segmentation_engine import SegmentationEngine
from database.repositories import ApprovalRepository
from frontend.components import render_header, render_kpi, create_pie_chart, create_bar_chart

def render_dashboard_page():
    render_header(
        "Executive Dashboard",
        "Strategic overview of customer segments, contract commitments, and AI campaign approvals."
    )

    # 1. Ground-Truth Data & Segments
    df = ProfileBuilder.get_enriched_dataframe()
    total_customers = len(df)
    active_customers = int(df["has_active_service"].sum())
    active_pct = round((active_customers / total_customers) * 100, 1)

    # Segment counts from SegmentationEngine
    segment_counts = SegmentationEngine.generate_ground_truth_labels(df).value_counts()
    loyal_count = int(segment_counts.get("LONG_TERM_LOYAL", 0))
    loyal_pct = round((loyal_count / total_customers) * 100, 1)

    # Approvals queue metrics
    approvals = ApprovalRepository.get_approvals()
    pending_approvals = [a for a in approvals if a.get("reviewer_status") == "PENDING"]
    approved_count = sum(1 for a in approvals if a.get("reviewer_status") == "APPROVED")
    rejected_count = sum(1 for a in approvals if a.get("reviewer_status") == "REJECTED")
    edited_count = sum(1 for a in approvals if a.get("reviewer_status") == "EDITED")
    pending_count = len(pending_approvals)

    # 2. Executive KPI Summary (4 Core Cards Only)
    c1, c2, c3, c4 = st.columns(4)
    render_kpi(c1, "Total Customers", f"{total_customers:,}", "Telecom Dataset", "normal")
    render_kpi(c2, "Active Subscribers", f"{active_customers:,}", f"{active_pct}% Active Base", "success")
    render_kpi(c3, "Loyalty Candidates", f"{loyal_count:,}", f"{loyal_pct}% High-Value", "success")
    render_kpi(c4, "Pending Approvals", f"{pending_count}", "Awaiting Review" if pending_count > 0 else "Queue Cleared", "danger" if pending_count > 0 else "success")

    st.markdown("---")

    # 3. Strategic Visualizations (3 Essential Charts in 3 Columns)
    v1, v2, v3 = st.columns(3)

    with v1:
        # Segment Distribution Donut Chart
        seg_display_map = {
            "LONG_TERM_LOYAL": "Long-Term Loyal",
            "CONTRACT_RENEWAL_CANDIDATE": "Contract Renewal",
            "EARLY_STAGE_CUSTOMER": "Early-Stage",
            "CONSISTENT_SUBSCRIBER": "Consistent Sub",
            "UPSELL_OPPORTUNITY": "Upsell Opportunity",
            "STANDARD_SUBSCRIBER": "Standard Sub"
        }
        labels = [seg_display_map.get(k, k) for k in segment_counts.index]
        values = segment_counts.values.tolist()
        st.plotly_chart(create_pie_chart(labels, values, "Customer Portfolio Segments"), use_container_width=True)

    with v2:
        # Contract Commitment Breakdown
        contracts = df["Contract"].value_counts()
        st.plotly_chart(create_bar_chart(contracts.index.tolist(), contracts.values.tolist(), "Contract Commitment & Churn Risk"), use_container_width=True)

    with v3:
        # Approval Decisions Funnel
        if approvals:
            appr_labels = ["Approved", "Pending", "Rejected", "Human Edited"]
            appr_vals = [approved_count, pending_count, rejected_count, edited_count]
        else:
            appr_labels = ["No Drafts Yet"]
            appr_vals = [1]
        st.plotly_chart(create_pie_chart(appr_labels, appr_vals, "AI Campaign Approval Status"), use_container_width=True)

    # 4. Executive Action Queue Preview
    st.markdown("---")
    st.markdown('<div class="section-title">⚡ Pending Approvals & Quick Actions</div>', unsafe_allow_html=True)

    if pending_count > 0:
        st.info(f"🔔 **{pending_count} customer communication draft(s) require review** before sending.")
        
        # Compact table preview
        preview_data = []
        for a in pending_approvals[:5]:
            preview_data.append({
                "Approval ID": a.get("approval_id"),
                "Customer ID": a.get("customer_id"),
                "Target Segment": a.get("segment", "").replace("_", " "),
                "Channel": a.get("channel", "EMAIL"),
                "Subject Line": a.get("original_subject", "")[:45] + "...",
                "Date Created": a.get("created_at", "")[:16]
            })
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)

        col_act, _ = st.columns([1, 2])
        with col_act:
            if st.button("🚀 Open Approval Center", type="primary", use_container_width=True):
                st.session_state.active_page = "Approval Center"
                st.rerun()
    else:
        st.success("✅ **All customer engagement drafts have been reviewed and processed.** New drafts can be generated from the Email Generation page.")

    st.caption("ℹ️ Core demographic, tenure, and contract values are sourced from `Telecom_dataset.csv`. Review and delivery decisions are persisted in SQLite `assistant.db`.")
