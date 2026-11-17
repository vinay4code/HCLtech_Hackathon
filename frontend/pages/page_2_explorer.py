import streamlit as st
import pandas as pd
from backend.profile_builder import ProfileBuilder
from backend.segmentation_engine import SegmentationEngine
from backend.strategy_engine import StrategyEngine
from rag.retriever import OfferRetriever
from backend.offer_validator import OfferValidator
from frontend.components import render_header, segment_badge

def render_explorer_page():
    render_header(
        "Customer Explorer",
        "Inspect customer profiles, query service adoptions, and launch AI recommendations."
    )

    df = ProfileBuilder.get_enriched_dataframe()
    seg_engine = SegmentationEngine()

    # Filters Section in an expander or container
    with st.expander("🔍 Filter & Search Criteria", expanded=True):
        f1, f2, f3 = st.columns(3)
        
        with f1:
            search_id = st.text_input("Customer ID Search", placeholder="e.g. 7590-VHVEG").strip()
            
        with f2:
            contract_options = ["ALL"] + sorted(df["Contract"].unique().tolist())
            selected_contract = st.selectbox("Contract Type", contract_options)

        with f3:
            internet_options = ["ALL"] + sorted(df["InternetService"].unique().tolist())
            selected_internet = st.selectbox("Internet Service", internet_options)

        f4, f5 = st.columns(2)
        with f4:
            tenure_range = st.slider("Tenure Range (Months)", 0, int(df["tenure"].max()), (0, 72))
        with f5:
            charges_range = st.slider("Monthly Charges ($)", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()), (18.0, 120.0))

    # Apply filters
    filtered_df = df.copy()
    if search_id:
        filtered_df = filtered_df[filtered_df["customerID"].str.contains(search_id, case=False, na=False)]
    if selected_contract != "ALL":
        filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]
    if selected_internet != "ALL":
        filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]
    filtered_df = filtered_df[(filtered_df["tenure"] >= tenure_range[0]) & (filtered_df["tenure"] <= tenure_range[1])]
    filtered_df = filtered_df[(filtered_df["MonthlyCharges"] >= charges_range[0]) & (filtered_df["MonthlyCharges"] <= charges_range[1])]

    st.markdown(f"**Found {len(filtered_df):,} matching customers** (showing first 100 below):")

    # Display Customer Table
    display_cols = [
        "customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges",
        "InternetService", "PhoneService", "subscribed_addons_count"
    ]
    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown('<div class="section-title">👤 Detailed Customer Profile & AI Recommendation Trigger</div>', unsafe_allow_html=True)

    # Customer Selection
    available_cids = filtered_df["customerID"].head(100).tolist()
    if not available_cids:
        st.warning("No customers found matching the filter criteria.")
        return

    # Use session state to retain selection
    default_index = 0
    if "selected_customer_id" in st.session_state and st.session_state.selected_customer_id in available_cids:
        default_index = available_cids.index(st.session_state.selected_customer_id)

    selected_cid = st.selectbox("Select Customer to Inspect & Analyze:", available_cids, index=default_index)
    st.session_state.selected_customer_id = selected_cid

    # Build customer profile and run explainable classification
    profile = ProfileBuilder.build_profile(selected_cid)
    classification = seg_engine.classify_customer(profile)
    strategy = StrategyEngine.select_strategy(profile, classification)

    # Display Card Layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"### Account: `{profile.customer_id}`")
        st.markdown(f"**Tenure:** {profile.tenure_months} months ({profile.tenure_category})")
        st.markdown(f"**Contract:** `{profile.contract}`")
        st.markdown(f"**Monthly Charges:** `${profile.monthly_charges:.2f}` | **Total Charges:** `${profile.total_charges:.2f}`")
        st.markdown(f"**Internet Service:** `{profile.internet_service}` | **Phone Service:** `{profile.phone_service}`")
        st.markdown(f"**Paperless Billing:** `{profile.paperless_billing}` | **Payment Method:** `{profile.payment_method}`")

        st.markdown("#### Optional Service Subscriptions:")
        srv_cols = st.columns(3)
        for idx, (srv, status) in enumerate(profile.subscribed_services.items()):
            srv_cols[idx % 3].metric(label=srv, value="Active" if status.lower() == "yes" else "No")

    with col_right:
        st.markdown("### AI Segmentation & Reason Codes")
        st.markdown(f"**Primary Segment:** {segment_badge(classification['primary_segment'])}", unsafe_allow_html=True)
        
        st.markdown("**Assigned Segments:**")
        badges_html = " ".join([segment_badge(s) for s in classification["segments"]])
        st.markdown(badges_html, unsafe_allow_html=True)

        st.markdown("**Explainable Reason Codes:**")
        for r in classification["reasons"]:
            st.markdown(f"- 📌 {r}")

        st.markdown("---")
        st.markdown("#### Recommended Engagement Objective:")
        st.info(f"**{strategy['strategy_title']}**\n\n_{strategy['objective']}_")

        # The Trigger Button
        if st.button("🚀 Generate Engagement Recommendation", type="primary", use_container_width=True):
            st.session_state.active_page = "Approval Center"
            st.session_state.trigger_customer_id = selected_cid
            st.session_state.trigger_pipeline = True
            st.success(f"Pipeline initiated for customer {selected_cid}! Navigating to Approval Center...")
            st.rerun()
