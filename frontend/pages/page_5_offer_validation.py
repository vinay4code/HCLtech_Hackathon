import streamlit as st
import json
import pandas as pd
from database.repositories import OfferRepository
from backend.profile_builder import ProfileBuilder
from backend.offer_validator import OfferValidator
from frontend.components import render_header

def render_offer_validation_page():
    render_header(
        "Offer Eligibility & Validation Engine",
        "Deterministic SQLite offer registry management and strict rule-based verification."
    )

    tab_registry, tab_test, tab_add = st.tabs([
        "📋 Offer Registry (SQLite)",
        "🧪 Test Deterministic Validation",
        "➕ Add New Promotional Offer"
    ])

    with tab_registry:
        st.markdown("### Approved Promotional Offers Registry")
        offers = OfferRepository.get_all_offers(active_only=False)

        if not offers:
            st.warning("No offers found in SQLite database.")
        else:
            table_rows = []
            for o in offers:
                table_rows.append({
                    "Offer ID": o["offer_id"],
                    "Name": o["offer_name"],
                    "Category": o["offer_category"],
                    "Discount Type": o["discount_type"],
                    "Discount Value": f"{o['discount_value']}%" if o["discount_type"] == "PERCENTAGE" else f"${o['discount_value']}",
                    "Min Tenure": f"{o['min_tenure']}m",
                    "Contracts": ", ".join(o["eligible_contract_types"]),
                    "Target Segment": o["target_segment"],
                    "Active": "✅ Active" if o["is_active"] else "❌ Inactive",
                    "Validity": f"{o['validity_start']} to {o['validity_end']}"
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # Quick status toggle
            st.markdown("#### Toggle Offer Active Status")
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                selected_offer_id = st.selectbox("Select Offer ID:", [o["offer_id"] for o in offers])
            with col_t2:
                current_status = next(o["is_active"] for o in offers if o["offer_id"] == selected_offer_id)
                new_status = st.checkbox("Active Status", value=bool(current_status))
                if st.button("Update Status"):
                    OfferRepository.update_offer_status(selected_offer_id, new_status)
                    st.success(f"Status for `{selected_offer_id}` updated.")
                    st.rerun()

    with tab_test:
        st.markdown("### Run Deterministic Eligibility Check")
        st.caption("Verify customer attributes against offer constraints before any LLM generation occurs.")

        df = ProfileBuilder.get_enriched_dataframe()
        c_cust, c_off = st.columns(2)

        with c_cust:
            test_cid = st.selectbox("Select Customer ID to test:", df["customerID"].head(50).tolist())
            profile = ProfileBuilder.build_profile(test_cid)
            st.markdown(f"**Customer Tenure:** {profile.tenure_months} months")
            st.markdown(f"**Contract:** `{profile.contract}`")
            st.markdown(f"**Internet Service:** `{profile.internet_service}`")
            st.markdown(f"**Service Gaps:** `{', '.join(profile.service_gaps) if profile.service_gaps else 'None'}`")

        with c_off:
            offers_list = OfferRepository.get_all_offers(active_only=False)
            offer_map = {f"{o['offer_id']} - {o['offer_name']}": o for o in offers_list}
            selected_choice = st.selectbox("Select Offer to evaluate:", list(offer_map.keys()))
            chosen_offer = offer_map[selected_choice]
            st.markdown(f"**Required Min Tenure:** {chosen_offer['min_tenure']} months")
            st.markdown(f"**Eligible Contracts:** `{chosen_offer['eligible_contract_types']}`")
            st.markdown(f"**Discount:** `{chosen_offer['discount_value']} ({chosen_offer['discount_type']})`")
            st.markdown(f"**Active Status:** `{'Active' if chosen_offer['is_active'] else 'Inactive'}`")

        if st.button("⚡ Run Deterministic Verification", type="primary", use_container_width=True):
            result = OfferValidator.validate_offer(chosen_offer, profile)

            if result.is_eligible:
                st.success(f"🎉 **ELIGIBILITY PASSED:** Customer `{profile.customer_id}` is 100% verified and eligible for `{chosen_offer['offer_name']}`.")
            else:
                st.error(f"🚫 **ELIGIBILITY FAILED:** Offer rejected by business policy. It will NOT be passed to LLM.")

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("#### ✅ Passed Rules")
                for p in result.passed_checks:
                    st.markdown(f"- {p}")

            with r2:
                st.markdown("#### ❌ Failed Rules")
                if not result.failed_checks:
                    st.markdown("- None. All conditions met.")
                for f in result.failed_checks:
                    st.markdown(f"- {f}")

            st.json(result.model_dump())

    with tab_add:
        st.markdown("### Register New Approved Offer in SQLite")
        with st.form("add_offer_form"):
            new_id = st.text_input("Offer ID (e.g. OFFER-STREAM-BOOST)", "").strip().upper()
            new_name = st.text_input("Offer Name", "")
            new_cat = st.selectbox("Category", ["UPGRADE", "LOYALTY", "RENEWAL", "SECURITY", "ONBOARDING", "UPSELL"])
            new_type = st.selectbox("Discount Type", ["PERCENTAGE", "FIXED_AMOUNT", "MONTHLY_CREDIT"])
            new_val = st.number_input("Discount Value (e.g. 20 for 20% or 15 for $15)", min_value=1.0, max_value=50.0, value=15.0)
            new_min_tenure = st.number_input("Minimum Tenure (Months)", min_value=0, max_value=72, value=12)
            new_contracts = st.multiselect("Eligible Contract Types", ["Month-to-month", "One year", "Two year"], default=["Month-to-month"])
            new_internet = st.multiselect("Eligible Internet Services", ["DSL", "Fiber optic", "No"], default=["DSL", "Fiber optic"])
            new_target_seg = st.selectbox("Target Segment", ["UPSELL_OPPORTUNITY", "LONG_TERM_LOYAL", "CONTRACT_RENEWAL_CANDIDATE", "EARLY_STAGE_CUSTOMER", "CONSISTENT_SUBSCRIBER", "ALL"])
            new_terms = st.text_area("Terms and Conditions", "Promotional rate valid for 12 months. Requires active account in good standing.")
            submitted = st.form_submit_button("Register Offer in Database")

            if submitted:
                if not new_id or not new_name:
                    st.error("Offer ID and Name are required.")
                else:
                    offer_payload = {
                        "offer_id": new_id,
                        "offer_name": new_name,
                        "offer_category": new_cat,
                        "discount_type": new_type,
                        "discount_value": new_val,
                        "min_tenure": int(new_min_tenure),
                        "eligible_contract_types": new_contracts,
                        "eligible_internet_services": new_internet,
                        "required_missing_services": [],
                        "target_segment": new_target_seg,
                        "terms_and_conditions": new_terms,
                        "source_document": "manual_admin_entry",
                        "source_page": 1,
                        "is_active": 1
                    }
                    OfferRepository.add_offer(offer_payload)
                    st.success(f"Offer `{new_id}` successfully registered!")
                    st.rerun()
