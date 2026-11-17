import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend.profile_builder import ProfileBuilder
from backend.segmentation_engine import SegmentationEngine
from frontend.components import render_header, render_kpi, segment_badge

def render_segmentation_page():
    render_header(
        "Customer Segmentation Engine (Random Forest ML)",
        "Machine Learning-driven multi-class segmentation with tree-based explainability and live-updating evaluation metrics."
    )

    engine = SegmentationEngine.get_instance()
    metrics_info = engine.get_model_metrics(force_reload=True)
    metrics = metrics_info["metrics"]
    importances = metrics_info["feature_importances"]
    classes = metrics_info["classes"]

    # Active Model State Banner
    trained_at = metrics.get("trained_at", "Baseline Initialized")
    train_acc = metrics.get("train_accuracy", metrics["accuracy"])
    test_acc = metrics["accuracy"]
    train_time = metrics.get("train_duration_sec", "< 0.5")
    n_trees = metrics.get("n_estimators", 100)
    max_d = metrics.get("max_depth", 10)
    seed = metrics.get("random_state", 42)

    st.markdown(f"""
    <div style="background: #1e293b; border-left: 4px solid #3b82f6; padding: 0.8rem 1.2rem; border-radius: 6px; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
        <div>
            <span style="color: #60a5fa; font-weight: 600;">Active Model State:</span>
            <span style="color: #f8fafc; margin-left: 0.5rem;">Random Forest (Trees: <b>{n_trees}</b>, Max Depth: <b>{max_d}</b>, Seed: <b>{seed}</b>)</span>
        </div>
        <div style="color: #94a3b8; font-size: 0.85rem;">
            🕒 Last Trained: <b style="color: #38bdf8;">{trained_at}</b> | ⚡ Train Time: <b style="color: #38bdf8;">{train_time}s</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_model, tab_inspect, tab_tune = st.tabs([
        "🧠 Random Forest Model Performance",
        "🔍 Customer Inference & Probabilities",
        "⚙️ Hyperparameter Tuning & Retraining"
    ])

    with tab_model:
        st.markdown("### 1. Model Governance & Evaluation Metrics")
        st.caption("Random Forest ensemble evaluated on Stratified Test Split (20% holdout, zero protected demographic features).")

        m1, m2, m3, m4 = st.columns(4)
        render_kpi(m1, "Holdout Test Accuracy", f"{test_acc * 100:.2f}%", f"Test N={metrics['test_samples']}", "success" if test_acc >= 0.95 else "normal")
        render_kpi(m2, "Training Set Accuracy", f"{train_acc * 100:.2f}%", f"Train N={metrics['train_samples']}", "success")
        render_kpi(m3, "Weighted Precision", f"{metrics['precision_weighted'] * 100:.2f}%", "Macro Balanced", "success")
        render_kpi(m4, "Weighted F1-Score", f"{metrics['f1_weighted'] * 100:.2f}%", "Harmonic Mean", "success")

        st.markdown("---")
        c_imp, c_rep = st.columns([1.2, 1])

        with c_imp:
            st.markdown("#### Global Feature Importances (Gini Impurity Reduction)")
            st.caption("Derived from all decision trees across the active Random Forest ensemble:")
            
            # Top 10 features bar chart
            top_features = list(importances.items())[:10]
            f_names = [f[0] for f in top_features][::-1]
            f_vals = [f[1] * 100 for f in top_features][::-1]

            fig_imp = px.bar(
                x=f_vals,
                y=f_names,
                orientation="h",
                labels={"x": "Feature Importance (%)", "y": "Feature"},
                color=f_vals,
                color_continuous_scale="Blues"
            )
            fig_imp.update_layout(
                height=340,
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with c_rep:
            st.markdown("#### Per-Segment Classification Report")
            report = metrics.get("classification_report", {})
            rows = []
            for cls_name in classes:
                if cls_name in report:
                    r = report[cls_name]
                    rows.append({
                        "Segment": cls_name,
                        "Precision": f"{r.get('precision', 0.0) * 100:.1f}%",
                        "Recall": f"{r.get('recall', 0.0) * 100:.1f}%",
                        "F1-Score": f"{r.get('f1-score', 0.0) * 100:.1f}%",
                        "Support": int(r.get("support", 0))
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_inspect:
        st.markdown("### 2. Live Random Forest Inference Inspector")
        st.caption("Select any customer to evaluate their tree-ensemble voting distribution and local feature contributions.")

        df = ProfileBuilder.get_enriched_dataframe()
        selected_cid = st.selectbox("Select Customer ID:", df["customerID"].head(100).tolist(), index=0)

        profile = ProfileBuilder.build_profile(selected_cid)
        prediction = engine.classify_customer(profile)

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown(f"#### Account: `{profile.customer_id}`")
            st.markdown(f"**Predicted Primary Segment:** {segment_badge(prediction['primary_segment'])}", unsafe_allow_html=True)
            st.markdown(f"**Model Confidence:** `{prediction['confidence_percentage']}`")

            # Progress bar for confidence
            st.progress(prediction["confidence_score"])

            st.markdown(f"**Multi-Segment Memberships:**")
            badges_html = " ".join([segment_badge(s) for s in prediction["segments"]])
            st.markdown(badges_html, unsafe_allow_html=True)

            st.markdown("#### Explainable Reasoning & Tree Drivers:")
            for r in prediction["reasons"]:
                st.markdown(f"- 📌 {r}")

        with col_right:
            st.markdown("#### Calibrated Class Probability Distribution")
            probs = prediction["probabilities"]
            p_segs = list(probs.keys())
            p_vals = [probs[s] * 100 for s in p_segs]

            fig_p = px.bar(
                x=p_vals,
                y=p_segs,
                orientation="h",
                labels={"x": "Class Probability (%)", "y": ""},
                color=p_vals,
                color_continuous_scale="Viridis",
                text=[f"{v:.1f}%" for v in p_vals]
            )
            fig_p.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False
            )
            fig_p.update_traces(textposition="outside")
            st.plotly_chart(fig_p, use_container_width=True)

            st.caption(f"ℹ️ Multi-segment threshold active at **{engine._multi_threshold * 100:.0f}%**. Classes exceeding this threshold are assigned as secondary memberships.")

    with tab_tune:
        st.markdown("### 3. Hyperparameter Tuning & Model Retraining")
        st.caption("Adjust Random Forest architecture hyperparameters and retrain on the complete customer base. Evaluation metrics update immediately.")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            current_n = int(metrics.get("n_estimators", 100))
            current_d = int(metrics.get("max_depth", 10))
            new_n_trees = st.slider("Ensemble Estimators (Number of Trees):", 25, 200, current_n, step=25)
            new_max_depth = st.slider("Maximum Tree Depth (Regularization):", 2, 20, current_d, step=1)
        with col_t2:
            current_th = float(metrics.get("multi_threshold", 0.20))
            current_s = int(metrics.get("random_state", 42))
            new_multi_thresh = st.slider("Multi-Segment Probability Threshold:", 0.05, 0.40, current_th, step=0.05)
            new_seed = st.number_input("Random State Seed:", value=current_s, step=1)

        if st.button("⚡ Retrain Random Forest Ensemble", type="primary", use_container_width=True):
            with st.spinner("Retraining Random Forest model across 7,043 customer records..."):
                new_metrics = engine.train_model(
                    n_estimators=new_n_trees,
                    max_depth=new_max_depth,
                    random_state=new_seed,
                    multi_threshold=new_multi_thresh
                )
            st.success(
                f"🎉 **Retrained Successfully!** Test Accuracy: **{new_metrics['accuracy'] * 100:.2f}%** | "
                f"Train Accuracy: **{new_metrics['train_accuracy'] * 100:.2f}%** | "
                f"Time: **{new_metrics['train_duration_sec']}s**."
            )
            st.rerun()
