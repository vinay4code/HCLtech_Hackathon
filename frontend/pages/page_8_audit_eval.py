import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.repositories import AuditRepository, ApprovalRepository, BenchmarkRepository
from evaluation.rag_evaluator import RAGEvaluator
from evaluation.email_evaluator import EmailQualityEvaluator
from frontend.components import render_header, render_kpi

def render_audit_eval_page():
    render_header(
        "Audit, Evaluation & Governance",
        "Continuous GenAI evaluation, 4-way RAG retrieval benchmarking, and immutable compliance audit trails."
    )

    tab_eval, tab_audit, tab_benchmarks = st.tabs([
        "📊 Comparative RAG Evaluation",
        "📜 System Audit Trail",
        "🎯 Benchmark Queries"
    ])

    with tab_eval:
        st.markdown("### 1. Comparative 4-Way Retrieval Architecture Benchmark")
        st.caption("Empirical comparison across BM25 Only, Vector Only, Hybrid (RRF Fusion), and Hybrid + Cross-Encoder Reranker.")

        top_k = st.slider("Benchmark top_k parameter", 1, 5, 3)
        if st.button("🚀 Run Live 4-Way Comparative Benchmark", type="primary"):
            with st.spinner("Evaluating BM25, Vector, Hybrid RRF, and Reranking across benchmark test sets..."):
                comp_results = RAGEvaluator.run_comparative_benchmark(top_k=top_k)
                st.session_state.active_comp_results = comp_results

        comp_res = st.session_state.get("active_comp_results")
        if not comp_res:
            comp_res = RAGEvaluator.run_comparative_benchmark(top_k=top_k)

        summary_data = comp_res.get("summary_table", [])
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Comparative Visualizations
            st.markdown("#### Performance Comparison Visualizations")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # MRR and nDCG comparison chart
                approaches = [r["Approach"] for r in summary_data]
                mrr_vals = [float(r["MRR"]) for r in summary_data]
                ndcg_vals = [float(r[f"nDCG@{top_k}"]) for r in summary_data]

                fig1 = go.Figure(data=[
                    go.Bar(name="MRR", x=approaches, y=mrr_vals, marker_color="#2563eb"),
                    go.Bar(name=f"nDCG@{top_k}", x=approaches, y=ndcg_vals, marker_color="#10b981")
                ])
                fig1.update_layout(
                    barmode="group",
                    title="Ranking Quality: MRR & nDCG@K by Approach",
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig1, use_container_width=True)

            with chart_col2:
                # Latency Comparison Chart
                lat_vals = [float(r["Latency (ms)"].replace(" ms", "")) for r in summary_data]
                fig2 = px.bar(
                    x=approaches,
                    y=lat_vals,
                    labels={"x": "", "y": "Latency (ms)"},
                    title="Average Query Latency (ms) by Retrieval Mode",
                    color=approaches,
                    color_discrete_sequence=["#0d9488", "#6366f1", "#f59e0b", "#ec4899"]
                )
                fig2.update_layout(
                    showlegend=False,
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### 2. Email Quality & Safety Governance Metrics")
        gov_res = EmailQualityEvaluator.compute_governance_metrics()

        gc1, gc2, gc3, gc4 = st.columns(4)
        render_kpi(gc1, "Groundedness Score", f"{gov_res['groundedness_score'] * 100:.1f}%", "Fact verification", "success")
        render_kpi(gc2, "Compliance Rate", f"{gov_res['compliance_rate'] * 100:.1f}%", "Policy passed", "success")
        render_kpi(gc3, "Hallucination Rate", f"{gov_res['hallucination_rate'] * 100:.1f}%", "Target < 5%", "normal")
        render_kpi(gc4, "Human Acceptance", f"{gov_res['human_acceptance_rate'] * 100:.1f}%", "Approved / Total", "normal")

    with tab_audit:
        st.markdown("### Immutable System Audit Log")
        st.caption("Every retrieval query, validation check, generation prompt, policy failure, and human decision is recorded.")

        # Filters
        c_act, c_st, c_srch = st.columns([1, 1, 2])
        with c_act:
            action_filter = st.selectbox("Action Type", ["ALL", "RETRIEVAL", "VALIDATION", "GENERATION", "APPROVAL", "REJECTION", "POLICY_FAILURE", "EDIT"])
        with c_st:
            status_filter = st.selectbox("Compliance Status", ["ALL", "PASSED", "FAILED", "WARNING"])
        with c_srch:
            cust_filter = st.text_input("Filter by Customer ID", "").strip()

        logs = AuditRepository.get_recent_logs(limit=250)

        # Apply filtering
        filtered_logs = []
        for l in logs:
            if action_filter != "ALL" and l.get("action_type") != action_filter:
                continue
            if status_filter != "ALL" and l.get("compliance_status") != status_filter:
                continue
            if cust_filter and cust_filter.lower() not in str(l.get("customer_id", "")).lower():
                continue
            filtered_logs.append(l)

        st.markdown(f"**Showing {len(filtered_logs)} log events:**")

        if filtered_logs:
            display_rows = []
            for l in filtered_logs:
                display_rows.append({
                    "Log ID": l["log_id"],
                    "Timestamp": l["timestamp"],
                    "Action": l["action_type"],
                    "Customer": l["customer_id"] or "N/A",
                    "User": l["user_id"],
                    "Compliance": "✅ PASS" if l["compliance_status"] == "PASSED" else "❌ FAIL",
                    "Details": json.dumps(l["details"])[:90] + "..."
                })
            log_df = pd.DataFrame(display_rows)
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            # Export Audit Log
            st.markdown("#### Export Audit Records")
            exp1, exp2 = st.columns(2)
            with exp1:
                csv_data = pd.DataFrame(filtered_logs).to_csv(index=False)
                st.download_button(
                    "📥 Download Audit Trail (CSV)",
                    data=csv_data,
                    file_name="assistant_audit_log.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with exp2:
                json_data = json.dumps(filtered_logs, indent=2)
                st.download_button(
                    "📥 Download Audit Trail (JSON)",
                    data=json_data,
                    file_name="assistant_audit_log.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("No audit logs matching selected filters.")

    with tab_benchmarks:
        st.markdown("### Configured Benchmark Test Pairs")
        st.caption("Predefined evaluation cases mapping representative customer intents to expected offer IDs.")
        benchmarks = BenchmarkRepository.get_all_benchmarks()
        st.dataframe(pd.DataFrame(benchmarks), use_container_width=True, hide_index=True)
