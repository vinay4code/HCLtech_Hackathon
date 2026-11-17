import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

def render_header(title: str, subtitle: str, badge: str = "HCLTech AI Hackathon"):
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div>
                <span class="badge badge-primary">{badge}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi(col, title: str, value: Any, delta: Optional[str] = None, delta_color: str = "normal"):
    delta_html = ""
    if delta:
        color = "#16a34a" if delta_color == "success" else ("#dc2626" if delta_color == "danger" else "#64748b")
        delta_html = f'<div class="metric-delta" style="color: {color};">{delta}</div>'

    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def segment_badge(segment_name: str) -> str:
    color_map = {
        "LONG_TERM_LOYAL": "badge-success",
        "CONSISTENT_SUBSCRIBER": "badge-primary",
        "UPSELL_OPPORTUNITY": "badge-warning",
        "EARLY_STAGE_CUSTOMER": "badge-neutral",
        "CONTRACT_RENEWAL_CANDIDATE": "badge-danger",
        "STANDARD_SUBSCRIBER": "badge-neutral"
    }
    css_class = color_map.get(segment_name, "badge-primary")
    clean_name = segment_name.replace("_", " ")
    return f'<span class="badge {css_class}">{clean_name}</span>'

def create_pie_chart(labels: List[str], values: List[int], title: str, hole: float = 0.45):
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=hole,
        marker=dict(colors=["#2563eb", "#0d9488", "#f59e0b", "#8b5cf6", "#ec4899", "#64748b"]),
        textinfo="percent+label",
        hoverinfo="label+value+percent"
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e293b")),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        height=320
    )
    return fig

def create_bar_chart(x: List[str], y: List[Any], title: str, color_seq=None):
    fig = px.bar(
        x=x, y=y,
        labels={"x": "", "y": "Customers"},
        color=x,
        color_discrete_sequence=color_seq or ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e293b")),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        height=320
    )
    return fig
