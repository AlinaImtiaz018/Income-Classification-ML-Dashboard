import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.styles import load_css
from utils import (
    constants as c,
    colors as col, 
    layout as l, 
    sidebar as sb,
    data as d
    )


def display():
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div class="hero-title">Income <span class="hero-accent">Classification</span><br>ML Dashboard</div>
        <div class="hero-subtitle">Binary prediction: ≤50K vs &gt;50K annual income &nbsp;·&nbsp; Gradient Boosting best performer</div>
    </div>""", unsafe_allow_html=True)

    if d.DATA_LOADED:
        labeled    = d.df[d.df["Income"].isin(["<=50K", ">50K"])]
        n_total    = len(d.df)
        n_labeled  = len(labeled)
        cls_counts = labeled["Income"].value_counts()
        n_low  = int(cls_counts.get("<=50K", 3779))
        n_high = int(cls_counts.get(">50K",  1221))
    else:
        n_total, n_labeled, n_low, n_high = 30000, 5000, 3779, 1221

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(l.make_kpi("Total Instances", f"{n_total:,}", "Full dataset size"), unsafe_allow_html=True)
    with k2: st.markdown(l.make_kpi("Labeled Rows",    f"{n_labeled:,}", "Used for training"), unsafe_allow_html=True)
    with k3: st.markdown(l.make_kpi("Best Accuracy",   "87.1%", "Gradient Boosting · Dropped"), unsafe_allow_html=True)
    with k4: st.markdown(l.make_kpi("Best ROC AUC",    "0.784", "Gradient Boosting · Dropped"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-eyebrow">Model Performance</div>'
                    '<div class="section-header">Accuracy across all configurations</div>', unsafe_allow_html=True)
        methods = ["KNN", "Iterative", "Dropped"]
        fig = go.Figure()
        for i, m_full in enumerate(c.MODEL_NAMES_FULL):
            accs = [c.RESULTS[mth][m_full]["accuracy"] for mth in methods]
            fig.add_trace(go.Bar(
                name=c.MODEL_NAMES_SHORT[i], x=methods, y=accs,
                marker_color=col.MODEL_COLORS[i], marker_line_width=0,
                text=[f"{v:.1%}" for v in accs], textposition="outside", textfont=dict(size=10),
                offsetgroup=i, width=0.18,
            ))
        l.apply_layout(fig, height=360,
                        yaxis_kw=dict(range=[0.6, 0.97], tickformat=".0%", gridcolor=col.COLORS["border"]),
                        barmode="group", bargap=0.2, bargroupgap=0.05)
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown('<div class="section-eyebrow">Dataset</div>'
                    '<div class="section-header">Class distribution (labeled)</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["≤50K", ">50K"], values=[n_low, n_high],
            hole=0.62,
            marker=dict(colors=[col.COLORS["teal"], col.COLORS["red"]], line=dict(color=col.COLORS["bg"], width=3)),
            textinfo="percent+label", textfont=dict(size=12, color=col.COLORS["text"]),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"{n_labeled:,}<br><span style='font-size:10px'>labeled</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=col.COLORS["text"]), align="center",
        )
        l.apply_layout(fig2, height=360, legend_kw=dict(orientation="h", y=-0.05), showlegend=True)
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-header">Project Summary</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="info-box"><strong>Problem Setting</strong><br>
        Binary classification to predict whether a person earns <strong>≤50K</strong> or <strong>>50K</strong> per year
        using 14 demographic and employment features.</div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="info-box"><strong>Missing Data Strategy</strong><br>
        Three approaches compared: <strong>KNN Imputation</strong>, <strong>Iterative Imputation</strong>,
        and <strong>Dropping</strong> rows with missing categorical values.</div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="info-box"><strong>Validation Strategy</strong><br>
        <strong>10-fold cross-validation</strong> on labeled training data. The test set (25K rows) has no income
        labels, so all reported metrics come from the validation folds.</div>""", unsafe_allow_html=True)
