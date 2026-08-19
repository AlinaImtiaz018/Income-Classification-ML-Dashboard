import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import (
    constants as c, 
    layout as s, 
    sidebar as sb, 
    colors as co)

def display():
    st.markdown('<div class="hero-title">Model <span class="hero-accent">Comparison</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Cross-imputation performance &nbsp;·&nbsp; 10-fold cross-validation</div><br>', unsafe_allow_html=True)

    metric_opt = st.selectbox(
        "Compare metric", ["accuracy", "f1_score", "recall", "precision", "roc_auc"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    methods = ["KNN", "Iterative", "Dropped"]
    fig_cmp = go.Figure()
    for i, m_full in enumerate(c.MODEL_NAMES_FULL):
        vals = [c.RESULTS[mth][m_full][metric_opt] for mth in methods]
        fig_cmp.add_trace(go.Bar(
            name=c.MODEL_NAMES_SHORT[i], x=methods, y=vals,
            marker_color=co.MODEL_COLORS[i], marker_line_width=0,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            offsetgroup=i, width=0.18,
        ))
    s.apply_layout(fig_cmp, title=f"{metric_opt.replace('_',' ').title()} by imputation method", height=400,
                    yaxis_kw=dict(gridcolor=co.COLORS["border"],
                                range=[0, max(c.RESULTS[m][n][metric_opt] for m in methods for n in c.MODEL_NAMES_FULL) * 1.2]),
                    barmode="group", bargap=0.2, bargroupgap=0.05)
    st.plotly_chart(fig_cmp, width="stretch")

    # ── Radar ──
    st.markdown('<div class="section-header">Multi-metric radar — Gradient Boosting across imputation methods</div>', unsafe_allow_html=True)
    radar_metrics = ["accuracy", "f1_score", "recall", "precision", "roc_auc"]
    radar_labels  = ["Accuracy", "F1-Score", "Recall", "Precision", "ROC AUC"]
    best_model    = "Gradient Boosting Classification"
    fig_radar = go.Figure()
    for i, mth in enumerate(methods):
        vals   = [c.RESULTS[mth][best_model][m] for m in radar_metrics] + [c.RESULTS[mth][best_model][radar_metrics[0]]]
        r, g, b = int(co.MODEL_COLORS[i][1:3],16), int(co.MODEL_COLORS[i][3:5],16), int(co.MODEL_COLORS[i][5:],16)
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels + [radar_labels[0]],
            name=mth, fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            line=dict(color=co.MODEL_COLORS[i], width=2),
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor=co.COLORS["border"],
                            tickfont=dict(size=9, color=co.COLORS["muted"])),
            angularaxis=dict(gridcolor=co.COLORS["border"], tickfont=dict(size=11, color=co.COLORS["text"])),
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=420,
        font=dict(family="Inter", color=co.COLORS["muted"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=50, r=50, t=40, b=30),
    )
    st.plotly_chart(fig_radar, width="stretch")

    # ── Heatmap ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Performance heatmap — all models × all methods</div>', unsafe_allow_html=True)
    heat_metric = st.selectbox("Heatmap metric", ["accuracy", "f1_score", "roc_auc"],
                                format_func=lambda x: x.replace("_"," ").title(), key="heat")
    z_vals   = [[c.RESULTS[mth][m][heat_metric] for mth in methods] for m in c.MODEL_NAMES_FULL]
    y_labels = [n.replace(" Classification","").replace(" (SVMs)","") for n in c.MODEL_NAMES_FULL]
    fig_heat = go.Figure(go.Heatmap(
        z=z_vals, x=methods, y=y_labels,
        colorscale=[[0, co.COLORS["card"]], [0.5, "#3d1520"], [1, co.COLORS["primary"]]],
        text=[[f"{v:.3f}" for v in row] for row in z_vals],
        texttemplate="%{text}", textfont=dict(size=13),
        xgap=4, ygap=4,
    ))
    s.apply_layout(fig_heat, height=300,
                    xaxis_kw=dict(gridcolor="rgba(0,0,0,0)"),
                    yaxis_kw=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_heat, width="stretch")
