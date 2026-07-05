import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import (
    constants as c, 
    layout as l, 
    sidebar as sb, 
    colors as co,
    data as d
)

from utils.styles import load_css


def display(imputation, model_choice):

    metrics = c.RESULTS[imputation][model_choice]

    st.markdown('<div class="hero-title">Model <span class="hero-accent">Results</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-subtitle">Imputation: <strong style="color:#F8FAFC;">{imputation}</strong>'
        f' &nbsp;·&nbsp; Model: <strong style="color:#F8FAFC;">{model_choice}</strong></div><br>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        ("Accuracy",  f"{metrics['accuracy']:.1%}",  ""),
        ("F1-Score",  f"{metrics['f1_score']:.3f}",  "For >50K class"),
        ("Recall",    f"{metrics['recall']:.3f}",    "Sensitivity"),
        ("Precision", f"{metrics['precision']:.3f}", "Pos. predictive value"),
        ("ROC AUC",   f"{metrics['roc_auc']:.3f}",   "Discrimination ability"),
    ]
    for col, (label, val, sub) in zip([k1, k2, k3, k4, k5], cards):
        with col:
            st.markdown(l.make_kpi(label, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_bar = st.columns(2)

    with col_cm:
        st.markdown('<div class="section-eyebrow">Evaluation</div>'
                    '<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = metrics["cm"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        text_vals = [
            [f"TN<br>{tn}<br>({tn/total:.1%})", f"FP<br>{fp}<br>({fp/total:.1%})"],
            [f"FN<br>{fn}<br>({fn/total:.1%})", f"TP<br>{tp}<br>({tp/total:.1%})"],
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]], text=text_vals, texttemplate="%{text}",
            colorscale=[[0, co.COLORS["card"]], [0.5, "#3d1520"], [1, co.COLORS["primary"]]],
            showscale=False, xgap=3, ygap=3,
            hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_traces(textfont=dict(size=12, color="white"))
        l.apply_layout(fig_cm, height=320,
                        xaxis_kw=dict(tickvals=[0,1], ticktext=["≤50K",">50K"],
                                    title="Predicted", gridcolor="rgba(0,0,0,0)"),
                        yaxis_kw=dict(tickvals=[0,1], ticktext=["≤50K",">50K"],
                                    title="Actual", gridcolor="rgba(0,0,0,0)", autorange="reversed"))
        st.plotly_chart(fig_cm, width="stretch")

    with col_bar:
        st.markdown('<div class="section-eyebrow">Evaluation</div>'
                    '<div class="section-header">All metrics at a glance</div>', unsafe_allow_html=True)
        metric_names = ["Accuracy", "F1-Score", "Recall", "Precision", "ROC AUC"]
        metric_vals  = [metrics["accuracy"], metrics["f1_score"], metrics["recall"],
                        metrics["precision"], metrics["roc_auc"]]
        bar_colors   = [co.COLORS["primary"] if v >= 0.7 else co.COLORS["amber"] if v >= 0.5 else co.COLORS["red"]
                        for v in metric_vals]
        fig_bar = go.Figure(go.Bar(
            x=metric_names, y=metric_vals,
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in metric_vals], textposition="outside",
        ))
        l.apply_layout(fig_bar, height=320,
                        yaxis_kw=dict(range=[0, 1.18], gridcolor=co.COLORS["border"], tickformat=".0%"))
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-eyebrow">All Models</div>'
                f'<div class="section-header">Full comparison — {imputation} imputation</div>', unsafe_allow_html=True)
    rows = []
    for mname in c.MODEL_NAMES_FULL:
        m = c.RESULTS[imputation][mname]
        is_best = mname == "Gradient Boosting Classification"
        rows.append({
            "Model":     mname.replace(" Classification","").replace(" (SVMs)","") + (" ★" if is_best else ""),
            "Accuracy":  f"{m['accuracy']:.1%}",
            "F1-Score":  f"{m['f1_score']:.3f}",
            "Recall":    f"{m['recall']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "ROC AUC":   f"{m['roc_auc']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)