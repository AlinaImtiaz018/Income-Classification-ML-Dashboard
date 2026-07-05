import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import (constants as con, 
                   layout as l, 
                   colors as co)


def display():
    st.markdown('<div class="hero-title">Test Set <span class="hero-accent">Predictions</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Gradient Boosting applied to 25,000 unlabeled test instances</div><br>', unsafe_allow_html=True)

    methods = ["KNN", "Iterative", "Dropped"]
    p1, p2, p3 = st.columns(3)
    for col, mth in zip([p1, p2, p3], methods):
        data      = con.PREDICTED_COUNTS[mth]
        total     = data["<=50K"] + data[">50K"]
        pct_high  = data[">50K"] / total
        with col:
            st.markdown(l.make_kpi(f"{mth} Imputation", f"{data['>50K']:,}", f"{pct_high:.1%} predicted >50K"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Predicted class counts — Gradient Boosting Classifier</div>', unsafe_allow_html=True)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Bar(
        name="≤50K", x=methods,
        y=[con.PREDICTED_COUNTS[m]["<=50K"] for m in methods],
        marker_color=co.COLORS["green"], marker_line_width=0,
        text=[con.PREDICTED_COUNTS[m]["<=50K"] for m in methods], textposition="outside",
    ))
    fig_pred.add_trace(go.Bar(
        name=">50K", x=methods,
        y=[con.PREDICTED_COUNTS[m][">50K"] for m in methods],
        marker_color=co.COLORS["red"], marker_line_width=0,
        text=[con.PREDICTED_COUNTS[m][">50K"] for m in methods], textposition="outside",
    ))
    l.apply_layout(fig_pred, height=380, yaxis_kw=dict(gridcolor=co.COLORS["border"]),
                    barmode="group", bargap=0.3)
    st.plotly_chart(fig_pred, width="stretch")

    st.markdown('<div class="section-header">Class share by imputation method</div>', unsafe_allow_html=True)
    fig_stk = go.Figure()
    totals  = {m: con.PREDICTED_COUNTS[m]["<=50K"] + con.PREDICTED_COUNTS[m][">50K"] for m in methods}
    fig_stk.add_trace(go.Bar(
        name="≤50K", x=methods,
        y=[con.PREDICTED_COUNTS[m]["<=50K"] / totals[m] for m in methods],
        marker_color=co.COLORS["teal"], marker_line_width=0,
        text=[f"{con.PREDICTED_COUNTS[m]['<=50K']/totals[m]:.1%}" for m in methods], textposition="inside",
    ))
    fig_stk.add_trace(go.Bar(
        name=">50K", x=methods,
        y=[con.PREDICTED_COUNTS[m][">50K"] / totals[m] for m in methods],
        marker_color=co.COLORS["red"], marker_line_width=0,
        text=[f"{con.PREDICTED_COUNTS[m]['>50K']/totals[m]:.1%}" for m in methods], textposition="inside",
    ))
    l.apply_layout(fig_stk, height=300,
                    yaxis_kw=dict(tickformat=".0%", gridcolor=co.COLORS["border"]),
                    barmode="stack")
    st.plotly_chart(fig_stk, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <strong>Interpretation</strong><br>
    Gradient Boosting consistently predicts ~18–23% of test instances as earning >50K across all imputation
    strategies. The <strong>Iterative</strong> method predicts the most ≤50K cases (20,955), while
    <strong>KNN</strong> and <strong>Dropped</strong> show similar distributions. Imputation method does
    not strongly shift prediction proportions, suggesting robust model behaviour.
    </div>""", unsafe_allow_html=True)