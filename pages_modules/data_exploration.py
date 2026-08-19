import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    constants as con, 
    layout as l, 
    sidebar as sb, 
    data as d, 
    colors as co)

def display():
    st.markdown('<div class="hero-title">Data <span class="hero-accent">Exploration</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Dataset structure, missing values, and feature distributions</div><br>', unsafe_allow_html=True)

    if not d.DATA_LOADED:
        st.warning("Place `einkommen.train` inside a `data/` folder next to `app.py` to enable live EDA.")
        st.stop()

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown('<div class="section-eyebrow">Schema</div>'
                    '<div class="section-header">Feature overview</div>', unsafe_allow_html=True)
        miss = d.df.isna().sum()
        schema = pd.DataFrame({
            "Feature": con.COLUMNS,
            "Type":    [str(d.df[c].dtype) for c in con.COLUMNS],
            "Missing": [f"{miss[c]:,}" if miss[c] > 0 else "—" for c in con.COLUMNS],
            "Unique":  [str(d.df[c].nunique()) for c in con.COLUMNS],
        })
        st.dataframe(schema, width="stretch", hide_index=True)

    with col_b:
        st.markdown('<div class="section-eyebrow">Missing Values</div>'
                    '<div class="section-header">Columns with nulls</div>', unsafe_allow_html=True)
        miss_df  = miss[miss > 0].sort_values(ascending=True)
        miss_pcts = (miss_df / len(d.df)).tolist()
        fig_miss = go.Figure(go.Bar(
            x=miss_df.values, y=miss_df.index.tolist(),
            orientation="h",
            marker=dict(color=[co.COLORS["primary"], co.COLORS["secondary"], co.COLORS["amber"], co.COLORS["red"]][:len(miss_df)]),
            text=[f"{v:,}  ({p:.1%})" for v, p in zip(miss_df.values, miss_pcts)],
            textposition="outside", textfont=dict(size=10),
        ))
        l.apply_layout(fig_miss, height=320,
                        xaxis_kw=dict(title="Missing count", gridcolor=co.COLORS["border"]),
                        yaxis_kw=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_miss, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Numerical distributions ──
    st.markdown('<div class="section-eyebrow">EDA</div>'
                '<div class="section-header">Numerical feature distributions</div>', unsafe_allow_html=True)
    fig_num = make_subplots(rows=2, cols=3, subplot_titles=con.NUM_FEATURES,
                                vertical_spacing=0.16, horizontal_spacing=0.08)
    for idx, feat in enumerate(con.NUM_FEATURES):
        r, c = idx // 3 + 1, idx % 3 + 1
        series = d.df[feat].dropna().astype(float)
        mu     = series.mean()
        counts, bin_edges = np.histogram(series, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig_num.add_trace(go.Scatter(
            x=bin_centers, y=counts / counts.max(), fill="tozeroy",
            fillcolor="rgba(0,179,179,0.12)",
            line=dict(color=co.COLORS["teal"], width=1.5),
            showlegend=False,
        ), row=r, col=c)
        fig_num.add_vline(x=mu, line_dash="dot", line_color=co.COLORS["red"], line_width=1.5, row=r, col=c)

    fig_num.update_layout(height=440, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", color=co.COLORS["muted"], size=10),
                            margin=dict(l=10, r=10, t=40, b=10))
    fig_num.update_xaxes(gridcolor=co.COLORS["border"], linecolor=co.COLORS["border"])
    fig_num.update_yaxes(gridcolor=co.COLORS["border"], linecolor=co.COLORS["border"], showticklabels=False)
    st.plotly_chart(fig_num, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Categorical distributions ──
    st.markdown('<div class="section-eyebrow">Categorical</div>'
                '<div class="section-header">Key categorical feature distributions</div>', unsafe_allow_html=True)
    cat_choice  = st.selectbox("Select feature", con.CAT_FEATURES, index=con.CAT_FEATURES.index("Gender"))
    cat_counts  = d.df[cat_choice].value_counts().head(15)
    fig_cat = go.Figure(go.Bar(
        x=cat_counts.index.tolist(), y=cat_counts.values,
        marker_color=co.COLORS["primary"], marker_line_width=0,
        text=cat_counts.values, textposition="outside",
    ))
    l.apply_layout(fig_cat, title=f"{cat_choice} distribution", height=340,
                    yaxis_kw=dict(gridcolor=co.COLORS["border"]))
    st.plotly_chart(fig_cat, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Bivariate ──
    st.markdown('<div class="section-eyebrow">Bivariate</div>'
                '<div class="section-header">Income class vs numerical feature</div>', unsafe_allow_html=True)
    labeled    = d.df[d.df["Income"].isin(["<=50K", ">50K"])]
    num_choice = st.selectbox("Select numerical feature", con.NUM_FEATURES, index=0)
    fig_biv = go.Figure()
    for cls, col in [("<=50K", co.COLORS["teal"]), (">50K", co.COLORS["red"])]:
        vals = labeled[labeled["Income"] == cls][num_choice].dropna()
        counts, edges = np.histogram(vals, bins=40)
        centers = (edges[:-1] + edges[1:]) / 2
        r2, g2, b2 = int(col[1:3], 16), int(col[3:5], 16), int(col[5:], 16)
        fig_biv.add_trace(go.Scatter(
            x=centers, y=counts / counts.max(),
            fill="tozeroy", name=cls,
            fillcolor=f"rgba({r2},{g2},{b2},0.15)",
            line=dict(color=col, width=2),
        ))
    l.apply_layout(fig_biv, title=f"{num_choice} by income class", height=320,
                    yaxis_kw=dict(showticklabels=False, gridcolor=co.COLORS["border"]))
    st.plotly_chart(fig_biv, width="stretch")