import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import constants as c, colors as col

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def base_layout(title="", height=400) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="DM Sans", size=14, color=col.COLORS["text"]), x=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=col.COLORS["muted"], size=11),
        margin=dict(l=10, r=10, t=40, b=10),
    )

def axis_style(**kwargs) -> dict:
    base = dict(gridcolor=col.COLORS["border"], linecolor=col.COLORS["border"], tickfont=dict(size=11))
    base.update(kwargs)
    return base

DEFAULT_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor=col.COLORS["border"], font=dict(size=11))

def apply_layout(fig, title="", height=400, xaxis_kw=None, yaxis_kw=None, legend_kw=None, **extra):
    layout = base_layout(title, height)
    layout["xaxis"]  = axis_style(**(xaxis_kw or {}))
    layout["yaxis"]  = axis_style(**(yaxis_kw or {}))
    layout["legend"] = {**DEFAULT_LEGEND, **(legend_kw or {})}
    layout.update(extra)
    fig.update_layout(**layout)

def make_kpi(label, value, sub="", warn=False):
    sub_class = "kpi-sub-warn" if warn else "kpi-sub"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="{sub_class}">{sub if sub else "&nbsp;"}</div>
    </div>"""
