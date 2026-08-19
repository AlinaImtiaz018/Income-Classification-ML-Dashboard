import pandas as pd
import numpy as np
import streamlit as st

from utils import (
    constants as c, 
    layout as l, 
    data as d
)


def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.1rem 0 1.4rem 0; 
                    border-bottom:1px solid #5c2432; 
                    margin-bottom:0.5rem;">
            <div style="font-family:'DM Sans',
                    sans-serif; 
                    font-size:1.30rem; 
                    font-weight:700;
                    color:#F8FAFC; 
                    letter-spacing:-0.01em; 
                    line-height:1.2;">
                Income Predict
            </div>
            <div style="font-size:0.725rem; 
                    color:#e8b4be; 
                    margin-top:0.3rem; 
                    letter-spacing:0.05em;">
                ML Classification Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><p>Navigation</p>", unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["Overview", "Data Exploration", "Model Results", "Model Comparison", "Predictions"],
            label_visibility="hidden",
        )

        # divider
        st.markdown("<hr style='border-color:#5c2432; margin:0.75rem 0;'>", unsafe_allow_html=True)

        st.markdown("<br><p>Imputation Method</p>", unsafe_allow_html=True)
        imputation = st.selectbox("Imputation Method", ["KNN", "Iterative", "Dropped"], label_visibility="hidden")

        st.markdown("<br><p>Primary Model</p>", unsafe_allow_html=True)
        model_choice = st.selectbox("Primary Model", c.MODEL_NAMES_FULL, index=2, label_visibility="hidden")

        # divider
        st.markdown("<hr style='border-color:#5c2432; margin:0.75rem 0;'>", unsafe_allow_html=True)

        if d.DATA_LOADED:
            labeled = d.df[d.df["Income"].isin(["<=50K", ">50K"])]
            n_total   = len(d.df)
            n_labeled = len(labeled)
            st.markdown(f"""
            <div style="font-size:0.72rem; color:#e8b4be; line-height:1.85;">
                <br><span style="color:#22C55E; font-weight:600;">Dataset loaded ✓</span><br>
                {n_total:,} total rows &nbsp;<br>{n_labeled:,} labeled rows<br><br>
                <span style="color:#F8FAFC; font-weight:600;">Best model</span><br>
                Gradient Boosting · Dropped<br>
                Accuracy 87.1% &nbsp;·&nbsp; F1 0.694
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-size:0.72rem; color:#EF4444; line-height:1.8;">
                ⚠ data/einkommen.train not found.<br>
                Place the file at <code>data/einkommen.train</code>
                to enable live charts.
            </div>""", unsafe_allow_html=True)

        return page, imputation, model_choice
