import streamlit as st

from utils.styles import load_css
from utils.sidebar import show_sidebar

from pages_modules import (
    overview,
    data_exploration,
    model_results,
    model_comparison,
    predictions,
)

st.set_page_config(
    page_title="Income Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

page, imputation, model_choice = show_sidebar()

if page == "📊 Overview":
    overview.display()

elif page == "🔍 Data Exploration":
    data_exploration.display()

elif page == "🧠 Model Results":
    model_results.display(imputation, model_choice)

elif page == "📈 Model Comparison":
    model_comparison.display()

elif page == "🎯 Predictions":
    predictions.display()