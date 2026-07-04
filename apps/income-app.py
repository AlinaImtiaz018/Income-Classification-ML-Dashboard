import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0F172A; }
    .block-container {
        padding-top: 2.5rem;
        padding-right: 2.5rem;
        padding-left: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #ad1f42; /* #4a1c28*/
        /* Subtle right border that blends — much darker than before */
        border-right: 1px solid #321018;
        /* Inward shadow gives depth without a harsh line */
        box-shadow: 4px 0 24px rgba(0,0,0,0.55);
    }

    /* Sidebar label typography — section headings above widgets */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] p {
        color: #ffffff !important; /* #E8CAD0*/
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0 !important;
    }

    /* Fixed, uniform spacing: label → widget → next label */
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        margin-top: 0.1rem;
        margin-bottom: 0.2rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] {
        margin-bottom: 0;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] {
        margin-bottom: 0;
    }
    /* Radio option labels */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        font-size: 0.84rem !important;
        font-weight: 400 !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
        color: #dba8b4 !important;
        padding: 0.18rem 0;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        color: #F8FAFC !important;
    }
    /* Sidebar top padding */
    section[data-testid="stSidebar"] .block-container,
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0rem !important;
    }

    /* ── Hero ── */
    .hero-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.025em;
        line-height: 1.18;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 0;
        line-height: 1.5;
    }
    .hero-accent { color: #dd3c64; }

    /* ── KPI cards ── */
    .kpi-card {
        background: #161e2e;
        border: 1px solid #243044;
        border-top: 2px solid #dd3c64;
        border-radius: 10px;
        padding: 1.2rem 1.4rem 1rem 1.4rem;
        position: relative;
        height: 126px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0,0,0,0.45), 0 1px 4px rgba(0,0,0,0.3);
    }
    .kpi-label {
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1;
    }
    .kpi-sub      { font-size: 0.73rem; color: #22C55E; margin-top: 0.25rem; min-height: 1rem; }
    .kpi-sub-warn { font-size: 0.73rem; color: #EF04444; margin-top: 0.25rem; min-height: 1rem; }

    /* ── Section headings ── */
    .section-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    .section-eyebrow {
        font-size: 0.63rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #db5771;
        margin-bottom: 0.18rem;
    }

    /* ── Dividers ── */
    hr { border: none; border-top: 1px solid #334155; margin: 1.5rem 0; }

    /* ── Info boxes ── */
    .info-box {
        background: #0d1117;
        border: 1px solid #334155;
        border-left: 3px solid #dd3c64;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        font-size: 0.83rem;
        color: #94A3B8;
        line-height: 1.65;
        min-height: 120px;
    }
    .info-box strong { color: #F8FAFC; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Color palette ────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#dd3c64",
    "secondary": "#c04a6a",
    "teal":      "#00b3b3",
    "green":     "#22C55E",
    "red":       "#EF4444",
    "amber":     "#F59E0B",
    "bg":        "#0F172A",
    "card":      "#0d1117",
    "border":    "#334155",
    "text":      "#F8FAFC",
    "muted":     "#94A3B8",
}
MODEL_COLORS = ["#c36ce0", "#22C55E", "#F59E0B", "#EF4444"]
MODEL_NAMES_SHORT = ["Logistic Reg.", "Random Forest", "Grad. Boosting", "SVM"]
MODEL_NAMES_FULL = [
    "Linear Classification",
    "Random Forest Classification",
    "Gradient Boosting Classification",
    "Support Vector Machines (SVMs)",
]
COLUMNS = [
    "Age", "Employment_type", "Weighting_factor", "Education_level",
    "Schooling_years", "Marital_status", "Employment_area", "Partnership",
    "Ethnicity", "Gender", "Gains", "Losses", "Weekly_working_time",
    "Country_of_birth", "Income",
]
NUM_FEATURES = ["Age", "Weighting_factor", "Schooling_years", "Gains", "Losses", "Weekly_working_time"]
CAT_FEATURES = ["Employment_type", "Education_level", "Marital_status", "Employment_area",
                "Partnership", "Ethnicity", "Gender", "Country_of_birth"]

RESULTS = {
    "KNN": {
        "Linear Classification":            {"accuracy": 0.806, "f1_score": 0.503, "recall": 0.415, "precision": 0.636, "roc_auc": 0.671, "cm": [[354,28],[69,49]]},
        "Random Forest Classification":     {"accuracy": 0.844, "f1_score": 0.639, "recall": 0.585, "precision": 0.704, "roc_auc": 0.754, "cm": [[353,29],[49,69]]},
        "Gradient Boosting Classification": {"accuracy": 0.870, "f1_score": 0.680, "recall": 0.585, "precision": 0.812, "roc_auc": 0.771, "cm": [[366,16],[49,69]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.772, "f1_score": 0.066, "recall": 0.034, "precision": 1.000, "roc_auc": 0.517, "cm": [[382,0],[114,4]]},
    },
    "Iterative": {
        "Linear Classification":            {"accuracy": 0.806, "f1_score": 0.481, "recall": 0.381, "precision": 0.652, "roc_auc": 0.659, "cm": [[358,24],[73,45]]},
        "Random Forest Classification":     {"accuracy": 0.842, "f1_score": 0.636, "recall": 0.585, "precision": 0.697, "roc_auc": 0.753, "cm": [[352,30],[49,69]]},
        "Gradient Boosting Classification": {"accuracy": 0.868, "f1_score": 0.680, "recall": 0.593, "precision": 0.795, "roc_auc": 0.773, "cm": [[364,18],[48,70]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.772, "f1_score": 0.066, "recall": 0.034, "precision": 1.000, "roc_auc": 0.517, "cm": [[382,0],[114,4]]},
    },
    "Dropped": {
        "Linear Classification":            {"accuracy": 0.814, "f1_score": 0.465, "recall": 0.343, "precision": 0.725, "roc_auc": 0.651, "cm": [[336,14],[71,37]]},
        "Random Forest Classification":     {"accuracy": 0.849, "f1_score": 0.657, "recall": 0.611, "precision": 0.710, "roc_auc": 0.767, "cm": [[323,27],[42,66]]},
        "Gradient Boosting Classification": {"accuracy": 0.871, "f1_score": 0.694, "recall": 0.620, "precision": 0.788, "roc_auc": 0.784, "cm": [[332,18],[41,67]]},
        "Support Vector Machines (SVMs)":   {"accuracy": 0.769, "f1_score": 0.036, "recall": 0.019, "precision": 1.000, "roc_auc": 0.509, "cm": [[350,0],[106,2]]},
    },
}
PREDICTED_COUNTS = {
    "KNN":       {"<=50K": 20324, ">50K": 4676},
    "Iterative": {"<=50K": 20955, ">50K": 4045},
    "Dropped":   {"<=50K": 18812, ">50K": 4412},
}

# ─── Data loading ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "data/einkommen.train") -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS)
    df.replace(" ?", pd.NA, inplace=True)
    df["Income"] = df["Income"].str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df

try:
    df = load_data()
    DATA_LOADED = True
except FileNotFoundError:
    df = None
    DATA_LOADED = False

# ─── Layout helpers ───────────────────────────────────────────────────────────────
def base_layout(title="", height=400) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="DM Sans", size=14, color=COLORS["text"]), x=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["muted"], size=11),
        margin=dict(l=10, r=10, t=40, b=10),
    )

def axis_style(**kwargs) -> dict:
    base = dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], tickfont=dict(size=11))
    base.update(kwargs)
    return base

DEFAULT_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"], font=dict(size=11))

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

# ─── Sidebar ───────────────────────────────────────────────────────────────────── ⚡ 
with st.sidebar:
    st.markdown("""
    <div style="padding:1.1rem 0 1.4rem 0; border-bottom:1px solid #5c2432; margin-bottom:0.5rem;">
        <div style="font-family:'DM Sans',sans-serif; font-size:1.30rem; font-weight:700;
                    color:#F8FAFC; letter-spacing:-0.01em; line-height:1.2;">
            Income Predict
        </div>
        <div style="font-size:0.725rem; color:#e8b4be; margin-top:0.3rem; letter-spacing:0.05em;">
            ML Classification Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><p>Navigation</p>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Data Exploration", "🧠 Model Results", "📈 Model Comparison", "🎯 Predictions"],
        label_visibility="hidden",
    )

    st.markdown("<hr style='border-color:#5c2432; margin:0.75rem 0;'>", unsafe_allow_html=True)

    st.markdown("<br><p>Imputation Method</p>", unsafe_allow_html=True)
    imputation = st.selectbox("Imputation Method", ["KNN", "Iterative", "Dropped"], label_visibility="hidden")

    st.markdown("<br><p>Primary Model</p>", unsafe_allow_html=True)
    model_choice = st.selectbox("Primary Model", MODEL_NAMES_FULL, index=2, label_visibility="hidden")

    st.markdown("<hr style='border-color:#5c2432; margin:0.75rem 0;'>", unsafe_allow_html=True)

    if DATA_LOADED:
        labeled = df[df["Income"].isin(["<=50K", ">50K"])]
        n_total   = len(df)
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

# ─── Page: Overview ──────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div class="hero-title">Income <span class="hero-accent">Classification</span><br>ML Dashboard</div>
        <div class="hero-subtitle">Binary prediction: ≤50K vs &gt;50K annual income &nbsp;·&nbsp; Gradient Boosting best performer</div>
    </div>""", unsafe_allow_html=True)

    if DATA_LOADED:
        labeled    = df[df["Income"].isin(["<=50K", ">50K"])]
        n_total    = len(df)
        n_labeled  = len(labeled)
        cls_counts = labeled["Income"].value_counts()
        n_low  = int(cls_counts.get("<=50K", 3779))
        n_high = int(cls_counts.get(">50K",  1221))
    else:
        n_total, n_labeled, n_low, n_high = 30000, 5000, 3779, 1221

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(make_kpi("Total Instances", f"{n_total:,}", "Full dataset size"), unsafe_allow_html=True)
    with k2: st.markdown(make_kpi("Labeled Rows",    f"{n_labeled:,}", "Used for training"), unsafe_allow_html=True)
    with k3: st.markdown(make_kpi("Best Accuracy",   "87.1%", "Gradient Boosting · Dropped"), unsafe_allow_html=True)
    with k4: st.markdown(make_kpi("Best ROC AUC",    "0.784", "Gradient Boosting · Dropped"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-eyebrow">Model Performance</div>'
                    '<div class="section-header">Accuracy across all configurations</div>', unsafe_allow_html=True)
        methods = ["KNN", "Iterative", "Dropped"]
        fig = go.Figure()
        for i, m_full in enumerate(MODEL_NAMES_FULL):
            accs = [RESULTS[mth][m_full]["accuracy"] for mth in methods]
            fig.add_trace(go.Bar(
                name=MODEL_NAMES_SHORT[i], x=methods, y=accs,
                marker_color=MODEL_COLORS[i], marker_line_width=0,
                text=[f"{v:.1%}" for v in accs], textposition="outside", textfont=dict(size=10),
                offsetgroup=i, width=0.18,
            ))
        apply_layout(fig, height=360,
                     yaxis_kw=dict(range=[0.6, 0.97], tickformat=".0%", gridcolor=COLORS["border"]),
                     barmode="group", bargap=0.2, bargroupgap=0.05)
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown('<div class="section-eyebrow">Dataset</div>'
                    '<div class="section-header">Class distribution (labeled)</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["≤50K", ">50K"], values=[n_low, n_high],
            hole=0.62,
            marker=dict(colors=[COLORS["teal"], COLORS["red"]], line=dict(color=COLORS["bg"], width=3)),
            textinfo="percent+label", textfont=dict(size=12, color=COLORS["text"]),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"{n_labeled:,}<br><span style='font-size:10px'>labeled</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=COLORS["text"]), align="center",
        )
        apply_layout(fig2, height=360, legend_kw=dict(orientation="h", y=-0.05), showlegend=True)
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

# ─── Page: Data Exploration ───────────────────────────────────────────────────────
elif page == "🔍 Data Exploration":
    st.markdown('<div class="hero-title">Data <span class="hero-accent">Exploration</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Dataset structure, missing values, and feature distributions</div><br>', unsafe_allow_html=True)

    if not DATA_LOADED:
        st.warning("Place `einkommen.train` inside a `data/` folder next to `app.py` to enable live EDA.")
        st.stop()

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown('<div class="section-eyebrow">Schema</div>'
                    '<div class="section-header">Feature overview</div>', unsafe_allow_html=True)
        miss = df.isna().sum()
        schema = pd.DataFrame({
            "Feature": COLUMNS,
            "Type":    [str(df[c].dtype) for c in COLUMNS],
            "Missing": [f"{miss[c]:,}" if miss[c] > 0 else "—" for c in COLUMNS],
            "Unique":  [str(df[c].nunique()) for c in COLUMNS],
        })
        st.dataframe(schema, width="stretch", hide_index=True)

    with col_b:
        st.markdown('<div class="section-eyebrow">Missing Values</div>'
                    '<div class="section-header">Columns with nulls</div>', unsafe_allow_html=True)
        miss_df  = miss[miss > 0].sort_values(ascending=True)
        miss_pcts = (miss_df / len(df)).tolist()
        fig_miss = go.Figure(go.Bar(
            x=miss_df.values, y=miss_df.index.tolist(),
            orientation="h",
            marker=dict(color=[COLORS["primary"], COLORS["secondary"], COLORS["amber"], COLORS["red"]][:len(miss_df)]),
            text=[f"{v:,}  ({p:.1%})" for v, p in zip(miss_df.values, miss_pcts)],
            textposition="outside", textfont=dict(size=10),
        ))
        apply_layout(fig_miss, height=320,
                     xaxis_kw=dict(title="Missing count", gridcolor=COLORS["border"]),
                     yaxis_kw=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_miss, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Numerical distributions ──
    st.markdown('<div class="section-eyebrow">EDA</div>'
                '<div class="section-header">Numerical feature distributions</div>', unsafe_allow_html=True)
    fig_num = make_subplots(rows=2, cols=3, subplot_titles=NUM_FEATURES,
                             vertical_spacing=0.16, horizontal_spacing=0.08)
    for idx, feat in enumerate(NUM_FEATURES):
        r, c = idx // 3 + 1, idx % 3 + 1
        series = df[feat].dropna().astype(float)
        mu     = series.mean()
        counts, bin_edges = np.histogram(series, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig_num.add_trace(go.Scatter(
            x=bin_centers, y=counts / counts.max(), fill="tozeroy",
            fillcolor="rgba(0,179,179,0.12)",
            line=dict(color=COLORS["teal"], width=1.5),
            showlegend=False,
        ), row=r, col=c)
        fig_num.add_vline(x=mu, line_dash="dot", line_color=COLORS["red"], line_width=1.5, row=r, col=c)

    fig_num.update_layout(height=440, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter", color=COLORS["muted"], size=10),
                           margin=dict(l=10, r=10, t=40, b=10))
    fig_num.update_xaxes(gridcolor=COLORS["border"], linecolor=COLORS["border"])
    fig_num.update_yaxes(gridcolor=COLORS["border"], linecolor=COLORS["border"], showticklabels=False)
    st.plotly_chart(fig_num, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Categorical distributions ──
    st.markdown('<div class="section-eyebrow">Categorical</div>'
                '<div class="section-header">Key categorical feature distributions</div>', unsafe_allow_html=True)
    cat_choice  = st.selectbox("Select feature", CAT_FEATURES, index=CAT_FEATURES.index("Gender"))
    cat_counts  = df[cat_choice].value_counts().head(15)
    fig_cat = go.Figure(go.Bar(
        x=cat_counts.index.tolist(), y=cat_counts.values,
        marker_color=COLORS["primary"], marker_line_width=0,
        text=cat_counts.values, textposition="outside",
    ))
    apply_layout(fig_cat, title=f"{cat_choice} distribution", height=340,
                 yaxis_kw=dict(gridcolor=COLORS["border"]))
    st.plotly_chart(fig_cat, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Bivariate ──
    st.markdown('<div class="section-eyebrow">Bivariate</div>'
                '<div class="section-header">Income class vs numerical feature</div>', unsafe_allow_html=True)
    labeled    = df[df["Income"].isin(["<=50K", ">50K"])]
    num_choice = st.selectbox("Select numerical feature", NUM_FEATURES, index=0)
    fig_biv = go.Figure()
    for cls, col in [("<=50K", COLORS["teal"]), (">50K", COLORS["red"])]:
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
    apply_layout(fig_biv, title=f"{num_choice} by income class", height=320,
                 yaxis_kw=dict(showticklabels=False, gridcolor=COLORS["border"]))
    st.plotly_chart(fig_biv, width="stretch")

# ─── Page: Model Results ──────────────────────────────────────────────────────────
elif page == "🧠 Model Results":
    st.markdown('<div class="hero-title">Model <span class="hero-accent">Results</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-subtitle">Imputation: <strong style="color:#F8FAFC;">{imputation}</strong>'
        f' &nbsp;·&nbsp; Model: <strong style="color:#F8FAFC;">{model_choice}</strong></div><br>',
        unsafe_allow_html=True,
    )

    metrics = RESULTS[imputation][model_choice]

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
            st.markdown(make_kpi(label, val, sub), unsafe_allow_html=True)

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
            colorscale=[[0, COLORS["card"]], [0.5, "#3d1520"], [1, COLORS["primary"]]],
            showscale=False, xgap=3, ygap=3,
            hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_traces(textfont=dict(size=12, color="white"))
        apply_layout(fig_cm, height=320,
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
        bar_colors   = [COLORS["primary"] if v >= 0.7 else COLORS["amber"] if v >= 0.5 else COLORS["red"]
                        for v in metric_vals]
        fig_bar = go.Figure(go.Bar(
            x=metric_names, y=metric_vals,
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in metric_vals], textposition="outside",
        ))
        apply_layout(fig_bar, height=320,
                     yaxis_kw=dict(range=[0, 1.18], gridcolor=COLORS["border"], tickformat=".0%"))
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-eyebrow">All Models</div>'
                f'<div class="section-header">Full comparison — {imputation} imputation</div>', unsafe_allow_html=True)
    rows = []
    for mname in MODEL_NAMES_FULL:
        m       = RESULTS[imputation][mname]
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

# ─── Page: Model Comparison ───────────────────────────────────────────────────────
elif page == "📈 Model Comparison":
    st.markdown('<div class="hero-title">Model <span class="hero-accent">Comparison</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Cross-imputation performance &nbsp;·&nbsp; 10-fold cross-validation</div><br>', unsafe_allow_html=True)

    metric_opt = st.selectbox(
        "Compare metric", ["accuracy", "f1_score", "recall", "precision", "roc_auc"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    methods = ["KNN", "Iterative", "Dropped"]
    fig_cmp = go.Figure()
    for i, m_full in enumerate(MODEL_NAMES_FULL):
        vals = [RESULTS[mth][m_full][metric_opt] for mth in methods]
        fig_cmp.add_trace(go.Bar(
            name=MODEL_NAMES_SHORT[i], x=methods, y=vals,
            marker_color=MODEL_COLORS[i], marker_line_width=0,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            offsetgroup=i, width=0.18,
        ))
    apply_layout(fig_cmp, title=f"{metric_opt.replace('_',' ').title()} by imputation method", height=400,
                 yaxis_kw=dict(gridcolor=COLORS["border"],
                               range=[0, max(RESULTS[m][n][metric_opt] for m in methods for n in MODEL_NAMES_FULL) * 1.2]),
                 barmode="group", bargap=0.2, bargroupgap=0.05)
    st.plotly_chart(fig_cmp, width="stretch")

    # ── Radar ──
    st.markdown('<div class="section-header">Multi-metric radar — Gradient Boosting across imputation methods</div>', unsafe_allow_html=True)
    radar_metrics = ["accuracy", "f1_score", "recall", "precision", "roc_auc"]
    radar_labels  = ["Accuracy", "F1-Score", "Recall", "Precision", "ROC AUC"]
    best_model    = "Gradient Boosting Classification"
    fig_radar = go.Figure()
    for i, mth in enumerate(methods):
        vals   = [RESULTS[mth][best_model][m] for m in radar_metrics] + [RESULTS[mth][best_model][radar_metrics[0]]]
        r, g, b = int(MODEL_COLORS[i][1:3],16), int(MODEL_COLORS[i][3:5],16), int(MODEL_COLORS[i][5:],16)
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels + [radar_labels[0]],
            name=mth, fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.12)",
            line=dict(color=MODEL_COLORS[i], width=2),
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor=COLORS["border"],
                            tickfont=dict(size=9, color=COLORS["muted"])),
            angularaxis=dict(gridcolor=COLORS["border"], tickfont=dict(size=11, color=COLORS["text"])),
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=420,
        font=dict(family="Inter", color=COLORS["muted"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=50, r=50, t=40, b=30),
    )
    st.plotly_chart(fig_radar, width="stretch")

    # ── Heatmap ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Performance heatmap — all models × all methods</div>', unsafe_allow_html=True)
    heat_metric = st.selectbox("Heatmap metric", ["accuracy", "f1_score", "roc_auc"],
                                format_func=lambda x: x.replace("_"," ").title(), key="heat")
    z_vals   = [[RESULTS[mth][m][heat_metric] for mth in methods] for m in MODEL_NAMES_FULL]
    y_labels = [n.replace(" Classification","").replace(" (SVMs)","") for n in MODEL_NAMES_FULL]
    fig_heat = go.Figure(go.Heatmap(
        z=z_vals, x=methods, y=y_labels,
        colorscale=[[0, COLORS["card"]], [0.5, "#3d1520"], [1, COLORS["primary"]]],
        text=[[f"{v:.3f}" for v in row] for row in z_vals],
        texttemplate="%{text}", textfont=dict(size=13),
        xgap=4, ygap=4,
    ))
    apply_layout(fig_heat, height=300,
                 xaxis_kw=dict(gridcolor="rgba(0,0,0,0)"),
                 yaxis_kw=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_heat, width="stretch")

# ─── Page: Predictions ────────────────────────────────────────────────────────────
elif page == "🎯 Predictions":
    st.markdown('<div class="hero-title">Test Set <span class="hero-accent">Predictions</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Gradient Boosting applied to 25,000 unlabeled test instances</div><br>', unsafe_allow_html=True)

    methods = ["KNN", "Iterative", "Dropped"]
    p1, p2, p3 = st.columns(3)
    for col, mth in zip([p1, p2, p3], methods):
        data      = PREDICTED_COUNTS[mth]
        total     = data["<=50K"] + data[">50K"]
        pct_high  = data[">50K"] / total
        with col:
            st.markdown(make_kpi(f"{mth} Imputation", f"{data['>50K']:,}", f"{pct_high:.1%} predicted >50K"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Predicted class counts — Gradient Boosting Classifier</div>', unsafe_allow_html=True)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Bar(
        name="≤50K", x=methods,
        y=[PREDICTED_COUNTS[m]["<=50K"] for m in methods],
        marker_color=COLORS["green"], marker_line_width=0,
        text=[PREDICTED_COUNTS[m]["<=50K"] for m in methods], textposition="outside",
    ))
    fig_pred.add_trace(go.Bar(
        name=">50K", x=methods,
        y=[PREDICTED_COUNTS[m][">50K"] for m in methods],
        marker_color=COLORS["red"], marker_line_width=0,
        text=[PREDICTED_COUNTS[m][">50K"] for m in methods], textposition="outside",
    ))
    apply_layout(fig_pred, height=380, yaxis_kw=dict(gridcolor=COLORS["border"]),
                 barmode="group", bargap=0.3)
    st.plotly_chart(fig_pred, width="stretch")

    st.markdown('<div class="section-header">Class share by imputation method</div>', unsafe_allow_html=True)
    fig_stk = go.Figure()
    totals  = {m: PREDICTED_COUNTS[m]["<=50K"] + PREDICTED_COUNTS[m][">50K"] for m in methods}
    fig_stk.add_trace(go.Bar(
        name="≤50K", x=methods,
        y=[PREDICTED_COUNTS[m]["<=50K"] / totals[m] for m in methods],
        marker_color=COLORS["teal"], marker_line_width=0,
        text=[f"{PREDICTED_COUNTS[m]['<=50K']/totals[m]:.1%}" for m in methods], textposition="inside",
    ))
    fig_stk.add_trace(go.Bar(
        name=">50K", x=methods,
        y=[PREDICTED_COUNTS[m][">50K"] / totals[m] for m in methods],
        marker_color=COLORS["red"], marker_line_width=0,
        text=[f"{PREDICTED_COUNTS[m]['>50K']/totals[m]:.1%}" for m in methods], textposition="inside",
    ))
    apply_layout(fig_stk, height=300,
                 yaxis_kw=dict(tickformat=".0%", gridcolor=COLORS["border"]),
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
