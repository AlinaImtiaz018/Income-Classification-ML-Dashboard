import pandas as pd
import streamlit as st

from utils import constants as c

@st.cache_data
def load_data(path: str = "data/einkommen.train") -> pd.DataFrame:
    df = pd.read_csv(path, names=c.COLUMNS)
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