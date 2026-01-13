import streamlit as st
import pandas as pd
import numpy as np
import os

# HARUS di awal sebelum perintah streamlit lainnya
st.set_page_config(layout="wide", page_title="Dashboard Clustering Kesehatan & Lingkungan")

# CSS untuk UI aesthetic dengan tema autumn/nature
# Palette: #606c38 (olive), #283618 (dark green), #fefae0 (cream), #dda15e (orange), #bc6c25 (terracotta)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
    
    /* Font Global */
    * {
        font-family: 'Poppins', 'Trebuchet MS', sans-serif !important;
    }
    
    /* Font Cambria Math untuk rumus matematika - override global font */
    .katex, .katex *, .katex-display, .katex-display *, 
    .katex .mathdefault, .katex .mathnormal, .katex .mathit,
    .katex-html, .katex .base, .katex .strut,
    mjx-container, mjx-container *, .MathJax, .MathJax *,
    span.katex-mathml, .katex .mord, .katex .mbin, .katex .mrel,
    .katex .mopen, .katex .mclose, .katex .mpunct, .katex .minner {
        font-family: 'Cambria Math', 'Cambria', 'Latin Modern Math', 'STIX Two Math', 'Times New Roman', serif !important;
    }
    
    /* Container utama */
    .main .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Background halaman - cream solid */
    .stApp {
        background-color: #fefae0 !important;
    }
    
    /* Custom Header */
    .main-header {
        background-color: #3d4a2d;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        border-left: 6px solid #dda15e;
    }
    
    .main-header h1 {
        color: #fefae0 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .main-header p {
        color: #dda15e !important;
        font-size: 0.95rem !important;
        margin: 0.2rem 0 !important;
    }
    
    /* Section Headers */
    .section-header {
        background-color: #bc6c25;
        color: #fefae0 !important;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .section-header-green {
        background-color: #606c38;
        color: #fefae0 !important;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Cards */
    .info-card {
        background-color: #fff;
        border: 2px solid #dda15e;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .info-card h4 {
        color: #283618 !important;
        margin-bottom: 0.8rem !important;
    }
    
    .feature-card {
        background-color: #fff;
        border: 2px solid #606c38;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.3rem 0;
        text-align: center;
    }
    
    /* Judul */
    h1, h2, h3 {
        color: #283618 !important;
        font-weight: 700 !important;
    }
    
    h4, h5, h6 {
        color: #606c38 !important;
        font-weight: 600 !important;
    }
    
    /* Tombol */
    .stButton > button {
        background-color: #bc6c25 !important;
        border: none !important;
        border-radius: 10px !important;
        color: #fefae0 !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    
    .stButton > button:hover {
        background-color: #dda15e !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 6px;
        background-color: #fff;
        padding: 10px;
        border-radius: 12px;
        border: 2px solid #dda15e;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #283618 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #fefae0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #bc6c25 !important;
        color: #fefae0 !important;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #fff !important;
        border: 2px solid #606c38 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        position: relative !important;
    }
    
    [data-testid="metric-container"] label {
        color: #606c38 !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
        position: relative !important;
        display: block !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #283618 !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        position: relative !important;
    }
    
    /* Fix text overlap globally */
    p, span, div, label {
        line-height: 1.5 !important;
    }
    
    /* Fix stacked elements */
    .element-container {
        position: relative !important;
        z-index: auto !important;
    }
    
    /* Expander styling - Fix keyboard_arrow_right text */
    [data-testid="stExpander"] {
        background-color: #fff !important;
        border: 2px solid #606c38 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    [data-testid="stExpander"] summary {
        padding: 0.8rem 1rem !important;
        background-color: #fff !important;
        display: flex !important;
        align-items: center !important;
    }
    
    [data-testid="stExpander"] summary:hover {
        background-color: #fefae0 !important;
    }
    
    /* Hide ONLY the toggle icon text (keyboard_arrow_right) */
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
        font-size: 0 !important;
        line-height: 0 !important;
        width: 24px !important;
        height: 24px !important;
        min-width: 24px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-right: 8px !important;
    }
    
    /* Add SVG arrow icon as replacement */
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"]::before {
        content: '▶' !important;
        font-size: 12px !important;
        color: #606c38 !important;
    }
    
    /* When expanded, rotate the arrow */
    [data-testid="stExpander"][open] [data-testid="stExpanderToggleIcon"]::before {
        content: '▼' !important;
    }
    
    /* Style the expander title - SHOW IT */
    [data-testid="stExpander"] summary p {
        font-size: 1rem !important;
        color: #283618 !important;
        font-weight: 600 !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }
    
    [data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] {
        display: inline !important;
        flex: 1 !important;
    }
    
    /* Expander content */
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #fff !important;
        padding: 1rem !important;
        border-top: 1px solid #dda15e !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 2px solid #dda15e !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    
    /* Inputs */
    .stSelectbox > div > div,
    .stMultiselect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #fff !important;
        border: 2px solid #dda15e !important;
        border-radius: 8px !important;
        color: #283618 !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within {
        border-color: #606c38 !important;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background-color: #fff !important;
        border: 2px dashed #bc6c25 !important;
        border-radius: 10px !important;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 10px !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background-color: #dda15e;
        margin: 1.5rem 0;
        opacity: 0.5;
    }
    
    /* Containers with border */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #fff !important;
        border: 2px solid #606c38 !important;
        border-radius: 12px !important;
    }
    
    /* Link buttons */
    .stLinkButton a {
        background-color: #606c38 !important;
        color: #fefae0 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }
    
    .stLinkButton a:hover {
        background-color: #283618 !important;
    }
    
    /* Caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #606c38 !important;
    }
    
    /* Info/Success/Warning boxes */
    .stInfo, [data-testid="stInfo"] {
        background-color: rgba(96, 108, 56, 0.1) !important;
        border-left: 4px solid #606c38 !important;
    }
    
    .stSuccess, [data-testid="stSuccess"] {
        background-color: rgba(96, 108, 56, 0.15) !important;
        border-left: 4px solid #283618 !important;
    }
    
    .stWarning, [data-testid="stWarning"] {
        background-color: rgba(221, 161, 94, 0.15) !important;
        border-left: 4px solid #dda15e !important;
    }
</style>
""", unsafe_allow_html=True)
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch,
    DBSCAN, OPTICS, SpectralClustering, AffinityPropagation
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pydeck as pdk

# optional hdbscan and geopy
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    from geopy.geocoders import Nominatim  # type: ignore
    from geopy.extra.rate_limiter import RateLimiter  # type: ignore
    HAS_GEOPY = True
except Exception:
    HAS_GEOPY = False

# Paths (update these if needed)
# Gunakan relative path - folder yang sama dengan script ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_HEALTH = os.path.join(BASE_DIR, "child_mortality.xlsx")
PATH_ENV = os.path.join(BASE_DIR, "deforestasi.xlsx")

# --- Header Utama ---
st.markdown("""
<div class="main-header">
    <h1>🌿 Dashboard Clustering Kesehatan & Lingkungan</h1>
    <p>🎓 UAS Machine Learning — Universitas Muhammadiyah Semarang</p>
    <p>📅 Semarang, 17 Desember 2025</p>
</div>
""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['📖 About', 
                            '📊 Dataset', 
                            '🔧 Preprocessing', 
                            '🤖 Machine Learning',
                            '📈 Visualisasi',
                            '🔮 Prediksi',
                            '👤 Contact Me'])

with tab0:
    import about_project
    about_project.about_tab()

with tab1:
    import about
    about.about_dataset()

with tab2:
    import preprocessing
    preprocessing.preprocessing_tab()

with tab3:
    import ml_5_algoritma as machine_learning
    machine_learning.ml_tab()

with tab4:
    import visualisasi
    visualisasi.visualisasi_tab()

with tab5:
    import prediksi
    prediksi.prediksi_tab()

with tab6:
    import contact
    contact.contact_tab()