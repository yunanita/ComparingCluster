import streamlit as st
import pandas as pd
import numpy as np
import os

# HARUS di awal sebelum perintah streamlit lainnya
st.set_page_config(layout="wide", page_title="Dashboard Clustering Kesehatan & Lingkungan")

# CSS untuk UI aesthetic dengan tema autumn/nature
# Palette: #606c38 (olive), #283618 (dark green), #fefae0 (cream), #dda15e (orange), #bc6c25 (terracotta)
# FIXED COLORS - Warna font SOLID tanpa bergantung tema, selalu terbaca
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
    
    /* Font Global */
    * {
        font-family: 'Poppins', 'Trebuchet MS', sans-serif !important;
    }
    
    /* Font Cambria Math untuk rumus matematika */
    .katex, .katex *, .katex-display, .katex-display *, 
    mjx-container, mjx-container *, .MathJax, .MathJax * {
        font-family: 'Cambria Math', 'Cambria', 'Times New Roman', serif !important;
    }
    
    /* Container utama */
    .main .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* ============================================
       CUSTOM HEADER - FIXED COLORS (TIDAK BERUBAH)
    ============================================ */
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
        color: #fefae0 !important;
        font-size: 0.95rem !important;
        margin: 0.2rem 0 !important;
    }
    
    /* Section Headers - FIXED COLORS */
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
    
    /* ============================================
       TOMBOL - FIXED COLORS
    ============================================ */
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
    
    /* Link buttons - FIXED COLORS */
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
    
    /* ============================================
       TABS - dengan border dan styling
    ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 6px;
        padding: 10px;
        border-radius: 12px;
        border: 2px solid #dda15e;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(221, 161, 94, 0.2) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #bc6c25 !important;
        color: #fefae0 !important;
    }
    
    /* ============================================
       METRIC CARDS - dengan border
    ============================================ */
    [data-testid="metric-container"] {
        border: 2px solid #606c38 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    /* ============================================
       EXPANDER - styling
    ============================================ */
    [data-testid="stExpander"] {
        border: 2px solid #606c38 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
        font-size: 0 !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"]::before {
        content: '▶' !important;
        font-size: 12px !important;
    }
    
    [data-testid="stExpander"][open] [data-testid="stExpanderToggleIcon"]::before {
        content: '▼' !important;
    }
    
    /* ============================================
       CONTAINERS & CARDS - dengan border
    ============================================ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 2px solid #606c38 !important;
        border-radius: 12px !important;
    }
    
    .stDataFrame {
        border: 2px solid #dda15e !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    
    /* ============================================
       INPUTS - dengan border
    ============================================ */
    .stSelectbox > div > div,
    .stMultiselect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border: 2px solid #dda15e !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader > div {
        border: 2px dashed #bc6c25 !important;
        border-radius: 10px !important;
    }
    
    /* ============================================
       ALERT BOXES
    ============================================ */
    .stAlert {
        border-radius: 10px !important;
    }
    
    [data-testid="stInfo"] {
        border-left: 4px solid #606c38 !important;
    }
    
    [data-testid="stSuccess"] {
        border-left: 4px solid #283618 !important;
    }
    
    [data-testid="stWarning"] {
        border-left: 4px solid #dda15e !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background-color: #dda15e;
        margin: 1.5rem 0;
        opacity: 0.5;
    }
    
    /* Fix line height */
    p, span, div, label {
        line-height: 1.5 !important;
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
    <p>📅 Semarang, 15 Januari 2026</p>
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