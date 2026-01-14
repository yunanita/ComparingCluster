import streamlit as st
import pandas as pd
import numpy as np
import os

# HARUS di awal sebelum perintah streamlit lainnya
st.set_page_config(layout="wide", page_title="Dashboard Clustering Kesehatan & Lingkungan")

# CSS untuk UI aesthetic dengan tema autumn/nature
# Palette: #606c38 (olive), #283618 (dark green), #fefae0 (cream), #dda15e (orange), #bc6c25 (terracotta)
# ADAPTIVE THEME: Font colors akan adaptif berdasarkan tema user (light/dark mode)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
    
    /* ============================================
       ADAPTIVE COLOR VARIABLES - THEME AWARE
       Font akan adaptif berdasarkan tema user
    ============================================ */
    :root {
        --adaptive-text-primary: #283618;
        --adaptive-text-secondary: #606c38;
        --adaptive-bg-card: #fff;
        --adaptive-bg-page: #fefae0;
        --adaptive-bg-input: #fff;
        --adaptive-tab-bg: #fff;
    }
    
    /* Dark mode detection via media query */
    @media (prefers-color-scheme: dark) {
        :root {
            --adaptive-text-primary: #f0f0f0;
            --adaptive-text-secondary: #c5d4a0;
            --adaptive-bg-card: #262730;
            --adaptive-bg-page: #0e1117;
            --adaptive-bg-input: #262730;
            --adaptive-tab-bg: #262730;
        }
    }
    
    /* Streamlit Dark Theme - detect via background color on stApp */
    /* Streamlit dark theme uses #0e1117 or similar dark backgrounds */
    .stApp[style*="background-color: rgb(14, 17, 23)"],
    .stApp[style*="background: rgb(14, 17, 23)"] {
        --adaptive-text-primary: #fafafa !important;
        --adaptive-text-secondary: #c5d4a0 !important;
        --adaptive-bg-card: #262730 !important;
        --adaptive-bg-page: #0e1117 !important;
        --adaptive-bg-input: #262730 !important;
        --adaptive-tab-bg: #262730 !important;
    }
    
    /* Alternative dark mode class selectors for Streamlit */
    [data-theme="dark"] {
        --adaptive-text-primary: #fafafa;
        --adaptive-text-secondary: #c5d4a0;
        --adaptive-bg-card: #262730;
        --adaptive-bg-page: #0e1117;
        --adaptive-bg-input: #262730;
        --adaptive-tab-bg: #262730;
    }
    
    /* Force dark theme styles when dark background detected */
    body:has(.stApp[style*="background-color: rgb(14"]),
    body:has(.stApp[style*="background: rgb(14"]) {
        --adaptive-text-primary: #fafafa !important;
        --adaptive-text-secondary: #c5d4a0 !important;
    }
    
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
    
    /* Background halaman - REMOVE FORCED BACKGROUND to allow theme adaptation */
    /* .stApp background removed to respect user theme */
    
    /* Custom Header - TETAP SAMA (tidak berubah) */
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
    
    /* Section Headers - TETAP SAMA (warna fixed di dalam header) */
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
        background-color: var(--adaptive-bg-card);
        border: 2px solid #dda15e;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .info-card h4 {
        color: var(--adaptive-text-primary) !important;
        margin-bottom: 0.8rem !important;
    }
    
    .feature-card {
        background-color: var(--adaptive-bg-card);
        border: 2px solid #606c38;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.3rem 0;
        text-align: center;
    }
    
    /* Judul - ADAPTIVE */
    h1, h2, h3 {
        color: var(--adaptive-text-primary) !important;
        font-weight: 700 !important;
    }
    
    h4, h5, h6 {
        color: var(--adaptive-text-secondary) !important;
        font-weight: 600 !important;
    }
    
    /* Tombol - TETAP SAMA */
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
        background-color: var(--adaptive-tab-bg);
        padding: 10px;
        border-radius: 12px;
        border: 2px solid #dda15e;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--adaptive-text-primary) !important;
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
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: var(--adaptive-bg-card) !important;
        border: 2px solid #606c38 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        position: relative !important;
    }
    
    [data-testid="metric-container"] label {
        color: var(--adaptive-text-secondary) !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
        position: relative !important;
        display: block !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--adaptive-text-primary) !important;
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
        background-color: var(--adaptive-bg-card) !important;
        border: 2px solid #606c38 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    [data-testid="stExpander"] summary {
        padding: 0.8rem 1rem !important;
        background-color: var(--adaptive-bg-card) !important;
        display: flex !important;
        align-items: center !important;
    }
    
    [data-testid="stExpander"] summary:hover {
        background-color: rgba(221, 161, 94, 0.15) !important;
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
        color: var(--adaptive-text-secondary) !important;
    }
    
    /* When expanded, rotate the arrow */
    [data-testid="stExpander"][open] [data-testid="stExpanderToggleIcon"]::before {
        content: '▼' !important;
    }
    
    /* Style the expander title - SHOW IT */
    [data-testid="stExpander"] summary p {
        font-size: 1rem !important;
        color: var(--adaptive-text-primary) !important;
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
        background-color: var(--adaptive-bg-card) !important;
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
        background-color: var(--adaptive-bg-input) !important;
        border: 2px solid #dda15e !important;
        border-radius: 8px !important;
        color: var(--adaptive-text-primary) !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within {
        border-color: #606c38 !important;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background-color: var(--adaptive-bg-card) !important;
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
        background-color: var(--adaptive-bg-card) !important;
        border: 2px solid #606c38 !important;
        border-radius: 12px !important;
    }
    
    /* Link buttons - TETAP SAMA */
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
    
    /* Caption - ADAPTIVE */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--adaptive-text-secondary) !important;
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
    
    /* ============================================
       GLOBAL TEXT ADAPTIVE - CRITICAL FOR THEME
       Semua text content harus adaptif
    ============================================ */
    
    /* Markdown text content */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] td,
    [data-testid="stMarkdownContainer"] th,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span {
        color: var(--adaptive-text-primary) !important;
    }
    
    /* Table text */
    [data-testid="stTable"] td,
    [data-testid="stTable"] th,
    .stDataFrame td,
    .stDataFrame th {
        color: var(--adaptive-text-primary) !important;
    }
    
    /* Alert/Info box text */
    [data-testid="stInfo"] p,
    [data-testid="stSuccess"] p,
    [data-testid="stWarning"] p,
    [data-testid="stError"] p,
    .stAlert p {
        color: var(--adaptive-text-primary) !important;
    }
    
    /* Container border wrapper text */
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] span,
    [data-testid="stVerticalBlockBorderWrapper"] li {
        color: var(--adaptive-text-primary) !important;
    }
    
    /* Selectbox and input labels */
    .stSelectbox label,
    .stMultiselect label,
    .stTextInput label,
    .stNumberInput label,
    .stSlider label,
    .stCheckbox label,
    .stRadio label {
        color: var(--adaptive-text-primary) !important;
    }
    
    /* Dropdown options */
    [data-baseweb="select"] [data-baseweb="menu"] {
        background-color: var(--adaptive-bg-card) !important;
    }
    
    [data-baseweb="select"] [data-baseweb="menu"] li {
        color: var(--adaptive-text-primary) !important;
    }
    
    [data-baseweb="select"] [data-baseweb="menu"] li:hover {
        background-color: rgba(221, 161, 94, 0.2) !important;
    }
    
    /* General paragraph and text */
    p, span, div.stText, .element-container p {
        color: var(--adaptive-text-primary);
    }
    
    /* Strong/bold text inside markdown */
    [data-testid="stMarkdownContainer"] strong,
    .stMarkdown strong {
        color: var(--adaptive-text-primary) !important;
    }
    
    /* ============================================
       DIRECT DARK MODE OVERRIDES 
       Fallback untuk browser yang tidak support CSS variables dengan benar
    ============================================ */
    
    /* Ketika Streamlit dalam dark mode, background akan gelap */
    /* Override langsung untuk dark mode */
    @media (prefers-color-scheme: dark) {
        h1, h2, h3 { color: #fafafa !important; }
        h4, h5, h6 { color: #c5d4a0 !important; }
        p, span, li, td, th { color: #fafafa !important; }
        label { color: #fafafa !important; }
        
        .stTabs [data-baseweb="tab"] { color: #fafafa !important; }
        [data-testid="metric-container"] label { color: #c5d4a0 !important; }
        [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #fafafa !important; }
        [data-testid="stExpander"] summary p { color: #fafafa !important; }
        
        .info-card, .feature-card, [data-testid="stExpander"],
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="metric-container"],
        .stTabs [data-baseweb="tab-list"] {
            background-color: #262730 !important;
        }
        
        .stSelectbox > div > div,
        .stMultiselect > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
    }
    
    /* ============================================
       JAVASCRIPT-CONTROLLED DARK MODE CLASS
       Diaktifkan oleh JavaScript saat dark mode terdeteksi
    ============================================ */
    body.streamlit-dark-mode h1,
    body.streamlit-dark-mode h2,
    body.streamlit-dark-mode h3 { 
        color: #fafafa !important; 
    }
    
    body.streamlit-dark-mode h4,
    body.streamlit-dark-mode h5,
    body.streamlit-dark-mode h6 { 
        color: #c5d4a0 !important; 
    }
    
    body.streamlit-dark-mode p,
    body.streamlit-dark-mode span,
    body.streamlit-dark-mode li,
    body.streamlit-dark-mode td,
    body.streamlit-dark-mode th,
    body.streamlit-dark-mode label,
    body.streamlit-dark-mode [data-testid="stMarkdownContainer"] p,
    body.streamlit-dark-mode [data-testid="stMarkdownContainer"] li,
    body.streamlit-dark-mode [data-testid="stMarkdownContainer"] span,
    body.streamlit-dark-mode [data-testid="stMarkdownContainer"] strong { 
        color: #fafafa !important; 
    }
    
    body.streamlit-dark-mode .stTabs [data-baseweb="tab"] { 
        color: #fafafa !important; 
    }
    
    body.streamlit-dark-mode [data-testid="metric-container"] label { 
        color: #c5d4a0 !important; 
    }
    
    body.streamlit-dark-mode [data-testid="metric-container"] [data-testid="stMetricValue"] { 
        color: #fafafa !important; 
    }
    
    body.streamlit-dark-mode [data-testid="stExpander"] summary p { 
        color: #fafafa !important; 
    }
    
    body.streamlit-dark-mode [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"]::before {
        color: #c5d4a0 !important;
    }
    
    body.streamlit-dark-mode .info-card,
    body.streamlit-dark-mode .feature-card,
    body.streamlit-dark-mode [data-testid="stExpander"],
    body.streamlit-dark-mode [data-testid="stExpander"] summary,
    body.streamlit-dark-mode [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    body.streamlit-dark-mode [data-testid="stVerticalBlockBorderWrapper"],
    body.streamlit-dark-mode [data-testid="metric-container"],
    body.streamlit-dark-mode .stTabs [data-baseweb="tab-list"],
    body.streamlit-dark-mode .stFileUploader > div {
        background-color: #262730 !important;
    }
    
    body.streamlit-dark-mode .stSelectbox > div > div,
    body.streamlit-dark-mode .stMultiselect > div > div,
    body.streamlit-dark-mode .stTextInput > div > div > input,
    body.streamlit-dark-mode .stNumberInput > div > div > input {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    
    body.streamlit-dark-mode .stCaption,
    body.streamlit-dark-mode [data-testid="stCaptionContainer"] {
        color: #c5d4a0 !important;
    }
    
    body.streamlit-dark-mode [data-baseweb="select"] [data-baseweb="menu"] {
        background-color: #262730 !important;
    }
    
    body.streamlit-dark-mode [data-baseweb="select"] [data-baseweb="menu"] li {
        color: #fafafa !important;
    }
    
    /* 
       ELEMENT YANG WARNANYA TETAP (tidak berubah di dark/light):
       - .main-header (background hijau tua, text cream)
       - .section-header (background terracotta, text cream)  
       - .section-header-green (background olive, text cream)
       - .stButton > button (background terracotta, text cream)
       - .stLinkButton a (background olive, text cream)
       - .stTabs [aria-selected="true"] (background terracotta, text cream)
    */
</style>
""", unsafe_allow_html=True)

# JavaScript untuk deteksi tema Streamlit secara dinamis
# Ini akan mendeteksi apakah user menggunakan dark mode atau light mode
st.markdown("""
<script>
(function() {
    function detectAndApplyTheme() {
        const root = document.documentElement;
        const stApp = document.querySelector('.stApp');
        
        // Check multiple indicators for dark mode
        let isDark = false;
        
        // Method 1: Check body background color
        const bodyBg = window.getComputedStyle(document.body).backgroundColor;
        if (bodyBg) {
            const rgb = bodyBg.match(/\\d+/g);
            if (rgb && rgb.length >= 3) {
                const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
                if (brightness < 128) isDark = true;
            }
        }
        
        // Method 2: Check stApp background
        if (stApp) {
            const appBg = window.getComputedStyle(stApp).backgroundColor;
            if (appBg) {
                const rgb = appBg.match(/\\d+/g);
                if (rgb && rgb.length >= 3) {
                    const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
                    if (brightness < 128) isDark = true;
                }
            }
        }
        
        // Method 3: Check for Streamlit's dark theme class or prefers-color-scheme
        if (document.body.classList.contains('dark') || 
            document.documentElement.getAttribute('data-theme') === 'dark' ||
            stApp?.getAttribute('data-theme') === 'dark' ||
            window.matchMedia('(prefers-color-scheme: dark)').matches) {
            // Only set isDark if not already determined by background check
            if (!isDark) {
                // Double check with background - Streamlit light theme has light background
                const mainBg = stApp ? window.getComputedStyle(stApp).backgroundColor : '';
                if (mainBg && mainBg !== 'rgba(0, 0, 0, 0)') {
                    const rgb = mainBg.match(/\\d+/g);
                    if (rgb && rgb.length >= 3) {
                        const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
                        isDark = brightness < 128;
                    }
                }
            }
        }
        
        // Apply theme with Streamlit dark mode colors
        if (isDark) {
            root.style.setProperty('--adaptive-text-primary', '#fafafa');
            root.style.setProperty('--adaptive-text-secondary', '#c5d4a0');
            root.style.setProperty('--adaptive-bg-card', '#262730');
            root.style.setProperty('--adaptive-bg-page', '#0e1117');
            root.style.setProperty('--adaptive-bg-input', '#262730');
            root.style.setProperty('--adaptive-tab-bg', '#262730');
            document.body.classList.add('streamlit-dark-mode');
            document.body.classList.remove('streamlit-light-mode');
        } else {
            root.style.setProperty('--adaptive-text-primary', '#283618');
            root.style.setProperty('--adaptive-text-secondary', '#606c38');
            root.style.setProperty('--adaptive-bg-card', '#fff');
            root.style.setProperty('--adaptive-bg-page', '#fefae0');
            root.style.setProperty('--adaptive-bg-input', '#fff');
            root.style.setProperty('--adaptive-tab-bg', '#fff');
            document.body.classList.add('streamlit-light-mode');
            document.body.classList.remove('streamlit-dark-mode');
        }
    }
    
    // Run on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', detectAndApplyTheme);
    } else {
        detectAndApplyTheme();
    }
    
    // Run periodically to catch theme changes
    setInterval(detectAndApplyTheme, 1000);
    
    // Also observe for DOM changes
    const observer = new MutationObserver(detectAndApplyTheme);
    observer.observe(document.body, { attributes: true, attributeFilter: ['class', 'data-theme'] });
})();
</script>
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