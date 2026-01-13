import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import plotly.express as px

def preprocessing_tab():
    st.markdown('<div class="section-header">🔧 Preprocessing & EDA</div>', unsafe_allow_html=True)
    st.caption("Exploratory Data Analysis, Feature Engineering dan Normalisasi Data")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Pilih dataset di tab Dataset terlebih dahulu.")
        return
    df = st.session_state['df']
    
    # ===================== EDA SECTION =====================
    st.markdown('<div class="section-header-green">🔍 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.info("Eksplorasi awal untuk memahami struktur, distribusi, dan kualitas data sebelum preprocessing.")
    
    # EDA Tabs
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["📋 Info Data", "📊 Statistik", "📈 Distribusi", "🔗 Korelasi"])
    
    with eda_tab1:
        st.markdown("##### 📋 Informasi Dataset")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("📁 Jumlah Baris", f"{len(df):,}")
        with col_info2:
            st.metric("📊 Jumlah Kolom", f"{len(df.columns)}")
        
        # Detail per kolom
        info_list = []
        for col in df.columns:
            non_missing = df[col].notna().sum()
            missing = df[col].isna().sum()
            missing_pct = (missing / len(df)) * 100
            dtype = str(df[col].dtype)
            unique_vals = df[col].nunique()
            info_list.append([col, dtype, non_missing, missing, f"{missing_pct:.1f}%", unique_vals])
        info_df = pd.DataFrame(info_list, columns=["Variabel", "Tipe Data", "Terisi", "Missing", "Missing %", "Unique"])
        st.dataframe(info_df, use_container_width=True)
        
        # Missing values summary
        total_missing = df.isna().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        st.info(f"📊 **Ringkasan Missing Values:** {total_missing:,} dari {total_cells:,} sel ({(total_missing/total_cells)*100:.2f}%)")
    
    with eda_tab2:
        st.markdown("##### 📊 Statistik Deskriptif")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            desc_stats = df[numeric_cols].describe().T
            desc_stats['range'] = desc_stats['max'] - desc_stats['min']
            desc_stats['cv'] = (desc_stats['std'] / desc_stats['mean']).abs() * 100
            st.dataframe(desc_stats.style.format("{:.2f}"), use_container_width=True)
            
            st.info("""
            **Keterangan:**
            - **CV (Coefficient of Variation)**: Variabilitas relatif terhadap mean (%). Nilai tinggi = data lebih tersebar
            - **Range**: Selisih antara nilai maksimum dan minimum
            """)
        else:
            st.info("Tidak ada kolom numerik untuk statistik deskriptif.")
    
    with eda_tab3:
        st.markdown("##### 📈 Distribusi Data")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Convert column names to string for display
            numeric_cols_str = [str(c) for c in numeric_cols]
            selected_col_str = st.selectbox("Pilih kolom untuk histogram:", numeric_cols_str)
            # Find the original column (could be int or str)
            selected_col = numeric_cols[numeric_cols_str.index(selected_col_str)]
            if selected_col is not None:
                # Create a copy with string column names for plotly
                df_plot = df.copy()
                df_plot.columns = [str(c) for c in df_plot.columns]
                
                fig = px.histogram(df_plot, x=str(selected_col), nbins=30, title=f"Distribusi {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                fig_box = px.box(df_plot, y=str(selected_col), title=f"Box Plot {selected_col}")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Tidak ada kolom numerik untuk visualisasi distribusi.")
    
    with eda_tab4:
        st.markdown("##### 🔗 Analisis Korelasi")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, title="Heatmap Korelasi", color_continuous_scale='RdBu_r', aspect='auto')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info("""
            **Interpretasi Korelasi:**
            - **Merah (positif)**: Kedua variabel bergerak searah
            - **Biru (negatif)**: Kedua variabel bergerak berlawanan
            - **Putih (~0)**: Tidak ada korelasi linier yang signifikan
            """)
        else:
            st.info("Diperlukan minimal 2 kolom numerik untuk analisis korelasi.")
    
    st.divider()
    
    # ===================== FEATURE ENGINEERING =====================
    selected_dataset = st.session_state.get('selected_dataset', '')
    
    if "kesehatan" in selected_dataset.lower():
        st.markdown("#### 🧬 Feature Engineering - Dataset Kesehatan")
        
        df.columns = df.columns.astype(str).str.strip().str.lower()
        year_cols = [c for c in df.columns if c.isdigit()]
        if not year_cols:
            st.error("Dataset kesehatan tidak memiliki kolom tahun numerik (misal '1990', '2000').")
            return
        df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=year_cols, how="all").reset_index(drop=True)
        if df.empty:
            st.error("Setelah cleaning, dataset kesehatan kosong.")
            return
        
        years = np.array(year_cols, dtype=int)
        df["value_1990"] = df[str(years.min())]
        df["value_2019"] = df[str(years.max())]
        df["mean_value"] = df[year_cols].mean(axis=1)
        df["std_value"] = df[year_cols].std(axis=1)
        
        def compute_trend(row):
            y = row[year_cols].astype(float).values
            if np.isnan(y).all() or len(y) < 2:
                return np.nan
            model = LinearRegression()
            model.fit(years.reshape(-1, 1), y)
            return model.coef_[0]
        df["trend"] = df.apply(compute_trend, axis=1)
        df["relative_change"] = (df["value_2019"] - df["value_1990"]) / df["value_1990"].replace(0, np.nan)
        
        features = ["country", "value_1990", "value_2019", "mean_value", "std_value", "trend", "relative_change"]
        df = df[features].dropna().reset_index(drop=True)
        if df.empty:
            st.error("Setelah FE, dataset kesehatan kosong.")
            return
        st.session_state['df_clean'] = df
        st.success("✅ Feature Engineering untuk dataset Kesehatan selesai.")
        
        with st.container(border=True):
            st.markdown("**📖 Penjelasan Fitur yang Digunakan**")
            st.markdown("""
            **📊 Fitur-fitur untuk Clustering Dataset Kesehatan:**
            
            | Fitur | Deskripsi | Interpretasi |
            |-------|-----------|--------------|
            | **value_1990** | Tingkat kematian anak tahun 1990 (per 1000 kelahiran) | Baseline/kondisi awal |
            | **value_2019** | Tingkat kematian anak tahun 2019 (per 1000 kelahiran) | Kondisi terkini |
            | **mean_value** | Rata-rata tingkat kematian selama 1990-2019 | Gambaran umum selama periode |
            | **std_value** | Standar deviasi tingkat kematian | Variabilitas/stabilitas data |
            | **trend** | Slope regresi linear dari waktu ke waktu | Negatif = membaik, Positif = memburuk |
            | **relative_change** | (value_2019 - value_1990) / value_1990 | Persentase perubahan dari baseline |
            
            **💡 Insight:**
            - Negara dengan **trend negatif** dan **relative_change negatif** menunjukkan perbaikan kesehatan anak
            - Nilai **mean_value tinggi** dengan **trend negatif** = progress signifikan dari kondisi buruk
            - **std_value tinggi** menunjukkan fluktuasi/ketidakstabilan selama periode
            """)
    
    elif "lingkungan" in selected_dataset.lower():
        st.markdown("#### 🧬 Feature Engineering - Dataset Lingkungan")
        
        df.columns = df.columns.str.strip().str.lower()
        if "country" not in df.columns:
            st.error("Dataset lingkungan harus memiliki kolom 'country'.")
            return
        num_cols = df.columns.drop("country")
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
        if "area_ha" not in df.columns:
            st.error("Dataset lingkungan harus memiliki kolom 'area_ha'.")
            return
        df = df.dropna(subset=["area_ha"]).reset_index(drop=True)
        if df.empty:
            st.error("Setelah cleaning, dataset lingkungan kosong.")
            return
        
        loss_cols = [c for c in df.columns if c.startswith("tc_loss_ha_")]
        if not loss_cols:
            st.error("Dataset lingkungan tidak memiliki kolom loss (misal 'tc_loss_ha_2001').")
            return
        
        years = np.arange(2001, 2001 + len(loss_cols))
        df["total_loss_ha"] = df[loss_cols].sum(axis=1)
        df["mean_annual_loss"] = df["total_loss_ha"] / len(loss_cols)
        df["loss_intensity"] = df["total_loss_ha"] / df["area_ha"]
        
        def loss_trend(row):
            y = row[loss_cols].astype(float).values
            if np.isnan(y).all() or len(y) < 2:
                return np.nan
            model = LinearRegression()
            model.fit(years.reshape(-1, 1), y)
            return model.coef_[0]
        df["loss_trend"] = df.apply(loss_trend, axis=1)
        
        if "gain_2000-2012_ha" in df.columns:
            df["net_forest_change"] = df["gain_2000-2012_ha"] - df["total_loss_ha"]
        
        features = ["country", "area_ha", "total_loss_ha", "mean_annual_loss", "loss_intensity", "loss_trend"]
        if "net_forest_change" in df.columns:
            features.append("net_forest_change")
        df = df[features].dropna().reset_index(drop=True)
        if df.empty:
            st.error("Setelah FE, dataset lingkungan kosong.")
            return
        st.session_state['df_clean'] = df
        st.success("✅ Feature Engineering untuk dataset Lingkungan selesai.")
        
        with st.container(border=True):
            st.markdown("**📖 Penjelasan Fitur yang Digunakan**")
            st.markdown("""
            **🌳 Fitur-fitur untuk Clustering Dataset Lingkungan:**
            
            | Fitur | Deskripsi | Interpretasi |
            |-------|-----------|--------------|
            | **area_ha** | Total area hutan dalam hektar | Luas wilayah hutan |
            | **total_loss_ha** | Total kehilangan hutan 2001-2022 (hektar) | Akumulasi deforestasi |
            | **mean_annual_loss** | Rata-rata kehilangan per tahun | Laju deforestasi tahunan |
            | **loss_intensity** | total_loss_ha / area_ha | Proporsi hutan yang hilang |
            | **loss_trend** | Slope regresi kehilangan tahunan | Positif = laju meningkat |
            | **net_forest_change** | Gain - Total Loss (jika tersedia) | Perubahan bersih tutupan |
            
            **💡 Insight:**
            - **loss_intensity tinggi** = deforestasi parah relatif terhadap luas hutan
            - **loss_trend positif** = laju deforestasi semakin cepat setiap tahun
            - **loss_trend negatif** = laju deforestasi melambat (membaik)
            - Negara dengan **area_ha besar** tapi **loss_intensity rendah** = konservasi baik
            """)
    
    st.divider()
    
    # ===================== PREPROCESSING OPTIONS =====================
    st.markdown("#### ⚙️ Opsi Preprocessing")
    st.info("Pilih fitur dan konfigurasi preprocessing untuk persiapan analisis clustering.")
    
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Info dataset
    st.markdown("##### 📋 Informasi Dataset (Post-FE)")
    info_list = []
    for col in df.columns:
        non_missing = df[col].notna().sum()
        missing = df[col].isna().sum()
        dtype = str(df[col].dtype)
        info_list.append([col, dtype, non_missing, missing])
    info_df = pd.DataFrame(info_list, columns=["Variabel", "Tipe Data", "Terisi", "Missing"])
    st.dataframe(info_df, use_container_width=True)
    
    # Statistik deskriptif
    st.markdown("##### 📊 Statistik Deskriptif (Post-FE)")
    st.dataframe(df.describe().T, use_container_width=True)
    
    st.divider()
    
    # Pilih fitur
    if 'select_all' not in st.session_state:
        st.session_state.select_all = False
    select_all = st.checkbox("Select All Numeric Features", value=st.session_state.select_all)
    if select_all:
        selected_features = all_numeric
        st.session_state.selected_features = all_numeric
    else:
        selected_features = st.multiselect("Fitur numerik untuk clustering", options=all_numeric, default=st.session_state.get('selected_features', []))
        st.session_state.selected_features = selected_features
    st.session_state.select_all = select_all
    
    if not selected_features:
        st.warning("⚠️ Pilih minimal satu fitur.")
        return
    
    # Preprocessing options
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        impute_strategy = st.selectbox("Strategi imputasi", ["median", "mean", "most_frequent"])
        drop_na = st.checkbox("Drop NA instead of impute")
    with col_opt2:
        scale_method = st.selectbox("Scaling", ["standard", "minmax", "none"])
        use_pca = st.checkbox("Gunakan PCA")
    
    pca_n = None
    if use_pca:
        pca_n = st.slider("Komponen PCA", 1, min(len(selected_features), 5), 2)
    
    if st.button("🚀 Proses Preprocessing", type="primary", use_container_width=True):
        X = df[selected_features].copy()
        if drop_na:
            X = X.dropna()
        else:
            imputer = SimpleImputer(strategy=impute_strategy)
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        if scale_method == "standard":
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            st.session_state['scaler'] = scaler
        elif scale_method == "minmax":
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            st.session_state['scaler'] = scaler
        else:
            st.session_state['scaler'] = None
        
        if pca_n:
            pca = PCA(n_components=pca_n)
            X = pd.DataFrame(pca.fit_transform(X), columns=[f"PC{i+1}" for i in range(pca_n)])
        
        st.session_state['X_proc'] = X
        st.session_state['selected_features'] = selected_features
        st.session_state['preprocessing_done'] = True  # Flag: preprocessing completed
        # Clear ML results when preprocessing changes
        for key in ['clustering_results', 'best_method', 'exploration_results', 'comparison_just_run', 'pred_result']:
            if key in st.session_state: del st.session_state[key]
        st.success("✅ Preprocessing selesai!")
    
    # Only show results if preprocessing was just done
    if st.session_state.get('preprocessing_done', False) and 'X_proc' in st.session_state:
        X = st.session_state['X_proc']
        
        # Tampilkan statistik setelah preprocessing
        st.markdown("##### ✅ Statistik Setelah Preprocessing")
        st.dataframe(X.describe().T, use_container_width=True)
        
        # Download data bersih
        st.markdown("##### 📥 Download Data Bersih")
        df_clean = st.session_state.get('df_clean', df)
        csv_clean = df_clean.to_csv(index=False)
        st.download_button(
            label="📥 Download Data Bersih (CSV)",
            data=csv_clean,
            file_name="data_bersih.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.divider()
    st.info("💡 **Langkah berikutnya:** Klik tab **Machine Learning** untuk melanjutkan analisis clustering.")
