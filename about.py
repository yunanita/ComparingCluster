import streamlit as st
import pandas as pd
import os

# Gunakan relative path - folder yang sama dengan script ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_HEALTH = os.path.join(BASE_DIR, "child_mortality.xlsx")
PATH_ENV = os.path.join(BASE_DIR, "deforestasi.xlsx")

@st.cache_data
def load_preview_health():
    return pd.read_excel(PATH_HEALTH).head(5)

@st.cache_data
def load_preview_env():
    return pd.read_excel(PATH_ENV).head(5)

def about_dataset():
    st.markdown('<div class="section-header">📊 Pilih Dataset untuk Analisis</div>', unsafe_allow_html=True)
    st.caption("Pilih salah satu dataset atau upload dataset custom Anda")
    
    # =====================================================================
    # DATASET CARDS - 2 KOLOM
    # =====================================================================
    col_health, col_env = st.columns(2, gap="large")
    
    # ===================== DATASET KESEHATAN =====================
    with col_health:
        st.markdown("### 🏥 Dataset Kesehatan")
        st.caption("Child Mortality Data")
        
        # Display NutriData logo jika ada
        nutridata_path = os.path.join(BASE_DIR, "NutriData.png")
        if os.path.exists(nutridata_path):
            st.image(nutridata_path, width=150)
        
        with st.container(border=True):
            st.markdown("**📋 Informasi Dataset**")
            
            st.markdown("**📍 Sumber Data:**")
            st.markdown("""
            **NutriData - Global Health Database**  
            Dataset dikumpulkan dari WHO, UNICEF, dan World Bank. 
            Data mencakup statistik mortalitas anak dari 195+ negara (1990-2023).
            """)
            
            st.markdown("**📊 Deskripsi:**")
            st.markdown("""
            Dataset berisi data tingkat kematian anak di bawah 5 tahun (Under-5 Mortality Rate) 
            per 1000 kelahiran hidup. Digunakan untuk menganalisis pola kesehatan anak.
            """)
        
        with st.container(border=True):
            st.markdown("**🔍 Penjelasan Fitur:**")
            st.markdown("""
            - **code**: Kode negara (ISO 3166-1 alpha-3)
            - **Country**: Nama negara
            - **1990, 1991, ..., 2019**: Tingkat kematian anak (per 1000 kelahiran hidup) untuk setiap tahun
            """)
        
        st.markdown("**📋 Preview Data:**")
        try:
            df_health = load_preview_health()
            st.dataframe(df_health, height=150, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat: {e}")
        
        st.link_button("🔗 Lihat Sumber Data", "https://aschimmenti.github.io/NutriData/metadata.html")
        
        if st.button("✅ Pilih Dataset Kesehatan", key="btn_health", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.session_state['selected_dataset'] = "kesehatan"
            try:
                df = pd.read_excel(PATH_HEALTH)
                st.session_state['df'] = df
                st.session_state['df_clean'] = df.copy()
                # Clear all previous results and flags
                for key in ['X_proc', 'clustering_results', 'best_method', 'exploration_results', 'selected_features', 'select_all', 'scaler', 'model_bundle', 'preprocessing_done', 'comparison_just_run', 'pred_result']:
                    if key in st.session_state: del st.session_state[key]
                st.success("✅ Dataset Kesehatan berhasil dipilih!")
            except Exception as e:
                st.error(f"❌ Gagal: {e}")
    
    # ===================== DATASET LINGKUNGAN =====================
    with col_env:
        st.markdown("### 🌲 Dataset Lingkungan")
        st.caption("Deforestation Data")
        
        # Display GFW (Global Forest Watch) logo jika ada
        gfw_path = os.path.join(BASE_DIR, "GFW.png")
        if os.path.exists(gfw_path):
            st.image(gfw_path, width=150)
        
        with st.container(border=True):
            st.markdown("**📋 Informasi Dataset**")
            
            st.markdown("**📍 Sumber Data:**")
            st.markdown("""
            **Global Forest Watch - World Resources Institute**  
            Data dari sistem pemantauan satelit global yang melacak perubahan 
            tutupan hutan secara real-time dari NASA dan University of Maryland.
            """)
            
            st.markdown("**📊 Deskripsi:**")
            st.markdown("""
            Dataset berisi data tingkat deforestasi dari berbagai negara. 
            Mencakup luas hutan yang hilang, persentase tutupan hutan, dan tren perubahan.
            """)
        
        with st.container(border=True):
            st.markdown("**🔍 Penjelasan Fitur:**")
            st.markdown("""
            - **country**: Nama negara
            - **area_ha**: Total area dalam hektar
            - **extent_2000_ha**: Luas tutupan hutan tahun 2000 (ha)
            - **extent_2010_ha**: Luas tutupan hutan tahun 2010 (ha)
            - **gain_2000-2020_ha**: Pertambahan hutan 2000-2020 (ha)
            - **tc_loss_ha_2001 s/d tc_loss_ha_2022**: Kehilangan hutan per tahun (ha)
            """)
        
        st.markdown("**📋 Preview Data:**")
        try:
            df_env = load_preview_env()
            st.dataframe(df_env, height=150, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat: {e}")
        
        st.link_button("🔗 Lihat Sumber Data", "https://www.globalforestwatch.org/dashboards/global/")
        
        if st.button("✅ Pilih Dataset Lingkungan", key="btn_env", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.session_state['selected_dataset'] = "lingkungan"
            try:
                df = pd.read_excel(PATH_ENV)
                st.session_state['df'] = df
                st.session_state['df_clean'] = df.copy()
                # Clear all previous results and flags
                for key in ['X_proc', 'clustering_results', 'best_method', 'exploration_results', 'selected_features', 'select_all', 'scaler', 'model_bundle', 'preprocessing_done', 'comparison_just_run', 'pred_result']:
                    if key in st.session_state: del st.session_state[key]
                st.success("✅ Dataset Lingkungan berhasil dipilih!")
            except Exception as e:
                st.error(f"❌ Gagal: {e}")
    
    # =====================================================================
    # CURRENT DATASET STATUS
    # =====================================================================
    st.divider()
    
    current_dataset = st.session_state.get('selected_dataset', 'Belum dipilih')
    status_icon = "🏥" if current_dataset == "kesehatan" else "🌲" if current_dataset == "lingkungan" else "❓"
    
    col_status1, col_status2, col_status3 = st.columns([1,2,1])
    with col_status2:
        st.info(f"{status_icon} **Dataset Aktif:** {current_dataset.title()}")
    
    # =====================================================================
    # UPLOAD CUSTOM DATASET
    # =====================================================================
    st.divider()
    
    st.markdown("#### 📁 Upload Dataset Custom")
    
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if st.button("📤 Gunakan Dataset Upload", use_container_width=True, type="primary"):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['df'] = df
                st.session_state['df_clean'] = df.copy()
                st.session_state['selected_dataset'] = f"Upload ({uploaded_file.name})"
                for key in ['X_proc', 'clustering_results', 'best_method', 'exploration_results', 'selected_features', 'select_all', 'scaler', 'model_bundle']:
                    if key in st.session_state: del st.session_state[key]
                st.success(f"✅ Dataset '{uploaded_file.name}' berhasil di-upload!")
            except Exception as e:
                st.error(f"❌ Gagal upload: {e}")
