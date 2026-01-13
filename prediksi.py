"""
Prediksi Tab - Predict/Assign Cluster untuk Data Baru
=====================================================
Fitur:
- Auto-select model terbaik dari hasil clustering
- Simplified input: hanya input nilai utama
- Feature engineering otomatis (hidden from user)
- Native OOS explanation yang jelas
- Hanya tersedia untuk dataset Kesehatan dan Lingkungan (bukan custom upload)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import pickle
from typing import Any, Dict, List, Optional, Tuple

import plotly.express as px


# ---------------------------------------------------------------------
# Feature Engineering Functions (hidden from user)
# ---------------------------------------------------------------------
def _compute_fe_kesehatan(value_awal: float, value_akhir: float) -> Dict[str, float]:
    """
    Hitung fitur turunan untuk dataset Kesehatan.
    User hanya input: value_1990 dan value_2019
    Sisanya dihitung otomatis dengan asumsi linear progression.
    """
    mean_value = (value_awal + value_akhir) / 2
    std_value = abs(value_akhir - value_awal) / 4
    trend = (value_akhir - value_awal) / 29.0
    
    if value_awal != 0:
        relative_change = (value_akhir - value_awal) / value_awal
    else:
        relative_change = 0.0
    
    return {
        "value_1990": value_awal,
        "value_2019": value_akhir,
        "mean_value": mean_value,
        "std_value": std_value,
        "trend": trend,
        "relative_change": relative_change,
    }


def _compute_fe_lingkungan(area_ha: float, total_loss_ha: float, n_years: int = 22) -> Dict[str, float]:
    """
    Hitung fitur turunan untuk dataset Lingkungan.
    User hanya input: area_ha dan total_loss_ha
    Sisanya dihitung otomatis.
    """
    mean_annual_loss = total_loss_ha / n_years if n_years > 0 else 0.0
    loss_intensity = total_loss_ha / area_ha if area_ha > 0 else 0.0
    loss_trend = mean_annual_loss * 0.01
    
    return {
        "area_ha": area_ha,
        "total_loss_ha": total_loss_ha,
        "mean_annual_loss": mean_annual_loss,
        "loss_intensity": loss_intensity,
        "loss_trend": loss_trend,
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _find_country_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in ["country", "negara", "country.name", "name"]):
            return col
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _load_pickle_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def _available_model_files() -> List[str]:
    candidates = ["model_kesehatan.pkl", "model_lingkungan.pkl", "model.pkl", "model_clustering.pkl"]
    return [p for p in candidates if os.path.exists(p)]


def _bundle_from_session(method_name: str) -> Optional[Dict[str, Any]]:
    """
    Convert exploration_results entry to a 'deployment-like' bundle on the fly.
    """
    results = st.session_state.get("exploration_results", {})
    if method_name not in results:
        return None
    r = results[method_name]
    if not r.get("success"):
        return None

    bundle = {
        "bundle_version": 2,
        "created_at": None,
        "dataset": st.session_state.get("selected_dataset", "Unknown"),
        "method": method_name,
        "model": r.get("model"),
        "input_columns": r.get("input_columns") or (list(st.session_state["X_proc"].columns) if "X_proc" in st.session_state else None),
        "selected_features": st.session_state.get("selected_features", []),
        "scaler": st.session_state.get("scaler", None),
        "metrics": {
            "n_clusters": r.get("n_clusters"),
            "noise_ratio": r.get("noise_ratio"),
            "silhouette": r.get("silhouette"),
            "davies_bouldin": r.get("davies_bouldin"),
            "calinski_harabasz": r.get("calinski_harabasz"),
        },
        "risk_feature": r.get("risk_feature"),
        "risk_mapping": r.get("risk_mapping", {}),
        "supports_predict": bool(r.get("supports_predict")),
        "native_predict": bool(r.get("native_predict", r.get("supports_predict", False))),
        "centroids": r.get("centroids", {}),
        "radii": r.get("radii", {}),
        "notes": {},
    }
    return bundle


def _detect_dataset_type(bundle: Dict[str, Any]) -> str:
    """Detect dataset type from bundle."""
    dataset = str(bundle.get("dataset", "")).lower()
    if "kesehatan" in dataset or "health" in dataset:
        return "kesehatan"
    elif "lingkungan" in dataset or "environment" in dataset or "deforest" in dataset:
        return "lingkungan"
    
    # Detect from input columns
    input_cols = bundle.get("input_columns", [])
    if input_cols:
        col_str = " ".join([str(c).lower() for c in input_cols])
        if "value_1990" in col_str or "value_2019" in col_str:
            return "kesehatan"
        elif "area_ha" in col_str or "loss" in col_str:
            return "lingkungan"
    
    return "unknown"


def _transform_input(bundle: Dict[str, Any], input_df: pd.DataFrame) -> np.ndarray:
    """
    Transform raw input_df into the same feature space used during training.
    """
    input_cols = bundle.get("input_columns") or []
    scaler = bundle.get("scaler", None)
    
    if not input_cols:
        raise ValueError("Bundle tidak memiliki input_columns yang valid.")
    
    # Ensure all columns exist
    missing = [c for c in input_cols if c not in input_df.columns]
    if missing:
        raise ValueError(f"Kolom yang hilang: {missing}")
    
    Xraw = input_df[input_cols].to_numpy(dtype=float)
    
    if scaler is not None:
        try:
            return scaler.transform(Xraw)
        except:
            pass  # If scaler fails, return raw
    
    return Xraw


def _predict_cluster(bundle: Dict[str, Any], X_in: np.ndarray, allow_approx: bool = True) -> Tuple[int, Dict[str, Any]]:
    """
    Predict cluster label for a single sample X_in shape (1, n_features).
    Returns (cluster_label, extra_info).
    """
    model = bundle.get("model", None)
    if model is None:
        raise ValueError("Model tidak tersedia di bundle.")

    # 1) Native predict if supported
    if callable(getattr(model, "predict", None)):
        try:
            lab = int(model.predict(X_in)[0])
            extra: Dict[str, Any] = {"mode": "native_predict"}

            # For GMM, add probabilities if available
            if callable(getattr(model, "predict_proba", None)):
                try:
                    proba = model.predict_proba(X_in)[0]
                    extra["membership_proba"] = [float(p) for p in proba]
                except Exception:
                    pass

            return lab, extra
        except:
            pass  # Fall through to approximate

    # 2) Approximate assignment via nearest centroid
    if allow_approx:
        centroids = bundle.get("centroids", {}) or {}
        radii = bundle.get("radii", {}) or {}
        if not centroids:
            raise ValueError("Centroid tidak tersedia untuk approximate assignment.")

        x = X_in.reshape(-1)
        best_lab = None
        best_dist = None

        for k, c in centroids.items():
            c_vec = np.asarray(c, dtype=float)
            dist = float(np.linalg.norm(x - c_vec))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_lab = int(k)

        if best_dist is None or best_lab is None:
            raise ValueError("Gagal menghitung nearest centroid.")
        
        # Outlier check: distance must be <= radius (quantile) for that cluster, else -1
        thr = float(radii.get(best_lab, np.inf))
        if best_dist > thr:
            return -1, {"mode": "nearest_centroid", "distance": best_dist, "radius_thr": thr, "note": "classified as outlier (-1)"}

        return best_lab, {"mode": "nearest_centroid", "distance": best_dist, "radius_thr": thr}

    raise ValueError("Model tidak mendukung predict. Aktifkan mode approximate jika ingin tetap assign cluster.")


def _risk_label(bundle: Dict[str, Any], cluster_id: int, dataset_type: str = "kesehatan") -> Tuple[str, str]:
    """
    Get risk label for cluster based on ACTUAL cluster characteristics (from risk_mapping).
    Label ditentukan berdasarkan nilai rata-rata fitur di setiap cluster, BUKAN nomor cluster.
    Returns (label, emoji)
    """
    # Gunakan risk_mapping dari bundle jika tersedia (ini sudah di-sort berdasarkan nilai)
    risk_mapping = bundle.get("risk_mapping", {})
    
    if risk_mapping and cluster_id in risk_mapping:
        risk_level = risk_mapping[cluster_id]
        
        if dataset_type == "kesehatan" or "health" in dataset_type.lower():
            if risk_level == "Rendah":
                return "Low-Burden (Risiko Rendah)", "🔵"
            elif risk_level == "Sedang":
                return "Medium-Burden (Risiko Sedang)", "🟢"
            elif risk_level == "Tinggi":
                return "High-Burden (Risiko Tinggi)", "🟡"
            else:
                return f"Cluster {cluster_id} ({risk_level})", "⚪"
        elif dataset_type == "lingkungan" or "deforest" in dataset_type.lower() or "environment" in dataset_type.lower():
            if risk_level == "Rendah":
                return "Low Deforestation Zone (Konservasi Baik)", "🔵"
            elif risk_level == "Sedang":
                return "Moderate Deforestation", "🟢"
            elif risk_level == "Tinggi":
                return "Deforestation Hotspot (Deforestasi Tinggi)", "🟡"
            else:
                return f"Cluster {cluster_id} ({risk_level})", "⚪"
        else:
            if risk_level == "Rendah":
                return f"Cluster {cluster_id} (Kategori Rendah)", "🔵"
            elif risk_level == "Sedang":
                return f"Cluster {cluster_id} (Kategori Sedang)", "🟢"
            elif risk_level == "Tinggi":
                return f"Cluster {cluster_id} (Kategori Tinggi)", "🟡"
            else:
                return f"Cluster {cluster_id}", "⚪"
    
    # Fallback: jika risk_mapping tidak tersedia, gunakan heuristic berdasarkan centroids
    centroids = bundle.get("centroids", {})
    if centroids and len(centroids) >= 2:
        # Hitung rata-rata nilai per centroid
        centroid_avgs = {cid: np.mean(cvec) for cid, cvec in centroids.items() if cid != -1}
        if centroid_avgs:
            sorted_clusters = sorted(centroid_avgs.items(), key=lambda x: x[1])
            n_clusters = len(sorted_clusters)
            
            # Temukan rank cluster_id
            rank = -1
            for i, (cid, _) in enumerate(sorted_clusters):
                if int(cid) == int(cluster_id):
                    rank = i
                    break
            
            if rank >= 0:
                if dataset_type == "kesehatan" or "health" in dataset_type.lower():
                    if n_clusters == 2:
                        if rank == 0:
                            return "Low-Burden (Risiko Rendah)", "🔵"
                        else:
                            return "High-Burden (Risiko Tinggi)", "🟡"
                    else:
                        if rank < n_clusters / 3:
                            return "Low-Burden (Risiko Rendah)", "🔵"
                        elif rank < 2 * n_clusters / 3:
                            return "Medium-Burden (Risiko Sedang)", "🟢"
                        else:
                            return "High-Burden (Risiko Tinggi)", "🟡"
                elif dataset_type == "lingkungan" or "deforest" in dataset_type.lower():
                    if n_clusters == 2:
                        if rank == 0:
                            return "Low Deforestation Zone (Konservasi Baik)", "🔵"
                        else:
                            return "Deforestation Hotspot (Deforestasi Tinggi)", "🟡"
                    else:
                        if rank < n_clusters / 3:
                            return "Low Deforestation Zone", "🔵"
                        elif rank < 2 * n_clusters / 3:
                            return "Moderate Deforestation", "🟢"
                        else:
                            return "Deforestation Hotspot", "🟡"
    
    # Ultimate fallback
    if cluster_id == -1:
        return "Outlier", "⚫"
    return f"Cluster {cluster_id}", "⚪"


# ---------------------------------------------------------------------
# Streamlit tab
# ---------------------------------------------------------------------
def prediksi_tab():
    # Simple title
    st.markdown('<div class="section-header">🔮 Prediksi Cluster</div>', unsafe_allow_html=True)
    st.caption("Prediksi cluster untuk data baru berdasarkan model yang sudah dilatih")

    # ==================== VALIDATION: Dataset Type Check ====================
    selected_dataset = st.session_state.get("selected_dataset", "").lower()
    
    # Check if this is a custom upload (not kesehatan or lingkungan)
    is_custom_upload = "kesehatan" not in selected_dataset and "lingkungan" not in selected_dataset and "health" not in selected_dataset and "deforest" not in selected_dataset
    
    if is_custom_upload and selected_dataset != "":
        st.warning("""
        **Fitur Prediksi Tidak Tersedia untuk Dataset Custom**
        
        Fitur prediksi hanya tersedia untuk dataset bawaan (Kesehatan atau Lingkungan).
        
        Alasan: Dataset custom tidak memiliki struktur fitur yang dapat diprediksi secara otomatis.
        Silakan gunakan salah satu dataset bawaan jika ingin menggunakan fitur prediksi.
        """)
        return

    # ==================== VALIDATION ====================
    if "X_proc" not in st.session_state:
        st.warning("Jalankan preprocessing terlebih dahulu di tab Preprocessing.")
        return
    if "exploration_results" not in st.session_state:
        st.warning("Jalankan clustering di tab Machine Learning terlebih dahulu.")
        return

    df_base = st.session_state.get("df_clean", None)
    df_raw = st.session_state.get("df", None)
    if df_base is None:
        df_base = df_raw

    # ==================== MODEL INFO ====================
    results = st.session_state.get("exploration_results", {})
    ok_methods = [m for m, r in results.items() if isinstance(r, dict) and r.get("success")]
    
    if not ok_methods:
        st.error("Tidak ada hasil clustering yang valid.")
        return
    
    best_method = st.session_state.get("best_method")
    if best_method not in ok_methods:
        best_method = ok_methods[0]
    
    bundle = _bundle_from_session(best_method)
    
    if bundle is None:
        st.error("Bundle model tidak tersedia.")
        return

    dataset_type = _detect_dataset_type(bundle)
    
    # Final check: ensure dataset type is valid
    if dataset_type == "unknown":
        st.warning("""
        **Fitur Prediksi Tidak Tersedia**
        
        Tipe dataset tidak dikenali. Fitur prediksi hanya tersedia untuk:
        - Dataset Kesehatan (Child Mortality)
        - Dataset Lingkungan (Deforestasi)
        """)
        return

    # Model Info Section
    with st.container(border=True):
        st.markdown("##### 🤖 Model yang Digunakan")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algoritma", str(bundle.get("method")))
        with col2:
            st.metric("Clusters", str(bundle.get("metrics", {}).get("n_clusters", "N/A")))
        with col3:
            sil = bundle.get('metrics', {}).get('silhouette')
            st.metric("Silhouette", f"{sil:.3f}" if isinstance(sil, (int, float)) else "N/A")

        with st.container(border=True):
            st.markdown("**🔄 Ganti Model**")
            selected_method = st.selectbox("Pilih model:", ok_methods, 
                                           index=ok_methods.index(best_method) if best_method in ok_methods else 0,
                                           key="pred_model_select")
            if selected_method != best_method:
                bundle = _bundle_from_session(selected_method)
                dataset_type = _detect_dataset_type(bundle)
                st.info(f"Model diubah ke: **{selected_method}**")
        
        with st.container(border=True):
            st.markdown("**📊 Karakteristik Cluster (Centroid)**")
            centroids = bundle.get("centroids", {})
            risk_mapping = bundle.get("risk_mapping", {})
            input_cols = bundle.get("input_columns", [])
            
            if centroids and input_cols:
                centroid_data = []
                for cid, cvec in centroids.items():
                    row = {"Cluster": int(cid)}
                    if isinstance(cvec, (list, np.ndarray)):
                        for i, col in enumerate(input_cols):
                            if i < len(cvec):
                                row[col] = f"{cvec[i]:.4f}"
                    risk = risk_mapping.get(int(cid), "N/A")
                    row["Risk Label"] = risk
                    centroid_data.append(row)
                
                if centroid_data:
                    centroid_df = pd.DataFrame(centroid_data)
                    st.dataframe(centroid_df, use_container_width=True, hide_index=True)
                    
                    st.caption("""
                    **Penjelasan:**
                    - Centroid adalah titik pusat dari setiap cluster dalam ruang fitur yang sudah di-scale
                    - Prediksi dilakukan dengan mengukur jarak data baru ke setiap centroid
                    - Data akan masuk ke cluster dengan centroid terdekat
                    """)
            else:
                st.info("Centroid tidak tersedia untuk model ini.")

    input_cols = bundle.get("input_columns") or []

    # ==================== INPUT PREDIKSI ====================
    with st.container(border=True):
        st.markdown("##### 📝 Input Prediksi")
        
        # Initialize session state for prediction results
        if "pred_result" not in st.session_state:
            st.session_state.pred_result = None
        
        country_col = _find_country_column(df_raw) if df_raw is not None else None
        country_col_base = _find_country_column(df_base) if df_base is not None else None
        
        if not country_col or df_raw is None:
            st.warning("Kolom negara tidak ditemukan di dataset.")
            return
        
        country_list = sorted(df_raw[country_col].dropna().astype(str).unique().tolist())
        
        if dataset_type == "kesehatan":
            st.info("""
            **Petunjuk Input - Dataset Kesehatan**
            
            1. Pilih negara dari daftar
            2. Masukkan tahun prediksi (untuk referensi)
            3. Masukkan nilai mortality yang ingin diprediksi (per 1000 kelahiran hidup)
            
            Sistem akan otomatis menghitung fitur-fitur turunan berdasarkan data baseline negara tersebut.
            """)
            
            # Use form to allow multiple predictions
            with st.form(key="kesehatan_pred_form"):
                selected_country = st.selectbox("Pilih Negara:", country_list, key="pred_country_kes")
                
                col1, col2 = st.columns(2)
                with col1:
                    pred_year = st.number_input("Tahun Prediksi:", 
                                                min_value=1900, max_value=2100, value=2023, step=1,
                                                help="Tahun untuk prediksi cluster")
                with col2:
                    pred_mortality = st.number_input("Nilai Mortality:", 
                                                     min_value=0.0, value=50.0, step=0.1,
                                                     help="Tingkat kematian anak per 1000 kelahiran hidup")
                
                submit_btn = st.form_submit_button("Prediksi Cluster", type="primary", use_container_width=True)
            
            if submit_btn:
                try:
                    # Get baseline value_1990
                    if df_base is not None and country_col_base:
                        base_data = df_base[df_base[country_col_base].astype(str) == selected_country]
                        if not base_data.empty:
                            value_1990 = _safe_float(base_data.iloc[0].get("value_1990", 100))
                        else:
                            value_1990 = 100.0
                    else:
                        value_1990 = 100.0
                    
                    # Compute features (hidden from user)
                    computed = _compute_fe_kesehatan(value_1990, pred_mortality)
                    
                    # Build final feature dict
                    final_vals = {}
                    for col in input_cols:
                        if col in computed:
                            final_vals[col] = computed[col]
                        else:
                            final_vals[col] = 0.0
                    
                    input_df = pd.DataFrame([final_vals])
                    X_in = _transform_input(bundle, input_df)
                    
                    cluster_id, extra = _predict_cluster(bundle, X_in, allow_approx=True)
                    risk = _risk_label(bundle, cluster_id, dataset_type)
                    
                    # Store result in session state
                    st.session_state.pred_result = {
                        "success": True,
                        "country": selected_country,
                        "year": pred_year,
                        "mortality": pred_mortality,
                        "cluster_id": cluster_id,
                        "risk": risk,
                        "final_vals": final_vals,
                        "extra": extra,
                        "X_in": X_in,
                        "dataset_type": dataset_type
                    }
                except Exception as e:
                    st.session_state.pred_result = {"success": False, "error": str(e)}
                    
        elif dataset_type == "lingkungan":
            st.info("""
            **Petunjuk Input - Dataset Lingkungan**
            
            1. Pilih negara dari daftar (area hutan akan diambil dari data negara tersebut)
            2. Masukkan total kehilangan hutan (dalam hektar)
            
            Sistem akan menghitung intensitas deforestasi dan fitur turunan lainnya secara otomatis.
            """)
            
            # Pilih negara di LUAR form agar bisa dinamis update
            selected_country = st.selectbox("Pilih Negara:", country_list, key="pred_country_ling_select")
            
            # Get area from baseline berdasarkan negara yang dipilih
            default_area = 1000000.0
            if df_base is not None and country_col_base and selected_country:
                base_data = df_base[df_base[country_col_base].astype(str) == selected_country]
                if not base_data.empty:
                    default_area = _safe_float(base_data.iloc[0].get("area_ha", 1000000))
            
            # Tampilkan info area hutan dari data
            st.success(f"📍 **{selected_country}** - Luas Total Hutan: **{default_area:,.0f} ha** (dari dataset)")
            
            with st.form(key="lingkungan_pred_form"):
                # Simpan area dari baseline sebagai hidden variable
                pred_area = default_area
                
                pred_loss = st.number_input("Total Kehilangan Hutan (ha):", 
                                            min_value=0.0, value=50000.0, step=1000.0,
                                            help="Total kehilangan hutan dalam hektar yang ingin diprediksi")
                
                submit_btn = st.form_submit_button("Prediksi Cluster", type="primary", use_container_width=True)
            
            if submit_btn:
                try:
                    # Use user input area instead of just baseline
                    area_ha = pred_area
                    
                    # Compute features (hidden from user)
                    computed = _compute_fe_lingkungan(area_ha, pred_loss)
                    
                    final_vals = {}
                    for col in input_cols:
                        if col in computed:
                            final_vals[col] = computed[col]
                        else:
                            final_vals[col] = 0.0
                    
                    input_df = pd.DataFrame([final_vals])
                    X_in = _transform_input(bundle, input_df)
                    
                    cluster_id, extra = _predict_cluster(bundle, X_in, allow_approx=True)
                    risk = _risk_label(bundle, cluster_id, dataset_type)
                    
                    st.session_state.pred_result = {
                        "success": True,
                        "country": selected_country,
                        "area_ha": pred_area,
                        "total_loss": pred_loss,
                        "cluster_id": cluster_id,
                        "risk": risk,
                        "final_vals": final_vals,
                        "extra": extra,
                        "X_in": X_in,
                        "dataset_type": dataset_type
                    }
                except Exception as e:
                    st.session_state.pred_result = {"success": False, "error": str(e)}
        else:
            st.warning("Tipe dataset tidak dikenali.")
            return

    # ==================== DISPLAY RESULT ====================
    if st.session_state.get("pred_result") is not None:
        result = st.session_state.pred_result
        
        if not result.get("success", False):
            st.error(f"Gagal prediksi: {result.get('error', 'Unknown error')}")
        else:
            with st.container(border=True):
                st.markdown("##### ✅ Hasil Prediksi")
                
                cluster_id = result["cluster_id"]
                # Handle both old format (string) and new format (tuple)
                risk_val = result.get("risk", ("Unknown", "⚪"))
                if isinstance(risk_val, tuple) and len(risk_val) == 2:
                    risk_label, risk_emoji = risk_val
                elif isinstance(risk_val, str):
                    risk_label = risk_val
                    risk_emoji = "⚪"
                else:
                    risk_label = "Unknown"
                    risk_emoji = "⚪"
                country = result["country"]
                final_vals = result["final_vals"]
                X_in = result["X_in"]
                res_dataset_type = result["dataset_type"]
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Negara", country)
                with c2:
                    st.metric("Cluster", int(cluster_id))
                with c3:
                    st.metric("Kategori", f"{risk_emoji} {risk_label}")
                
                if res_dataset_type == "kesehatan":
                    st.info(f"**Tahun Prediksi:** {result.get('year')} | **Mortality:** {result.get('mortality'):.2f}")
                elif res_dataset_type == "lingkungan":
                    area = result.get('area_ha', 0)
                    loss = result.get('total_loss', 0)
                    intensity = (loss / area * 100) if area > 0 else 0
                    st.info(f"**Area Hutan:** {area:,.0f} ha | **Total Loss:** {loss:,.0f} ha | **Intensitas:** {intensity:.2f}%")
                
                # ==================== INTERPRETASI HASIL ====================
                st.markdown("---")
                st.markdown("##### 📖 Interpretasi Hasil Prediksi")
                
                if cluster_id == -1:
                    st.warning("""
                    **Hasil: Data teridentifikasi sebagai OUTLIER (Noise)**
                    
                    Data yang Anda masukkan memiliki karakteristik yang berbeda secara signifikan dari 
                    pola-pola yang ditemukan dalam data training. Ini bisa berarti:
                    - Nilai input sangat ekstrem dibandingkan data historis
                    - Kombinasi nilai yang tidak umum ditemukan dalam dataset
                    
                    **Rekomendasi:** Periksa kembali nilai input atau pertimbangkan data ini sebagai kasus khusus.
                    """)
                else:
                    # Generate interpretation based on dataset type and cluster label
                    # Ambil statistik cluster untuk insight yang lebih kaya
                    centroids = bundle.get("centroids", {})
                    cluster_centroid = centroids.get(cluster_id, []) if centroids else []
                    
                    if res_dataset_type == "kesehatan":
                        mortality_val = result.get('mortality', 0)
                        
                        # Hitung perbandingan dengan cluster lain
                        all_cluster_means = {}
                        if centroids:
                            for cid, cvec in centroids.items():
                                if isinstance(cvec, (list, np.ndarray)) and len(cvec) > 0:
                                    all_cluster_means[cid] = np.mean(cvec)
                        
                        avg_all = np.mean(list(all_cluster_means.values())) if all_cluster_means else mortality_val
                        diff_from_avg = ((mortality_val - avg_all) / avg_all * 100) if avg_all > 0 else 0
                        
                        if "High-Burden" in risk_label:
                            interpretation = f"""
                            **🟡 Hasil: Cluster {cluster_id} - {risk_label}**
                            
                            **📊 Analisis Data:**
                            - Negara: **{country}**
                            - Tingkat Mortality: **{mortality_val:.2f} per 1000 kelahiran hidup**
                            - Perbandingan dengan rata-rata cluster: **{abs(diff_from_avg):.1f}% {'di atas' if diff_from_avg > 0 else 'di bawah'} rata-rata**
                            
                            **🔍 Insight Berdasarkan Data:**
                            - Dengan mortality {mortality_val:.2f}, negara ini berada di **kelompok risiko tinggi**
                            - Cluster ini umumnya berisi negara-negara dengan tantangan kesehatan struktural
                            - Angka mortality tinggi mengindikasikan perlunya penguatan sistem kesehatan dasar
                            
                            **💡 Rekomendasi Berbasis Data:**
                            
                            Negara dalam cluster ini menunjukkan angka kematian anak yang tinggi, yang umumnya disebabkan oleh kombinasi faktor seperti keterbatasan akses layanan kesehatan primer, infrastruktur yang belum memadai, serta tantangan sosial-ekonomi. Negara-negara di cluster High-Burden cenderung menghadapi tantangan struktural seperti keterbatasan fasilitas kesehatan, tingkat kemiskinan tinggi, konflik, atau ketidakstabilan politik yang menghambat layanan kesehatan dasar. Berdasarkan pola data, prioritas intervensi sebaiknya difokuskan pada penguatan sistem kesehatan dasar di tingkat komunitas, peningkatan akses terhadap layanan prenatal dan postnatal, serta perbaikan kondisi sanitasi dan nutrisi. Negara-negara di cluster ini dapat belajar dari keberhasilan negara di cluster Low-Burden yang telah berhasil menurunkan angka mortalitas melalui pendekatan kesehatan masyarakat yang komprehensif.
                            """
                        else:  # Low-Burden
                            interpretation = f"""
                            **🔵 Hasil: Cluster {cluster_id} - {risk_label}**
                            
                            **📊 Analisis Data:**
                            - Negara: **{country}**
                            - Tingkat Mortality: **{mortality_val:.2f} per 1000 kelahiran hidup**
                            - Perbandingan dengan rata-rata cluster: **{abs(diff_from_avg):.1f}% {'di atas' if diff_from_avg > 0 else 'di bawah'} rata-rata**
                            
                            **🔍 Insight Berdasarkan Data:**
                            - Negara berada di **kelompok performa terbaik** dalam hal kesehatan anak
                            - Angka mortality rendah mengindikasikan sistem kesehatan yang efektif
                            - Cluster ini umumnya berisi negara dengan infrastruktur kesehatan mapan
                            
                            **💡 Rekomendasi Berbasis Data:**
                            
                            Negara-negara di cluster Low-Burden (biru) umumnya berasal dari kawasan dengan infrastruktur kesehatan yang mapan seperti Eropa, Amerika Utara, dan sebagian Asia Timur—dimana akses terhadap vaksinasi, air bersih, sanitasi, dan tenaga medis terlatih sudah tersedia luas. Untuk mempertahankan capaian ini, fokus dapat diarahkan pada pemeliharaan kualitas layanan, inovasi berkelanjutan, dan antisipasi tantangan kesehatan baru. Pengalaman dan model keberhasilan dari cluster ini dapat menjadi referensi bagi negara-negara di cluster High-Burden yang ingin memperbaiki sistem kesehatannya.
                            """
                    elif res_dataset_type == "lingkungan":
                        area = result.get('area_ha', 0)
                        loss = result.get('total_loss', 0)
                        intensity = (loss / area * 100) if area > 0 else 0
                        annual_loss = loss / 22  # Asumsi 22 tahun data
                        
                        # Perkirakan tahun hingga hutan habis jika trend berlanjut
                        years_to_depletion = (area / annual_loss) if annual_loss > 0 else float('inf')
                        
                        if "Hotspot" in risk_label:
                            interpretation = f"""
                            **🟡 Hasil: Cluster {cluster_id} - {risk_label}**
                            
                            **📊 Analisis Data:**
                            - Negara: **{country}**
                            - Total Area Hutan: **{area:,.0f} ha**
                            - Total Kehilangan: **{loss:,.0f} ha** ({intensity:.2f}%)
                            - Rata-rata Kehilangan/Tahun: **{annual_loss:,.0f} ha/tahun**
                            - ⚠️ Estimasi waktu hingga hutan habis (jika trend berlanjut): **{years_to_depletion:.0f} tahun**
                            
                            **🔍 Insight Berdasarkan Data:**
                            - Intensitas deforestasi {intensity:.2f}% termasuk **tinggi** dibanding rata-rata global
                            - Dengan laju kehilangan {annual_loss:,.0f} ha/tahun, diperlukan perhatian serius
                            - Cluster ini umumnya berisi negara dengan tekanan ekonomi tinggi terhadap sumber daya hutan
                            
                            **💡 Rekomendasi Berbasis Data:**
                            
                            Negara di cluster Hotspot (kuning) menghadapi tekanan besar dari ekspansi pertanian, pertambangan, perkebunan komersial, dan pembalakan liar yang didorong oleh permintaan pasar global dan kebutuhan ekonomi lokal. Untuk pemerintah dan pembuat kebijakan, langkah prioritas adalah memperkuat penegakan hukum kehutanan, memberikan insentif ekonomi bagi masyarakat yang menjaga hutan, dan mengembangkan alternatif mata pencaharian berkelanjutan. Negara-negara di cluster ini dapat belajar dari praktik konservasi di negara dengan tingkat deforestasi rendah.
                            """
                        else:  # Low Deforestation Zone
                            interpretation = f"""
                            **🔵 Hasil: Cluster {cluster_id} - {risk_label}**
                            
                            **📊 Analisis Data:**
                            - Negara: **{country}**
                            - Total Area Hutan: **{area:,.0f} ha**
                            - Total Kehilangan: **{loss:,.0f} ha** ({intensity:.2f}%)
                            - Rata-rata Kehilangan/Tahun: **{annual_loss:,.0f} ha/tahun**
                            
                            **🔍 Insight Berdasarkan Data:**
                            - Intensitas {intensity:.2f}% menunjukkan **pengelolaan hutan yang relatif baik**
                            - Cluster ini berisi negara dengan tingkat kehilangan hutan yang terkendali
                            - Pola data mengindikasikan keberhasilan kebijakan konservasi atau kondisi geografis yang mendukung
                            
                            **💡 Rekomendasi Berbasis Data:**
                            
                            Negara di cluster Low Deforestation (biru) umumnya memiliki regulasi kehutanan yang ketat, penegakan hukum yang efektif, atau telah melewati fase industrialisasi intensif sehingga tekanan terhadap hutan berkurang. Model pengelolaan dari cluster ini dapat menjadi referensi bagi negara-negara di cluster Hotspot yang ingin memperbaiki kondisi hutannya. Bagi investor dan perusahaan, hasil clustering ini dapat menjadi panduan due diligence untuk menghindari rantai pasok yang terkait deforestasi ilegal.
                            """
                    else:
                        interpretation = f"""
                        **Hasil: Cluster {cluster_id}** - {risk_label}
                        
                        Data berhasil diklasifikasikan ke dalam cluster {cluster_id}.
                        """
                    
                    st.success(interpretation)

                # Show distance to all centroids
                centroids = bundle.get("centroids", {})
                if centroids:
                    st.markdown("**Jarak ke Setiap Centroid (scaled space):**")
                    x_vec = X_in.reshape(-1)
                    dist_data = []
                    for cid, cvec in centroids.items():
                        c_arr = np.asarray(cvec, dtype=float)
                        dist = float(np.linalg.norm(x_vec - c_arr))
                        clabel, cemoji = _risk_label(bundle, int(cid), res_dataset_type)
                        is_chosen = "(Terpilih)" if int(cid) == int(cluster_id) else ""
                        dist_data.append({
                            "Cluster": int(cid),
                            "Kategori": f"{cemoji} {clabel}",
                            "Jarak": f"{dist:.4f}",
                            "Status": is_chosen
                        })
                    dist_df = pd.DataFrame(dist_data)
                    st.dataframe(dist_df, use_container_width=True, hide_index=True)
                    st.caption("Cluster dengan jarak terkecil akan dipilih sebagai hasil prediksi.")

                if "membership_proba" in result.get("extra", {}):
                    probs = result["extra"]["membership_proba"]
                    st.markdown("**Probabilitas Keanggotaan Cluster:**")
                    prob_df = pd.DataFrame({"Cluster": range(len(probs)), "Probability": probs})
                    fig = px.bar(prob_df, x="Cluster", y="Probability", title="Membership Probability")
                    fig.update_layout(font_family="Trebuchet MS")
                    st.plotly_chart(fig, use_container_width=True)

                # Download result
                out = {
                    "negara": [country],
                    "method": [bundle.get("method")],
                    "cluster": [int(cluster_id)],
                    "kategori": [risk_label],
                }
                if res_dataset_type == "kesehatan":
                    out["tahun"] = [result.get("year")]
                    out["mortality"] = [result.get("mortality")]
                elif res_dataset_type == "lingkungan":
                    out["total_loss_ha"] = [result.get("total_loss")]
                
                out_df = pd.DataFrame(out)
                st.download_button(
                    "Download Hasil (CSV)",
                    data=out_df.to_csv(index=False),
                    file_name="prediksi_single.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ==================== BATCH PREDICTION ====================
    with st.container(border=True):
        st.markdown("##### 📤 Prediksi Batch (Upload CSV)")
        
        if dataset_type == "kesehatan":
            st.info("""
            **Format CSV yang Diperlukan:**
            - `country` atau `name`: Nama negara (harus ada di dataset training)
            - `mortality`: Nilai mortality yang ingin diprediksi
            """)
        elif dataset_type == "lingkungan":
            st.info("""
            **Format CSV yang Diperlukan:**
            - `country` atau `name`: Nama negara (harus ada di dataset training)
            - `total_loss_ha`: Total kehilangan hutan yang ingin diprediksi
            """)

        up = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload_pred")
        if up is not None:
            try:
                batch_df = pd.read_csv(up)
                st.write("Preview:", batch_df.head())

                if st.button("Jalankan Prediksi Batch", use_container_width=True, key="run_batch_pred"):
                    country_col_batch = _find_country_column(batch_df)
                    
                    processed_rows = []
                    
                    for idx, row in batch_df.iterrows():
                        try:
                            country_name = row.get(country_col_batch) if country_col_batch else None
                            
                            if dataset_type == "kesehatan":
                                if country_name and df_base is not None and country_col_base:
                                    base_data = df_base[df_base[country_col_base].astype(str) == str(country_name)]
                                    if not base_data.empty:
                                        value_1990 = _safe_float(base_data.iloc[0].get("value_1990", 100))
                                    else:
                                        value_1990 = 100.0
                                else:
                                    value_1990 = 100.0
                                
                                mortality = _safe_float(row.get("mortality", row.get("value_2019", 50)))
                                fe_vals = _compute_fe_kesehatan(value_1990, mortality)
                                
                            elif dataset_type == "lingkungan":
                                if country_name and df_base is not None and country_col_base:
                                    base_data = df_base[df_base[country_col_base].astype(str) == str(country_name)]
                                    if not base_data.empty:
                                        area_ha = _safe_float(base_data.iloc[0].get("area_ha", 1000000))
                                    else:
                                        area_ha = 1000000.0
                                else:
                                    area_ha = 1000000.0
                                
                                loss = _safe_float(row.get("total_loss_ha", 50000))
                                fe_vals = _compute_fe_lingkungan(area_ha, loss)
                            else:
                                fe_vals = {col: _safe_float(row.get(col, 0)) for col in input_cols}
                            
                            final_vals = {col: fe_vals.get(col, 0.0) for col in input_cols}
                            processed_rows.append(final_vals)
                        except:
                            processed_rows.append({col: 0.0 for col in input_cols})
                    
                    processed_df = pd.DataFrame(processed_rows)
                    
                    try:
                        Xin = _transform_input(bundle, processed_df)
                    except Exception as e:
                        st.error(f"Error transformasi: {e}")
                        return

                    preds = []
                    for i in range(Xin.shape[0]):
                        try:
                            lab, _ = _predict_cluster(bundle, Xin[i:i+1], allow_approx=True)
                            preds.append(lab)
                        except:
                            preds.append(-1)

                    out_df = batch_df.copy()
                    out_df["predicted_cluster"] = preds
                    out_df["kategori"] = [_risk_label(bundle, int(lab), dataset_type)[0] for lab in preds]

                    st.success(f"Selesai. Total: {len(out_df)} baris.")
                    st.dataframe(out_df.head(20), use_container_width=True)

                    st.download_button(
                        "Download Hasil Batch (CSV)",
                        data=out_df.to_csv(index=False),
                        file_name="prediksi_batch.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_batch_pred"
                    )

                    vc = out_df["predicted_cluster"].value_counts().sort_index()
                    fig = px.bar(x=vc.index.astype(str), y=vc.values, 
                                 labels={"x": "Cluster", "y": "Jumlah"}, 
                                 title="Distribusi Cluster (Batch)")
                    fig.update_layout(font_family="Trebuchet MS")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
