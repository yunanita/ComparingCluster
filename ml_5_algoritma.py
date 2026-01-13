"""
Machine Learning Module - Perbandingan 5 Algoritma Clustering TANPA Parameter K
================================================================================
Semua algoritma menentukan jumlah cluster secara OTOMATIS:

1. DBSCAN - Density-Based (eps, min_samples)
2. OPTICS - Hierarchical Density-Based (xi, min_samples)
3. MeanShift - Mode-seeking (bandwidth)
4. AffinityPropagation - Message-passing (preference, damping)
5. GMM (BIC) - Gaussian Mixture dengan BIC selection

Hasil: Tabel perbandingan jumlah cluster optimal + metrik evaluasi
"""

import streamlit as st
import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    MeanShift,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
RANDOM_STATE_DEFAULT = 42
N_TRIALS = 80  # Lebih banyak trial untuk memastikan semua metode menemukan konfigurasi valid
DISTANCE_METRIC = "euclidean"

MIN_CLUSTERS = 2
MAX_CLUSTERS = 5  # Maksimal cluster yang diizinkan (sesuai permintaan dosen)
MAX_CLUSTER_RATIO = 0.5
MIN_CLUSTER_SIZE_RATIO = 0.01

ALL_METHODS = ["DBSCAN", "OPTICS", "MeanShift", "AffinityPropagation", "GMM_BIC"]

METHOD_DESCRIPTIONS = {
    "DBSCAN": "Density-Based Spatial Clustering. Jumlah cluster ditentukan oleh eps dan min_samples.",
    "OPTICS": "Ordering Points To Identify Clustering Structure. Jumlah cluster ditentukan oleh xi.",
    "MeanShift": "Mode-seeking algorithm. Jumlah cluster ditentukan oleh bandwidth.",
    "AffinityPropagation": "Message-passing clustering. Jumlah cluster ditentukan oleh preference.",
    "GMM_BIC": "Gaussian Mixture Model. K optimal dipilih otomatis berdasarkan BIC terendah.",
}

# =====================================================================
# Utilities
# =====================================================================

def _to_numpy(X: Any) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


def _count_clusters(labels: np.ndarray) -> Tuple[int, float, int]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0, 1.0, 0
    
    unique = set(labels.tolist())
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / len(labels)
    
    if -1 in unique:
        unique.remove(-1)
    
    min_size = 0
    if unique:
        sizes = [np.sum(labels == lab) for lab in unique]
        min_size = min(sizes) if sizes else 0
    
    return len(unique), noise_ratio, min_size


def _validate_clustering_result(labels: np.ndarray, n_samples: int, strict: bool = True) -> Tuple[bool, str]:
    n_clusters, noise_ratio, min_cluster_size = _count_clusters(labels)
    
    if n_clusters < MIN_CLUSTERS:
        return False, f"n_clusters={n_clusters} < {MIN_CLUSTERS}"
    
    # Batasi maksimal cluster ke MAX_CLUSTERS (5)
    if n_clusters > MAX_CLUSTERS:
        return False, f"n_clusters={n_clusters} > max={MAX_CLUSTERS}"
    
    return True, ""


def _euclidean_centroids_and_radii(X: np.ndarray, labels: np.ndarray, radius_q: float = 0.95):
    centroids, radii = {}, {}
    labels = np.asarray(labels)
    for lab in np.unique(labels):
        if lab == -1:
            continue
        mask = labels == lab
        if mask.sum() == 0:
            continue
        pts = X[mask]
        c = np.mean(pts, axis=0)
        d = np.linalg.norm(pts - c, axis=1)
        centroids[int(lab)] = c.astype(float).tolist()
        radii[int(lab)] = float(np.quantile(d, radius_q)) if d.size else 0.0
    return centroids, radii


# =====================================================================
# Evaluation Metrics
# =====================================================================

def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels)
    n_clusters, noise_ratio, min_cluster_size = _count_clusters(labels)

    out = {
        "n_clusters": int(n_clusters),
        "noise_ratio": float(noise_ratio),
        "noise_count": int(np.sum(labels == -1)),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "silhouette_penalized": None,
        "valid": False,
    }

    if n_clusters < 2:
        return out

    mask = labels != -1
    Xv = X[mask]
    lv = labels[mask]

    if len(Xv) < 3 or len(np.unique(lv)) < 2:
        return out

    try:
        sil = float(silhouette_score(Xv, lv, metric=DISTANCE_METRIC))
        out["silhouette"] = sil
        out["silhouette_penalized"] = sil * (1.0 - noise_ratio)
    except:
        pass

    try:
        out["davies_bouldin"] = float(davies_bouldin_score(Xv, lv))
    except:
        pass

    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xv, lv))
    except:
        pass

    out["valid"] = out["silhouette"] is not None
    return out


# =====================================================================
# Parameter Grids
# =====================================================================
"""
REFERENSI PARAMETER OPTIMAL:

1. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
   - Referensi: Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A density-based algorithm 
     for discovering clusters in large spatial databases with noise." KDD.
   - eps: Ditentukan menggunakan k-distance graph (Schubert et al., 2017)
   - min_samples: Umumnya 2*n_features (rule of thumb dari Scikit-learn documentation)
   
2. OPTICS (Ordering Points To Identify the Clustering Structure)
   - Referensi: Ankerst, M., Breunig, M. M., Kriegel, H. P., & Sander, J. (1999). 
     "OPTICS: Ordering Points To Identify the Clustering Structure." SIGMOD.
   - xi: Range [0.01, 0.2] umum digunakan untuk reachability-based clustering
   - min_samples: Sama dengan DBSCAN, 2*n_features

3. MeanShift
   - Referensi: Comaniciu, D., & Meer, P. (2002). "Mean shift: A robust approach toward 
     feature space analysis." IEEE Transactions on Pattern Analysis and Machine Intelligence.
   - bandwidth: Ditentukan otomatis menggunakan estimate_bandwidth dari sklearn 
     dengan quantile [0.1-0.5] (Scikit-learn documentation)

4. AffinityPropagation
   - Referensi: Frey, B. J., & Dueck, D. (2007). "Clustering by Passing Messages Between 
     Data Points." Science, 315(5814), 972-976.
   - preference: Defaultnya median similarity, nilai lebih rendah = lebih sedikit cluster
   - damping: [0.5, 0.9] untuk konvergensi stabil (sklearn default 0.5)

5. Gaussian Mixture Model (GMM) dengan BIC
   - Referensi: Schwarz, G. (1978). "Estimating the Dimension of a Model." The Annals of Statistics.
   - n_components: Dipilih berdasarkan BIC (Bayesian Information Criterion) terendah
   - covariance_type: "full", "tied", "diag" - tergantung struktur data

Semua parameter di-tune menggunakan grid search dengan metrik Silhouette Score.
"""

def _sample_param_configs(param_grid: Dict[str, List], n_trials: int, rng) -> List[Dict]:
    keys = list(param_grid.keys())
    all_values = list(param_grid.values())
    all_combinations = list(product(*all_values))
    
    if len(all_combinations) <= n_trials:
        return [dict(zip(keys, combo)) for combo in all_combinations]
    
    indices = rng.choice(len(all_combinations), size=n_trials, replace=False)
    return [dict(zip(keys, all_combinations[i])) for i in indices]


def _estimate_dbscan_eps(X: np.ndarray, min_samples: int) -> float:
    n_samples = len(X)
    if n_samples <= min_samples:
        return float(np.percentile(np.linalg.norm(X - np.mean(X, axis=0), axis=1), 90))

    nn = NearestNeighbors(n_neighbors=min(min_samples, n_samples - 1))
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = np.sort(distances[:, -1])

    if len(k_dist) < 10:
        return float(np.percentile(k_dist, 90))

    x = np.linspace(0, 1, len(k_dist))
    y = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min() + 1e-12)
    dy = np.gradient(y, x)
    ddy = np.gradient(dy, x)
    curvature = np.abs(ddy) / (1 + dy ** 2) ** 1.5
    idx = int(np.argmax(curvature))
    
    return max(float(k_dist[idx]), 1e-6)


def get_param_grid_dbscan(X, n_samples, n_features):
    """
    DBSCAN parameter grid - SANGAT LEBAR untuk memastikan tidak gagal
    eps lebih besar = lebih sedikit cluster, noise lebih sedikit
    """
    # Range min_samples yang sangat lebar
    min_samples_opts = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 
                        max(3, n_features), max(5, n_features + 1), max(8, 2*n_features),
                        max(10, n_samples // 20), max(15, n_samples // 15), 
                        max(20, n_samples // 10), max(25, n_samples // 8)]
    min_samples_opts = [min(m, max(5, n_samples // 2)) for m in min_samples_opts]
    min_samples_opts = sorted(set([m for m in min_samples_opts if 2 <= m <= n_samples // 2]))
    
    # eps range yang SANGAT lebar - kunci untuk mendapatkan 2-5 cluster
    eps_base = _estimate_dbscan_eps(X, max(5, 2*n_features))
    # Tambah banyak multiplier untuk mencari sweet spot
    multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                   1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 
                   5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 50.0]
    eps_opts = [eps_base * m for m in multipliers]
    eps_opts = sorted(set([e for e in eps_opts if e > 0]))
    
    return {"eps": eps_opts, "min_samples": min_samples_opts}


def get_param_grid_optics(n_samples, n_features):
    """
    OPTICS parameter grid - xi lebih besar = lebih sedikit cluster
    Range yang lebar untuk memastikan menemukan konfigurasi dengan 2-5 cluster
    """
    # min_samples range yang lebar
    min_samples_opts = [2, 3, 4, 5, 6, 8, max(5, n_features), max(8, 2*n_features), max(12, 3*n_features)]
    min_samples_opts = [min(m, max(5, n_samples // 3)) for m in min_samples_opts]
    min_samples_opts = sorted(set([m for m in min_samples_opts if m >= 2]))
    
    # xi range yang lebar - dari kecil (banyak cluster) sampai besar (sedikit cluster)
    return {
        "min_samples": min_samples_opts,
        "xi": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
        "min_cluster_size": min_samples_opts,
    }


def get_param_grid_meanshift(X, n_samples):
    """
    MeanShift parameter grid - bandwidth lebih besar = lebih sedikit cluster
    """
    # Quantile lebih besar = bandwidth lebih besar = cluster lebih sedikit
    quantiles = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
    bandwidths = []
    for q in quantiles:
        try:
            bw = estimate_bandwidth(X, quantile=q, n_samples=min(500, n_samples))
            if np.isfinite(bw) and bw > 0:
                bandwidths.append(bw)
        except:
            pass
    
    # Fallback dengan bandwidth yang cukup besar
    if not bandwidths:
        bandwidths = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    else:
        # Tambah multiplier yang lebih besar untuk cluster lebih sedikit
        base_bw = np.median(bandwidths)
        for mult in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            bw = base_bw * mult
            if bw > 0:
                bandwidths.append(bw)
    
    return {"bandwidth": sorted(set(bandwidths))}


def get_param_grid_affinity(X):
    """
    AffinityPropagation preference parameter menentukan jumlah cluster.
    - Preference LEBIH RENDAH (negatif besar) = LEBIH SEDIKIT cluster
    
    Untuk mendapat ≤5 cluster, fokus ke preference yang lebih rendah (negatif besar).
    """
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(X, metric='euclidean')
    similarities = -dists
    median_sim = np.median(similarities)
    min_sim = np.min(similarities)
    
    # Fokus ke preference yang lebih rendah untuk cluster lebih sedikit
    # Lower preference = fewer clusters
    pref_opts = [
        min_sim * 3.0,            # Very very few clusters (2-3)
        min_sim * 2.5,            # Very few clusters
        min_sim * 2.0,            # Very few clusters
        min_sim * 1.5,            # Few clusters
        min_sim * 1.2,            # Few clusters
        min_sim,                  # Few clusters  
        min_sim * 0.8,            # Some clusters
        min_sim * 0.5 + median_sim * 0.5,  # Medium-few
        median_sim,               # Medium clusters
    ]
    # Filter valid values and remove duplicates
    pref_opts = sorted(set([p for p in pref_opts if np.isfinite(p) and p < 0]))
    
    return {"preference": pref_opts, "damping": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]}


def get_param_grid_gmm_bic(n_samples):
    # Batasi maksimal k ke MAX_CLUSTERS (5)
    max_k = min(MAX_CLUSTERS, int(np.sqrt(n_samples)) + 3, n_samples - 1)
    return {
        "n_components": list(range(2, max_k + 1)),
        "covariance_type": ["full", "tied", "diag"],
    }


# =====================================================================
# Fitting Functions
# =====================================================================

def _fit_single_config(method, X, config, random_state):
    if method == "DBSCAN":
        model = DBSCAN(eps=config["eps"], min_samples=config["min_samples"], metric=DISTANCE_METRIC)
        labels = model.fit_predict(X)
        
    elif method == "OPTICS":
        model = OPTICS(
            min_samples=config["min_samples"],
            xi=config.get("xi", 0.05),
            min_cluster_size=config.get("min_cluster_size", config["min_samples"]),
            metric=DISTANCE_METRIC,
        )
        labels = model.fit_predict(X)
        
    elif method == "MeanShift":
        bw = config.get("bandwidth", None)
        model = MeanShift(bandwidth=bw if bw and bw > 0 else None, bin_seeding=True)
        labels = model.fit_predict(X)
        
    elif method == "AffinityPropagation":
        model = AffinityPropagation(
            preference=config.get("preference", None),
            damping=config.get("damping", 0.9),
            random_state=random_state,
            max_iter=500,
        )
        labels = model.fit_predict(X)
        
    elif method == "GMM_BIC":
        model = GaussianMixture(
            n_components=config["n_components"],
            covariance_type=config.get("covariance_type", "full"),
            n_init=5,
            reg_covar=1e-6,
            random_state=random_state,
        )
        model.fit(X)
        labels = model.predict(X)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return model, labels


def tune_and_fit_method(method, X, n_trials, random_state):
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)
    
    result = {
        "method": method,
        "description": METHOD_DESCRIPTIONS.get(method, ""),
        "success": False,
        "native_predict": method == "GMM_BIC",
        "tuning_info": {},
    }
    
    try:
        if method == "DBSCAN":
            param_grid = get_param_grid_dbscan(X, n_samples, n_features)
        elif method == "OPTICS":
            param_grid = get_param_grid_optics(n_samples, n_features)
        elif method == "MeanShift":
            param_grid = get_param_grid_meanshift(X, n_samples)
        elif method == "AffinityPropagation":
            param_grid = get_param_grid_affinity(X)
        elif method == "GMM_BIC":
            param_grid = get_param_grid_gmm_bic(n_samples)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        configs = _sample_param_configs(param_grid, n_trials, rng)
        result["tuning_info"]["n_configs_tried"] = len(configs)
        
        best_score = -np.inf
        best_config = None
        best_labels = None
        best_model = None
        best_metrics = None
        best_bic = np.inf
        
        # Fair tuning: hanya terima hasil dengan 2-5 cluster
        for config in configs:
            try:
                model, labels = _fit_single_config(method, X, config, random_state)
                n_clusters, noise_ratio, _ = _count_clusters(labels)
                
                # Skip jika cluster < 2 atau > MAX_CLUSTERS (5)
                if n_clusters < MIN_CLUSTERS or n_clusters > MAX_CLUSTERS:
                    continue
                
                metrics = evaluate_clustering(X, labels)
                if not metrics["valid"]:
                    continue
                
                if method == "GMM_BIC":
                    bic = model.bic(X)
                    if bic < best_bic:
                        best_bic = bic
                        best_score = metrics.get("silhouette_penalized", -np.inf)
                        best_config = config
                        best_labels = labels
                        best_model = model
                        best_metrics = metrics
                        best_metrics["bic"] = bic
                else:
                    score = metrics.get("silhouette_penalized", -np.inf)
                    if score is not None and score > best_score:
                        best_score = score
                        best_config = config
                        best_labels = labels
                        best_model = model
                        best_metrics = metrics
                        
            except:
                continue
        
        if best_config is None or best_labels is None:
            result["success"] = False
            result["error"] = f"Tidak ada konfigurasi yang menghasilkan 2-{MAX_CLUSTERS} cluster. Metode ini tidak cocok untuk dataset ini."
            return result
        
        centroids, radii = _euclidean_centroids_and_radii(X, best_labels)
        
        result.update({
            "success": True,
            "model": best_model,
            "labels": best_labels,
            "best_config": best_config,
            "best_score": best_score,
            "supports_predict": method == "GMM_BIC",
            "input_columns": None,
            **best_metrics,
            "centroids": centroids,
            "radii": radii,
        })
        
        return result
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        return result


def fit_with_custom_params(method: str, X: np.ndarray, config: Dict, random_state: int) -> Dict:
    """
    Fit clustering dengan parameter custom dari user.
    """
    n_samples = X.shape[0]
    
    result = {
        "method": method,
        "description": METHOD_DESCRIPTIONS.get(method, ""),
        "success": False,
        "tuning_info": {"custom_params": True},
    }
    
    try:
        model, labels = _fit_single_config(method, X, config, random_state)
        
        is_valid, msg = _validate_clustering_result(labels, n_samples, strict=False)
        if not is_valid:
            result["error"] = f"Clustering tidak valid: {msg}"
            return result
        
        metrics = evaluate_clustering(X, labels)
        if not metrics.get("valid"):
            result["error"] = "Metrik evaluasi gagal dihitung."
            return result
        
        centroids, radii = _euclidean_centroids_and_radii(X, labels)
        
        result.update({
            "success": True,
            "model": model,
            "labels": labels,
            "best_config": config,
            "best_score": metrics.get("silhouette_penalized", 0),
            "supports_predict": method == "GMM_BIC",
            "input_columns": None,
            **metrics,
            "centroids": centroids,
            "radii": radii,
        })
        
        return result
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        return result


# =====================================================================
# Risk Mapping
# =====================================================================

def _infer_risk_feature(df):
    if df is None or df.empty:
        return None
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return None
    preferred = ["value_2019", "mean_value", "loss_intensity", "total_loss_ha"]
    for p in preferred:
        if p in df.columns:
            return p
    return cols[0]


def _cluster_risk_mapping(df_raw, labels):
    """
    Create risk mapping based on cluster characteristics.
    
    For health data: Higher mortality = Higher risk
    For environment data: Higher loss = Higher risk
    
    The mapping is based on the mean value of the risk feature per cluster,
    sorted from lowest (Rendah) to highest (Tinggi).
    """
    labels = np.asarray(labels)
    if df_raw is None or df_raw.empty or len(df_raw) != len(labels):
        return {}

    risk_feature = _infer_risk_feature(df_raw)
    if risk_feature is None:
        return {}

    tmp = df_raw.copy()
    tmp["_cluster"] = labels
    tmp = tmp[tmp["_cluster"] != -1]
    if tmp.empty:
        return {}

    # Sort clusters by mean risk feature value (ascending)
    grp = tmp.groupby("_cluster")[risk_feature].mean().sort_values()
    cluster_order = list(grp.index.astype(int).tolist())

    n = len(cluster_order)
    if n == 1:
        return {cluster_order[0]: "Sedang"}

    mapping = {}
    
    if n == 2:
        # Untuk 2 cluster: cluster dengan nilai rendah = Rendah, tinggi = Tinggi
        mapping[cluster_order[0]] = "Rendah"  # Lower mean value = Lower risk
        mapping[cluster_order[1]] = "Tinggi"  # Higher mean value = Higher risk
    elif n == 3:
        # Untuk 3 cluster: Rendah, Sedang, Tinggi
        mapping[cluster_order[0]] = "Rendah"
        mapping[cluster_order[1]] = "Sedang"
        mapping[cluster_order[2]] = "Tinggi"
    else:
        # Untuk >3 cluster: proportional
        for rank, cid in enumerate(cluster_order):
            q = rank / (n - 1) if n > 1 else 0
            if q <= 0.33:
                mapping[cid] = "Rendah"
            elif q <= 0.66:
                mapping[cid] = "Sedang"
            else:
                mapping[cid] = "Tinggi"
    
    return mapping


# =====================================================================
# Ranking
# =====================================================================

def rank_methods(results):
    rows = []
    for method, r in results.items():
        if not r.get("success"):
            continue
        rows.append({
            "Method": method,
            "Clusters": r.get("n_clusters"),
            "Noise%": r.get("noise_ratio", 0) * 100,
            "Silhouette": r.get("silhouette"),
            "Sil_Penalized": r.get("silhouette_penalized"),
            "DB": r.get("davies_bouldin"),
            "CH": r.get("calinski_harabasz"),
        })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    df["Rank_Sil"] = df["Sil_Penalized"].rank(ascending=False, na_option='bottom')
    df["Rank_DB"] = df["DB"].rank(ascending=True, na_option='bottom')
    df["Rank_CH"] = df["CH"].rank(ascending=False, na_option='bottom')
    df["Rank_Noise"] = df["Noise%"].rank(ascending=True, na_option='bottom')
    
    df["Avg_Rank"] = (df["Rank_Sil"] + df["Rank_DB"] + df["Rank_CH"] + df["Rank_Noise"]) / 4
    
    return df.sort_values("Avg_Rank").reset_index(drop=True)


# =====================================================================
# Streamlit Tab
# =====================================================================

def ml_tab():
    # Simple title
    st.markdown('<div class="section-header">🤖 Perbandingan 5 Algoritma Clustering</div>', unsafe_allow_html=True)
    st.caption("DBSCAN - OPTICS - MeanShift - Affinity Propagation - GMM")
    
    # ==================== VALIDATION ====================
    if "X_proc" not in st.session_state:
        st.warning("Lakukan preprocessing di tab Preprocessing terlebih dahulu.")
        return

    X_df = st.session_state["X_proc"]
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)

    X = _to_numpy(X_df)
    n_samples, n_features = X.shape

    if n_samples < 3:
        st.error("Jumlah sampel terlalu sedikit (minimal 3).")
        return

    df_raw = st.session_state.get("df_clean", st.session_state.get("df", None))

    # ==================== INFO ALGORITMA ====================
    st.markdown('<div class="section-header-green">📖 Metodologi 5 Algoritma Clustering</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("### � Penjelasan Singkat 5 Algoritma")
        st.caption("Semua algoritma menentukan jumlah cluster (K) secara otomatis tanpa input manual.")
        
        # Simple cards for each algorithm
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("**🔵 1. DBSCAN**")
                st.markdown("Mengelompokkan data berdasarkan **kepadatan (density)**. Titik yang berdekatan dan padat membentuk cluster, sedangkan titik yang terisolasi dianggap noise/outlier.")
                st.caption("Parameter: eps (radius), min_samples")
                
            with st.container(border=True):
                st.markdown("**🟢 2. OPTICS**")
                st.markdown("Pengembangan DBSCAN yang lebih fleksibel. Menghasilkan **ordering** data berdasarkan kepadatan sehingga dapat menemukan cluster dengan kepadatan berbeda.")
                st.caption("Parameter: xi (threshold), min_samples")
                
            with st.container(border=True):
                st.markdown("**🟡 3. MeanShift**")
                st.markdown("Mencari **mode (puncak)** distribusi data menggunakan kernel. Data bergerak menuju puncak kepadatan tertinggi dan membentuk cluster natural.")
                st.caption("Parameter: bandwidth (ukuran kernel)")
                
        with col2:
            with st.container(border=True):
                st.markdown("**🟠 4. Affinity Propagation**")
                st.markdown("Menggunakan **message-passing** antar data untuk menentukan exemplar (pusat cluster). Setiap data saling bertukar informasi tentang kelayakan menjadi pusat.")
                st.caption("Parameter: preference, damping")
                
            with st.container(border=True):
                st.markdown("**🔴 5. GMM-BIC**")
                st.markdown("Memodelkan data sebagai campuran **distribusi Gaussian**. Jumlah cluster optimal dipilih otomatis berdasarkan skor BIC (Bayesian Information Criterion) terendah.")
                st.caption("Parameter: n_components (otomatis via BIC)")
                
        st.divider()
        
        st.markdown("**📊 Metrik Evaluasi:**")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.info("**Silhouette Score**\n\n-1 s.d. 1, lebih tinggi lebih baik")
        with m2:
            st.info("**Davies-Bouldin**\n\n≥0, lebih rendah lebih baik")
        with m3:
            st.info("**Calinski-Harabasz**\n\n≥0, lebih tinggi lebih baik")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ==================== DATA INFO ====================
    st.markdown('<div class="section-header-green">📊 Info Data</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Sampel", n_samples)
        with col2:
            st.metric("Jumlah Fitur", n_features)
        with col3:
            st.metric("Dataset", st.session_state.get("selected_dataset", "Unknown"))

    # ==================== PENGATURAN UMUM ====================
    with st.container(border=True):
        st.markdown("##### ⚙️ Pengaturan Umum")
        
        col1, col2 = st.columns(2)
        with col1:
            random_state = st.number_input("Random State", value=RANDOM_STATE_DEFAULT, step=1,
                                           help="Seed untuk reproducibility")
        with col2:
            n_trials = st.slider("Budget Tuning", 10, 100, N_TRIALS,
                                help="Jumlah kombinasi parameter yang dicoba")

        selected_methods = st.multiselect(
            "Pilih Algoritma",
            options=ALL_METHODS,
            default=ALL_METHODS,
        )
        
        if not selected_methods:
            st.warning("Pilih minimal satu algoritma.")
            return

    # ==================== PARAMETER PER ALGORITMA ====================
    with st.container(border=True):
        st.markdown("##### 🔧 Parameter Algoritma (Opsional)")
        st.caption("Default adalah parameter optimal. Ubah jika ingin eksperimen.")
        
        user_params = {}
        
        tabs = st.tabs(selected_methods)
        
        for i, method in enumerate(selected_methods):
            with tabs[i]:
                if method == "DBSCAN":
                    st.markdown("**DBSCAN** - Density-Based Clustering")
                    col1, col2 = st.columns(2)
                    with col1:
                        eps = st.number_input("eps (radius neighborhood)", 
                                             value=0.5, min_value=0.01, max_value=10.0, step=0.1,
                                             help="Radius untuk menentukan neighborhood. Lebih kecil = cluster lebih padat.",
                                             key="dbscan_eps")
                    with col2:
                        min_samples = st.number_input("min_samples", 
                                                      value=max(3, n_features), min_value=2, max_value=50, step=1,
                                                      help="Minimum titik untuk membentuk cluster.",
                                                      key="dbscan_min_samples")
                    use_custom = st.checkbox("Gunakan parameter custom", key="dbscan_custom")
                    user_params["DBSCAN"] = {"eps": eps, "min_samples": int(min_samples)} if use_custom else None
                    
                elif method == "OPTICS":
                    st.markdown("**OPTICS** - Ordering Points To Identify Clustering Structure")
                    col1, col2 = st.columns(2)
                    with col1:
                        xi = st.slider("xi (steepness threshold)", 
                                      0.01, 0.5, 0.05, 0.01,
                                      help="Threshold untuk menentukan cluster boundary.",
                                      key="optics_xi")
                    with col2:
                        min_samples_optics = st.number_input("min_samples", 
                                                             value=max(3, n_features), min_value=2, max_value=50, step=1,
                                                             help="Minimum titik untuk core point.",
                                                             key="optics_min_samples")
                    use_custom = st.checkbox("Gunakan parameter custom", key="optics_custom")
                    user_params["OPTICS"] = {"xi": xi, "min_samples": int(min_samples_optics)} if use_custom else None
                    
                elif method == "MeanShift":
                    st.markdown("**MeanShift** - Mode-Seeking Algorithm")
                    bandwidth = st.slider("bandwidth", 
                                         0.1, 5.0, 1.0, 0.1,
                                         help="Bandwidth kernel. Lebih kecil = lebih banyak cluster.",
                                         key="meanshift_bandwidth")
                    use_custom = st.checkbox("Gunakan parameter custom", key="meanshift_custom")
                    user_params["MeanShift"] = {"bandwidth": bandwidth} if use_custom else None
                    
                elif method == "AffinityPropagation":
                    st.markdown("**AffinityPropagation** - Message-Passing Clustering")
                    st.info("""
                    💡 **Tentang Preference:**
                    - Preference **lebih tinggi** (mendekati 0) = **lebih banyak** cluster
                    - Preference **lebih rendah** (negatif besar) = **lebih sedikit** cluster
                    """)
                    col1, col2 = st.columns(2)
                    with col1:
                        preference = st.number_input("preference", 
                                                     value=-50.0, min_value=-1000.0, max_value=0.0, step=10.0,
                                                     help="Preference untuk semua titik. Negatif = fewer clusters.",
                                                     key="affinity_preference")
                    with col2:
                        damping = st.slider("damping", 
                                           0.5, 0.99, 0.9, 0.01,
                                           help="Damping factor untuk konvergensi.",
                                           key="affinity_damping")
                    use_custom = st.checkbox("Gunakan parameter custom", key="affinity_custom")
                    user_params["AffinityPropagation"] = {"preference": preference, "damping": damping} if use_custom else None
                    
                elif method == "GMM_BIC":
                    st.markdown("**GMM (BIC)** - Gaussian Mixture Model")
                    col1, col2 = st.columns(2)
                    with col1:
                        max_k = min(15, int(np.sqrt(n_samples)) + 3)
                        n_components = st.slider("n_components (max K to try)", 
                                                2, max_k, max_k,
                                                help="Maksimum K yang dicoba. BIC akan memilih K optimal.",
                                                key="gmm_n_components")
                    with col2:
                        cov_type = st.selectbox("covariance_type",
                                               ["full", "tied", "diag", "spherical"],
                                               help="Tipe covariance matrix.",
                                               key="gmm_cov_type")
                    use_custom = st.checkbox("Gunakan parameter custom", key="gmm_custom")
                    user_params["GMM_BIC"] = {"n_components": n_components, "covariance_type": cov_type} if use_custom else None
        
        # Store user params in session
        st.session_state["user_params"] = user_params

    # ==================== RUN BUTTON ====================
    with st.container(border=True):
        if st.button("🚀 Jalankan Perbandingan", type="primary", use_container_width=True):
            results = {}
            progress = st.progress(0.0)
            status = st.empty()

            for i, method in enumerate(selected_methods):
                status.text(f"Processing: {method} ({i+1}/{len(selected_methods)})...")
                
                # Check if user provided custom params
                custom_params = user_params.get(method)
                
                if custom_params:
                    # Use custom params directly
                    res = fit_with_custom_params(method, X, custom_params, int(random_state))
                else:
                    # Use auto-tuning
                    res = tune_and_fit_method(method, X, n_trials, int(random_state))
                
                res["input_columns"] = list(X_df.columns)
                
                if res.get("success") and isinstance(df_raw, pd.DataFrame):
                    if len(df_raw) == len(res.get("labels", [])):
                        res["risk_mapping"] = _cluster_risk_mapping(df_raw, res["labels"])
                        res["risk_feature"] = _infer_risk_feature(df_raw)
                
                results[method] = res
                progress.progress((i + 1) / len(selected_methods))

            progress.empty()
            status.empty()

            st.session_state["exploration_results"] = results
            st.session_state["clustering_results"] = results
            st.session_state["comparison_just_run"] = True  # Flag: user just clicked run
            st.session_state["ml_artifacts"] = {
                "random_state": int(random_state),
                "n_trials": n_trials,
                "input_columns": list(X_df.columns),
                "selected_features": st.session_state.get("selected_features", []),
                "scaler": st.session_state.get("scaler", None),
            }

            ranking_df = rank_methods(results)
            best_method = ranking_df.iloc[0]["Method"] if not ranking_df.empty else None
            st.session_state["best_method"] = best_method

            st.success("✅ Perbandingan selesai!")
            
    # ==================== HASIL ====================
    # Only show results if user clicked "Jalankan Perbandingan" in this session
    if "exploration_results" in st.session_state and st.session_state.get("comparison_just_run", False):
        results = st.session_state["exploration_results"]
        
        with st.container(border=True):
            st.markdown("##### 📊 Tabel Perbandingan Algoritma")
            
            summary_rows = []
            for m, r in results.items():
                if not r.get("success"):
                    summary_rows.append({
                        "Algoritma": m,
                        "Status": "❌ GAGAL",
                        "Cluster": "-",
                        "Noise (%)": "-",
                        "Silhouette": "-",
                        "Davies-Bouldin": "-",
                        "Calinski-Harabasz": "-",
                    })
                else:
                    summary_rows.append({
                        "Algoritma": m,
                        "Status": "✅ OK",
                        "Cluster": r.get("n_clusters"),
                        "Noise (%)": f"{r.get('noise_ratio', 0)*100:.1f}",
                        "Silhouette": f"{r.get('silhouette', 0):.4f}" if r.get('silhouette') else "-",
                        "Davies-Bouldin": f"{r.get('davies_bouldin', 0):.4f}" if r.get('davies_bouldin') else "-",
                        "Calinski-Harabasz": f"{r.get('calinski_harabasz', 0):.1f}" if r.get('calinski_harabasz') else "-",
                    })

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.caption("""
            **Keterangan:** Cluster = jumlah cluster optimal | Noise = outlier (DBSCAN/OPTICS) | 
            Silhouette: ↑ lebih baik | Davies-Bouldin: ↓ lebih baik | Calinski-Harabasz: ↑ lebih baik
            """)

        # ==================== RANKING ====================
        ranking_df = rank_methods(results)
        
        with st.container(border=True):
            st.markdown("##### 🏆 Ranking dan Algoritma Terbaik")
            
            if not ranking_df.empty:
                display_df = ranking_df[["Method", "Clusters", "Noise%", "Silhouette", "DB", "CH", "Avg_Rank"]].copy()
                display_df.columns = ["Algoritma", "Clusters", "Noise%", "Silhouette", "DB", "CH", "Avg Rank"]
                
                for col in ["Silhouette", "DB", "Avg Rank"]:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
                display_df["CH"] = display_df["CH"].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "-")
                display_df["Noise%"] = display_df["Noise%"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                best_method = st.session_state.get("best_method")
                if best_method and best_method in results:
                    best_r = results[best_method]
                    
                    st.markdown(f"### 🏆 Algoritma Terbaik: **{best_method}**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Cluster Optimal", best_r.get("n_clusters", "-"))
                    with col2:
                        st.metric("Silhouette", f"{best_r.get('silhouette', 0):.4f}")
                    with col3:
                        st.metric("Davies-Bouldin", f"{best_r.get('davies_bouldin', 0):.4f}")
                    with col4:
                        st.metric("Calinski-Harabasz", f"{best_r.get('calinski_harabasz', 0):.1f}")
                    
                    st.info(f"**Deskripsi:** {METHOD_DESCRIPTIONS.get(best_method, '')}")
                    
                    # Format parameter optimal
                    st.markdown("**Parameter Optimal:**")
                    config = best_r.get("best_config", {})
                    if config:
                        param_items = []
                        for k, v in config.items():
                            if isinstance(v, float):
                                param_items.append(f"- **{k}**: {v:.4f}")
                            else:
                                param_items.append(f"- **{k}**: {v}")
                        st.markdown("\n".join(param_items))

        # ==================== VISUALISASI ====================
        with st.container(border=True):
            st.markdown("##### 📈 Visualisasi Perbandingan")
            
            import plotly.express as px
            
            ok_methods = [m for m, r in results.items() if r.get("success")]
            
            if ok_methods:
                col1, col2 = st.columns(2)
                
                with col1:
                    cluster_data = pd.DataFrame([
                        {"Algoritma": m, "Jumlah Cluster": results[m].get("n_clusters", 0)}
                        for m in ok_methods
                    ])
                    fig = px.bar(cluster_data, x="Algoritma", y="Jumlah Cluster",
                                title="Jumlah Cluster per Algoritma", color="Jumlah Cluster",
                                color_continuous_scale="Viridis")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    sil_data = pd.DataFrame([
                        {"Algoritma": m, "Silhouette": results[m].get("silhouette", 0) or 0}
                        for m in ok_methods
                    ])
                    fig = px.bar(sil_data.sort_values("Silhouette", ascending=False),
                                x="Algoritma", y="Silhouette", title="Silhouette Score",
                                color="Silhouette", color_continuous_scale="RdYlGn")
                    st.plotly_chart(fig, use_container_width=True)

        # ==================== SIMPAN MODEL ====================
        with st.container(border=True):
            st.markdown("##### 📦 Simpan Model")

            ok_methods = [m for m, r in results.items() if r.get("success")]

            if ok_methods:
                default_method = st.session_state.get("best_method") or ok_methods[0]
                method_to_deploy = st.selectbox("Pilih model untuk disimpan:", ok_methods, 
                    index=ok_methods.index(default_method) if default_method in ok_methods else 0)

                if method_to_deploy:
                    r = results[method_to_deploy]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Algoritma", method_to_deploy)
                    with col2:
                        st.metric("Clusters", r.get("n_clusters", "-"))

                    if st.button("📦 Simpan Model", type="primary", use_container_width=True):
                        import pickle
                        import io
                        from datetime import datetime

                        dataset_type = st.session_state.get("selected_dataset", "Unknown")

                        bundle = {
                            "bundle_version": 6,
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "dataset": dataset_type,
                            "method": method_to_deploy,
                            "model": r.get("model"),
                            "input_columns": r.get("input_columns"),
                            "selected_features": st.session_state.get("selected_features", []),
                            "scaler": st.session_state.get("scaler", None),
                            "best_config": r.get("best_config"),
                            "metrics": {
                                "n_clusters": r.get("n_clusters"),
                                "noise_ratio": r.get("noise_ratio"),
                                "silhouette": r.get("silhouette"),
                                "davies_bouldin": r.get("davies_bouldin"),
                                "calinski_harabasz": r.get("calinski_harabasz"),
                            },
                            "risk_mapping": r.get("risk_mapping", {}),
                            "centroids": r.get("centroids", {}),
                            "radii": r.get("radii", {}),
                        }

                        st.session_state["deployed_model"] = bundle

                        filename = "model_clustering.pkl"
                        if "Kesehatan" in str(dataset_type):
                            filename = "model_kesehatan.pkl"
                        elif "Lingkungan" in str(dataset_type):
                            filename = "model_lingkungan.pkl"

                        try:
                            with open(filename, "wb") as f:
                                pickle.dump(bundle, f)
                            st.success(f"✅ Model tersimpan: {filename}")
                        except Exception as e:
                            st.warning(f"Error: {e}")

                        buf = io.BytesIO()
                        pickle.dump(bundle, buf)
                        st.download_button("⬇️ Download Model", data=buf.getvalue(),
                                          file_name=filename, mime="application/octet-stream",
                                          use_container_width=True)
