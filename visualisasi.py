import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

def visualisasi_tab():
    # Cek data tersedia
    if 'exploration_results' not in st.session_state:
        st.warning("⚠️ Jalankan clustering di tab Machine Learning terlebih dahulu.")
        return
    
    results = st.session_state['exploration_results']
    
    # Ambil best_method
    if 'best_method' in st.session_state:
        best_method = st.session_state['best_method']
    elif results:
        best_method = list(results.keys())[0]
        st.session_state['best_method'] = best_method
    else:
        st.error("Tidak ada hasil clustering yang valid.")
        return
    
    if 'X_proc' not in st.session_state:
        st.warning("⚠️ Data preprocessing belum dilakukan.")
        return
    
    X = st.session_state['X_proc']
    
    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']
    elif 'df' in st.session_state:
        df = st.session_state['df']
    else:
        st.warning("⚠️ Dataset tidak ditemukan.")
        return
    
    selected_features = st.session_state.get('selected_features', [])
    
    # ====== DEFINISI WARNA CLUSTER ======
    # Cluster 0 = Biru, Cluster 1 = Kuning, sisanya warna lain
    CLUSTER_COLORS = {
        "0": "#1E90FF",  # Biru (DodgerBlue)
        "1": "#FFD700",  # Kuning (Gold)
        "2": "#32CD32",  # Hijau (LimeGreen)
        "3": "#FF6347",  # Merah (Tomato)
        "4": "#9370DB",  # Ungu (MediumPurple)
        "-1": "#808080", # Abu-abu untuk noise
    }
    
    # Title
    st.markdown('<div class="section-header">📈 Visualisasi Hasil Clustering</div>', unsafe_allow_html=True)
    st.caption("Dashboard untuk eksplorasi visual dan interpretasi hasil analisis clustering")
    st.info("**Keterangan Warna:** 🔵 Cluster 0 = Biru | 🟡 Cluster 1 = Kuning")
    
    # Validasi results
    if best_method not in results or "error" in results.get(best_method, {}):
        st.error(f"Hasil clustering untuk {best_method} tidak valid.")
        valid_methods = [m for m in results.keys() if "error" not in results.get(m, {})]
        if valid_methods:
            best_method = valid_methods[0]
            st.info(f"Beralih ke metode: {best_method}")
            st.session_state['best_method'] = best_method
        else:
            return
    
    labels = results[best_method]["labels"]
    
    # Buat dataframe clustered
    df_clustered = df.copy()
    if len(df_clustered) == len(labels):
        df_clustered["cluster"] = labels
    else:
        st.warning("Jumlah sampel tidak cocok antara data dan labels.")
        return
    
    # ===================== SCATTER PLOT =====================
    st.markdown('<div class="section-header-green">🔵 Scatter Plot Clusters</div>', unsafe_allow_html=True)
    st.info("Scatter plot menampilkan distribusi data dalam ruang 2D/3D menggunakan PCA. Setiap warna mewakili cluster yang berbeda.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        use_pca = st.checkbox("Gunakan PCA untuk 2D visualization", value=True)
    with col2:
        if use_pca:
            n_components = st.slider("Jumlah komponen PCA", 2, 5, 2)
    
    if use_pca:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        if n_components >= 2:
            df_plot = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
            df_plot["cluster"] = labels.astype(str)
            
            var_exp = pca.explained_variance_ratio_[:2].sum() * 100
            fig = px.scatter(df_plot, x="PC1", y="PC2", color="cluster",
                           title=f"Cluster (PCA) - {best_method} | Variance: {var_exp:.1f}%",
                           color_discrete_map=CLUSTER_COLORS)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretasi Scatter Plot
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = np.sum(labels == -1)
            with st.container(border=True):
                st.markdown("**📊 Interpretasi Scatter Plot:**")
                overlap_analysis = "terpisah dengan baik" if var_exp > 70 else "ada overlap (variasi data kompleks)"
                st.markdown(f"""
                - **Variance Explained:** {var_exp:.1f}% - Semakin tinggi, semakin baik PCA merepresentasikan data asli
                - **Jumlah Cluster:** {n_clusters} cluster teridentifikasi
                - **Pola Visual:** Cluster {overlap_analysis}
                - **Noise/Outlier:** {noise_count} data point ({noise_count/len(labels)*100:.1f}%)
                
                💡 *Jika cluster terlihat overlap, bisa jadi data memiliki variasi tinggi atau perlu fitur tambahan*
                """)
        
        if n_components >= 3:
            df_plot_3d = pd.DataFrame(X_pca[:, :3], columns=["PC1", "PC2", "PC3"])
            df_plot_3d["cluster"] = labels.astype(str)
            fig_3d = px.scatter_3d(df_plot_3d, x="PC1", y="PC2", z="PC3", color="cluster",
                                  title=f"3D Cluster (PCA) - {best_method}",
                                  color_discrete_map=CLUSTER_COLORS)
            st.plotly_chart(fig_3d, use_container_width=True)
    else:
        if len(selected_features) >= 2:
            feat1 = st.selectbox("Fitur sumbu X", selected_features, index=0)
            feat2 = st.selectbox("Fitur sumbu Y", selected_features, index=min(1, len(selected_features)-1))
            
            df_plot = df_clustered.copy()
            df_plot["cluster"] = df_plot["cluster"].astype(str)
            fig = px.scatter(df_plot, x=feat1, y=feat2, color="cluster",
                           title=f"Cluster - {best_method}",
                           color_discrete_map=CLUSTER_COLORS)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Pilih minimal 2 fitur untuk scatter plot.")
    
    st.divider()
    
    # ===================== CLUSTER SIZES =====================
    st.markdown("#### 📊 Distribusi Ukuran Cluster")
    st.info("Distribusi yang seimbang mengindikasikan segmentasi yang baik. Cluster kecil mungkin outlier/noise.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        cluster_df = pd.DataFrame({
            "Cluster": cluster_sizes.index.astype(str),
            "Count": cluster_sizes.values,
            "Percentage": (cluster_sizes.values / len(labels) * 100).round(1)
        })
        
        fig_bar = px.bar(cluster_df, x="Cluster", y="Count", text="Count",
                        title=f"Cluster Sizes - {best_method}", color="Cluster",
                        color_discrete_map=CLUSTER_COLORS)
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        if len(cluster_df) <= 10:
            fig_pie = px.pie(cluster_df, values="Count", names="Cluster",
                           title=f"Cluster Distribution - {best_method}", hole=0.3,
                           color="Cluster", color_discrete_map=CLUSTER_COLORS)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info(f"Terlalu banyak cluster ({len(cluster_df)}) untuk pie chart.")
    
    # Interpretasi Distribusi Cluster
    with st.container(border=True):
        st.markdown("**📊 Interpretasi Distribusi Cluster:**")
        max_cluster = cluster_df.loc[cluster_df['Count'].idxmax()]
        min_cluster = cluster_df.loc[cluster_df['Count'].idxmin()]
        balance_ratio = min_cluster['Count'] / max_cluster['Count'] if max_cluster['Count'] > 0 else 0
        
        if balance_ratio > 0.5:
            balance_status = "Seimbang ✅"
            balance_note = "Distribusi cluster cukup merata, menandakan segmentasi yang baik."
        elif balance_ratio > 0.2:
            balance_status = "Cukup Seimbang 🟡"
            balance_note = "Ada variasi ukuran cluster, beberapa kelompok lebih dominan."
        else:
            balance_status = "Tidak Seimbang ⚠️"
            balance_note = "Distribusi tidak merata. Cluster kecil mungkin outlier atau niche group."
        
        st.markdown(f"""
        - **Status Distribusi:** {balance_status}
        - **Cluster Terbesar:** Cluster {max_cluster['Cluster']} ({max_cluster['Count']} data, {max_cluster['Percentage']}%)
        - **Cluster Terkecil:** Cluster {min_cluster['Cluster']} ({min_cluster['Count']} data, {min_cluster['Percentage']}%)
        - **Rasio Balance:** {balance_ratio:.2f}
        
        💡 *{balance_note}*
        """)
    
    st.divider()
    
    # ===================== MAP =====================
    country_col = None
    for col in df_clustered.columns:
        col_lower = str(col).lower()
        # Lebih lengkap deteksi kolom negara
        if any(keyword in col_lower for keyword in ["country", "negara", "nation", "wilayah", "region", "state"]):
            country_col = col
            break
        # Cek jika kolom bernama persis "name" atau mengandung "country"
        if col_lower == "name" or "country" in col_lower:
            country_col = col
            break
    
    if country_col and len(df_clustered[country_col].notna()) > 0:
        st.markdown("#### 🗺️ Distribusi Geografis Cluster")
        st.info("Peta choropleth menampilkan distribusi cluster berdasarkan wilayah geografis.")
        
        df_map = df_clustered[[country_col, "cluster"]].copy()
        df_map = df_map.dropna(subset=[country_col])
        
        # Normalisasi nama negara ke title case untuk kompatibilitas Plotly
        df_map[country_col] = df_map[country_col].astype(str).str.strip().str.title()
        df_map["cluster"] = df_map["cluster"].astype(str)
        
        # Debug info - tampilkan jika ada masalah
        if len(df_map) == 0:
            st.warning("⚠️ Tidak ada data negara yang valid untuk ditampilkan di peta.")
        else:
            st.caption(f"📍 Menampilkan {len(df_map)} negara dari kolom: `{country_col}`")
        
        try:
            fig_map = px.choropleth(df_map, locations=country_col, locationmode="country names",
                                   color="cluster", title=f"Cluster Map - {best_method}",
                                   color_discrete_map=CLUSTER_COLORS)
            fig_map.update_layout(
                geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
                height=500,
                legend_title_text="Cluster"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Interpretasi Peta
            with st.container(border=True):
                st.markdown("**🗺️ Interpretasi Peta Geografis:**")
                
                # Analisis per region
                cluster_by_region = df_map.groupby("cluster")[country_col].apply(list).to_dict()
                
                st.markdown(f"""
                - **Total Negara:** {len(df_map)} negara tervisualisasi
                - **Distribusi Cluster:** {len(cluster_by_region)} cluster terdistribusi di peta
                
                **Insight Geografis:**
                """)
                
                for cluster_id, countries in cluster_by_region.items():
                    sample_countries = countries[:5]
                    more_text = f" dan {len(countries)-5} lainnya" if len(countries) > 5 else ""
                    st.markdown(f"- **Cluster {cluster_id}:** {', '.join(sample_countries)}{more_text}")
                
                st.markdown("""
                
                💡 *Pola geografis dapat menunjukkan faktor regional seperti kebijakan, kondisi ekonomi, atau faktor lingkungan yang mempengaruhi clustering.*
                """)
        except Exception as e:
            st.warning(f"⚠️ Tidak dapat membuat peta: {str(e)}")
            st.info("💡 Tips: Pastikan nama negara dalam bahasa Inggris (contoh: 'Indonesia', 'Malaysia', 'United States')")
        
        st.divider()
    else:
        st.info("ℹ️ Peta tidak ditampilkan karena kolom negara/wilayah tidak ditemukan di dataset.")
        if df_clustered is not None:
            st.caption(f"Kolom yang tersedia: {list(df_clustered.columns)}")
    
    # ===================== CLUSTER CHARACTERISTICS =====================
    st.markdown("#### 🔍 Karakteristik & Profil Cluster")
    st.info("Rata-rata (mean) membantu memahami profil tiap cluster, standar deviasi menunjukkan variabilitas.")
    
    if selected_features and len(selected_features) > 0:
        try:
            cluster_stats = df_clustered.groupby("cluster")[selected_features].agg(['mean', 'std', 'count']).round(3)
            
            tab1, tab2, tab3 = st.tabs(["📊 Means", "📈 Comparison", "📋 Summary"])
            
            with tab1:
                means_df = cluster_stats.xs('mean', axis=1, level=1)
                st.dataframe(means_df, use_container_width=True)
            
            with tab2:
                top_features = selected_features[:min(4, len(selected_features))]
                for feature in top_features:
                    means = cluster_stats[(feature, 'mean')]
                    stds = cluster_stats[(feature, 'std')]
                    fig = px.bar(x=means.index.astype(str), y=means.values, error_y=stds.values,
                               title=f"{feature} by Cluster", labels={'x': 'Cluster', 'y': f'Mean {feature}'},
                               color=means.index.astype(str), color_discrete_map=CLUSTER_COLORS)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("**Summary per Cluster:**")
                summary_data = []
                for cluster in sorted(df_clustered["cluster"].unique()):
                    cluster_data = df_clustered[df_clustered["cluster"] == cluster]
                    summary_data.append({
                        "Cluster": str(cluster),
                        "Size": len(cluster_data),
                        "Percentage": f"{len(cluster_data)/len(df_clustered)*100:.1f}%"
                    })
                st.table(pd.DataFrame(summary_data))
            
            # Cluster profiles
            st.markdown("##### 🏷️ Label & Interpretasi Cluster")
            means_df = cluster_stats.xs('mean', axis=1, level=1)
            
            # Determine dataset type and create cluster labels
            selected_dataset = st.session_state.get('selected_dataset', '').lower()
            
            # Define cluster labels based on dataset
            def get_cluster_label(cluster_id, dataset_type, means_df, selected_features):
                """Generate cluster label based on dataset type and ACTUAL cluster characteristics
                   Label ditentukan berdasarkan nilai rata-rata fitur, BUKAN nomor cluster
                """
                # Hitung rata-rata semua fitur per cluster
                cluster_means_avg = means_df.mean(axis=1)
                sorted_clusters = cluster_means_avg.sort_values()
                n_clusters = len(sorted_clusters)
                
                # Tentukan rank cluster (0 = terendah, n-1 = tertinggi berdasarkan nilai)
                rank = list(sorted_clusters.index).index(cluster_id)
                
                if dataset_type == "kesehatan" or "health" in dataset_type:
                    # For health data: mortality TINGGI = High-Burden (buruk)
                    # Cluster dengan rata-rata TINGGI = risiko tinggi
                    if n_clusters == 2:
                        if rank == 0:  # Nilai terendah = risiko rendah
                            return "Low-Burden (Risiko Rendah)", "🔵"
                        else:  # Nilai tertinggi = risiko tinggi
                            return "High-Burden (Risiko Tinggi)", "🟡"
                    else:
                        # More than 2 clusters
                        if rank < n_clusters / 3:
                            return "Low-Burden (Risiko Rendah)", "🔵"
                        elif rank < 2 * n_clusters / 3:
                            return "Medium-Burden (Risiko Sedang)", "🟢"
                        else:
                            return "High-Burden (Risiko Tinggi)", "🟡"
                            
                elif dataset_type == "lingkungan" or "deforest" in dataset_type or "environment" in dataset_type:
                    # For environment data: loss TINGGI = Hotspot (buruk)
                    # Cluster dengan rata-rata TINGGI = deforestasi tinggi
                    if n_clusters == 2:
                        if rank == 0:  # Nilai terendah = konservasi baik
                            return "Low Deforestation Zone (Konservasi Baik)", "🔵"
                        else:  # Nilai tertinggi = hotspot
                            return "Deforestation Hotspot (Deforestasi Tinggi)", "🟡"
                    else:
                        if rank < n_clusters / 3:
                            return "Low Deforestation Zone", "🔵"
                        elif rank < 2 * n_clusters / 3:
                            return "Moderate Deforestation", "🟢"
                        else:
                            return "Deforestation Hotspot", "🟡"
                else:
                    # Default
                    if n_clusters == 2:
                        if rank == 0:
                            return f"Cluster (Kategori Rendah)", "🔵"
                        else:
                            return f"Cluster (Kategori Tinggi)", "🟡"
                    else:
                        return f"Cluster {cluster_id}", "⚪"
            
            for cluster_id in sorted(df_clustered["cluster"].unique()):
                cluster_means = means_df.loc[cluster_id]
                cluster_size = len(df_clustered[df_clustered["cluster"] == cluster_id])
                pct = cluster_size / len(df_clustered) * 100
                
                # Get cluster label
                cluster_label, emoji = get_cluster_label(cluster_id, selected_dataset, means_df, selected_features)
                
                high_features = []
                low_features = []
                for feat in selected_features[:5]:
                    if feat in means_df.columns:
                        overall_mean = df_clustered[feat].mean()
                        if cluster_means[feat] > overall_mean * 1.2:
                            high_features.append(feat)
                        elif cluster_means[feat] < overall_mean * 0.8:
                            low_features.append(feat)
                
                if high_features:
                    profile = f"Tinggi: {', '.join(high_features[:3])}"
                elif low_features:
                    profile = f"Rendah: {', '.join(low_features[:3])}"
                else:
                    profile = "Rata-rata (mendekati nilai tengah)"
                
                with st.container(border=True):
                    st.markdown(f"**{emoji} Cluster {cluster_id}: {cluster_label}** ({cluster_size} data • {pct:.1f}%)")
                    st.markdown(f"📊 **Karakteristik:** {profile}")
                
        except Exception as e:
            st.warning(f"Tidak dapat menghitung karakteristik cluster: {str(e)}")
    
    st.divider()
    
    # ===================== METRICS COMPARISON =====================
    st.markdown("#### ⚖️ Perbandingan Metrik Evaluasi Antar Algoritma")
    st.info("""
    - **Silhouette Score** (-1 s.d. 1): Semakin tinggi semakin baik
    - **Davies-Bouldin Index**: Semakin rendah semakin baik
    - **Calinski-Harabasz Index**: Semakin tinggi semakin baik
    """)
    
    if 'exploration_results' in st.session_state:
        metrics_data = []
        for method, res in st.session_state['exploration_results'].items():
            if res.get("success", False) or "error" not in res:
                sil = res.get("silhouette_excl_noise") or res.get("silhouette", 0)
                db = res.get("davies_bouldin_excl_noise") or res.get("davies_bouldin", 0)
                ch = res.get("calinski_harabasz_excl_noise") or res.get("calinski_harabasz", 0)
                noise = res.get("noise_ratio", 0)
                sil_pen = res.get("silhouette_penalized", sil * (1 - noise) if sil else 0)
                
                metrics_data.append({
                    "Method": method,
                    "Clusters": res.get("n_clusters", 0),
                    "Noise%": noise * 100 if noise else 0,
                    "Silhouette": sil if isinstance(sil, (int, float)) else 0,
                    "Sil(penalized)": sil_pen if isinstance(sil_pen, (int, float)) else 0,
                    "Davies-Bouldin": db if isinstance(db, (int, float)) else 0,
                    "Calinski-Harabasz": ch if isinstance(ch, (int, float)) else 0,
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            st.markdown("**Tabel Perbandingan:**")
            st.dataframe(metrics_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_sil = px.bar(metrics_df.sort_values("Sil(penalized)", ascending=False),
                               x="Method", y="Sil(penalized)", title="Silhouette Score",
                               color="Sil(penalized)", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig_sil, use_container_width=True)
            with col2:
                fig_noise = px.bar(metrics_df.sort_values("Noise%", ascending=True),
                                 x="Method", y="Noise%", title="Noise Ratio (%)",
                                 color="Noise%", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig_noise, use_container_width=True)
            
            st.divider()
            
            # ===================== KESIMPULAN =====================
            st.markdown("#### 📋 KESIMPULAN & REKOMENDASI")
            
            best_row = metrics_df.loc[metrics_df['Sil(penalized)'].idxmax()]
            best_method_name = best_row['Method']
            best_sil = best_row['Sil(penalized)']
            best_db = best_row['Davies-Bouldin']
            best_ch = best_row['Calinski-Harabasz']
            best_clusters = best_row['Clusters']
            
            if best_sil >= 0.5:
                quality = "Sangat Baik"
            elif best_sil >= 0.3:
                quality = "Baik"
            elif best_sil >= 0.1:
                quality = "Cukup"
            else:
                quality = "Perlu Perbaikan"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏆 Algoritma Terbaik", best_method_name)
                st.metric("📊 Jumlah Cluster", int(best_clusters))
            with col2:
                st.metric("⭐ Kualitas Clustering", quality)
                st.metric("📈 Silhouette Score", f"{best_sil:.4f}")
            
            st.success(f"""
            **Interpretasi:** Algoritma **{best_method_name}** memberikan hasil clustering terbaik 
            dengan kualitas **{quality}**. Data berhasil dibagi menjadi **{int(best_clusters)} cluster** 
            dengan Silhouette Score **{best_sil:.4f}**, Davies-Bouldin **{best_db:.4f}**, 
            dan Calinski-Harabasz **{best_ch:.1f}**.
            """)
            
            # ===================== KESIMPULAN DETAIL =====================
            st.markdown("---")
            st.markdown("##### 📖 Kesimpulan Detail & Insight")
            
            # Get dataset type
            selected_dataset = st.session_state.get('selected_dataset', '').lower()
            
            with st.container(border=True):
                st.markdown("**🎯 Ringkasan Analisis Clustering:**")
                
                # Analisis per cluster
                cluster_insights = []
                for cluster_id in sorted(df_clustered["cluster"].unique()):
                    cluster_data = df_clustered[df_clustered["cluster"] == cluster_id]
                    cluster_size = len(cluster_data)
                    pct = cluster_size / len(df_clustered) * 100
                    
                    # Determine cluster characteristic
                    if selected_features:
                        cluster_means = cluster_data[selected_features].mean()
                        overall_means = df_clustered[selected_features].mean()
                        
                        high_vals = []
                        low_vals = []
                        for feat in selected_features[:4]:
                            if cluster_means[feat] > overall_means[feat] * 1.2:
                                high_vals.append(feat)
                            elif cluster_means[feat] < overall_means[feat] * 0.8:
                                low_vals.append(feat)
                        
                        if high_vals:
                            characteristic = f"Nilai tinggi pada: {', '.join(high_vals[:2])}"
                        elif low_vals:
                            characteristic = f"Nilai rendah pada: {', '.join(low_vals[:2])}"
                        else:
                            characteristic = "Nilai mendekati rata-rata"
                    else:
                        characteristic = "Profil cluster"
                    
                    cluster_insights.append({
                        "id": cluster_id,
                        "size": cluster_size,
                        "pct": pct,
                        "char": characteristic
                    })
                
                for insight in cluster_insights:
                    cid = insight["id"]
                    if cid == -1:
                        st.markdown(f"- **Noise/Outlier:** {insight['size']} data ({insight['pct']:.1f}%) - Data dengan pola tidak umum")
                    else:
                        st.markdown(f"- **Cluster {cid}:** {insight['size']} data ({insight['pct']:.1f}%) - {insight['char']}")
            
            # Dataset-specific conclusions
            with st.container(border=True):
                if "kesehatan" in selected_dataset or "health" in selected_dataset:
                    st.markdown("**🏥 Kesimpulan Dataset Kesehatan (Child Mortality):**")
                    st.markdown(f"""
                    Berdasarkan analisis clustering menggunakan **{best_method_name}**, negara-negara di dunia 
                    dapat dikelompokkan menjadi **{int(best_clusters)} cluster** berdasarkan tingkat kematian anak.
                    
                    **Insight Utama:**
                    - Cluster dengan nilai tinggi menunjukkan negara dengan tingkat kematian anak yang masih tinggi
                    - Cluster dengan nilai rendah menunjukkan negara dengan sistem kesehatan yang lebih baik
                    - Trend (kecenderungan penurunan/kenaikan) membantu mengidentifikasi progress setiap negara
                    
                    **Implikasi Kebijakan:**
                    - Negara di cluster risiko tinggi membutuhkan intervensi kesehatan prioritas
                    - Program dapat difokuskan berdasarkan karakteristik cluster
                    - Monitoring progress dapat dilakukan dengan membandingkan perpindahan cluster antar waktu
                    """)
                    
                elif "lingkungan" in selected_dataset or "deforest" in selected_dataset:
                    st.markdown("**🌳 Kesimpulan Dataset Lingkungan (Deforestasi):**")
                    st.markdown(f"""
                    Berdasarkan analisis clustering menggunakan **{best_method_name}**, negara-negara di dunia 
                    dapat dikelompokkan menjadi **{int(best_clusters)} cluster** berdasarkan tingkat deforestasi.
                    
                    **Insight Utama:**
                    - Cluster dengan loss intensity tinggi menunjukkan negara dengan deforestasi parah
                    - Cluster dengan loss intensity rendah menunjukkan negara dengan konservasi lebih baik
                    - Area hutan dan total loss menentukan posisi relatif setiap negara
                    
                    **Implikasi Kebijakan:**
                    - Negara di cluster deforestasi tinggi membutuhkan intervensi konservasi segera
                    - Best practice dari negara cluster rendah dapat diadopsi
                    - Program reboisasi dapat diprioritaskan berdasarkan cluster
                    """)
                else:
                    st.markdown("**📊 Kesimpulan Analisis:**")
                    st.markdown(f"""
                    Data berhasil dikelompokkan menjadi **{int(best_clusters)} cluster** menggunakan algoritma **{best_method_name}**.
                    
                    **Insight Utama:**
                    - Setiap cluster memiliki karakteristik berbeda berdasarkan kombinasi fitur yang digunakan
                    - Kualitas clustering **{quality}** menunjukkan pemisahan cluster yang {'optimal' if best_sil >= 0.3 else 'cukup baik'}
                    - Hasil ini dapat digunakan untuk segmentasi dan analisis lebih lanjut
                    """)
            
            st.info("""
            **Langkah Selanjutnya:**
            - Gunakan hasil clustering untuk analisis lanjutan dan pengambilan keputusan berbasis data
            - Perhatikan karakteristik setiap cluster untuk menyusun strategi yang tepat sasaran
            - Gunakan fitur Prediksi untuk mengklasifikasikan data baru ke cluster yang sesuai
            """)
    
    st.divider()
    
    # Download
    col1, col2 = st.columns(2)
    with col1:
        csv = df_clustered.to_csv(index=False)
        st.download_button("📥 Download Cluster Results (CSV)", csv,
                          f"clustering_results_{best_method}.csv", "text/csv",
                          use_container_width=True)
    with col2:
        if st.button("🔄 Refresh Visualizations", use_container_width=True):
            st.rerun()
