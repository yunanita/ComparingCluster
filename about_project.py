import streamlit as st
import pandas as pd

def about_tab():
    """
    Tab About - Menjelaskan pendahuluan project dan 5 metodologi algoritma clustering
    """
    
    st.markdown('<div class="section-header">📖 Tentang Project</div>', unsafe_allow_html=True)
    st.caption("Dashboard Clustering Kesehatan & Lingkungan - UAS Machine Learning")
    
    # =====================================================================
    # PENDAHULUAN PROJECT
    # =====================================================================
    with st.container(border=True):
        st.markdown("#### 📋 Pendahuluan Project")
        st.markdown("""
        Project ini merupakan implementasi **Machine Learning** untuk melakukan 
        **analisis clustering** pada dataset Kesehatan (Child Mortality) dan 
        Lingkungan (Deforestasi).
        """)
    
    # =====================================================================
    # INFORMASI DATASET
    # =====================================================================
    st.markdown('<div class="section-header-green">📊 Informasi Dataset</div>', unsafe_allow_html=True)
    
    tab_kes, tab_ling = st.tabs(["🏥 Dataset Kesehatan", "🌳 Dataset Lingkungan"])
    
    with tab_kes:
        st.markdown("##### 🏥 Dataset Child Mortality (Kematian Anak)")
        st.markdown("""
        Dataset ini berisi data tingkat kematian anak (per 1000 kelahiran hidup) dari berbagai negara 
        selama periode **1990-2019**.
        
        **Sumber Data:** World Health Organization (WHO) / UNICEF
        
        **Tujuan Analisis:**
        - Mengelompokkan negara berdasarkan pola kematian anak
        - Mengidentifikasi negara dengan trend perbaikan atau penurunan
        - Memberikan insight untuk prioritas intervensi kesehatan global
        """)
    
    with tab_ling:
        st.markdown("##### 🌳 Dataset Deforestasi (Kehilangan Hutan)")
        st.markdown("""
        Dataset ini berisi data kehilangan tutupan hutan dari berbagai negara 
        selama periode **2001-2022**.
        
        **Sumber Data:** Global Forest Watch / Hansen et al.
        
        **Tujuan Analisis:**
        - Mengelompokkan negara berdasarkan pola deforestasi
        - Mengidentifikasi hotspot deforestasi yang memerlukan perhatian
        - Evaluasi efektivitas konservasi hutan
        """)
    
    st.divider()
    
    st.markdown("##### 🎯 Tujuan Utama")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        with st.container(border=True):
            st.markdown("**1️⃣ Mengelompokkan Negara**")
            st.caption("Berdasarkan karakteristik data kesehatan dan lingkungan")
        with st.container(border=True):
            st.markdown("**2️⃣ Mengidentifikasi Pola**")
            st.caption("Hubungan antar variabel menggunakan algoritma clustering")
    with col_t2:
        with st.container(border=True):
            st.markdown("**3️⃣ Membandingkan Performa**")
            st.caption("5 algoritma clustering yang berbeda secara otomatis")
        with st.container(border=True):
            st.markdown("**4️⃣ Memberikan Insight**")
            st.caption("Untuk pengambilan keputusan berbasis data")
    
    # Features menggunakan st.columns
    st.markdown('<div class="section-header-green">🎯 Fitur Utama Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        with st.container(border=True):
            st.markdown("📊")
            st.markdown("**Dataset**")
            st.caption("Pilih & preview")
    with col2:
        with st.container(border=True):
            st.markdown("🔧")
            st.markdown("**Preprocessing**")
            st.caption("Clean & normalize")
    with col3:
        with st.container(border=True):
            st.markdown("🤖")
            st.markdown("**ML**")
            st.caption("5 algoritma")
    with col4:
        with st.container(border=True):
            st.markdown("📈")
            st.markdown("**Visualisasi**")
            st.caption("Charts interaktif")
    with col5:
        with st.container(border=True):
            st.markdown("🔮")
            st.markdown("**Prediksi**")
            st.caption("Data baru")
    
    # =====================================================================
    # 5 METODOLOGI ALGORITMA CLUSTERING
    # =====================================================================
    st.markdown('<div class="section-header-green">🔬 5 Metodologi Algoritma Clustering</div>', unsafe_allow_html=True)
    
    st.success("""
    ✨ Dalam project ini, kami mengimplementasikan **5 algoritma clustering** yang 
    **tidak memerlukan parameter jumlah cluster (K) secara manual**. 
    Setiap algoritma menentukan jumlah cluster secara **OTOMATIS** berdasarkan karakteristik data.
    """)
    
    # =====================================================================
    # 1. DBSCAN
    # =====================================================================
    with st.container(border=True):
        st.markdown("##### 1️⃣ DBSCAN - Density-Based Spatial Clustering")
        st.caption("Density-Based Spatial Clustering of Applications with Noise")
        
        st.markdown("""
        **DBSCAN** adalah algoritma clustering berbasis *kepadatan (density-based)* yang 
        mengelompokkan titik-titik data yang berdekatan. Algoritma ini dapat menemukan cluster dengan 
        **bentuk arbitrer** dan mengidentifikasi **outlier (noise)** secara otomatis.
        """)
        
        st.markdown("**📐 Rumus Matematika:**")
        st.latex(r"N_{\varepsilon}(p) = \{q \in D \mid dist(p,q) \leq \varepsilon\}")
        st.markdown("Dimana $N_ε(p)$ adalah neighborhood dari titik p dengan radius ε")
        
        st.markdown("**Definisi Core Point:**")
        st.latex(r"|N_{\varepsilon}(p)| \geq MinPts")
        
        st.markdown("**Jarak Euclidean:**")
        st.latex(r"dist(p,q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}")
        
        st.markdown("**⚙️ Parameter Utama:**")
        param_dbscan = pd.DataFrame({
            "Parameter": ["eps (ε)", "min_samples"],
            "Deskripsi": ["Radius maksimum untuk mencari tetangga dari suatu titik", 
                         "Jumlah minimum titik dalam radius eps untuk membentuk core point"]
        })
        st.table(param_dbscan)
        
        st.markdown("**📋 Langkah-langkah Metodologi:**")
        st.markdown("""
        1. **Inisialisasi Parameter**: Tentukan nilai `eps` (radius) dan `min_samples`. Semua titik ditandai sebagai "belum dikunjungi".
        2. **Pilih Titik Acak**: Pilih satu titik yang belum dikunjungi secara acak.
        3. **Cari Tetangga**: Hitung semua titik dalam radius `eps` dari titik tersebut.
        4. **Evaluasi Core Point**: Jika jumlah tetangga ≥ `min_samples`, titik ini adalah **Core Point** → buat cluster baru.
        5. **Ekspansi Cluster**: Untuk setiap tetangga dari core point, jika belum dikunjungi, tambahkan ke cluster. Jika tetangga juga core point, ekspansi rekursif.
        6. **Identifikasi Border Point**: Titik yang bukan core point tapi berada dalam radius core point → **Border Point**.
        7. **Identifikasi Noise**: Titik yang tidak termasuk cluster manapun → **Noise** (label = -1).
        8. **Ulangi**: Kembali ke langkah 2 hingga semua titik dikunjungi.
        9. **Output**: Setiap titik memiliki label cluster (0, 1, 2, ...) atau noise (-1). Jumlah cluster ditentukan otomatis.
        """)
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("**✅ Kelebihan:**")
            st.markdown("""
            - Tidak perlu menentukan K
            - Menemukan cluster bentuk arbitrer
            - Identifikasi outlier otomatis
            """)
        with col_k2:
            st.markdown("**❌ Kekurangan:**")
            st.markdown("""
            - Sensitif terhadap parameter eps
            - Kesulitan dengan densitas bervariasi
            - Membutuhkan tuning parameter
            """)
    
    # =====================================================================
    # 2. OPTICS
    # =====================================================================
    with st.container(border=True):
        st.markdown("##### 2️⃣ OPTICS - Ordering Points To Identify Clustering Structure")
        st.caption("Ordering Points To Identify the Clustering Structure")
        
        st.markdown("""
        **OPTICS** adalah pengembangan dari DBSCAN yang dapat menangani cluster dengan 
        **kepadatan bervariasi**. Algoritma ini menghasilkan *reachability plot* yang 
        memvisualisasikan struktur cluster dalam data.
        """)
        
        st.markdown("**📐 Rumus Matematika:**")
        st.markdown("**Core Distance:**")
        st.latex(r"core\text{-}dist_{\varepsilon,MinPts}(p) = \begin{cases} UNDEFINED & \text{if } |N_\varepsilon(p)| < MinPts \\ d(p, N_{MinPts}(p)) & \text{otherwise} \end{cases}")
        
        st.markdown("**Reachability Distance:**")
        st.latex(r"reach\text{-}dist_{\varepsilon,MinPts}(o, p) = \max(core\text{-}dist(o), dist(o,p))")
        
        st.markdown("Dimana $d(p, N_{MinPts}(p))$ adalah jarak ke tetangga ke-MinPts terdekat")
        
        st.markdown("**⚙️ Parameter Utama:**")
        param_optics = pd.DataFrame({
            "Parameter": ["min_samples", "max_eps", "xi"],
            "Deskripsi": ["Jumlah minimum sampel untuk core point", 
                         "Radius maksimum (default: ∞)",
                         "Parameter steepness untuk ekstraksi cluster"]
        })
        st.table(param_optics)
        
        st.markdown("**📋 Langkah-langkah Metodologi:**")
        st.markdown("""
        1. **Inisialisasi**: Tentukan `min_samples`. Set `reachability-distance` = ∞ dan `core-distance` = undefined untuk semua titik.
        2. **Hitung Core Distance**: Untuk setiap titik, hitung jarak ke tetangga ke-`min_samples` terdekat. Ini adalah core distance.
        3. **Pilih Titik Awal**: Pilih titik yang belum diproses dengan reachability-distance terkecil.
        4. **Update Reachability**: Untuk setiap tetangga yang belum diproses, update reachability-distance = max(core-distance, jarak ke titik saat ini).
        5. **Simpan Urutan**: Simpan titik dalam ordered list beserta reachability-distance-nya.
        6. **Ulangi**: Kembali ke langkah 3 hingga semua titik diproses.
        7. **Buat Reachability Plot**: Plot reachability-distance untuk setiap titik dalam urutan pemrosesan.
        8. **Ekstraksi Cluster**: Gunakan parameter `xi` untuk mendeteksi "lembah" dalam reachability plot → setiap lembah = 1 cluster.
        9. **Output**: Setiap titik memiliki label cluster. Cluster hierarkis dapat diidentifikasi dari plot.
        """)
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("**✅ Kelebihan:**")
            st.markdown("""
            - Menangani densitas bervariasi
            - Visualisasi struktur cluster
            - Lebih fleksibel dari DBSCAN
            """)
        with col_k2:
            st.markdown("**❌ Kekurangan:**")
            st.markdown("""
            - Lebih lambat dari DBSCAN
            - Parameter xi perlu tuning
            - Interpretasi plot kompleks
            """)
    
    # =====================================================================
    # 3. K-Means dengan Elbow/Silhouette
    # =====================================================================
    with st.container(border=True):
        st.markdown("##### 3️⃣ K-Means - Automatic K Selection")
        st.caption("Partitioning Clustering dengan Elbow Method & Silhouette Score")
        
        st.markdown("""
        **K-Means** adalah algoritma clustering partitional yang membagi data menjadi K cluster.
        Dalam implementasi ini, nilai **K optimal ditentukan secara otomatis** menggunakan 
        kombinasi **Elbow Method** dan **Silhouette Score**.
        """)
        
        st.markdown("**📐 Rumus Matematika:**")
        st.markdown("**Objective Function (Inertia/WCSS):**")
        st.latex(r"J = \sum_{j=1}^{K} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2")
        
        st.markdown("**Update Centroid:**")
        st.latex(r"\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i")
        
        st.markdown("**Assignment Rule:**")
        st.latex(r"C_j = \{x_i : \|x_i - \mu_j\|^2 \leq \|x_i - \mu_l\|^2, \forall l\}")
        
        st.markdown("Dimana $J$ adalah total within-cluster sum of squares, $μ_j$ adalah centroid cluster j")
        
        st.markdown("**⚙️ Parameter Utama:**")
        param_kmeans = pd.DataFrame({
            "Parameter": ["n_clusters (K)", "max_iter", "n_init"],
            "Deskripsi": ["Jumlah cluster (ditentukan otomatis)", 
                         "Iterasi maksimum per run",
                         "Jumlah inisialisasi dengan centroid berbeda"]
        })
        st.table(param_kmeans)
        
        st.markdown("**📋 Langkah-langkah Metodologi:**")
        st.markdown("""
        1. **Tentukan Range K**: Set range K yang akan diuji, misal K = 2 sampai K = 10.
        2. **Loop untuk Setiap K**: Untuk setiap nilai K dalam range:
           - Inisialisasi K centroid secara acak
           - **Assignment Step**: Assign setiap titik ke centroid terdekat (berdasarkan jarak Euclidean)
           - **Update Step**: Hitung ulang posisi centroid = mean dari semua titik dalam cluster
           - Ulangi assignment & update hingga konvergen (centroid tidak berubah)
           - Hitung **Inertia** (sum of squared distances) dan **Silhouette Score**
        3. **Elbow Method**: Plot Inertia vs K. Identifikasi titik "siku" dimana penurunan inertia mulai melambat.
        4. **Silhouette Validation**: Pilih K dengan silhouette score tertinggi sebagai validasi tambahan.
        5. **Pilih K Optimal**: Kombinasikan hasil Elbow dan Silhouette untuk menentukan K terbaik.
        6. **Final Clustering**: Jalankan K-Means dengan K optimal yang telah ditentukan.
        7. **Output**: Setiap titik memiliki label cluster (0 hingga K-1).
        """)
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("**✅ Kelebihan:**")
            st.markdown("""
            - Cepat dan efisien
            - K ditentukan otomatis
            - Mudah diinterpretasi
            """)
        with col_k2:
            st.markdown("**❌ Kekurangan:**")
            st.markdown("""
            - Asumsi cluster spherical
            - Sensitif terhadap outlier
            - Hasil bisa berbeda tiap run
            """)
    
    # =====================================================================
    # 4. Agglomerative Clustering
    # =====================================================================
    with st.container(border=True):
        st.markdown("##### 4️⃣ Agglomerative - Hierarchical Clustering")
        st.caption("Bottom-Up Hierarchical Clustering dengan Dendrogram")
        
        st.markdown("""
        **Agglomerative Clustering** adalah algoritma *hierarchical clustering* yang 
        menggunakan pendekatan **bottom-up**. Dimulai dengan setiap titik sebagai cluster 
        individual, kemudian secara iteratif menggabungkan cluster terdekat.
        """)
        
        st.markdown("**📐 Rumus Matematika:**")
        st.markdown("**Single Linkage:**")
        st.latex(r"d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)")
        
        st.markdown("**Complete Linkage:**")
        st.latex(r"d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)")
        
        st.markdown("**Average Linkage:**")
        st.latex(r"d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)")
        
        st.markdown("**Ward's Method (Minimize Variance):**")
        st.latex(r"\Delta(C_i, C_j) = \frac{|C_i||C_j|}{|C_i|+|C_j|} \|\mu_i - \mu_j\|^2")
        
        st.markdown("**⚙️ Parameter Utama:**")
        param_agglo = pd.DataFrame({
            "Parameter": ["n_clusters", "linkage", "distance_threshold"],
            "Deskripsi": ["Jumlah cluster (otomatis via dendrogram)", 
                         "Metode linkage: ward, complete, average, single",
                         "Threshold jarak untuk penentuan K otomatis"]
        })
        st.table(param_agglo)
        
        st.markdown("**📋 Langkah-langkah Metodologi:**")
        st.markdown("""
        1. **Inisialisasi**: Setiap data point menjadi cluster individual. Jika ada N data, maka ada N cluster.
        2. **Hitung Matriks Jarak**: Hitung jarak antar semua pasangan cluster menggunakan metode linkage:
           - **Single**: Jarak minimum antar titik dari 2 cluster
           - **Complete**: Jarak maksimum antar titik dari 2 cluster
           - **Average**: Rata-rata jarak antar titik dari 2 cluster
           - **Ward**: Minimize variance dalam cluster (paling umum digunakan)
        3. **Cari Pasangan Terdekat**: Temukan dua cluster dengan jarak terkecil.
        4. **Merge Cluster**: Gabungkan dua cluster tersebut menjadi satu cluster baru.
        5. **Update Matriks Jarak**: Hitung ulang jarak antara cluster baru dengan cluster lainnya.
        6. **Ulangi**: Kembali ke langkah 3 hingga tersisa 1 cluster atau mencapai threshold.
        7. **Buat Dendrogram**: Visualisasikan hierarki penggabungan cluster dalam bentuk tree diagram.
        8. **Tentukan K Optimal**: Cut dendrogram di level tertentu berdasarkan jarak atau jumlah cluster yang diinginkan.
        9. **Output**: Setiap titik memiliki label cluster berdasarkan hasil cut dendrogram.
        """)
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("**✅ Kelebihan:**")
            st.markdown("""
            - Menghasilkan dendrogram
            - Tidak perlu tentukan K awal
            - Cluster hierarkis
            """)
        with col_k2:
            st.markdown("**❌ Kekurangan:**")
            st.markdown("""
            - Kompleksitas O(n²) atau lebih
            - Tidak bisa undo merge
            - Sensitif terhadap noise
            """)
    
    # =====================================================================
    # 5. Mean Shift
    # =====================================================================
    with st.container(border=True):
        st.markdown("##### 5️⃣ Mean Shift - Mode-Seeking Algorithm")
        st.caption("Non-parametric Mode-Seeking Clustering Algorithm")
        
        st.markdown("""
        **Mean Shift** adalah algoritma *non-parametric* yang mencari **mode** (puncak) dari 
        distribusi data. Algoritma ini **tidak memerlukan jumlah cluster** dan menemukan 
        cluster berdasarkan kepadatan lokal data.
        """)
        
        st.markdown("**📐 Rumus Matematika:**")
        st.markdown("**Mean Shift Vector:**")
        st.latex(r"m(x) = \frac{\sum_{x_i \in N(x)} K(x_i - x) \cdot x_i}{\sum_{x_i \in N(x)} K(x_i - x)} - x")
        
        st.markdown("**Gaussian Kernel:**")
        st.latex(r"K(x) = e^{-\frac{\|x\|^2}{2h^2}}")
        
        st.markdown("**Update Posisi:**")
        st.latex(r"x^{(t+1)} = x^{(t)} + m(x^{(t)})")
        
        st.markdown("**⚙️ Parameter Utama:**")
        param_meanshift = pd.DataFrame({
            "Parameter": ["bandwidth", "bin_seeding", "cluster_all"],
            "Deskripsi": ["Radius kernel RBF (otomatis via estimate_bandwidth)", 
                         "Seed initial centroids dari bins untuk mempercepat",
                         "Apakah semua titik harus masuk cluster"]
        })
        st.table(param_meanshift)
        
        st.markdown("**📋 Langkah-langkah Metodologi:**")
        st.markdown("""
        1. **Estimasi Bandwidth**: Hitung bandwidth optimal menggunakan metode seperti Scott's rule atau quantile-based estimation.
        2. **Inisialisasi Centroid**: Setiap data point (atau subset menggunakan bin_seeding) menjadi kandidat centroid awal.
        3. **Definisi Kernel**: Gunakan kernel (biasanya Gaussian/RBF) dengan radius = bandwidth.
        4. **Hitung Mean Shift Vector**: Untuk setiap centroid, hitung weighted mean dari semua titik dalam bandwidth:
           - Titik yang lebih dekat ke centroid memiliki bobot lebih tinggi
           - Mean shift vector = arah dari centroid ke weighted mean
        5. **Geser Centroid**: Pindahkan centroid ke posisi weighted mean yang baru dihitung.
        6. **Cek Konvergensi**: Jika perpindahan centroid < threshold → centroid sudah konvergen.
        7. **Ulangi**: Kembali ke langkah 4 hingga semua centroid konvergen.
        8. **Merge Centroid**: Gabungkan centroid yang berdekatan (jarak < bandwidth) menjadi satu.
        9. **Assign Label**: Setiap titik data di-assign ke centroid terdekat.
        10. **Output**: Jumlah cluster = jumlah centroid unik setelah merge. Setiap titik memiliki label cluster.
        """)
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.markdown("**✅ Kelebihan:**")
            st.markdown("""
            - K otomatis
            - Menemukan cluster arbitrer
            - Berbasis teori statistik
            """)
        with col_k2:
            st.markdown("**❌ Kekurangan:**")
            st.markdown("""
            - Bandwidth sensitif
            - Lambat untuk data besar
            - Tidak cocok untuk dimensi tinggi
            """)
    
    # =====================================================================
    # METRIK EVALUASI
    # =====================================================================
    st.markdown('<div class="section-header-green">📊 Metrik Evaluasi Clustering</div>', unsafe_allow_html=True)
    
    st.markdown("Untuk membandingkan performa kelima algoritma, digunakan **3 metrik evaluasi internal**:")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        with st.container(border=True):
            st.markdown("**📊 Silhouette Score**")
            st.caption("Range: -1 hingga 1")
            st.markdown("Semakin **tinggi** semakin baik")
            st.markdown("**Rumus:**")
            st.latex(r"s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}")
            st.caption("a(i) = rata-rata jarak ke titik dalam cluster yang sama")
            st.caption("b(i) = rata-rata jarak ke titik cluster terdekat lainnya")
    
    with col_m2:
        with st.container(border=True):
            st.markdown("**📉 Davies-Bouldin Index**")
            st.caption("Range: 0 hingga ∞")
            st.markdown("Semakin **rendah** semakin baik")
            st.markdown("**Rumus:**")
            st.latex(r"DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)} \right)")
            st.caption("σᵢ = rata-rata jarak titik ke centroid cluster i")
            st.caption("d(μᵢ,μⱼ) = jarak antar centroid")
    
    with col_m3:
        with st.container(border=True):
            st.markdown("**📈 Calinski-Harabasz**")
            st.caption("Range: 0 hingga ∞")
            st.markdown("Semakin **tinggi** semakin baik")
            st.markdown("**Rumus:**")
            st.latex(r"CH = \frac{SS_B / (k-1)}{SS_W / (n-k)}")
            st.caption("SSB = Between-cluster dispersion")
            st.caption("SSW = Within-cluster dispersion")
    
    # =====================================================================
    # WORKFLOW DASHBOARD
    # =====================================================================
    st.markdown('<div class="section-header-green">🔄 Workflow Dashboard</div>', unsafe_allow_html=True)
    
    # Workflow dengan visual cards
    wf1, wf2, wf3, wf4, wf5 = st.columns(5)
    
    with wf1:
        with st.container(border=True):
            st.markdown("📊 **Step 1**")
            st.markdown("Pilih Dataset")
            st.caption("Kesehatan/Lingkungan")
    
    with wf2:
        with st.container(border=True):
            st.markdown("🔧 **Step 2**")
            st.markdown("Preprocessing")
            st.caption("Cleaning & Normalisasi")
    
    with wf3:
        with st.container(border=True):
            st.markdown("🤖 **Step 3**")
            st.markdown("Run 5 Algoritma")
            st.caption("ML Clustering")
    
    with wf4:
        with st.container(border=True):
            st.markdown("📈 **Step 4**")
            st.markdown("Visualisasi")
            st.caption("Charts & Plots")
    
    with wf5:
        with st.container(border=True):
            st.markdown("🔮 **Step 5**")
            st.markdown("Prediksi")
            st.caption("Data Baru")
    
    st.info("💡 **Tips:** Mulai dari tab **Dataset** untuk memilih data, lalu lanjut ke **Preprocessing** untuk membersihkan data, dan jalankan **Machine Learning** untuk melihat hasil clustering!")
