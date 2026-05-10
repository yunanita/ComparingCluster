# 🌿 Dashboard Clustering Kesehatan & Lingkungan

**Tugas Akhir Semester - Machine Learning | Universitas Muhammadiyah Semarang**
akses public: https://comparecluster.streamlit.app/

## 📋 Deskripsi Project

Aplikasi web interaktif untuk menganalisis dan membandingkan 5 algoritma clustering machine learning pada dua dataset berbeda:

- **Dataset Kesehatan**: Data mortalitas anak dari 195+ negara (NutriData - WHO/UNICEF/World Bank)
- **Dataset Lingkungan**: Data deforestasi global (Global Forest Watch - WRI)

Dashboard ini memungkinkan pengguna untuk melakukan exploratory data analysis (EDA), preprocessing, clustering dengan berbagai algoritma, visualisasi hasil, dan prediksi cluster baru.

---

## 🎯 Fitur Utama

### 1. **📊 Dataset Management**

- Pilih antara 2 dataset bawaan (Kesehatan atau Lingkungan)
- Upload dataset custom dalam format CSV/Excel
- Preview data dengan statistik deskriptif
- Koneksi langsung ke sumber data asli

### 2. **🔧 Preprocessing & EDA**

- **Exploratory Data Analysis**: Informasi dataset, statistik, distribusi, dan korelasi
- **Data Cleaning**: Handling missing values dengan SimpleImputer
- **Feature Engineering**: Transformasi fitur khusus untuk setiap dataset
- **Normalisasi**: StandardScaler dan MinMaxScaler
- **Dimensionality Reduction**: PCA untuk visualisasi 2D/3D

### 3. **🤖 5 Algoritma Clustering**

1. **K-Means** - Partitioning based clustering
2. **Hierarchical Clustering** - Agglomerative approach
3. **DBSCAN** - Density-based clustering
4. **OPTICS** - Extended DBSCAN dengan density distance
5. **Gaussian Mixture Model (GMM)** - Probabilistic clustering

### 4. **📈 Evaluasi & Perbandingan**

- **Internal Validation Metrics**:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Inertia / Within-cluster sum of squares
- **Automatic K Selection**: Elbow method, Silhouette analysis
- **Comparative Dashboard**: Side-by-side comparison hasil clustering

### 5. **🗺️ Visualisasi Interaktif**

- **Scatter Plot 2D/3D**: Dengan PCA dimension reduction
- **Choropleth Map**: Distribusi cluster berdasarkan wilayah geografis
- **Cluster Distribution**: Bar chart dan summary statistics
- **Heatmap Korelasi**: Feature correlation analysis

### 6. **🔮 Prediksi Cluster**

- Prediksi cluster untuk data point baru
- Input manual atau batch upload
- Rekomendasi interpretasi hasil clustering
- Export prediksi dalam format CSV

---

## 📁 Struktur File

```
ComparingCluster/
├── uas.py                      # Main Streamlit app
├── about.py                    # Dataset selection & info
├── about_project.py            # Project documentation
├── preprocessing.py            # EDA & preprocessing
├── ml_5_algoritma.py          # Clustering algorithms
├── visualisasi.py             # Visualization & analysis
├── prediksi.py                # Prediction module
├── contact.py                 # Contact & creator info
│
├── requirements.txt            # Python dependencies
├── streamlit_config.toml      # Streamlit configuration
├── .gitignore                 # Git ignore rules
└── README.md                  # This file

# Data Files (Required)
├── child_mortality.xlsx       # Health dataset
├── deforestasi.xlsx           # Environment dataset
└── nutridata.png              # NutriData logo/reference image
```

---

## 🚀 Installation & Setup

### 1. Clone atau Download Repository

```bash
git clone <repository-url>
cd ComparingCluster
```

### 2. Create Virtual Environment (Optional but recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data Files

Pastikan file berikut ada di folder `ComparingCluster/`:

- `child_mortality.xlsx` - Dataset kesehatan
- `deforestasi.xlsx` - Dataset lingkungan

**Optional - Generate Image Files:**

Jika ingin menampilkan logo GFW (Global Forest Watch) di section dataset lingkungan, jalankan:

```bash
python generate_gfw_image.py
```

Ini akan menggenerate `GFW.png` dari base64 data.

### 5. Run Streamlit App

```bash
streamlit run uas.py
```

Aplikasi akan terbuka di browser: `http://localhost:8501`

---

## 📊 Dataset Details

### **Dataset Kesehatan (Child Mortality)**

- **Sumber**: NutriData - WHO, UNICEF, World Bank
- **Cakupan**: 195+ negara, 1990-2023
- **Metrik**: Under-5 Mortality Rate (per 1000 kelahiran hidup)
- **Fitur**: Kode negara, nama negara, data mortalitas per tahun
- **Link**: https://aschimmenti.github.io/NutriData/metadata.html

### **Dataset Lingkungan (Deforestation)**

- **Sumber**: Global Forest Watch - World Resources Institute
- **Teknologi**: Satelit NASA & University of Maryland
- **Metrik**: Luas hutan, kehilangan hutan, laju deforestasi
- **Fitur**: Area total (ha), extent 2000-2010, gain/loss data, trend per tahun
- **Link**: https://www.globalforestwatch.org/dashboards/global/

---

## 🛠️ Technologies & Libraries

**Backend**:

- Python 3.8+
- Streamlit 1.51.0 - Web framework
- Pandas 2.3.3 - Data manipulation
- NumPy 2.3.4 - Numerical computing

**Machine Learning**:

- Scikit-learn 1.7.2 - ML algorithms & metrics
- SciPy 1.16.3 - Scientific computing
- Scikit-learn modules:
  - `cluster`: KMeans, AgglomerativeClustering, DBSCAN, OPTICS
  - `mixture`: GaussianMixture
  - `preprocessing`: StandardScaler, MinMaxScaler
  - `decomposition`: PCA
  - `metrics`: Silhouette, Davies-Bouldin, Calinski-Harabasz

**Visualization**:

- Plotly 6.3.1 - Interactive charts
- PyDeck 0.9.1 - Geographic visualization
- Matplotlib 3.10.7 - Static plots
- Seaborn 0.13.2 - Statistical visualization

**Data Formats**:

- OpenPyXL 3.12.5 - Excel file handling

---

## 💡 Workflow & Usage Guide

### **Step 1: Select Dataset**

Go to **"📊 Dataset"** tab → Pilih Kesehatan atau Lingkungan → Click "✅ Pilih Dataset"

### **Step 2: Preprocessing**

Go to **"🔧 Preprocessing"** tab:

- Review EDA (Info, Statistik, Distribusi, Korelasi)
- Handle missing values (auto-impute atau drop)
- Normalize data (StandardScaler/MinMaxScaler)
- Select features untuk clustering

### **Step 3: Clustering**

Go to **"🤖 Machine Learning"** tab:

- Select 1 atau lebih algoritma
- Optimize K (jika perlu)
- Run clustering
- Review metrics & comparison

### **Step 4: Visualize**

Go to **"📈 Visualisasi"** tab:

- Scatter plot 2D/3D dengan PCA
- Choropleth map (untuk data geografis)
- Cluster distribution
- Export results sebagai CSV

### **Step 5: Predict (Optional)**

Go to **"🔮 Prediksi"** tab:

- Input data baru (manual atau batch)
- Prediksi cluster untuk data tersebut
- Lihat similarity dengan existing clusters
- Download hasil prediksi

---

## 📊 Algoritma Clustering - Penjelasan Singkat

| Algoritma        | Tipe          | Kelebihan                       | Kekurangan                           |
| ---------------- | ------------- | ------------------------------- | ------------------------------------ |
| **K-Means**      | Partitioning  | Cepat, scalable                 | Perlu set K dulu, spherical clusters |
| **Hierarchical** | Hierarchical  | Dendrogram visualization        | Komputasi mahal, irreversible        |
| **DBSCAN**       | Density-based | Arbitrary shapes, deteksi noise | Sensitif dengan parameter eps        |
| **OPTICS**       | Density-based | Extension DBSCAN, lebih robust  | Lebih kompleks                       |
| **GMM**          | Probabilistic | Probabilistic assignment        | Assuming Gaussian distribution       |

---

## 📈 Metrics Explanation

- **Silhouette Score** [-1 to 1]: Semakin tinggi semakin baik (well-separated clusters)
- **Davies-Bouldin Index** [0 to ∞]: Semakin rendah semakin baik (compact clusters)
- **Calinski-Harabasz Index** [0 to ∞]: Semakin tinggi semakin baik (dense & well-separated)
- **Inertia**: Sum of squared distances dari setiap point ke centroid terdekatnya

---

## 🐛 Troubleshooting

### **Error: File not found**

- Pastikan `child_mortality.xlsx` dan `deforestasi.xlsx` ada di folder yang sama dengan script

### **Error: Memory issue pada large dataset**

- Kurangi jumlah fitur atau gunakan sample data
- Coba PCA dengan component lebih kecil

### **Missing value handling**

- Pilih "SimpleImputer" untuk auto-fill missing values
- Atau pilih "Drop" untuk hapus baris/kolom dengan missing values

---

## 📝 Project Structure - Code Organization

- **uas.py**: Entry point, page navigation, CSS styling
- **about.py**: Dataset loader, preview, file uploader
- **preprocessing.py**: EDA & data preprocessing functions
- **ml_5_algoritma.py**: 5 clustering algorithms implementation
- **visualisasi.py**: Plotly & PyDeck visualizations
- **prediksi.py**: Prediction module dengan model loading
- **about_project.py**: Project documentation & algorithm explanations
- **contact.py**: Contact information

---

## 👤 Author

**Novia Yunanita**

- 🎓 Mahasiswa Sains Data - Universitas Muhammadiyah Semarang
- 📧 noviayuna4@gmail.com
- 📱 +62 859-7515-9194
- 🔗 LinkedIn: https://linkedin.com/in/noviayunanita
- 🐙 GitHub: https://github.com/yunanita

---

## 📄 License

Project ini dibuat untuk keperluan akademik (Tugas Akhir Semester).

---

## 🙏 Acknowledgments

- **NutriData**: Global Health Database (WHO, UNICEF, World Bank)
- **Global Forest Watch**: World Resources Institute (WRI)
- **Streamlit**: Web framework for data apps
- **Scikit-learn**: Machine learning library
- **Plotly**: Interactive visualization

---

**Last Updated**: January 2026
**Status**: ✅ Ready for Deployment
