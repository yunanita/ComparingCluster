import streamlit as st

def contact_tab():
    st.markdown('<div class="section-header">👤 Kontak & Informasi</div>', unsafe_allow_html=True)
    st.caption("Hubungi saya untuk pertanyaan atau kolaborasi")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            st.image("C:/Users/LENOVO/OneDrive/Pictures/Saved Pictures/aku.jpeg", use_container_width=True)
        except:
            st.info("📷 Foto tidak tersedia")
    
    with col2:
        st.markdown("### 🌿 Novia Yunanita")
        
        st.markdown("""
        - 🎓 **Mahasiswa Sains Data**
        - 🏫 Universitas Muhammadiyah Semarang
        - 📧 noviayuna4@gmail.com
        - 📱 +62 859-7515-9194
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.link_button("🔗 LinkedIn", "https://linkedin.com/in/noviayunanita", use_container_width=True)
        with col_b:
            st.link_button("🐙 GitHub", "https://github.com/yunanita", use_container_width=True)
    
    st.divider()
    
    # About me section
    st.markdown("#### 📝 Tentang Saya")
    
    st.info("""
    Saya adalah mahasiswa Sains Data di Universitas Muhammadiyah Semarang dengan minat kuat 
    di bidang Machine Learning dan Data Science. Dashboard ini dibuat sebagai bagian dari 
    tugas akhir semester (UAS) untuk mendemonstrasikan kemampuan analisis clustering pada 
    dataset kesehatan dan lingkungan. 🎯✨
    """)
    
    st.divider()
    
    # Project info
    st.markdown("#### 📌 Informasi Project")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        with st.container(border=True):
            st.markdown("**🎓 Mata Kuliah**")
            st.markdown("Machine Learning")
            st.markdown("**👨‍🏫 Dosen Pengampu**")
            st.markdown("Saeful Amri, S.Kom., M.Kom")
    
    with col_p2:
        with st.container(border=True):
            st.markdown("**📅 Tanggal Submit**")
            st.markdown("15 Januari 2026")
            st.markdown("**🏫 Universitas**")
            st.markdown("Universitas Muhammadiyah Semarang")
    
    st.divider()
    
    # Tools
    st.markdown("#### 🛠️ Tools & Technologies")
    
    tools = ["Python", "Streamlit", "Scikit-learn", "Pandas", "Plotly", "NumPy"]
    cols = st.columns(6)
    for i, tool in enumerate(tools):
        with cols[i]:
            st.metric(label="", value=tool)
