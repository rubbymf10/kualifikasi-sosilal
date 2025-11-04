import streamlit as st
import pandas as pd
import joblib
import io
import folium
from streamlit_folium import st_folium
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Konfigurasi halaman (cukup sekali di seluruh app)
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# CSS untuk styling dan animasi yang lebih menarik
st.markdown("""
<style>
body {
    background-color: #ffffff !important;
    color: #333333 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
}
@keyframes bounceIn {
    0% { transform: scale(0.3); opacity: 0; }
    50% { transform: scale(1.05); }
    70% { transform: scale(0.9); }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes slideInLeft {
    from { transform: translateX(-100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
@keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
@keyframes wiggle {
    0%, 7% { transform: rotateZ(0); }
    15% { transform: rotateZ(-15deg); }
    20% { transform: rotateZ(10deg); }
    25% { transform: rotateZ(-10deg); }
    30% { transform: rotateZ(6deg); }
    35% { transform: rotateZ(-4deg); }
    40%, 100% { transform: rotateZ(0); }
}
.bounce-in {
    animation: bounceIn 1s ease-out;
}
.slide-left {
    animation: slideInLeft 1s ease-out;
}
.slide-right {
    animation: slideInRight 1s ease-out;
}
.fade-up {
    animation: fadeInUp 1s ease-out;
}
.pulse {
    animation: pulse 2s infinite;
}
.wiggle {
    animation: wiggle 2s ease-in-out infinite;
}
.dashboard-header {
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
    padding: 30px;
    border-radius: 15px;
    color: #333333;
    text-align: center;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 30px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    animation: bounceIn 1.5s ease-out;
}
.card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border: 2px solid #e0e0e0;
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    transition: all 0.4s ease;
    color: #333333;
}
.card:hover {
    transform: translateY(-10px) rotate(1deg);
    box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    background: linear-gradient(135deg, #c3cfe2 0%, #f5f7fa 100%);
}
.sidebar .stRadio > div {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 10px;
    padding: 15px;
    color: #333333;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    transform: scale(1.05);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}
.stDataFrame, .stTable {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stSuccess, .stWarning, .stInfo {
    border-radius: 10px;
    padding: 15px;
    color: #333333;
}
.stExpander {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.icon-bounce {
    display: inline-block;
    animation: wiggle 2s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Fungsi buat label otomatis
# ================================
def buat_label_kelayakan(df):
    kondisi = (
        (df["Pendapatan_Bulanan"] < 1500000) |
        (df["Jumlah_Anggota_Keluarga"] >= 5) |
        (df["Kepemilikan_Rumah"] == "Tidak")
    )
    df["Status_Kelayakan"] = kondisi.map({True: "Layak", False: "Tidak Layak"})
    return df

# ================================
# Fungsi training model
# ================================
def train_model(df):
    le_rumah = LabelEncoder()
    le_target = LabelEncoder()

    df["Kepemilikan_Rumah_encoded"] = le_rumah.fit_transform(df["Kepemilikan_Rumah"])
    df = buat_label_kelayakan(df)
    df["Status_Kelayakan_encoded"] = le_target.fit_transform(df["Status_Kelayakan"])

    X = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
            "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]
    y = df["Status_Kelayakan_encoded"]

    model = GaussianNB()
    model.fit(X, y)

    return model, le_rumah, le_target

# ================================
# Fungsi alasan kelayakan
# ================================
def alasan_bansos_row(row):
    if row["Status_Kelayakan"] == "Layak":
        alasan = []
        if row["Pendapatan_Bulanan"] < 1500000:
            alasan.append(f"Pendapatan rendah (Rp {row['Pendapatan_Bulanan']:,})")
        if row["Jumlah_Anggota_Keluarga"] >= 5:
            alasan.append(f"Tanggungan keluarga besar ({row['Jumlah_Anggota_Keluarga']} orang)")
        if row["Kepemilikan_Rumah"] == "Tidak":
            alasan.append("Tidak memiliki rumah pribadi")
        if not alasan:
            alasan.append("Kondisi ekonomi terbatas")
        return ", ".join(alasan) + " → Layak menerima bansos."
    else:
        alasan = []
        if row["Pendapatan_Bulanan"] >= 1500000:
            alasan.append(f"Pendapatan cukup tinggi (Rp {row['Pendapatan_Bulanan']:,})")
        if row["Jumlah_Anggota_Keluarga"] < 5:
            alasan.append(f"Tanggungan keluarga kecil ({row['Jumlah_Anggota_Keluarga']} orang)")
        if row["Kepemilikan_Rumah"] == "Ya":
            alasan.append("Sudah memiliki rumah pribadi")
        if not alasan:
            alasan.append("Kondisi ekonomi memadai")
        return ", ".join(alasan) + " → Tidak Layak menerima bansos."

# ================================
# Sidebar Navigasi
# ================================
st.sidebar.title("🧭 Navigasi")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "🔮 Prediksi Kelayakan", "📊 Prioritas Penerima", "🏡 Profil Desa"],
    index=0,
    help="Pilih halaman yang ingin Anda kunjungi."
)
st.sidebar.markdown("---")
st.sidebar.info("💡 Sistem ini membantu klasifikasi bantuan sosial dengan AI.")

# ================================
# Halaman 1: Dashboard
# ================================
if page == "🏠 Dashboard":
    # Header dengan gradient background dan animasi bounce
    st.markdown("""
    <div class="dashboard-header">📊 <span class="icon-bounce">Sistem Klasifikasi Bantuan Sosial</span></div>
    """, unsafe_allow_html=True)
    
    # Layout dengan kolom dan animasi slide
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="card slide-left">', unsafe_allow_html=True)
        st.subheader("📌 Tentang Sistem")
        st.write("""
        Sistem ini menggunakan **Naive Bayes** dengan label otomatis
        (berdasarkan pendapatan, jumlah anggota keluarga, dan kepemilikan rumah) untuk menentukan
        apakah seorang warga **Layak** atau **Tidak Layak** menerima bansos.
        """)
        st.markdown("### 🚀 Fitur Utama")
        st.markdown("- 🔍 Prediksi Kelayakan Berdasarkan Data")
        st.markdown("- 📈 Prioritas Penerima Bansos")
        st.markdown("- 🏘️ Profil Desa Lengkap")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card slide-right pulse">', unsafe_allow_html=True)
        st.image("https://via.placeholder.com/300x200/4facfe/ffffff?text=Sistem+Bansos", use_column_width=True)
        st.caption("Ilustrasi Sistem Klasifikasi")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card fade-up">', unsafe_allow_html=True)
    st.subheader("🎯 Tujuan")
    st.write("""
    - Membantu perangkat desa menyalurkan bansos tepat sasaran  
    - Mengurangi subjektivitas  
    - Memanfaatkan data objektif untuk klasifikasi  
    """)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; animation: fadeInUp 1s ease-out;">
    <h4>💡 Mengapa Penting?</h4>
    <p>Dengan AI, proses klasifikasi menjadi lebih akurat dan efisien, memastikan bantuan sampai ke yang benar-benar membutuhkan.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# Halaman 2: Prediksi Kelayakan
# ================================
elif page == "🔮 Prediksi Kelayakan":
    st.markdown('<div class="fade-up">', unsafe_allow_html=True)
    st.title("🔮 Prediksi Kelayakan Penerima Bansos")
    st.markdown("---")
    
    # Upload section dengan styling dan animasi
    st.markdown("""
    <div class="card slide-left" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
    <h4>📤 Upload Dataset</h4>
    <p>Upload file Excel atau CSV berisi data penduduk untuk prediksi kelayakan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload dataset penduduk (Excel/CSV)", type=["csv", "xlsx"], key="prediksi")
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head(), use_container_width=True)

        model, le_rumah, le_target = train_model(df)
        df["Kepemilikan_Rumah_encoded"] = le_rumah.transform(df["Kepemilikan_Rumah"])
        X_all = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
                    "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]

        y_pred = model.predict(X_all)
        df["Status_Kelayakan"] = le_target.inverse_transform(y_pred)
        df["Alasan"] = df.apply(alasan_bansos_row, axis=1)

        st.session_state["hasil_prediksi"] = df

        st.subheader("📋 Hasil Prediksi")
        # Tambahkan filter untuk hasil
        status_filter = st.selectbox("Filter berdasarkan Status:", ["Semua", "Layak", "Tidak Layak"])
        if status_filter == "Layak":
            filtered_df = df[df["Status_Kelayakan"] == "Layak"]
        elif status_filter == "Tidak Layak":
            filtered_df = df[df["Status_Kelayakan"] == "Tidak Layak"]
        else:
            filtered_df = df
        st.dataframe(filtered_df[["Nama", "Status_Kelayakan", "Alasan"]], use_container_width=True)

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Hasil Prediksi", buffer,
                           file_name="hasil_prediksi_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# Halaman 3: Prioritas Penerima
# ================================
elif page == "📊 Prioritas Penerima":
    st.markdown('<div class="fade-up">', unsafe_allow_html=True)
    st.title("📊 Urutan Prioritas Penerima Bansos")
    st.markdown("---")
    
    if "hasil_prediksi" not in st.session_state:
        st.warning("⚠️ Belum ada hasil prediksi. Silakan lakukan prediksi terlebih dahulu di halaman **Prediksi Kelayakan**.")
        st.info("🔄 Navigasi ke halaman Prediksi untuk memulai.")
    else:
        df = st.session_state["hasil_prediksi"]

        penerima = df[df["Status_Kelayakan"] == "Layak"].copy()
        penerima = penerima.sort_values(
            by=["Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Usia_Kepala_Keluarga"],
            ascending=[True, False, True]
        )

        st.subheader("📋 Daftar Prioritas Penerima")
        # Tambahkan expander untuk detail
        with st.expander("🔍 Lihat Kriteria Prioritas"):
            st.write("Prioritas ditentukan berdasarkan: Pendapatan terendah, jumlah anggota keluarga terbanyak, usia kepala keluarga termuda.")
        st.dataframe(penerima[["Nama", "Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Usia_Kepala_Keluarga", "Alasan"]], use_container_width=True)

        buffer = io.BytesIO()
        penerima.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Daftar Prioritas", buffer,
                           file_name="prioritas_penerima_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# Halaman 4: Profil Desa Cikembar
# ================================
elif page == "🏡 Profil Desa":
    st.markdown('<div class="fade-up">', unsafe_allow_html=True)

    PROFILE = {
        "nama_desa": "Desa Cikembar",
        "kecamatan": "Cikembar",
        "kabupaten": "Sukabumi",
        "provinsi": "Jawa Barat",
        "kode_pos": "43157",
        "alamat_kantor_desa": "Jl. Pelabuhan II KM 18, Desa Cikembar",
        "koordinat": [-6.9393, 106.9153],
        "luas_sawah_ha": 1385.38,
        "luas_lahan_kering_ha": 5148.09,
        "suhu_min": 18,
        "suhu_max": 32,
        "curah_hujan_min": 1200,
        "curah_hujan_max": 2200,
        "jumlah_dusun": 44,
        "jumlah_rw": 103,
        "jumlah_rt": 438
    }

    STRUKTUR = [
        {"Jabatan": "Kepala Desa", "Nama": "Andi Rahmat Sanjaya, A.Md"},
        {"Jabatan": "Sekretaris Desa", "Nama": "Dian Purnama"},
        {"Jabatan": "Kaur Keuangan", "Nama": "Nining Sulastri"},
        {"Jabatan": "Kaur Umum", "Nama": "Ade Rohman"},
        {"Jabatan": "Kasi Pemerintahan", "Nama": "Iwan Setiawan"},
        {"Jabatan": "Kasi Kesejahteraan", "Nama": "Dede Komarudin"},
        {"Jabatan": "Kasi Pelayanan", "Nama": "Teti Nuraeni"},
    ]

    # Header & logo Desa dengan layout yang lebih menarik
