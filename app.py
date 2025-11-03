import streamlit as st
import pandas as pd
import joblib
import io
import folium
from streamlit_folium import st_folium
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# ================================
# 🌈 KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Klasifikasi Bantuan Sosial",
    layout="wide",
    page_icon="📊"
)

# 🌟 CSS Styling
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #fce4ec);
    font-family: "Poppins", sans-serif;
}

/* Sidebar */
.css-1d391kg, .css-1lcbmhc, .css-1v3fvcr {
    background-color: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(10px);
}

/* Header */
h1, h2, h3 {
    color: #2E86C1 !important;
    font-weight: 700;
}

/* Cards */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    background-color: white;
}

/* Buttons */
.stDownloadButton > button, .stButton > button {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.5em;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    transform: scale(1.05);
}

/* Table styling */
table {
    border-collapse: collapse;
    width: 100%;
}
th {
    background-color: #4facfe !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ================================
# 🔖 Fungsi Label Otomatis
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
# 🧠 Fungsi Training Model
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
# 💬 Fungsi Alasan Bansos
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
# 🧭 Sidebar Navigasi
# ================================
st.sidebar.markdown("## 🌍 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "🔮 Prediksi Kelayakan", "📊 Prioritas Penerima", "🏡 Profil Desa"]
)

# ================================
# 🏠 Dashboard
# ================================
if page == "🏠 Dashboard":
    st.markdown("<h1 style='text-align:center;'>📊 Sistem Klasifikasi Bantuan Sosial</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/5533/5533914.png", width=250)
    with col2:
        st.write("""
        Sistem ini menggunakan **Naive Bayes Classifier** dengan label otomatis untuk menentukan apakah seorang warga **Layak** atau **Tidak Layak** menerima bantuan sosial.

        🎯 **Tujuan Utama:**
        - Membantu perangkat desa menyalurkan bansos **tepat sasaran**  
        - Mengurangi **subjektivitas keputusan**  
        - Memanfaatkan data **objektif dan transparan**
        """)

# ================================
# 🔮 Prediksi Kelayakan
# ================================
elif page == "🔮 Prediksi Kelayakan":
    st.title("🔮 Prediksi Kelayakan Penerima Bansos")
    uploaded_file = st.file_uploader("📂 Upload dataset penduduk (Excel/CSV)", type=["csv", "xlsx"], key="prediksi")

    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head())

        model, le_rumah, le_target = train_model(df)
        df["Kepemilikan_Rumah_encoded"] = le_rumah.transform(df["Kepemilikan_Rumah"])
        X_all = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]

        y_pred = model.predict(X_all)
        df["Status_Kelayakan"] = le_target.inverse_transform(y_pred)
        df["Alasan"] = df.apply(alasan_bansos_row, axis=1)
        st.session_state["hasil_prediksi"] = df

        st.subheader("📋 Hasil Prediksi")
        st.dataframe(df[["Nama", "Status_Kelayakan", "Alasan"]])

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Hasil Prediksi", buffer, file_name="hasil_prediksi_bansos.xlsx")

# ================================
# 📊 Prioritas Penerima
# ================================
elif page == "📊 Prioritas Penerima":
    st.title("📊 Urutan Prioritas Penerima Bansos")
    if "hasil_prediksi" not in st.session_state:
        st.warning("⚠️ Belum ada hasil prediksi. Silakan lakukan prediksi terlebih dahulu di halaman **Prediksi Kelayakan**.")
    else:
        df = st.session_state["hasil_prediksi"]
        penerima = df[df["Status_Kelayakan"] == "Layak"].copy()
        penerima = penerima.sort_values(
            by=["Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Usia_Kepala_Keluarga"],
            ascending=[True, False, True]
        )

        st.subheader("📋 Daftar Prioritas Penerima")
        st.dataframe(penerima[["Nama", "Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Usia_Kepala_Keluarga", "Alasan"]])

        buffer = io.BytesIO()
        penerima.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Daftar Prioritas", buffer, file_name="prioritas_penerima_bansos.xlsx")

# ================================
# 🏡 Profil Desa
# ================================
elif page == "🏡 Profil Desa":
    st.title("🏡 Profil Desa Cikembar")
    col_logo, col_title = st.columns([1, 3])
    with col_logo:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/6a/Lambang_Kab_Sukabumi.svg", width=120)
    with col_title:
        st.markdown("""
        **Desa Cikembar**, Kecamatan Cikembar, Kabupaten Sukabumi, Provinsi Jawa Barat.  
        Kode Pos: **43157**  
        Alamat: *Jl. Pelabuhan II KM 18, Desa Cikembar*
        """)

    st.markdown("---")
    st.markdown("""
    🌾 **Potensi Desa:**  
    - Pertanian & perkebunan produktif  
    - Industri ringan dan logistik  
    - Warga aktif dan gotong royong  
    
    ⚠️ **Tantangan:**  
    - Banjir lokal & longsor  
    - Peningkatan kualitas jalan  
    - Digitalisasi administrasi publik
    """)

    st.markdown("---")
    st.header("🗺️ Peta Lokasi Desa")
    lat, lon = -6.9393, 106.9153
    tiles_url = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles=tiles_url)
    folium.Marker([lat, lon], popup="Kantor Desa Cikembar", tooltip="Kantor Desa Cikembar", icon=folium.Icon(color='green', icon='home')).add_to(m)
    st_folium(m, width=700, height=450)

    st.caption("Halaman profil resmi Desa Cikembar © 2025")
