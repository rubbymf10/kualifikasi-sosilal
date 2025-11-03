import streamlit as st
import pandas as pd
import io
import folium
from streamlit_folium import st_folium
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# ================================
# Konfigurasi Tampilan Global
# ================================
st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# CSS Desain Modern & Kontras Tinggi
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #e8f0fe 0%, #f9fbff 100%);
    color: #1c1c1c;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #004aad 0%, #007bff 100%);
    color: white;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: white !important;
}

/* Judul dan teks utama */
h1, h2, h3 {
    color: #004aad !important;
}
hr {
    border: 1px solid #e0e0e0;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Card konten */
.block-container {
    padding-top: 2rem;
}
div[data-testid="stVerticalBlock"] > div {
    background: #ffffff;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Tombol */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #007bff, #0056d6);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, #0056d6, #007bff);
    transform: scale(1.03);
}

/* Table */
thead tr th {
    background-color: #004aad !important;
    color: white !important;
}

/* Metric box */
[data-testid="stMetricValue"] {
    color: #004aad;
    font-weight: 700;
}

/* Info box */
div[data-testid="stAlert"] {
    background-color: #f0f6ff !important;
    color: #003b73 !important;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Fungsi pembuat label otomatis
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
# Fungsi model
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
# Fungsi alasan bansos
# ================================
def alasan_bansos_row(row):
    if row["Status_Kelayakan"] == "Layak":
        alasan = []
        if row["Pendapatan_Bulanan"] < 1500000:
            alasan.append(f"Pendapatan rendah (Rp {row['Pendapatan_Bulanan']:,})")
        if row["Jumlah_Anggota_Keluarga"] >= 5:
            alasan.append(f"Tanggungan besar ({row['Jumlah_Anggota_Keluarga']} orang)")
        if row["Kepemilikan_Rumah"] == "Tidak":
            alasan.append("Tidak memiliki rumah pribadi")
        return ", ".join(alasan) + " → Layak menerima bansos."
    else:
        alasan = []
        if row["Pendapatan_Bulanan"] >= 1500000:
            alasan.append(f"Pendapatan cukup tinggi (Rp {row['Pendapatan_Bulanan']:,})")
        if row["Jumlah_Anggota_Keluarga"] < 5:
            alasan.append(f"Tanggungan kecil ({row['Jumlah_Anggota_Keluarga']} orang)")
        if row["Kepemilikan_Rumah"] == "Ya":
            alasan.append("Sudah memiliki rumah pribadi")
        return ", ".join(alasan) + " → Tidak Layak menerima bansos."

# ================================
# Sidebar Navigasi
# ================================
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "🔮 Prediksi Kelayakan", "📊 Prioritas Penerima", "🏡 Profil Desa"]
)

# ================================
# Halaman Dashboard
# ================================
if page == "🏠 Dashboard":
    st.markdown("<h1 style='text-align:center;'>📊 Sistem Klasifikasi Bantuan Sosial</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3141/3141158.png", width=120)
    st.markdown("---")

    st.info("💡 Sistem ini membantu perangkat desa menentukan kelayakan penerima bantuan sosial berdasarkan data objektif dengan algoritma *Naive Bayes*.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👨‍👩‍👧 Jumlah Penduduk", "12,580")
    with col2:
        st.metric("🏘️ Rumah Tangga", "3,420")
    with col3:
        st.metric("🎯 Target Penerima", "870 Keluarga")

    st.subheader("🎯 Tujuan Sistem")
    st.write("""
    - Menentukan penerima bansos secara **objektif dan adil**  
    - Membantu desa menyalurkan **bantuan tepat sasaran**  
    - Menggunakan **data pendapatan, tanggungan, dan kepemilikan rumah**
    """)

# ================================
# Halaman Prediksi
# ================================
elif page == "🔮 Prediksi Kelayakan":
    st.title("🔮 Prediksi Kelayakan Penerima Bansos")
    uploaded_file = st.file_uploader("📂 Upload dataset penduduk (CSV/XLSX)", type=["csv", "xlsx"], key="prediksi")

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)
        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head())

        model, le_rumah, le_target = train_model(df)
        df["Kepemilikan_Rumah_encoded"] = le_rumah.transform(df["Kepemilikan_Rumah"])
        X_all = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
                    "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]
        y_pred = model.predict(X_all)
        df["Status_Kelayakan"] = le_target.inverse_transform(y_pred)
        df["Alasan"] = df.apply(alasan_bansos_row, axis=1)
        st.session_state["hasil_prediksi"] = df

        st.subheader("📋 Hasil Prediksi")
        st.dataframe(df[["Nama", "Status_Kelayakan", "Alasan"]])

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Hasil Prediksi", buffer,
                           file_name="hasil_prediksi_bansos.xlsx")

# ================================
# Halaman Prioritas
# ================================
elif page == "📊 Prioritas Penerima":
    st.title("📊 Urutan Prioritas Penerima Bansos")
    if "hasil_prediksi" not in st.session_state:
        st.warning("⚠️ Belum ada hasil prediksi.")
    else:
        df = st.session_state["hasil_prediksi"]
        penerima = df[df["Status_Kelayakan"] == "Layak"].copy()
        penerima = penerima.sort_values(
            by=["Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga"],
            ascending=[True, False]
        )
        st.dataframe(penerima[["Nama", "Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Alasan"]])
        buffer = io.BytesIO()
        penerima.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Daftar Prioritas", buffer,
                           file_name="prioritas_penerima_bansos.xlsx")

# ================================
# Halaman Profil
# ================================
elif page == "🏡 Profil Desa":
    st.title("🏡 Profil Desa Cikembar")
    col_logo, col_text = st.columns([1,3])
    with col_logo:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/6a/Lambang_Kab_Sukabumi.svg", width=120)
    with col_text:
        st.markdown("**Desa Cikembar, Kecamatan Cikembar, Kabupaten Sukabumi, Jawa Barat (43157)**")

    st.markdown("---")
    st.subheader("🌾 Potensi Desa")
    st.write("""
    - Pertanian padi & hortikultura  
    - Industri logistik di jalur Pelabuhan II  
    - Masyarakat aktif & gotong royong  
    """)

    st.subheader("🗺️ Peta Desa")
    lat, lon = -6.9393, 106.9153
    tiles = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles=tiles)
    folium.Marker([lat, lon], popup="Kantor Desa Cikembar",
                  icon=folium.Icon(color='green')).add_to(m)
    st_folium(m, width=700, height=450)
