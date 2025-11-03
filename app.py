import streamlit as st
import pandas as pd
import joblib
import io
import folium
from streamlit_folium import st_folium
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

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
    df = buat_label_kelayakan(df)  # label otomatis
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
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard", "🔮 Prediksi Kelayakan", "📊 Prioritas Penerima", "🏡 Profil Desa"]
)

# ================================
# Halaman 1: Dashboard
# ================================
if page == "🏠 Dashboard":
    st.markdown("<h1 style='text-align:center;color:#4facfe;'>📊 Sistem Klasifikasi Bantuan Sosial</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📌 Tentang Sistem")
    st.write("""
    Sistem ini menggunakan **Naive Bayes** dengan label otomatis
    (berdasarkan pendapatan, jumlah anggota keluarga, dan kepemilikan rumah) untuk menentukan
    apakah seorang warga **Layak** atau **Tidak Layak** menerima bansos.
    """)

    st.subheader("🎯 Tujuan")
    st.write("""
    - Membantu perangkat desa menyalurkan bansos tepat sasaran  
    - Mengurangi subjektivitas  
    - Memanfaatkan data objektif untuk klasifikasi  
    """)

# ================================
# Halaman 2: Prediksi Kelayakan
# ================================
elif page == "🔮 Prediksi Kelayakan":
    st.title("🔮 Prediksi Kelayakan Penerima Bansos")

    uploaded_file = st.file_uploader("Upload dataset penduduk (Excel/CSV)", type=["csv", "xlsx"], key="prediksi")
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head())

        # Train model & prediksi
        model, le_rumah, le_target = train_model(df)
        df["Kepemilikan_Rumah_encoded"] = le_rumah.transform(df["Kepemilikan_Rumah"])
        X_all = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
                    "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]

        y_pred = model.predict(X_all)
        df["Status_Kelayakan"] = le_target.inverse_transform(y_pred)
        df["Alasan"] = df.apply(alasan_bansos_row, axis=1)

        # Simpan hasil ke session_state
        st.session_state["hasil_prediksi"] = df

        st.subheader("📋 Hasil Prediksi")
        st.dataframe(df[["Nama", "Status_Kelayakan", "Alasan"]])

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Hasil Prediksi", buffer,
                           file_name="hasil_prediksi_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ================================
# Halaman 3: Prioritas Penerima
# ================================
elif page == "📊 Prioritas Penerima":
    st.title("📊 Urutan Prioritas Penerima Bansos")

    if "hasil_prediksi" not in st.session_state:
        st.warning("⚠️ Belum ada hasil prediksi. Silakan lakukan prediksi terlebih dahulu di halaman **Prediksi Kelayakan**.")
    else:
        df = st.session_state["hasil_prediksi"]

        # Filter hanya penerima layak
        penerima = df[df["Status_Kelayakan"] == "Layak"].copy()

        # Urutkan prioritas (pendapatan rendah, keluarga besar, usia tua)
        penerima = penerima.sort_values(
            by=["Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Usia_Kepala_Keluarga"],
            ascending=[True, False, True]
        )

        st.subheader("📋 Daftar Prioritas Penerima")
        st.dataframe(penerima[["Nama", "Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Usia_Kepala_Keluarga", "Alasan"]])

        buffer = io.BytesIO()
        penerima.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Daftar Prioritas", buffer,
                           file_name="prioritas_penerima_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ================================
# Halaman 4: Profil Desa Cikembar
# ================================
elif page == "🏡 Profil Desa":
    st.set_page_config(page_title="Profil Desa Cikembar", layout="centered")

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

    # Header & logo
    col_logo, col_title = st.columns([1,3])
    with col_logo:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/6/6a/Emblem_of_Sukabumi_Regency.png",
            width=120,
            caption="Logo Kabupaten Sukabumi"
        )
    with col_title:
        st.title(PROFILE['nama_desa'])
        st.markdown(f"**Kecamatan:** {PROFILE['kecamatan']}  \
**Kabupaten:** {PROFILE['kabupaten']}  \
**Provinsi:** {PROFILE['provinsi']}  \
**Kode Pos:** {PROFILE['kode_pos']}")
        st.write(PROFILE['alamat_kantor_desa'])

    st.markdown("---")
    st.header("Profil Singkat")
    st.markdown(f"""
    **Desa Cikembar** merupakan salah satu dari 10 desa di Kecamatan Cikembar, Kabupaten Sukabumi, Provinsi Jawa Barat. 
    Desa ini terletak strategis di jalur **Jl. Pelabuhan II KM 18**, yang menghubungkan pusat Kabupaten Sukabumi dengan kawasan Pelabuhanratu di pesisir selatan.

    Desa Cikembar memiliki lahan sawah seluas **{PROFILE['luas_sawah_ha']} hektare** dan lahan kering **{PROFILE['luas_lahan_kering_ha']} hektare**. 
    Kondisi iklim relatif sejuk, dengan suhu antara **{PROFILE['suhu_min']}°C – {PROFILE['suhu_max']}°C** dan curah hujan tahunan sekitar **{PROFILE['curah_hujan_min']} – {PROFILE['curah_hujan_max']} mm**.

    Jumlah wilayah administratif terdiri atas **{PROFILE['jumlah_dusun']} dusun**, **{PROFILE['jumlah_rw']} RW**, dan **{PROFILE['jumlah_rt']} RT**.

    ### Potensi Desa
    - **Pertanian & Perkebunan:** Lahan sawah dan kebun produktif untuk padi serta hortikultura.
    - **Perindustrian & Logistik:** Lokasi strategis di jalur Pelabuhan II membuka peluang industri ringan dan distribusi.
    - **Sosial & Budaya:** Warga aktif dalam gotong royong, kegiatan kemasyarakatan, dan pembangunan infrastruktur.

    ### Tantangan & Pengembangan
    - Risiko **banjir lokal dan longsor** di musim hujan.
    - Peningkatan kualitas jalan lingkungan dan drainase.
    - Perluasan akses layanan publik dan digitalisasi administrasi desa.
    """)

    st.markdown("---")
    st.header("Peta Desa Cikembar")
    lat, lon = PROFILE['koordinat']
    tiles_url = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
    attr = '&copy; <a href="https://carto.com/attributions">CartoDB</a> contributors'
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles=tiles_url, attr=attr)
    folium.Marker(
        [lat, lon],
        popup="Kantor Desa Cikembar",
        tooltip="Kantor Desa Cikembar",
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)
    st_folium(m, width=700, height=450)

    st.markdown("---")
    st.header("Struktur Pemerintahan Desa")
    df_struktur = pd.DataFrame(STRUKTUR)
    st.table(df_struktur)

    try:
        graph_lines = ['digraph {', 'node [shape=box, style=filled, fillcolor=lightyellow];']
        graph_lines.append('"Kepala Desa\\nAndi Rahmat Sanjaya, A.Md"')
        graph_lines.append('"Kepala Desa\\nAndi Rahmat Sanjaya, A.Md" -> "Sekretaris Desa\\nDian Purnama";')
        for r in STRUKTUR[2:]:
            jab = r['Jabatan']
            nama = r['Nama']
            graph_lines.append(f'"Sekretaris Desa\\nDian Purnama" -> "{jab}\\n{nama}";')
        graph_lines.append('}')
        graph = '\n'.join(graph_lines)
        st.graphviz_chart(graph)
    except Exception as e:
        st.warning(f"Gagal menampilkan diagram organisasi: {e}")

    st.markdown("---")
    st.caption("Halaman profil ini merupakan bagian dari website resmi Desa Cikembar, Kecamatan Cikembar, Kabupaten Sukabumi. Semua data bersumber dari administrasi desa dan ditampilkan secara online.")
