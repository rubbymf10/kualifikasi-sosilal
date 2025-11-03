import streamlit as st
import pandas as pd
import joblib
import io
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
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Dashboard", "🔮 Prediksi Kelayakan", "📊 Prioritas Penerima", "🏘️ Profil Desa"])

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
        st.dataframe(penerima[["Nama", "Pendapatan_Bulanan", "Jumlah_Anggota_Keluarga", "Usia_Kepala_Kepala_Keluarga", "Alasan"]])

        buffer = io.BytesIO()
        penerima.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Daftar Prioritas", buffer,
                           file_name="prioritas_penerima_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ================================
# Halaman 4: Profil Desa/Kecamatan
# ================================
elif page == "🏘️ Profil Desa":
    st.title("🏘️ Profil Kecamatan Cikembar")
    st.markdown("---")

    st.subheader("📍 Lokasi & Informasi Umum")
    st.write("""
    **Nama Kecamatan**: Kecamatan Cikembar, Kabupaten Sukabumi  
    **Luas Wilayah**: 8.651,83 Ha (Tanah Sawah: 1.385,38 Ha; Tanah Kering: 5.148,09 Ha)  
    **Desa/Kelurahan**: Terdiri atas 10 desa.  
    **Alamat Kantor**: Jalan Pelabuhan II Km 18 Desa Cikembar Kecamatan Cikembar Kabupaten Sukabumi  
    **Kode Pos**: 43157  
    Sumber: [web.sukabumikab.go.id](https://web.sukabumikab.go.id/web/detail_opd/cikembar.asp)
    """)

    st.subheader("📝 Deskripsi")
    st.write("""
    Kecamatan Cikembar memiliki potensi di bidang industri, pertanian, dan perkebunan. Dengan luas wilayah yang cukup besar dan kombinasi lahan sawah + lahan kering, wilayah ini terus berkembang.  
    Sumber: [web.sukabumikab.go.id](https://web.sukabumikab.go.id/web/detail_opd/cikembar.asp)
    """)

    st.subheader("📷 Galeri Kegiatan")
    st.write("Galeri kegiatan dan dokumentasi publik dari Kecamatan Cikembar / Desa-sekitarnya.")
    st.image([
        "https://sukabumizone.com/2025/08/25/ragam-kegiatan-meriahkan-hut-ri-ke-80-di-desa-cikembar/.jpg",  
        "https://www.beritaekspos.com/2024/02/ratusan-siswa-sekolah-dasar-di-kecamatan-cikembar-mengikuti-lomba/…jpg",
        "https://www.radarjabar.com/jawa-barat/95110273828/ratusan-warga-cikembar-sukabumi-gelar-salat-istisqa-minta-diturunkan-hujan-dan-perlindungan-dari-bencana/.jpg"
    ], width=700, caption=["HUT RI ke-80 Desa Cikembar","Pentas PAI Kecamatan Cikembar","Salat Istisqa Warga Cikembar"])

    st.subheader("🔗 Tautan Penting")
    st.write("""
    - Website resmi Kecamatan Cikembar: https://kec-cikembar.sukabumikab.go.id/album  
    - Informasi OPD terkait: https://web.sukabumikab.go.id/web/detail_opd/cikembar.asp  
    """)

    st.subheader("📌 Catatan Tambahan")
    st.write("""
    Data dan gambar di atas bersifat publik dan diambil dari sumber online. Untuk penggunaan resmi lebih lanjut, disarankan memastikan hak cipta dan lisensi masing-gambar.  
    Anda juga dapat menambahkan bagian potensi ekonomi, demografi, struktur organisasi, atau foto-lapangan lebih lengkap jika diperlukan.
    """)

