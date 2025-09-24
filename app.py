import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# ================================
# Fungsi Alasan Bansos
# ================================
def alasan_bansos_row(row):
    if row["Keterangan_Layak"] == "Layak":
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
# Fungsi Klasifikasi Rule-Based
# ================================
def klasifikasi_rule(df):
    hasil = []
    for _, row in df.iterrows():
        if (row["Pendapatan_Bulanan"] < 1500000) or (row["Jumlah_Anggota_Keluarga"] >= 5) or (row["Kepemilikan_Rumah"] == "Tidak"):
            hasil.append("Layak")
        else:
            hasil.append("Tidak Layak")
    df["Keterangan_Layak"] = hasil
    df["Keterangan_Alasan"] = df.apply(alasan_bansos_row, axis=1)
    return df

# ================================
# Sidebar Navigasi
# ================================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Dashboard", "📤 Upload Data", "📋 Prediksi Kelayakan"])

# ================================
# Halaman 1: Dashboard
# ================================
if page == "🏠 Dashboard":
    st.markdown("<h1 style='text-align:center;color:#4facfe;'>📊 Sistem Klasifikasi Bantuan Sosial</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📌 Tentang Sistem")
    st.write("""
    Sistem ini menentukan apakah seorang warga **Layak** atau **Tidak Layak** menerima bantuan sosial berdasarkan:
    - Usia Kepala Keluarga  
    - Jumlah Anggota Keluarga  
    - Pendapatan Bulanan  
    - Kepemilikan Rumah  
    """)

    st.subheader("🎯 Tujuan")
    st.write("""
    - Membantu perangkat desa menyalurkan bansos tepat sasaran  
    - Mengurangi subjektivitas & meningkatkan transparansi  
    - Memanfaatkan data untuk keputusan yang lebih adil  
    """)

# ================================
# Halaman 2: Upload Data
# ================================
elif page == "📤 Upload Data":
    st.title("📤 Upload Dataset Penduduk")

    uploaded_file = st.file_uploader("Upload dataset (Excel/CSV) dengan kolom: Nama, Jumlah_Anggota_Keluarga, Usia_Kepala_Keluarga, Pendapatan_Bulanan, Kepemilikan_Rumah", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head())

        st.info("Data siap digunakan untuk prediksi di halaman berikutnya.")

# ================================
# Halaman 3: Prediksi
# ================================
elif page == "📋 Prediksi Kelayakan":
    st.title("📋 Prediksi Kelayakan Penerima Bansos")

    uploaded_file = st.file_uploader("Upload dataset untuk prediksi", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload")

        required_cols = ["Nama", "Jumlah_Anggota_Keluarga", "Usia_Kepala_Keluarga", "Pendapatan_Bulanan", "Kepemilikan_Rumah"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Dataset tidak sesuai. Kolom hilang: {missing}")
            st.stop()

        df = klasifikasi_rule(df)

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df[["Nama", "Keterangan_Layak", "Keterangan_Alasan"]])

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Hasil Prediksi", buffer,
                           file_name="hasil_prediksi_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
