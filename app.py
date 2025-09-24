import streamlit as st
import pandas as pd
import joblib
import os
import io
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# ================================
# Fungsi Training Model
# ================================
def train_model(df):
    le_rumah = LabelEncoder()
    le_target = LabelEncoder()

    df["Kepemilikan_Rumah_encoded"] = le_rumah.fit_transform(df["Kepemilikan_Rumah"])
    df["Status_Kesejahteraan_encoded"] = le_target.fit_transform(df["Status_Kesejahteraan"])

    X = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
            "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]
    y = df["Status_Kesejahteraan_encoded"]

    model = GaussianNB()
    model.fit(X, y)

    joblib.dump(model, "model_bansos.pkl")
    joblib.dump(le_rumah, "encoder_rumah.pkl")
    joblib.dump(le_target, "encoder_target.pkl")

    return model, le_rumah, le_target

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
# Sidebar Navigasi
# ================================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Dashboard", "📤 Upload & Training", "🔮 Prediksi Penerima Bansos"])

# ================================
# Halaman 1: Dashboard
# ================================
if page == "🏠 Dashboard":
    st.markdown("<h1 style='text-align:center;color:#4facfe;'>📊 Sistem Klasifikasi Bantuan Sosial</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📌 Tentang Sistem")
    st.write("""
    Sistem ini menggunakan **Naive Bayes Classifier** untuk membantu perangkat desa menentukan
    apakah seorang warga **Layak** atau **Tidak Layak** menerima bantuan sosial berdasarkan:
    - Usia Kepala Keluarga  
    - Pendapatan Bulanan  
    - Jumlah Anggota Keluarga  
    - Kepemilikan Rumah  
    """)

    st.subheader("🎯 Tujuan")
    st.write("""
    - Membantu perangkat desa menyalurkan bansos tepat sasaran  
    - Mengurangi subjektivitas & meningkatkan transparansi  
    - Memanfaatkan data untuk keputusan yang lebih adil  
    """)

# ================================
# Halaman 2: Upload & Training
# ================================
elif page == "📤 Upload & Training":
    st.title("📤 Upload Dataset dengan Label (Training Model)")

    uploaded_file = st.file_uploader("Upload dataset (Excel/CSV) dengan kolom `Status_Kesejahteraan`", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head())

        if "Status_Kesejahteraan" not in df.columns:
            st.error("Dataset tidak memiliki kolom `Status_Kesejahteraan`. Tidak bisa training.")
        else:
            model, le_rumah, le_target = train_model(df)
            st.success("✅ Model berhasil dilatih dan disimpan sebagai `model_bansos.pkl`")

# ================================
# Halaman 3: Prediksi Penerima
# ================================
elif page == "🔮 Prediksi Penerima Bansos":
    st.title("🔮 Prediksi Penerima Bantuan Sosial")

    uploaded_file = st.file_uploader("Upload dataset tanpa label (Excel/CSV)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset berhasil diupload")
        st.dataframe(df.head())

        if not (os.path.exists("model_bansos.pkl") and 
                os.path.exists("encoder_rumah.pkl") and 
                os.path.exists("encoder_target.pkl")):
            st.error("❌ Belum ada model tersimpan. Silakan latih model terlebih dahulu di halaman 'Upload & Training'.")
            st.stop()

        # Load model
        model = joblib.load("model_bansos.pkl")
        le_rumah = joblib.load("encoder_rumah.pkl")
        le_target = joblib.load("encoder_target.pkl")

        required_cols = ["Nama", "Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
                         "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Dataset tidak sesuai. Kolom hilang: {missing}")
            st.stop()

        df["Kepemilikan_Rumah_encoded"] = le_rumah.transform(df["Kepemilikan_Rumah"])
        X_all = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
                    "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]

        y_pred = model.predict(X_all)
        df["Prediksi_Status"] = le_target.inverse_transform(y_pred)

        mapping = {
            "Miskin": "Layak",
            "Rentan Miskin": "Layak",
            "Sejahtera": "Tidak Layak",
            "Sangat Sejahtera": "Tidak Layak"
        }
        df["Keterangan_Layak"] = df["Prediksi_Status"].map(mapping)
        df["Keterangan_Alasan"] = df.apply(alasan_bansos_row, axis=1)

        st.subheader("📋 Hasil Prediksi")
        st.dataframe(df[["Nama", "Prediksi_Status", "Keterangan_Layak", "Keterangan_Alasan"]])

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Hasil Prediksi", buffer,
                           file_name="hasil_prediksi_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
