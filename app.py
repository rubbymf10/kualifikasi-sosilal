import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import io

st.set_page_config(page_title="Klasifikasi Bantuan Sosial", layout="wide")

# ================================
# Fungsi training model
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

    # Simpan model & encoder
    joblib.dump(model, "model_bansos.pkl")
    joblib.dump(le_rumah, "encoder_rumah.pkl")
    joblib.dump(le_target, "encoder_target.pkl")

    return model, le_rumah, le_target

# ================================
# Fungsi alasan bansos
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
# UI Streamlit
# ================================
st.title("📊 Prediksi Penerima Bantuan Sosial (Naive Bayes)")

uploaded_file = st.file_uploader("📁 Upload dataset penduduk (Excel/CSV)", type=["csv", "xlsx"])
if uploaded_file:
    # Baca dataset
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Dataset berhasil diupload")
    st.dataframe(df.head())

    # MODE TRAINING
    if "Status_Kesejahteraan" in df.columns:
        st.info("🔄 Dataset berisi label → model akan dilatih ulang.")
        model, le_rumah, le_target = train_model(df)
        st.success("✅ Model berhasil dilatih dan disimpan!")
        st.stop()

    # MODE PREDIKSI
    else:
        # Pastikan model sudah ada
        if not (os.path.exists("model_bansos.pkl") and 
                os.path.exists("encoder_rumah.pkl") and 
                os.path.exists("encoder_target.pkl")):
            st.error("❌ Belum ada model tersimpan. Upload dataset dengan label terlebih dahulu untuk melatih model.")
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

        # Transformasi
        df["Kepemilikan_Rumah_encoded"] = le_rumah.transform(df["Kepemilikan_Rumah"])
        X_all = df[["Usia_Kepala_Keluarga", "Pendapatan_Bulanan",
                    "Jumlah_Anggota_Keluarga", "Kepemilikan_Rumah_encoded"]]

        # Prediksi
        y_pred = model.predict(X_all)
        df["Prediksi_Status"] = le_target.inverse_transform(y_pred)

        # Mapping status → layak/tidak layak
        mapping = {
            "Miskin": "Layak",
            "Rentan Miskin": "Layak",
            "Sejahtera": "Tidak Layak",
            "Sangat Sejahtera": "Tidak Layak"
        }
        df["Keterangan_Layak"] = df["Prediksi_Status"].map(mapping)
        df["Keterangan_Alasan"] = df.apply(alasan_bansos_row, axis=1)

        # Tampilkan hasil
        st.subheader("📋 Hasil Prediksi")
        st.dataframe(df[["Nama", "Prediksi_Status", "Keterangan_Layak", "Keterangan_Alasan"]])

        # Download
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("📥 Download Hasil Prediksi", buffer,
                           file_name="hasil_prediksi_bansos.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
