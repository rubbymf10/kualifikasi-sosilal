import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import io

# =============================
# Konfigurasi halaman
# =============================
st.set_page_config(page_title="Sistem Klasifikasi Bansos - Desa Cikembar", layout="wide")

st.title("📊 Sistem Klasifikasi Penerima Bansos Desa Cikembar")
st.write("Unggah dataset penduduk untuk mengetahui siapa saja yang **Layak** atau **Tidak Layak** menerima bantuan sosial.")

# =============================
# Upload Dataset
# =============================
uploaded_file = st.file_uploader("📂 Upload dataset penduduk (CSV atau Excel)", type=["csv", "xlsx", "xls"])

# =============================
# Fungsi Training Model
# =============================
def train_model(df):
    le_rumah = LabelEncoder()
    le_target = LabelEncoder()

    df['Kepemilikan_Rumah_encoded'] = le_rumah.fit_transform(df['Kepemilikan_Rumah'])
    df['Status_Kesejahteraan_encoded'] = le_target.fit_transform(df['Status_Kesejahteraan'])

    X = df[['Usia_Kepala_Keluarga', 'Pendapatan_Bulanan', 'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah_encoded']]
    y = df['Status_Kesejahteraan_encoded']

    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, le_rumah, le_target, acc

# =============================
# Fungsi Prediksi dengan Alasan
# =============================
def prediksi_dengan_alasan(row, model, le_rumah, le_target):
    rumah_encoded = le_rumah.transform([row['Kepemilikan_Rumah']])[0]
    data_input = np.array([[row['Usia_Kepala_Keluarga'], row['Pendapatan_Bulanan'], row['Jumlah_Anggota_Keluarga'], rumah_encoded]])
    prediksi_encoded = model.predict(data_input)[0]
    prediksi = le_target.inverse_transform([prediksi_encoded])[0]

    # Buat alasan
    alasan = []
    if row['Pendapatan_Bulanan'] < 1000000:
        alasan.append("Pendapatan rendah (< 1 juta)")
    if row['Jumlah_Anggota_Keluarga'] > 5:
        alasan.append("Tanggungan keluarga banyak (> 5)")
    if row['Kepemilikan_Rumah'] == "Tidak":
        alasan.append("Tidak memiliki rumah")
    if row['Usia_Kepala_Keluarga'] > 60:
        alasan.append("Kepala keluarga lanjut usia")

    if not alasan:
        alasan.append("Kondisi ekonomi dianggap cukup")

    return prediksi, "; ".join(alasan)

# =============================
# Proses Dataset
# =============================
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Pastikan kolom wajib ada
        required_columns = ['Nama', 'Usia_Kepala_Keluarga', 'Pendapatan_Bulanan',
                            'Jumlah_Anggota_Keluarga', 'Kepemilikan_Rumah', 'Status_Kesejahteraan']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Dataset harus memiliki kolom: {', '.join(required_columns)}")
        else:
            # Latih model
            model, le_rumah, le_target, acc = train_model(df)
            st.success(f"✅ Model berhasil dilatih dengan akurasi {acc*100:.2f}%")

            # Prediksi semua warga
            hasil_prediksi = []
            for _, row in df.iterrows():
                prediksi, alasan = prediksi_dengan_alasan(row, model, le_rumah, le_target)
                hasil_prediksi.append([row['Nama'], prediksi, alasan])

            df_hasil = pd.DataFrame(hasil_prediksi, columns=["Nama", "Prediksi_Status", "Keterangan_Alasan"])

            # Gabungkan dengan dataset asli
            df_final = pd.concat([df, df_hasil.drop(columns="Nama")], axis=1)

            # Tampilkan semua hasil
            st.subheader("📋 Hasil Prediksi Semua Warga")
            st.dataframe(df_final)

            # Filter penerima bansos
            penerima = df_final[df_final["Prediksi_Status"] == "Layak"]

            st.subheader("✅ Daftar Penerima Bansos (Layak)")
            st.dataframe(penerima)

            # Tombol download
            csv_all = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Semua Hasil Prediksi", csv_all, "hasil_prediksi_semua.csv", "text/csv")

            csv_penerima = penerima.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Daftar Penerima Bansos", csv_penerima, "penerima_bansos.csv", "text/csv")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses dataset: {e}")
