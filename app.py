# profil_desa_cikembar.py
# Halaman profil resmi Desa Cikembar untuk ditampilkan di website (Streamlit)

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(page_title="Profil Desa Cikembar", layout="centered")

# ==============================
# DATA PROFIL DESA
# ==============================

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

# ==============================
# TAMPILAN PROFIL DESA
# ==============================

# Header & logo
col_logo, col_title = st.columns([1, 3])
with col_logo:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/0/00/Lambang_Kab_Sukabumi.svg",
        use_column_width=True,
        caption="Logo Kabupaten Sukabumi"
    )
with col_title:
    st.title(PROFILE['nama_desa'])
    st.markdown(f"""
    **Kecamatan:** {PROFILE['kecamatan']}  
    **Kabupaten:** {PROFILE['kabupaten']}  
    **Provinsi:** {PROFILE['provinsi']}  
    **Kode Pos:** {PROFILE['kode_pos']}
    """)
    st.write(PROFILE['alamat_kantor_desa'])

# ==============================
# PROFIL SINGKAT
# ==============================
st.markdown("---")
st.header("Profil Singkat")
st.markdown(f"""
**Desa Cikembar** merupakan salah satu dari 10 desa di Kecamatan Cikembar, Kabupaten Sukabumi, Provinsi Jawa Barat. 
Desa ini terletak strategis di jalur **Jl. Pelabuhan II KM 18**, yang menghubungkan pusat Kabupaten Sukabumi dengan kawasan Pelabuhanratu di pesisir selatan.

Desa Cikembar memiliki lahan sawah seluas **{PROFILE['luas_sawah_ha']} hektare** dan lahan kering **{PROFILE['luas_lahan_kering_ha']} hektare**. 
Kondisi iklim relatif sejuk, dengan suhu antara **{PROFILE['suhu_min']}°C – {PROFILE['suhu_max']}°C** dan curah hujan tahunan sekitar **{PROFILE['curah_hujan_min']} – {PROFILE['curah_hujan_max']} mm**.

Jumlah wilayah administratif terdiri atas **{PROFILE['jumlah_dusun']} dusun**, **{PROFILE['jumlah_rw']} RW**, dan **{PROFILE['jumlah_rt']} RT**.

### Potensi Desa
- 🌾 **Pertanian & Perkebunan:** Lahan sawah dan kebun produktif untuk padi serta hortikultura.
- 🏭 **Perindustrian & Logistik:** Lokasi strategis di jalur Pelabuhan II membuka peluang industri ringan dan distribusi.
- 🤝 **Sosial & Budaya:** Warga aktif dalam gotong royong, kegiatan kemasyarakatan, dan pembangunan infrastruktur.

### Tantangan & Pengembangan
- ⚠️ Risiko **banjir lokal dan longsor** di musim hujan.
- 🚧 Peningkatan kualitas jalan lingkungan dan drainase.
- 🌐 Perluasan akses layanan publik dan digitalisasi administrasi desa.
""")

# ==============================
# PETA DESA (ONLINE TILE)
# ==============================

st.markdown("---")
st.header("Peta Desa Cikembar")

lat, lon = PROFILE['koordinat']

# Menggunakan tile online dari CartoDB Positron untuk tampilan elegan
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

# ==============================
# STRUKTUR PEMERINTAHAN
# ==============================

st.markdown("---")
st.header("Struktur Pemerintahan Desa")

df_struktur = pd.DataFrame(STRUKTUR)
st.table(df_struktur)

# Diagram sederhana organisasi
try:
    graph_lines = [
        'digraph {',
        'node [shape=box, style=filled, fillcolor=lightyellow];'
    ]
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

# ==============================
# CATATAN
# ==============================

st.markdown("---")
st.caption("Halaman profil ini merupakan bagian dari website resmi Desa Cikembar, Kecamatan Cikembar, Kabupaten Sukabumi. Semua data bersumber dari administrasi desa dan ditampilkan secara online.")
