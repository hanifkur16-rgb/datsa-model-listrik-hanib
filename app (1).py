from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_joblib.pkl"

# Fitur ini harus SAMA dengan fitur yang dipakai saat model listrik dilatih.
# month_name_Apr dan tariff_class_R1 tidak dibuat karena diasumsikan sebagai
# kategori referensi (drop_first=True), sesuai struktur pada aplikasi awal.
EXPECTED_FEATURES = [
    "kwh",
    "ac_units",
    "ac_hours_per_day",
    "family_size",
    "month_name_Aug",
    "month_name_Dec",
    "month_name_Feb",
    "month_name_Jan",
    "month_name_Jul",
    "month_name_Jun",
    "month_name_Mar",
    "month_name_May",
    "month_name_Nov",
    "month_name_Oct",
    "month_name_Sep",
    "tariff_class_R2",
    "tariff_class_R3",
]

MONTH_DUMMY_FEATURES = [
    "month_name_Aug",
    "month_name_Dec",
    "month_name_Feb",
    "month_name_Jan",
    "month_name_Jul",
    "month_name_Jun",
    "month_name_Mar",
    "month_name_May",
    "month_name_Nov",
    "month_name_Oct",
    "month_name_Sep",
]

TARIFF_DUMMY_FEATURES = ["tariff_class_R2", "tariff_class_R3"]


# ---------------------------------------------------------------------
# Memuat model
# ---------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


st.title("Prediksi Tagihan Listrik Jakarta")
st.write(
    "Aplikasi untuk memprediksi jumlah tagihan listrik berdasarkan "
    "parameter yang diberikan."
)

if not MODEL_PATH.exists():
    st.error(
        f"File model tidak ditemukan: {MODEL_PATH.name}. "
        "Letakkan model hasil training listrik di folder yang sama dengan app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model gagal dimuat: {exc}")
    st.stop()


# ---------------------------------------------------------------------
# Validasi model: mencegah model dari dataset lain digunakan untuk listrik
# ---------------------------------------------------------------------
model_features = getattr(model, "feature_names_in_", None)

if model_features is None:
    st.error(
        "Model tidak menyimpan nama fitur saat training. "
        "Latih ulang model menggunakan DataFrame pandas, lalu simpan kembali "
        "dengan joblib.dump()."
    )
    st.stop()

model_features = list(model_features)
expected_set = set(EXPECTED_FEATURES)
