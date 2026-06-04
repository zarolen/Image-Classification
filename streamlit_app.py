import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

# =========================
# Download model dari Google Drive
# =========================

MODEL_FILE = "model_crack_beton.h5"
FILE_ID = "1IAbUUl0MVrhN6Hq9phGgDGm8SH9543Ga"

if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# =========================
# Load model
# =========================

@st.cache_resource
def load_my_model():
    return load_model(MODEL_FILE, compile=False)

model = load_my_model()

# =========================
# Konfigurasi halaman
# =========================

st.set_page_config(
    page_title="Deteksi Retak Beton",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Sistem Deteksi Retak Beton")
st.write(
    "Upload gambar beton untuk mendeteksi apakah terdapat retakan atau tidak menggunakan Deep Learning."
)

# =========================
# Upload gambar
# =========================

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Prediksi
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    img = image.resize((150, 150))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    CLASS_NAMES = [
        "Retak",
        "Tidak Retak"
    ]

    st.subheader("Hasil Prediksi")

    st.success(
        f"Kategori : {CLASS_NAMES[predicted_index]}"
    )

    st.info(
        f"Tingkat Keyakinan : {confidence:.2f}%"
    )
