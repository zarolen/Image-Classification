import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
import os

MODEL_FILE = "model_crack_beton.h5"
FILE_ID = "1IAbUUl0MVrhN6Hq9phGgDGm8SH9543Ga"

if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

@st.cache_resource
def load_my_model():
    return load_model(MODEL_FILE, compile=False)

model = load_my_model()

st.set_page_config(page_title="Deteksi Retak Beton", page_icon="🏗️")

st.title("🏗️ Sistem Deteksi Retak Beton")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    img = image.resize((150,150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    classes = ["Retak", "Tidak Retak"]
    idx = np.argmax(prediction)

    st.success(f"Hasil Prediksi: {classes[idx]}")
    st.info(f"Confidence: {np.max(prediction)*100:.2f}%")
