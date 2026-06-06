import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
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

class_names = ["Retak", "Tidak_Retak"]

st.set_page_config(
    page_title="Deteksi Retak Beton",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Deteksi Retak Beton")
st.write("Upload gambar beton untuk mendeteksi adanya retakan.")

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    img = image.resize((150,150))

    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class_index = np.argmax(prediction[0])

    predicted_class_name = class_names[predicted_class_index]

    confidence = prediction[0][predicted_class_index] * 100

    st.success(
        f"Hasil Prediksi : {predicted_class_name}"
    )

    st.info(
        f"Confidence : {confidence:.2f}%"
    )
