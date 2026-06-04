import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Image Classification Beton",
    page_icon="🏗️",
    layout="centered"
)

# =====================================================
# JUDUL & INFORMASI
# =====================================================
st.title("🏗️ Klasifikasi Retakan Beton")
st.markdown("""
**Nama :** Muhammad Reval Denta  
**NIM   :** 032400048  
**Prodi :** Elektro Mekanika  

Aplikasi ini menggunakan model CNN (Convolutional Neural Network) untuk mendeteksi apakah sebuah gambar beton **Retak** atau **Tidak Retak**.
""")
st.divider()

# =====================================================
# LOAD MODEL
# =====================================================
MODEL_PATH = "model_crack_beton.h5"

@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model_cached()

# =====================================================
# KONFIGURASI
# =====================================================
IMG_HEIGHT = 150
IMG_WIDTH  = 150
CLASS_NAMES = ["Retak", "Tidak_Retak"]

# =====================================================
# SIDEBAR – INFO MODEL
# =====================================================
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.markdown(f"""
    - **Arsitektur:** CNN (Sequential)
    - **Ukuran Input:** {IMG_HEIGHT}×{IMG_WIDTH} px
    - **Kelas:** {", ".join(CLASS_NAMES)}
    - **Optimizer:** Adam
    - **Epochs Latih:** 10
    """)

    st.divider()
    st.subheader("📊 Performa Training")
    st.markdown("""
    | Metrik | Nilai |
    |---|---|
    | Akurasi Train | ~99.00% |
    | Akurasi Val | ~98.75% |
    | Loss Train | ~0.034 |
    | Loss Val | ~0.078 |
    """)

    st.divider()
    st.caption("Upload file `model_crack_beton.h5` ke direktori yang sama dengan `streamlit_app.py` agar prediksi dapat berjalan.")

# =====================================================
# UPLOAD MODEL (jika tidak ditemukan)
# =====================================================
if model is None:
    st.warning("⚠️ File model `model_crack_beton.h5` tidak ditemukan di direktori saat ini.")
    uploaded_model = st.file_uploader("Upload file model (.h5)", type=["h5"])
    if uploaded_model is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("Model berhasil diupload! Silakan refresh halaman.")
        st.rerun()
    st.stop()
else:
    st.success("✅ Model berhasil dimuat!")

st.divider()

# =====================================================
# UPLOAD GAMBAR & PREDIKSI
# =====================================================
st.subheader("📷 Upload Gambar Beton")
uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa gambar beton (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("🔍 Hasil Prediksi")

    cols_per_row = 2
    rows = [uploaded_files[i:i+cols_per_row] for i in range(0, len(uploaded_files), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, uploaded_file in zip(cols, row):
            with col:
                # Tampilkan gambar
                img_pil = Image.open(uploaded_file).convert("RGB")
                st.image(img_pil, caption=uploaded_file.name, use_container_width=True)

                # Preprocessing
                img_resized = img_pil.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array  = img_to_array(img_resized)
                img_array  = tf.expand_dims(img_array, 0)  # [1, H, W, 3]

                # Prediksi
                prediction = model.predict(img_array, verbose=0)
                predicted_idx  = int(tf.argmax(prediction[0]).numpy())
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence      = float(prediction[0][predicted_idx]) * 100

                # Tampilkan hasil
                if predicted_class == "Retak":
                    st.error(f"**{predicted_class}** 🔴")
                else:
                    st.success(f"**{predicted_class}** 🟢")

                st.metric(label="Confidence", value=f"{confidence:.2f}%")

                # Bar probabilitas semua kelas
                with st.expander("Detail probabilitas"):
                    for i, cls in enumerate(CLASS_NAMES):
                        prob = float(prediction[0][i]) * 100
                        st.write(f"{cls}: **{prob:.2f}%**")
                        st.progress(prob / 100)

    st.divider()

    # =====================================================
    # RINGKASAN HASIL
    # =====================================================
    st.subheader("📋 Ringkasan Hasil")
    results = []
    for uploaded_file in uploaded_files:
        img_pil    = Image.open(uploaded_file).convert("RGB")
        img_resized = img_pil.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array  = img_to_array(img_resized)
        img_array  = tf.expand_dims(img_array, 0)
        prediction = model.predict(img_array, verbose=0)
        predicted_idx  = int(tf.argmax(prediction[0]).numpy())
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(prediction[0][predicted_idx]) * 100
        results.append({
            "Nama File": uploaded_file.name,
            "Prediksi": predicted_class,
            "Confidence (%)": f"{confidence:.2f}"
        })

    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # Statistik ringkas
    total   = len(df)
    retak   = (df["Prediksi"] == "Retak").sum()
    tidak   = total - retak
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Gambar", total)
    col2.metric("Retak 🔴", retak)
    col3.metric("Tidak Retak 🟢", tidak)

else:
    st.info("Silakan upload gambar beton untuk memulai prediksi.")

# =====================================================
# PEMBAHASAN HASIL TRAINING
# =====================================================
st.divider()
with st.expander("📈 Pembahasan Hasil Training"):
    st.markdown("""
    ### Hasil Pelatihan Model

    Dari grafik akurasi dan loss selama **10 epochs** pelatihan:

    | Metrik | Awal | Akhir |
    |---|---|---|
    | Training Accuracy | ~71.63% | ~99.00% |
    | Validation Accuracy | ~92.25% | ~98.75% |
    | Training Loss | 0.546 | 0.034 |
    | Validation Loss | 0.201 | 0.078 |

    **Kesimpulan:** Model berhasil mempelajari pola gambar beton dengan sangat baik tanpa tanda-tanda *overfitting* yang signifikan. Akurasi validasi yang tinggi dan stabil mengindikasikan kemampuan generalisasi yang kuat.
    """)

with st.expander("📝 Kesimpulan Akhir"):
    st.markdown("""
    1. **Akurasi dan Generalisasi Tinggi:** Model mencapai akurasi mendekati 99% dalam 10 epoch. Tidak ada tanda-tanda *overfitting* yang signifikan.
    2. **Prediksi Akurat dengan Kepercayaan Tinggi:** Model mampu mengklasifikasikan gambar beton baru sebagai *Retak* atau *Tidak Retak* dengan confidence di atas 90%, bahkan banyak mencapai 99%.

    Model ini sangat cocok untuk mengidentifikasi kondisi retak pada beton secara otomatis.
    """)
