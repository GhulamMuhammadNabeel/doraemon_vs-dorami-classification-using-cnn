import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model.keras")

st.title("Doraemon vs Dorami")
file = st.file_uploader("Upload Image")

if file:
    # RAW IMAGE (for display)
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img_raw = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Original Image")

    # MODEL INPUT
    img_resized = cv2.resize(img_rgb, (224,224))
    img_input = np.expand_dims(img_resized, axis=0).astype(np.float32)

    img_input = (img_input / 127.5) - 1  # normalization for model

    pred = model.predict(img_input)[0][0]

    label = "Dorami" if pred > 0.5 else "Doraemon"

    st.write("Prediction:", label, pred)