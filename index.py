import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps


st.set_option('deprecation.showfileUploaderEncoding', False)


st.title("🩻 Pneumonia Detection from Chest X-Rays")
st.markdown("""
This application uses a deep learning model to identify **pneumonia** from chest scan images.
Please upload a chest X-ray image in **JPG, JPEG, or PNG** format.
""")

@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5")
    return model

with st.spinner("Loading AI Model..."):
    model = load_trained_model()


class_labels = ['Normal', 'Pneumonia']


def process_and_predict(image: Image.Image, model: tf.keras.Model):
    target_size = (180, 180)
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    input_image = img_rgb[np.newaxis, ...]
    prediction = model.predict(input_image)
    return prediction

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)

    predictions = process_and_predict(image, model)
    confidence_scores = tf.nn.softmax(predictions[0])
    predicted_class = class_labels[np.argmax(confidence_scores)]

    st.subheader("Prediction Result")
    st.write(f"🧠 The model predicts: **{predicted_class}**")
    st.write(f"🧪 Confidence Score: **{100 * np.max(confidence_scores):.2f}%**")

    with st.expander("See raw model output"):
        st.write(predictions)
        st.write(confidence_scores)
else:
    st.info("Awaiting image upload...")

