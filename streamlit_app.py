import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Clothing Image Classification",
    page_icon="👕"
)

st.title("Clothing Image Classification")

st.write(
    "Upload a clothing image and the trained CNN model "
    "will predict its clothing category."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dropout_model.keras")

model = load_model()

class_names = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt"
]

uploaded_file = st.file_uploader(
    "Choose a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, width=300)

    # Preprocessing
    image = image.resize((128, 128))

    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)
    probabilities = predictions[0]

    # Top 3 predictions
    top_3_indices = np.argsort(probabilities)[-3:][::-1]

    predicted_index = top_3_indices[0]
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index] * 100

    st.subheader("Prediction")

    st.success(f"Predicted Class: {predicted_class}")

    st.write("Prediction Confidence")
    st.write(f"### {confidence:.2f}%")

    st.subheader("Top-3 Predictions")

    for rank, index in enumerate(top_3_indices, start=1):
        probability = probabilities[index]

        st.write(
            f"{rank}. {class_names[index]} "
            f"— {probability * 100:.2f}%"
        )

        st.progress(float(probability))
