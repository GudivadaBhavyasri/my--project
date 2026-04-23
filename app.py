import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="IndoFashion Classifier",
    page_icon="👗",
    layout="centered"
)

st.title("👗 IndoFashion Image Classifier")
st.write("Upload an image of Indian fashion clothing and get prediction.")

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "fashion_classifier.h5",
        compile=False
    )
    return model

model = load_model()

# ==============================
# DATASET CLASSES
# ==============================

class_names = [
    'blouse',
    'dhoti_pants',
    'dupattas',
    'gowns',
    'kurta_men',
    'leggings_and_salwars',
    'lehenga',
    'mojaris_men',
    'mojaris_women',
    'nehru_jackets',
    'palazzos',
    'petticoats',
    'saree',
    'sherwanis',
    'women_kurta'
]

# ==============================
# IMAGE PREPROCESS
# ==============================

IMG_SIZE = (96, 96)

def preprocess_image(image):

    image = image.resize(IMG_SIZE)

    img = np.array(image)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# ==============================
# FILE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        img = preprocess_image(image)

        prediction = model.predict(img)

        pred_index = np.argmax(prediction)

        confidence = np.max(prediction)

        predicted_class = class_names[pred_index]

        st.success(f"Prediction: **{predicted_class}**")

        st.write(f"Confidence: **{confidence*100:.2f}%**")

        st.subheader("All Class Probabilities")

        for i, cls in enumerate(class_names):
            st.write(f"{cls} : {prediction[0][i]*100:.2f}%")