'''import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Load Trained Model
# ==========================
MODEL_PATH = "models/mask_detector_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Streamlit App UI
# ==========================
st.set_page_config(page_title="Face Mask Detector 😷", layout="centered")
st.title("😷 Face Mask Detector")
st.write("Upload an image and check if the person is wearing a mask or not.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess for MobileNetV2
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "Mask 😷" if prediction < 0.5 else "No Mask ❌"

    # Show result
    st.subheader(f"Prediction: **{label}**")
'''


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
MODEL_PATH = "models/mask_detector_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="Face Mask Detector 😷", layout="wide")
st.title("😷 Face Mask Detector (Image & Webcam)")

# -------------------
# File upload section
# -------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -------------------
# Webcam section
# -------------------
camera_photo = st.camera_input("Or take a photo with your webcam")

def predict_mask(img: Image.Image):
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]
    label = "Mask 😷" if prediction < 0.5 else "No Mask ❌"
    return label

# Handle uploaded file
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.subheader(f"Prediction: **{predict_mask(img)}**")

# Handle camera input
elif camera_photo:
    img = Image.open(camera_photo).convert("RGB")
    st.image(img, caption="Webcam Snapshot", use_column_width=True)
    st.subheader(f"Prediction: **{predict_mask(img)}**")

