# app_realtime.py
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# -------- CONFIG ----------
MODEL_PATH = "models/mask_detector_mobilenetv2.h5"
INPUT_SIZE = (224, 224)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Face Mask Detector — Webcam", layout="wide")
st.title("😷 Face Mask Detector")
st.write("Choose between **Live Streaming** or **Snapshot Capture** for mask detection.")

# Load model once
@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    return tf.keras.models.load_model(path)

model = load_model()

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ----------- Real-time Transformer ------------
class MaskDetectorTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face, INPUT_SIZE)
            except Exception:
                continue

            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = face_rgb.astype("float32") / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            pred = model.predict(face_array, verbose=0)[0][0]
            label = "Mask" if pred < 0.5 else "No Mask"
            conf = (1 - pred) if pred < 0.5 else pred
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img


# ----------- UI ------------
mode = st.radio("Select Mode:", ["Live Webcam", "Snapshot Capture"])

if mode == "Live Webcam":
    st.info("Live streaming requires browser camera permission.")
    webrtc_ctx = webrtc_streamer(
        key="mask-detector",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=MaskDetectorTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

elif mode == "Snapshot Capture":
    st.info("Upload or capture a single frame below:")
    img_file = st.camera_input("Take a picture") or st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Captured Image", use_column_width=True)

        # Preprocess
        img_resized = img.resize(INPUT_SIZE)
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array, verbose=0)[0][0]
        label = "Mask 😷" if pred < 0.5 else "No Mask ❌"
        st.subheader(f"Prediction: **{label}**")
