import cv2
import numpy as np
import tensorflow as tf

# ==========================
# Load Trained Model
# ==========================
MODEL_PATH = "models/mask_detector_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# Load Haar Cascade for Face Detection
# ==========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==========================
# Start Webcam
# ==========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        # Crop face
        face = frame[y:y+h, x:x+w]

        # Preprocess for MobileNetV2 (224x224, normalization)
        face_resized = cv2.resize(face, (224, 224))
        face_resized = face_resized.astype("float32") / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        # Prediction
        pred = model.predict(face_resized, verbose=0)[0][0]
        label = "Mask" if pred < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw results
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Show frame
    cv2.imshow("Face Mask Detector", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
