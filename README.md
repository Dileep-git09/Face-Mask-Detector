# 😷 Face Mask Detector with MobileNetV2

A deep learning project to detect whether a person is wearing a face mask using **MobileNetV2** and **Streamlit**.

## 🚀 Features
- Image upload detection
- Real-time webcam detection
- Deployed online using **Streamlit Cloud**

## 🧠 Model
- Base: [MobileNetV2](https://arxiv.org/abs/1801.04381)
- Pre-trained on ImageNet
- Fine-tuned for **Mask vs. No Mask**

## 📊 Training
- Framework: TensorFlow / Keras
- Loss: Binary Crossentropy
- Optimizer: Adam

## Accuracy and Loss
<img width="1200" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/6d41ced6-94f0-4c57-a57e-e256f2e0b115" />

## ⚡ Quick Start
```bash
git clone https://github.com/Dileep-git09/mask-detector.git
cd mask-detector
pip install -r requirements.txt
streamlit run app.py
