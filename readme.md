# 😁 Live Smile Detection Using Custom CNN

A deep learning project to detect smiles in real-time using a **Convolutional Neural Network (CNN)** trained from scratch.  
Built using **Python**, **OpenCV**, **NumPy**, and **TensorFlow/Keras**.

---

## 🌐 Deployment & Notebook

- 🚀 **Live App:** [live-smile-detection.streamlit.app](https://live-smile-detection.streamlit.app/)
- 📓 **Kaggle Notebook:** [Smile Detection (92% Accuracy)](https://www.kaggle.com/code/rishabh2007/smile-detection-92)

---

## 📌 Project Overview

This project demonstrates a real-time smile detection system using computer vision and deep learning.  
A CNN model was trained on preprocessed facial images to classify whether a person is smiling or not.

<p align="center">
  <img src="./media/image.png" width="80%" alt="App Screenshot" />
</p>

---

## 🧠 Model Summary

- **Architecture:** 3 Convolutional layers (ReLU + MaxPooling)  
- **Dense Layers:** 2 Fully Connected layers with Dropout  
- **Activation Functions:** ReLU and Sigmoid  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Validation Accuracy:** ~94%

---

## 📊 Model Metrics & Results

<table>
  <tr>
    <td align="center">
      <h4>📈 Model Metrics</h4>
      <img src="https://github.com/user-attachments/assets/dd9d3718-02b1-4fb5-a156-fd1dc44cc7fa" width="100%" style="max-width: 500px;">
    </td>
  </tr>
  <tr>
    <td align="center">
      <h4>📸 Model Predictions</h4>
      <img src="https://github.com/user-attachments/assets/f2a303f4-6624-4d2d-911c-c5784b7f295e" width="100%" style="max-width: 500px;">
    </td>
  </tr>
</table>

---

## 🗂️ Dataset

- 📁 Used the **SMILES** subset from the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)  
- ✅ Balanced classes: smiling 😄 vs not smiling 😐  
- 📏 All images resized to `64x64` grayscale

---

## 🔬 Pipeline

```plaintext
Step 1: Face Detection (Haar Cascade - OpenCV)
Step 2: Smile Classification (CNN-based)
Step 3: Real-time Prediction (via Webcam or Image)
