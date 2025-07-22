# 😁 Live Smile Detection Using Custom CNN

A deep learning project to detect smiles in real-time images using a Convolutional Neural Network (CNN) trained from scratch. Built with Python, OpenCV, NumPy, and TensorFlow/Keras.


### The app is deployed at https://live-smile-detection.streamlit.app/
### Kaggle Notebook  : https://www.kaggle.com/code/rishabh2007/smile-detection-92

## 📌 Project Overview

This project demonstrates a real-time smile detection system using computer vision and deep learning. A CNN model was trained on preprocessed facial images to classify whether a person is smiling or not.


Here is the demo how the page looks like 

![alt text](./media/image.png)

---

## 🧠 Model Summary

- **Architecture**: 3 Convolutional layers with ReLU + MaxPooling  
- **Dense layers**: 2 Fully Connected layers with Dropout  
- **Activation**: ReLU and Sigmoid (binary classification)  
- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Accuracy Achieved**: ~94% on validation set

---

## 🗂️ Dataset

- Used the **SMILES** subset from the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)  
- Balanced classes (smiling, not smiling)  
- All images resized to `64x64` grayscale for uniformity.

---

## 🔬 Pipeline

1. **Face Detection** using Haar Cascades (OpenCV)
2. **Smile Classification** using the trained CNN
3. **Real-time Prediction** via webcam or test image input

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Rishabh-creator601/SmileDetection.git
cd SmileDetection

# Install dependencies
pip install -r requirements.txt

# Train the model (optional)
python train.py

# Run the detector
python detect_smile.py

