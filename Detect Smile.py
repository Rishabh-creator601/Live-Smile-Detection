import streamlit as st
import cv2
import av
import time
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode



def load_detector():
    return cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")



def load_model_file():
    return load_model("./smile.hdf5")


detector = load_detector()
model = load_model_file()

st.title("Live Smile Detection with WebRTC")
st.sidebar.success("Select a page above.")
st.markdown("* **To stop the webcam, switch the page or close the app**")
st.markdown("⚠️ **Please remove your glasses for best results (model not trained with spectacles)**")


class SmileDetector(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_time = time.time()

    def recv(self, frame):
        self.frame_count += 1
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        # FPS calculation
        now = time.time()
        fps = 1 / (now - self.last_time)
        self.last_time = now

        # Skip frames for performance
        

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = roi.reshape(1, 64, 64)

            prediction = model.predict(roi, verbose=0)[0]
            label = "Smiling" if prediction > 0.5 else "Not Smiling, Say Cheese !"

            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if self.frame_count % 5 != 0:
            return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="smile-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SmileDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
