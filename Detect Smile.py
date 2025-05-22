import streamlit as st
import cv2
import av
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


class SmileDetector(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

        for (x, y, w, h) in rects:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = roi.reshape(-1, 64, 64)

            prediction = model.predict(roi)[0]
            label = "Smiling" if prediction > 0.5 else "Not Smiling, Say Cheese!!"

            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="smile-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SmileDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
