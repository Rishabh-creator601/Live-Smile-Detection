import streamlit as st
import cv2 ,os
import numpy as  np 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



@st.cache_resource()
def load_detector():
    detector  =  cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    return detector
 
 
 
@st.cache_resource()
def load_Model():
    model = load_model("./smile.hdf5")
    return model  




detector = load_detector()
model = load_Model()


st.title("Live Smile Detection ")
st.sidebar.success("Please Select A Page Above ")
st.markdown("* **For Closing the WebCam switch the page or close the site**")

FRAME_WINDOW = st.empty()
camera = cv2.VideoCapture(0)

    
while camera.isOpened():
    ret, frame = camera.read()
    
    if not ret:
        st.warning("Failed to grab frame")
        break
    frame = cv2.flip(frame,1)
    frameClone  = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects  =  detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(64,64))
    for (X,Y,W,H) in rects:
        roi = gray[Y:Y+H,X:X+W]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = roi.reshape(-1,64,64)
        prediction = model.predict(roi)[0]
            #print(prediction)
        label  = "Smiling"  if prediction > 0.5 else "Not Smiling , Say Cheese !!"
            #print(label)
            #st.text(label)
        cv2.putText(frameClone, label, (X, Y  -10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0,255), 2)
        cv2.rectangle(frameClone,(X,Y),(X+W,Y+H),(0,0,255),2)
    FRAME_WINDOW.image(frameClone,channels="RGB")





