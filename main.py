import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tempfile

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Load emotion model and face detector
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
HAARCASCADE_PATH = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')

try:
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    if face_cascade.empty():
        st.error("Error: Haar Cascade file could not be loaded")
        model = None
        face_cascade = None
except Exception as e:
    st.error(f"Error loading model or face cascade: {str(e)}")
    model = None
    face_cascade = None

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def analyze_video(video_path):
    if model is None or face_cascade is None:
        st.error("Model or face detector not properly loaded. Cannot analyze video.")
        return None
    cap = cv2.VideoCapture(video_path)
    counts = {emotion: 0 for emotion in EMOTIONS}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum(roi) == 0:
                continue

            roi = roi.astype("float32") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            label = EMOTIONS[np.argmax(preds)]
            counts[label] += 1
            break  # Only first face per frame

    cap.release()
    return counts


def plot_emotions(counts):
    labels = [f"{k} ({v})" for k, v in counts.items() if v > 0]
    sizes = [v for v in counts.values() if v > 0]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    return fig


# ----------------------
# Streamlit UI
# ----------------------
st.title("🎥 Emotion Detection from Video")
st.write("Upload a recorded video file to analyze dominant emotions.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.info("Analyzing video, please wait...")
    emotion_counts = analyze_video(video_path)
    if emotion_counts is not None:
        fig = plot_emotions(emotion_counts)
        st.pyplot(fig)
    else:
        st.warning("Analysis failed. Please check the console for details.")
else:
    st.warning("Please upload a video to proceed.")

