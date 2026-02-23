import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load model
model = pickle.load(open("asl_model.pkl", "rb"))

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

st.title("Sign Language Interpreter")

option = st.radio("Choose input", ["Upload Image", "Webcam"])

# ------------------ Upload ------------------
if option == "Upload Image":
    file = st.file_uploader("Upload hand image")

    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                pred = model.predict([landmarks])
                st.success(f"Prediction: {pred[0]}")

                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        st.image(img_rgb)

# ------------------ Webcam ------------------
if option == "Webcam":
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                pred = model.predict([landmarks])

                cv2.putText(frame, str(pred[0]), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        FRAME_WINDOW.image(frame)

    cap.release()
