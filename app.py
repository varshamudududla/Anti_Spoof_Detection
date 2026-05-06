import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time

st.set_page_config(layout="wide")
st.title("🔐 Face Liveness Detection System")

# ---------------- SIDEBAR ----------------
threshold = st.sidebar.slider("Model Threshold", 0.0, 1.0, 0.7)
blink_required = st.sidebar.slider("Required Blinks", 1, 5, 2)
ear_threshold = st.sidebar.slider("Blink Sensitivity", 0.1, 0.3, 0.2)
time_limit = st.sidebar.slider("Time Limit", 1, 10, 5)

# ---------------- SESSION ----------------
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0
if "blink_flag" not in st.session_state:
    st.session_state.blink_flag = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "pred_buffer" not in st.session_state:
    st.session_state.pred_buffer = deque(maxlen=10)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("liveness_model_final_v3.keras", compile=False)

model = load_model()

# ---------------- MEDIAPIPE ----------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(0, 0.6)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def ear(landmarks, eye):
    p = [landmarks[i] for i in eye]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C)

run = st.checkbox("Start Camera")
frame_window = st.image([])
status = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)
        mesh = face_mesh.process(rgb)

        # ---------------- TIMER ----------------
        if results.detections and st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
        remaining = max(0, int(time_limit - elapsed))

        face_detected = False
        avg_pred = 0

        # ---------------- BLINK ----------------
        if mesh.multi_face_landmarks:
            for lm in mesh.multi_face_landmarks:
                face_detected = True

                pts = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]
                e = (ear(pts, LEFT_EYE) + ear(pts, RIGHT_EYE)) / 2

                if e < ear_threshold:
                    if not st.session_state.blink_flag:
                        st.session_state.blink_flag = True
                else:
                    if st.session_state.blink_flag:
                        st.session_state.blink_count += 1
                        st.session_state.blink_flag = False

        # ---------------- FACE + MODEL ----------------
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = frame[y1:y2, x1:x2]

                if face.size != 0:
                    face = cv2.resize(face, (128, 128))
                    face = face / 255.0
                    face = np.expand_dims(face, axis=0)

                    pred = model.predict(face, verbose=0)[0][0]
                    st.session_state.pred_buffer.append(pred)
                    avg_pred = np.mean(st.session_state.pred_buffer)

                # -------- LABEL --------
                if avg_pred > threshold:
                    label = "REAL"
                    color = (0, 255, 0)
                else:
                    label = "SPOOF"
                    color = (0, 0, 255)

                # -------- DRAW BOX --------
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = f"{label} {avg_pred:.2f}"

                (tw, th), _ = cv2.getTextSize(text,
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              0.7, 2)

                y_text = max(y1 - 10, th + 10)

                # background
                cv2.rectangle(frame,
                              (x1, y_text - th - 5),
                              (x1 + tw, y_text),
                              color, -1)

                # text
                cv2.putText(frame,
                            text,
                            (x1, y_text - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            2)

        # ---------------- FINAL DECISION ----------------
        if not face_detected:
            final = "SHOW YOUR FACE"

        elif st.session_state.blink_count >= blink_required:
            final = "AUTHENTICATED"

        elif elapsed > time_limit:
            final = "SPOOF"

        else:
            final = "BLINK REQUIRED"

        # ---------------- FRAME INFO ----------------
        cv2.putText(frame, f"Blinks: {st.session_state.blink_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2)

        cv2.putText(frame, f"Time: {remaining}s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        # ---------------- DISPLAY ----------------
        frame_window.image(frame, channels="BGR")

        # ---------------- STATUS ----------------
        if final == "AUTHENTICATED":
            status.success("✅ AUTHENTICATED")
            break
        elif final == "SPOOF":
            status.error("❌ SPOOF")
            break
        elif final == "BLINK REQUIRED":
            status.warning("⚠ BLINK NOW")
        else:
            status.info("ℹ SHOW YOUR FACE")

    cap.release()