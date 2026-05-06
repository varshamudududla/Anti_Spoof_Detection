import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("liveness_model_final_v3.keras", compile=False)

# ---------------- MEDIAPIPE ----------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye):
    p = [landmarks[i] for i in eye]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

blink_count = 0
blink_flag = False
EAR_THRESHOLD = 0.20

# smooth prediction
pred_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------------- DETECTION ----------------
    results = face_detection.process(rgb)
    mesh_results = face_mesh.process(rgb)

    # ---------------- BLINK DETECTION ----------------
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            landmarks = [(int(lm.x * w), int(lm.y * h))
                         for lm in face_landmarks.landmark]

            ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
            ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (ear_left + ear_right) / 2

            if ear < EAR_THRESHOLD:
                if not blink_flag:
                    blink_flag = True
            else:
                if blink_flag:
                    blink_count += 1
                    blink_flag = False

    avg_pred = 0

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

            if face.size == 0:
                continue

            # preprocess
            face = cv2.resize(face, (128, 128))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face, verbose=0)[0][0]
            pred_buffer.append(pred)
            avg_pred = np.mean(pred_buffer)

            # label
            if avg_pred > 0.7:
                label = "REAL"
                color = (0, 255, 0)
            else:
                label = "SPOOF"
                color = (0, 0, 255)

            # ---------------- DRAW BOUNDING BOX ----------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ---------------- TEXT ABOVE BOX ----------------
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

    # ---------------- BLINK COUNT DISPLAY ----------------
    cv2.putText(frame,
                f"Blinks: {blink_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2)

    # ---------------- SHOW FRAME ----------------
    cv2.imshow("Face Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()