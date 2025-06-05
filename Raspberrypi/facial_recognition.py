import cv2
import face_recognition
import numpy as np
import pickle
from fer import FER


print("[INFO] Loading face encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]


emotion_detector = FER()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv_scaler = 5  # for faster face recognition

print("[INFO] Face and Emotion detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (0, 0), fx=1 / cv_scaler, fy=1 / cv_scaler)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, top - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
        face_crop = frame[top:bottom, left:right]
        if face_crop.size > 0:
            emotions = emotion_detector.detect_emotions(face_crop)
            if emotions:
                top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
                cv2.putText(frame, f"{top_emotion}",
                            (left + 6, bottom + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Face & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

