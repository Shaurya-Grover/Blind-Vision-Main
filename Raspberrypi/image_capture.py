import cv2
import os
import time
from datetime import datetime
import subprocess
import speech_recognition as sr
import pyttsx3

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

DATASET_DIR = "dataset"
CAPTURE_DURATION = 45  # seconds
CAPTURE_INTERVAL = 1   # seconds
recognizer = sr.Recognizer()


def speak(text):
    print(f"[TTS] {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()


def get_voice_input(prompt):
    with sr.Microphone(device_index=1) as source:  # Adjust index if needed
        print(prompt)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"[You said]: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            print("Speech recognition service error.")
            return None


def create_folder(name):
    person_folder = os.path.join(DATASET_DIR, name)
    os.makedirs(person_folder, exist_ok=True)
    return person_folder


def capture_photos(name):
    speak(f"Taking datasets for {name}")
    folder = create_folder(name)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"[INFO] Capturing photos for '{name}' for {CAPTURE_DURATION} seconds...")
    photo_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow('Capture', frame)

        if int(time.time() - start_time) % CAPTURE_INTERVAL == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            photo_count += 1
            print(f"[INFO] Photo {photo_count} saved: {filepath}")
            time.sleep(1)

        if time.time() - start_time > CAPTURE_DURATION:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Finished capturing {photo_count} photos for '{name}'.")


if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)

    while True:
        name = get_voice_input("Say the name for this entry or say 'start' to begin training:")
        if name is None:
            continue
        if name == "start":
            speak("Model training starting")
            break
        capture_photos(name)

    print("[INFO] Starting model training...")
    subprocess.run(["python", "model_training.py"])
    print("[INFO] Training complete. Encodings saved to 'encodings.pickle'")

