import cv2
import os
import time
import threading
import queue
import sounddevice
import speech_recognition as sr
import google.generativeai as genai
from pathlib import Path
import pyttsx3

genai.configure(api_key='AIzaSyCN-P2GYhWLU1Ke68c5ZSl_CG6Dx21HB00')

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 1.0,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    },
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]
)

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

def input_image_setup(file_loc):
    img = Path(file_loc)
    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")
    return [{"mime_type": "image/jpeg", "data": img.read_bytes()}]

def generate_gemini_response(input_prompt, image_loc):
    image_prompt = input_image_setup(image_loc)
    system_instruction = (
        "Describe the image based on the spoken prompt. "
        "Please do not use any * (asterices sign)in your response. "
        "Use plain language without markdown or special formatting."
        "also try to use less number of words maximum 2-3 lines to describe"
    )
    prompt_parts = [input_prompt, image_prompt[0], system_instruction]
    response = model.generate_content(prompt_parts)
    return response.text

def listen_thread_fn(recognizer, mic, result_queue, stop_event):
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            if not stop_event.is_set():
                text = recognizer.recognize_google(audio)
                result_queue.put(text)
        except sr.WaitTimeoutError:
            result_queue.put(None)
        except sr.UnknownValueError:
            result_queue.put("")
        except sr.RequestError as e:
            print(f"[❌] Speech API error: {e}")
            result_queue.put(None)

folder_path = 'CapturedImages'
os.makedirs(folder_path, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=1)

counter = 0
listening_mode = False

print("🎯 System Ready.")
print("➡️ Press 'v' to start listening loop.")
print("➡️ Press 'q' to quit.\n")

result_queue = queue.Queue()
stop_event = threading.Event()
listener_thread = None
last_speech_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed (Press 'v' to start listening)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('v') and not listening_mode:
        listening_mode = True
        stop_event.clear()
        print("\n🔄 Listening loop started (5 sec window)...")
        last_speech_time = time.time()
        listener_thread = threading.Thread(target=listen_thread_fn, args=(recognizer, mic, result_queue, stop_event))
        listener_thread.start()

    if key == ord('q'):
        print("[👋] Quitting...")
        stop_event.set()
        break

    if listening_mode:
        if not result_queue.empty():
            prompt = result_queue.get()
            if prompt is None:
                print("🛑 No speech detected. Exiting listening loop.\n")
                listening_mode = False
                stop_event.set()
                continue

            if prompt == "":
                print("⚠️ Speech was unclear. Listening again...")
                stop_event.set()
                listening_mode = True
                result_queue = queue.Queue()
                stop_event.clear()
                listener_thread = threading.Thread(target=listen_thread_fn, args=(recognizer, mic, result_queue, stop_event))
                listener_thread.start()
                continue

            last_speech_time = time.time()
            counter += 1
            filename = f"{folder_path}/image_{counter}.jpg"
            cv2.imwrite(filename, frame.copy())
            print(f"[📸] Image saved: {filename}")
            print(f"[🤖] Prompt: {prompt}")
            print("[🔁] Sending to Gemini...")

            try:
                response = generate_gemini_response(prompt, filename)
                print(f"\n[🧠 Gemini Response]:\n{response}\n")
                tts_engine.say(response)
                tts_engine.runAndWait()
            except Exception as e:
                print(f"[❌] Gemini error: {e}")

            stop_event.set()
            result_queue = queue.Queue()
            stop_event = threading.Event()
            listener_thread = threading.Thread(target=listen_thread_fn, args=(recognizer, mic, result_queue, stop_event))
            listener_thread.start()

        if listener_thread and not listener_thread.is_alive() and result_queue.empty():
            if time.time() - last_speech_time >= 5:
                print("⏳ No speech in 5 seconds. Exiting listening mode.\n")
                listening_mode = False
                stop_event.set()

cap.release()
cv2.destroyAllWindows()
