import cv2
import google.generativeai as genai
from pathlib import Path
import threading
import os
import atexit
import pyttsx3
import time
from cvzone.HandTrackingModule import HandDetector
from queue import Queue
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)
SW_PIN = 17  # Joystick button connected to GPIO17
GPIO.setup(SW_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button input with pull-up

def button_pressed():
    return GPIO.input(SW_PIN) == GPIO.LOW  # Button is pressed if GPIO input is LOW

# Speech engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.4)

# List available voices and set them based on language
voices = engine.getProperty('voices')
hindi_voice = None
english_voice = None

# Find Hindi and English voices based on their language or name
for voice in voices:
    if "hi" in voice.languages or "Neerja" in voice.name or "Aditi" in voice.name:  # Change as per available voices
        hindi_voice = voice.id
    if "en-us" in voice.languages or "David" in voice.name or "Zira" in voice.name:  # Change as per available voices
        english_voice = voice.id

# Fallback if specific voices are not found
if hindi_voice is None:
    hindi_voice = voices[0].id
if english_voice is None:
    english_voice = voices[1].id

def set_voice(language):
    """Set the voice based on the language."""
    if language == "hindi":
        engine.setProperty('voice', hindi_voice)
    elif language == "english":
        engine.setProperty('voice', english_voice)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

genai.configure(api_key='AIzaSyCN-P2GYhWLU1Ke68c5ZSl_CG6Dx21HB00')

counter = 0
gesture_processed = False
gesture_timestamp = 0

generation_config = {
    "temperature": 1.0,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

detector = HandDetector(detectionCon=0.8, maxHands=2)

# Define gesture patterns for different prompts
patterns = {
    "Left": {
        (0, 1, 0, 0, 0): "hindi"  # Only index finger up (left hand) triggers Hindi
    },
    "Right": {
        (0, 1, 0, 0, 0): "english"  # Only index finger up (right hand) triggers English
    }
}

# Additional gesture for stopping the program
exit_gesture = {
    "Right": (1, 1, 0, 0, 1)  # Stop gesture with the right hand
}

speech_queue = Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:  # Stop signal
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def input_image_setup(file_loc):
    if not (img := Path(file_loc)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")
    image_parts = [{"mime_type": "image/jpeg", "data": Path(file_loc).read_bytes()}]
    return image_parts

def generate_gemini_response_async(input_prompt, image_loc, question_prompt):
    response = generate_gemini_response(input_prompt, image_loc, question_prompt)
    print(response)
    speech_queue.put(response)

def generate_gemini_response(input_prompt, image_loc, question_prompt):
    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt, image_prompt[0], question_prompt]
    response = model.generate_content(prompt_parts)
    return response.text

def delete_saved_images():
    for i in range(1, counter + 1):
        filename = f'Videos/image_{i}.png'
        if os.path.exists(filename):
            os.remove(filename)

# Define the prompts for Hindi and English
input_prompts = {
    "hindi": """
आप एक विशेषज्ञ हैं जो परिदृश्यों को समझने और चित्रों से वस्तुओं की पहचान करने में सक्षम हैं।
आपको छवियां प्राप्त होंगी और आपको 20 से 25 शब्दों में हिंदी में परिदृश्यों और वस्तुओं का वर्णन करना होगा। (keep the level of hindi not so hard for the person to understand, keep it easy to understand but also make it upto a level)
""",
    "english": """
You are an expert in understanding scenarios and identifying objects from images.
You will receive input images and need to describe the scenarios and objects in 20 to 40 words in English (if in case of money: tell the denotion and sum of the money and also tell if the money is real or fake, do not mention about it if there is none).
"""
}

atexit.register(delete_saved_images)
folder_path = 'Videos'
os.makedirs(folder_path, exist_ok=True)

# Initialize the debounce mechanism
last_button_press_time = 0  # Last time the button was pressed
button_debounce_time = 0.5  # Minimum time (in seconds) between presses

while True:
    ret, frame = cap.read()
    hands, frame = detector.findHands(frame)

    exit_program = False  # Flag to detect the stop condition

    if hands:
        for hand in hands:
            hand_type = hand["type"]  # Left or Right hand
            fingers = detector.fingersUp(hand)  # Check which fingers are up
            finger_tuple = tuple(fingers)

            # Check if the detected gesture is in patterns
            if hand_type in patterns and finger_tuple in patterns[hand_type] and not gesture_processed:
                language = patterns[hand_type][finger_tuple]  # Get the language prompt
                input_prompt = input_prompts[language]  # Select the correct prompt based on gesture
                set_voice(language)  # Set the voice based on the language
                gesture_processed = True
                gesture_timestamp = time.time()

            # Check if the exit gesture is detected
            if hand_type == "Right" and finger_tuple == exit_gesture["Right"]:
                print("Exit gesture detected. Closing the program.")
                exit_program = True
                break

    # Check for joystick button press
    if button_pressed() and (time.time() - last_button_press_time > button_debounce_time):
        last_button_press_time = time.time()  # Update the last button press time

        # Only take the image if a gesture hasn't already processed
        if not gesture_processed:
            language = "english"  # Default language (you can change to "hindi" if preferred)
            input_prompt = input_prompts[language]
            set_voice(language)

        gesture_processed = False  # Reset the gesture flag after button press
        counter += 1
        filename = f'{folder_path}/image_{counter}.png'
        cv2.imwrite(filename, frame)
        image_loc = filename
        question_prompt = "What is this image? Describe precisely."

        # Add confirmation speech after image is saved
        speech_queue.put(f"Image taken.")  # Confirmation message

        # Generate response in a separate thread to avoid blocking
        threading.Thread(target=generate_gemini_response_async, args=(input_prompt, image_loc, question_prompt)).start()
        print(f"Image {counter} saved as {filename}")

    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed or exit gesture is detected
    if cv2.waitKey(1) == ord('q') or exit_program:
        break

speech_queue.put(None)  # Stop the speech worker thread
speech_thread.join()
cv2.destroyAllWindows()



