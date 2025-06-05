import cv2
import os
import time
import speech_recognition as sr
import google.generativeai as genai
from pathlib import Path


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

def input_image_setup(file_loc):
    img = Path(file_loc)
    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")
    return [{"mime_type": "image/jpeg", "data": img.read_bytes()}]

def generate_gemini_response(input_prompt, image_loc):
    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt, image_prompt[0], "Describe the image based on the spoken prompt."]
    response = model.generate_content(prompt_parts)
    return response.text

def get_microphone_input():
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        print("\n Speak your prompt now...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print(" Sorry, could not understand.")
        except sr.RequestError:
            print(" Could not request results from Google Speech API.")
    return None

folder_path = 'CapturedImages'
os.makedirs(folder_path, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

counter = 0

print("Press 'v' to give voice prompt & capture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('v'):
        prompt = get_microphone_input()
        if prompt:
            counter += 1
            filename = f'{folder_path}/image_{counter}.jpg'
            cv2.imwrite(filename, frame)
            print(f"[📸] Image saved as {filename}")

            print("[🤖] Generating Gemini response, please wait...")
            try:
                result = generate_gemini_response(prompt, filename)
                print("\n[🧠 Gemini Response]:", result, "\n")
            except Exception as e:
                print("Error generating response:", e)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

