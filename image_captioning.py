import cv2
import google.generativeai as genai
from pathlib import Path
import os


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

def generate_gemini_response(input_prompt, image_loc, question_prompt):
    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt, image_prompt[0], question_prompt]
    response = model.generate_content(prompt_parts)
    return response.text

folder_path = 'Videos'
os.makedirs(folder_path, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

counter = 0

print("Press 's' to capture and describe an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        counter += 1
        filename = f'{folder_path}/image_{counter}.png'
        cv2.imwrite(filename, frame)
        print(f"Image {counter} saved as {filename}")

        input_prompt = "You are an expert at understanding visual scenes."
        question_prompt = "Please describe the contents of this image in clear and concise English."

        try:
            result = generate_gemini_response(input_prompt, filename, question_prompt)
            print("\n[Description]:", result, "\n")
        except Exception as e:
            print("Error generating response:", e)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
