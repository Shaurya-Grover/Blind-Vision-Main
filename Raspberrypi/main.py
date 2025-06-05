import tkinter as tk
from tkinter import messagebox
import subprocess
import pyttsx3
import os


engine = pyttsx3.init()
engine.setProperty('rate', 150)

task_process = None
current_task = None

TASK_COMMANDS = {
    "Object Detection": "Programs/object_detection.py",
    "Image Captioning": "Programs/clap.py",
    "Face Recognition": "Programs/facial_recognition.py",
    "Custom Face Training": "Programs/image_capture.py",
    "Navigation": "Programs/navigation.py"
}

def stop_current_task():
    global task_process
    if task_process and task_process.poll() is None:
        task_process.terminate()
        task_process.wait()
        print("[INFO] Previous process terminated.")

def run_task(task_name):
    global task_process, current_task

    if current_task == task_name:
        stop_current_task()
        engine.say(f"{task_name} stopped.")
        engine.runAndWait()
        current_task = None
        return

    stop_current_task()
    current_task = task_name

    engine.say(f"{task_name} running")
    engine.runAndWait()

    script = TASK_COMMANDS.get(task_name)
    if not script or not os.path.exists(script):
        messagebox.showerror("File Not Found", f"Script for {task_name} not found.")
        current_task = None
        return

    try:
        task_process = subprocess.Popen(["python", script])
    except Exception as e:
        messagebox.showerror("Error", str(e))
        current_task = None

app = tk.Tk()
app.title("Blind Vision")
app.geometry("400x500")
app.configure(bg="white")

style = {
    "font": ("Helvetica", 14, "bold"),
    "bg": "#4CAF50",
    "fg": "white",
    "activebackground": "#45a049",
    "bd": 0,
    "relief": tk.FLAT,
    "height": 2,
    "width": 25
}

title = tk.Label(app, text="Blind Vision", font=("Helvetica", 24, "bold"), bg="white", fg="#333")
title.pack(pady=30)

for task in TASK_COMMANDS.keys():
    btn = tk.Button(app, text=task, command=lambda t=task: run_task(t), **style)
    btn.pack(pady=10)

app.mainloop()

stop_current_task()

