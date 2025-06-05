import cv2
import time
import numpy as np
import requests
from ultralytics import YOLO
import pyttsx3

MODEL_PATH = "/home/gear5/Desktop/Blind-Vision-Genius-Olympiad/Main/Programs/yolo11n_ncnn_model"
CAMERA_ID = 0
RESOLUTION = (640, 640)
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

spoken_objects = set()

model = YOLO(MODEL_PATH, task='detect')
labels = model.names

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(3, RESOLUTION[0])
cap.set(4, RESOLUTION[1])

bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200

while True:
    t_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue

    results = model(frame, verbose=False)
    detections = results[0].boxes

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > 0.5:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if classname not in spoken_objects:
                print(f"[INFO] New object detected: {classname}")
                tts_engine.say(f"A {classname} is there infront of you")
                tts_engine.runAndWait()
                spoken_objects.add(classname)

    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("YOLO Detection - Local Camera", frame)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')

