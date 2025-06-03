# ⚠️ Prerequisite

**Ensure your system has a properly configured GPU** that can efficiently handle `yolov11l` large object detection weights and deliver good FPS during real-time inference.

---

#  AI Task Server Setup Guide

This guide outlines how to develop a modular client-server architecture for two AI tasks:

-  **Image Captioning**
-  **Object Detection**

---

## 📌 Workflow Overview

1. **Start by providing the following scripts to ChatGPT**:
   - `image_captioning.py`
   - `object_detection.py`

2. **Prompt ChatGPT to design both**:
   - **Client-side code**: Responsible for capturing image(s) or video stream and sending them to the server.
   - **Server-side code**: Responsible for processing the incoming data using AI models (captioning or detection) and returning results.

3. **For each module**:
   -  **Image Captioning**:
     - Client captures a single image and sends it to the server.
     - Server returns a descriptive caption.
   -  **Object Detection**:
     - Client sends live video frames in real-time.
     - Server performs object detection and responds with detected labels and bounding boxes.

4. ✅ Once both modules are working independently:
   - Continue submitting your **remaining program files** to ChatGPT.
   - ChatGPT will extend and refine the client-server architecture as needed.

---

