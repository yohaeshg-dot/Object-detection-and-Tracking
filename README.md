# Real-Time Object Detection and Tracking

## Project Overview

This project implements real-time object detection and tracking using YOLOv8, OpenCV, and Deep SORT.

The system detects multiple objects such as people, cars, bikes, and other common objects in a video stream and assigns unique tracking IDs to each detected object.

---

## Features

- Real-time object detection using YOLOv8
- Multi-object tracking using Deep SORT
- Unique tracking IDs for detected objects
- Video file input support
- Webcam input support
- Bounding box visualization

---

## Technologies Used

- Python
- OpenCV
- YOLOv8
- Deep SORT
- NumPy

---

## Project Structure

```text
object_detection_tracking
│
├── main.py
├── README.md
├── yolov8n.pt
│
└── videos
    └── houses.mp4
```

---

## Installation

Install the required packages:

```bash
pip install ultralytics
pip install opencv-python
pip install deep-sort-realtime
```

---

## How to Run

For video input:

```bash
python main.py
```

For webcam input:

```python
cap = cv2.VideoCapture(0)
```

---

## Workflow

```text
Video/Webcam
      ↓
OpenCV
      ↓
YOLOv8 Detection
      ↓
Bounding Boxes
      ↓
Deep SORT Tracking
      ↓
Tracking IDs
      ↓
Display Output
```

---

## Sample Output

- Person → ID 1
- Person → ID 2
- Car → ID 3

The system continuously tracks detected objects across video frames.

---

## Future Enhancements

- Object counting
- Vehicle counting
- Speed estimation
- Face detection and tracking
- Custom YOLO model training

---
