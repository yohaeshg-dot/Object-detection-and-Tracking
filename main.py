import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Video source
cap = cv2.VideoCapture("videos/pooshan.jpg")
# For webcam use:
# cap = cv2.VideoCapture(0)

cv2.namedWindow("Object Detection & Tracking", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]

    detections = []

    for box in results.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        # Filter weak detections
        if confidence > 0.5:

            detections.append(
                (
                    [x1, y1, x2 - x1, y2 - y1],
                    confidence,
                    class_id
                )
            )

    # Update tracker
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Object Detection & Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()