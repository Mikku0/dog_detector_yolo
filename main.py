import numpy as np
from ultralytics import YOLO
import cv2
from sort.sort import Sort  # SORT algorithm


model = YOLO('runs/detect/train3/weights/best.pt')  # dogs detection model

cap = cv2.VideoCapture('dogs.mp4')  # example video

if not cap.isOpened():
    print("can't find video file")
    exit()


frame_width = int(cap.get(3))  # video properties
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# result video
out = cv2.VideoWriter('output_dogs4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

tracker = Sort(max_age=30)  # SORT algorithm with 30 frames without new object id detections

frame_num = 0


while cap.isOpened():  # main detection loop
    ret, frame = cap.read()  # reading each frame

    if not ret:
        break  # the end of video

    results = model(frame)  # processing each frame basing on detection model

    detections = []
    for detection in results:
        for box in detection.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # box coordinates
            score = box.score[0]
            if score > 0.5:  # confidence threshold
                detections.append([x1, y1, x2, y2, score])

    # converting to numpy array for SORT
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    trackers = tracker.update(detections)  # trackers update

    # drawing boxes on detected frames
    for trk in trackers:
        x1, y1, x2, y2, obj_id = map(int, trk)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {obj_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)  # saving frame
    frame_num += 1


cap.release()
out.release()