import numpy as np
from ultralytics import YOLO
import cv2
import sys
from sort.sort import Sort  # SORT algorithm

# Check if file argument is provided
if len(sys.argv) < 2:
    print('No file argument provided.')
    sys.exit()

file_path = sys.argv[1]  # input argument

# Open the YOLO model for detection
model = YOLO('runs/detect/train3/weights/best.pt')  # Dog detection model

# Check the file extension
file_extension = file_path.split('.')[-1].lower()
file_name = file_path.split('.')[0]

# Initialize tracker
tracker = Sort(max_age=30)  # SORT algorithm with 30 frames without new object id detections

if file_extension in ['jpg', 'jpeg', 'png', 'jfif', 'bmp', 'tif', 'tiff', '.webp', 'ppm', 'pgm', 'jp2']:
    # Handle image files
    image = cv2.imread(file_path)
    if image is None:
        print("Can't open the image file.")
        sys.exit()

    # Process image
    results = model(image)  # processing the image based on detection model

    detections = []
    for detection in results:
        for box in detection.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # box coordinates
            score = box.conf[0]
            if score > 0.5:  # confidence threshold
                detections.append([x1, y1, x2, y2, score])

    # converting to numpy array for SORT
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    trackers = tracker.update(detections)  # trackers update

    # drawing boxes on detected image
    for trk in trackers:
        x1, y1, x2, y2, obj_id = map(int, trk)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {obj_id}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result image
    output_file = file_name + '_output.png'
    cv2.imwrite(output_file, image)

    print('Image processing complete. Saved as', output_file)

elif file_extension in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']:
    # Handle video files
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Can't find video file.")
        sys.exit()

    frame_width = int(cap.get(3))  # video properties
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Result video
    output_file = file_name + '_output.mp4'
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_num = 0

    while cap.isOpened():  # main detection loop
        ret, frame = cap.read()  # reading each frame

        if not ret:
            break  # the end of video

        results = model(frame)  # processing each frame based on detection model

        detections = []
        for detection in results:
            for box in detection.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # box coordinates
                score = box.conf[0]
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
    print('Video processing complete. Saved as output_dogs.mp4.')
