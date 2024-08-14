# 🐶 Dogs detector on YOLOv8

<div align="center">

<table>
  <tr>
    <td><img src="assets/dog_cropped.gif" alt="GIF 1" /></td>
    <td><img src="assets/dog_with_box_cropped.gif" alt="GIF 2" /></td>
  </tr>
  <tr>
    <td>original video</td>
    <td>after using algorithm</td>
  </tr>
</table>

</div>

## 🗃️ data

The dataset I used to train the model can be downloaded [here](https://universe.roboflow.com/yolo-lggkk/dogs-4i7ne)

## 🧠 model 

For dog detection, I used YOLOv8n. The model was trained using the following setup:

```python
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('yolov8n.yaml')

# Train model with dataset
dog_detector = model.train(data='config.yaml', epochs=30)
```
The model was trained for 30 epochs on a dataset of 949 images so it's not perfect but it performs reasonably well.

## 👨🏻‍💻 SORT algorithm

I used Simple Online and Realtime Tracking algorithm from [this](https://github.com/abewley/sort) repository created by [abewley](https://github.com/abewley)

[Licence](https://github.com/abewley/sort/blob/master/LICENSE)

## ✨ Features

- **Object detection**: Utilizes the YOLO model for detecting dogs in video clips. The model is loaded with pre-trained weights (`best.pt`).

- **Object tracking**: Implements the SORT algorithm to track objects across multiple video frames, assigning unique identifiers (IDs) to detected objects.

- **Detection visualization**: Draws bounding boxes around detected objects and displays their IDs on video frames. Visualization includes:
  - Colored bounding boxes (green) for detected objects.
  - Labels with object IDs.

- **Video processing**: Reads frames from a video file, processes them in real-time, and saves the results to a new video file (`output_dogs.mp4`).

- **Video handling**: Automatically detects the dimensions and frame rate (FPS) of the input video file and sets appropriate parameters for the output video.

- **Object management**: Allows tracking of objects for a specified number of frames (`max_age=30`), enabling tracking of objects that may briefly disappear from view.

- **Extensibility**: Easily modifiable and adaptable to different object detection models or tracking algorithms.

## 🔜 To be done

- **Model improving**: improving the detection model to detect the objects better and more efficiently

- **Data validation**: adjusting code to prevent boxes from disappearing between frames

- **Results file**: saving results to .csv file, assigned by object Id


