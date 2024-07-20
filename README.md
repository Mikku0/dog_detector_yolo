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
