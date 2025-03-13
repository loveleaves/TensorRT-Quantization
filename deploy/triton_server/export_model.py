from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load an official model

model.export(format="onnx",batch=1,simplify=True)