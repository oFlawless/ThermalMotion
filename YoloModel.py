from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="Beads Tracking.v1i.yolov11\data.yaml")