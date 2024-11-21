from ultralytics import YOLO

model = YOLO("best.onnx")

results = model("TestImage.jpg")

for detection in results:

    print(detection.boxes.conf)

