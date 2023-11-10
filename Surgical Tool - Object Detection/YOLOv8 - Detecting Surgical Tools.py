from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='data.yaml', epochs=10, imgsz=640)

# Run inference with the YOLOv8n model on the test image
results = model('C:\\Users\\cchan\\Laproscopic Surgery Work\\Surgical Tool - Object Detection\\train\\images\\002_jpg.rf.8f460f3798190c7107d647ba4eb12ec7.jpg')