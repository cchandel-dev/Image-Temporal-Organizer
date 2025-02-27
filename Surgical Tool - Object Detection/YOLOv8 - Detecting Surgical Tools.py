from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
 
# Display model information (optional)
model.info()


# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='data.yaml', epochs=1000, imgsz=640, verbose = True, patience = 100)
