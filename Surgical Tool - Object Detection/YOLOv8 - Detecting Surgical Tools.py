from ultralytics import YOLO, settings

# Load a COCO-pretrained YOLOv8n model
model = YOLO('yolov8n.pt')
 
# Display model information (optional)
model.info()

# Update a setting
settings.update(
        {
            'runs_dir': 'C:\\Users\\cchan\\Laproscopic Surgery Work\\runs\\detect'      
        }
    )

# Reset settings to default values
settings.reset()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='data.yaml', epochs=100, imgsz=640, verbose = True, patience = 20)
