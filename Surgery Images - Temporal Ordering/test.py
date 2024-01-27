from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict(source='images\\02142010_192913\\001.jpg', save=False, imgsz=640, conf=0.35, verbose = False)
for result in results:
    boxes = result.boxes.xywhn
    classes = result.boxes.cls
    print('boxes: ', len(boxes), boxes)
    print('classes: ', len(classes), classes)
