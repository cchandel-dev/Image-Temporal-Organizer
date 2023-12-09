from ultralytics import YOLO

model = YOLO('C:\\Users\\cchandel\\Laproscopic-Surgery-Work\\Surgery Images - Temporal Ordering\\best.pt')
results = model.predict(source='C:\\Users\\cchandel\\Laproscopic-Surgery-Work\\Surgical Tool - Object Detection\\data\\test\\images\\014.jpg', save=False, imgsz=640, conf=0.35, verbose = False)
for result in results:
    boxes = result.boxes.xywhn
    classes = result.boxes.cls
    print('boxes: ', len(boxes), boxes)
    print('classes: ', len(classes), classes)