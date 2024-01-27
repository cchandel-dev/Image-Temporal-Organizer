import os, time
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess images
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))  # Adjust the target size based on your requirements
    img_array = img_to_array(img) / 255.0  # Normalize the pixel values
    return img_array

# Read a results object class tensor and spit out a tensor of fixed length which represents frequency of each class
def class_tensor_frequency(class_tensor, length = 7):
    class_tensor = class_tensor.numpy()
    frequency = {i: 0 for i in range(length)}  # Initialize frequencies for each class label
    for cls in class_tensor:
        frequency[cls] += 1
    # Convert the frequency dictionary values to a tensor
    frequency_tensor = tf.constant([frequency[i] for i in range(length)], dtype=tf.int32)
    return frequency_tensor

yolo_model = YOLO("best.pt")

filenames = [
                '02142010_192913\\001.jpg',
                 '02142010_192913\\020.jpg',
                 '02142010_204825\\007.jpg',
                 '02142010_204825\\020.jpg',
                 '09092011_130836\\013.jpg'
             ]

for filename in filenames:
    # image = load_and_preprocess_image(os.path.join('images', filename))
    results = yolo_model.predict(os.path.join('images', filename), save=False, imgsz=640, show = True)
    time.sleep(5)
    num = []
    print('/n', '************', filename, '************')
    # convert the yolo output to a length tensor
    for result in results:
        class_tensor = result.boxes.cls
        print('class_tensor: ', class_tensor)
        num.append(class_tensor_frequency(class_tensor))
  
    num = tf.convert_to_tensor(num)
    print('end result num tensor: ', num)
