import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models, applications

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load and preprocess images
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))  # Adjust the target size based on your requirements
    img_array = img_to_array(img) / 255.0  # Normalize the pixel values
    return img_array

# Define a custom data generator
def data_generator(image_paths_1, image_paths_2, labels, batch_size=32):
    num_samples = len(labels)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_paths_1 = image_paths_1[offset:offset+batch_size]
            batch_paths_2 = image_paths_2[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]

            batch_images_1 = [load_and_preprocess_image(image_path) for image_path in batch_paths_1]
            batch_images_2 = [load_and_preprocess_image(image_path) for image_path in batch_paths_2]

            yield {
                        'input_1': [np.array(batch_images_1), np.array(batch_paths_1)],
                        'input_2': [np.array(batch_images_2), np.array(batch_paths_2)]
                    }, np.array(batch_labels)

def custom_fit(siamese_model, yolo_model, num_epochs=10, steps_per_epoch=100, batch_size = 32):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        data_gen = data_generator(train_image_paths_1, train_image_paths_2, train_labels, batch_size)
        for step in range(steps_per_epoch):
            inputs, labels = next(data_gen)
            
            input_1_images, input_1_paths = inputs['input_1']
            input_2_images, input_2_paths = inputs['input_2']

            # Load and preprocess images for YOLO input
            yolo_input_1 = [load_img(image_path) for image_path in input_1_paths]
            yolo_input_2 = [load_img(image_path) for image_path in input_2_paths]
            
            # Make YOLO predictions
            yolo_output_1 = yolo_model.predict(yolo_input_1, save=False, imgsz=640, conf=0.35, verbose = False)
            yolo_output_2 = yolo_model.predict(yolo_input_2, save=False, imgsz=640, conf=0.35, verbose = False)

            num1 = []
            num2 = []

            # convert the yolo output to a length tensor
            for idx in range(batch_size):
                num1.append([len(yolo_output_1[idx].boxes.xywhn)])
                num2.append([len(yolo_output_2[idx].boxes.xywhn)])

            # Train the siamese model
            with tf.GradientTape() as tape:
                num1 = np.array(num1)
                num2 = np.array(num2)
                labels = [[i] for i in labels]
                outputs = siamese_model([input_1_images, input_2_images, num1, num2])
                loss = loss_fn(labels, outputs)
                # Calculate accuracy
                predicted_labels = [1 if output >= 0.5 else 0 for output in outputs]
                true_labels = [label[0] for label in labels]
                accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
                # Compute gradients and update weights
                gradients = tape.gradient(loss, siamese_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, siamese_model.trainable_variables))

            # Display loss and accuracy
            print(f"Step {step}/{steps_per_epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

def create_siamese_model(input_shape):
    input_1 = tf.keras.Input(shape=input_shape)
    input_2 = tf.keras.Input(shape=input_shape)
    input_num1 = tf.keras.Input(shape=(1))  # Shape can be adjusted based on your data
    input_num2 = tf.keras.Input(shape=(1))  # Shape can be adjusted based on your data

    # Load a pre-trained image classification network as the shared subnetwork
    base_network = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape, pooling='max')
    base_network.trainable = False  # Freeze the pre-trained weights

    output_1 = base_network(input_1)
    output_2 = base_network(input_2)

    # Measure the similarity of the two outputs
    distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([output_1, output_2])

    # Concatenate the outputs and YOLO detections
    combined = tf.concat([output_1, output_2, input_num1, input_num2], axis=1)
    output_dense = layers.Dense(512, activation='relu')(combined)
    output_dense = layers.Dense(256, activation='relu')(output_dense)
    outputs = layers.Dense(1, activation='sigmoid')(output_dense)

    siamese_model = tf.keras.Model(inputs=[input_1, input_2, input_num1, input_num2], outputs=outputs)
    return siamese_model





# Read the CSV file
csv_file_path = './data.json'
data = pd.read_json(csv_file_path)
train = data['train']
test = data['test']
#define data_generator inputs
train_image_paths_1 = train['img_path_1']
train_image_paths_2 = train['img_path_2']
train_labels = train['labels']

test_image_paths_1 = test['img_path_1']
test_image_paths_2 = test['img_path_2']
test_labels = test['labels']

training_length = len(train_image_paths_1)
batch_size = 32
print("training length: ", len(train_image_paths_1), "testing length: ", len(test_image_paths_2))

# Compile the model
input_shape = (299, 299, 3)  # Adjust the input shape based on your images

yolo_model = YOLO("best.pt")
siamese_model = create_siamese_model(input_shape)
# custom_fit(siamese_model, yolo_model, data_gen)
custom_fit(siamese_model, yolo_model, num_epochs=10, steps_per_epoch=training_length//batch_size, batch_size = batch_size)