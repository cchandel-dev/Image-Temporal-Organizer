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


def create_siamese_model_with_yolo(input_shape, yolo_model):
    input_1 = tf.keras.Input(shape=input_shape)
    input_2 = tf.keras.Input(shape=input_shape)

    # Load a pre-trained image classification network as the shared subnetwork
    base_network = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape, pooling='max')
    base_network.trainable = False  # Freeze the pre-trained weights

    yolo_input_1 = tf.keras.Input(shape=(...))  # Shape for YOLO predictions
    yolo_input_2 = tf.keras.Input(shape=(...))  # Shape for YOLO predictions

    # Obtain YOLO detections for both inputs
    yolo_output_1 = yolo_model.predict(yolo_input_1)
    yolo_output_2 = yolo_model.predict(yolo_input_2)

    output_1 = base_network(input_1)
    output_2 = base_network(input_2)

    # Measure the similarity of the two outputs
    distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([output_1, output_2])

    # Concatenate the outputs and YOLO detections
    combined = tf.concat([output_1, output_2, distance, yolo_output_1, yolo_output_2], axis=1)

    # Final output layer
    output_dense = layers.Dense(512, activation='relu')(combined)
    output_dense = layers.Dense(256, activation='relu')(output_dense)
    output = layers.Dense(1, activation='sigmoid')(output_dense)

    siamese_model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
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
#print(labels[5:])
batch_size = 8
print(len(train_image_paths_1), len(test_image_paths_2))

#train_image_paths_1 = image_paths_1[int(len(image_paths_1)*0.8):]
#train_image_paths_2 = image_paths_2[int(len(image_paths_2)*0.8):]
#train_labels = labels[int(len(labels)*0.8):]



# Compile the model
input_shape = (299, 299, 3)  # Adjust the input shape based on your images
siamese_model = create_siamese_model_with_yolo(input_shape, YOLO("C:\\Users\\cchan\\computer-vision\\runs\\detect\\train5\\weights\\best.pt"))
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with your dataset
# Assuming you have `train_images_1` and `train_images_2` as input image pairs and `labels` as their corresponding temporal order labels
history = siamese_model.fit(data_generator(train_image_paths_1, train_image_paths_2, train_labels, batch_size), steps_per_epoch=len(train_labels) // batch_size, epochs=30, validation_data=data_generator(test_image_paths_1, test_image_paths_2, test_labels, batch_size), validation_steps = len(test_labels) // batch_size)

# Visualize the training history
plt.figure(figsize=(12, 6))

# Plot training and validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()
# Evaluate the model
# Assuming you have `test_images_1` and `test_images_2` for testing the model
#test_loss, test_accuracy = siamese_model.evaluate(data_generator(test_image_paths_1, test_image_paths_2, test_labels, batch_size), steps=len(test_labels)//batch_size)
#print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
