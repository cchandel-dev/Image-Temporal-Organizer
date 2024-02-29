import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
import os, sys, time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras import regularizers

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(title="Select a File")

    if file_path:
        return file_path


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
MODE = 0 # 0 is conv + yolo, 1 is just conv, 2 is just yolo
yolo_model = YOLO("best.pt")


def checkpoint(model_name, completed_training):
    # Define the ModelCheckpoint callback
    checkpoint_path = 'path/to/save/checkpoints/{model_name}_{completed_training}.h5'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                checkpoint_path,
                                                                monitor='val_loss',  # Monitor validation loss
                                                                save_best_only=True,  # Save only the best model
                                                                mode='min',  # Mode for monitoring the metric (minimize validation loss)
                                                                verbose=1  # Verbosity level
                                                            )
    
# Define a function to save the model
def save_model(model, file_path):
    model.save(file_path)
    print(f"Model saved at {file_path}")

def loading_bar(total, completed):
    progress = int((completed / total) * 100)
    toolbar_width = 100
    progress_length = int(toolbar_width * (completed / total))
    sys.stdout.write("[%s%s] %d%%" % ("=" * progress_length, " " * (toolbar_width - progress_length), progress))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 6))  # return to start of line, after ']'


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
            
            # Make YOLO predictions
            yolo_output_1 = yolo_model.predict(batch_paths_1, save=False, imgsz=640, conf=0.35, verbose = False)
            yolo_output_2 = yolo_model.predict(batch_paths_2, save=False, imgsz=640, conf=0.35, verbose = False)

            num1 = []
            num2 = []

            # convert the yolo output to a length tensor
            for idx in range(batch_size):
                class_tensor_1 = yolo_output_1[idx].boxes.cls
                class_tensor_2 = yolo_output_2[idx].boxes.cls
                
                num1.append(class_tensor_frequency(class_tensor_1))
                num2.append(class_tensor_frequency(class_tensor_2))

            yield {
                        'input_1': [np.array(batch_images_1), np.array(batch_paths_1), np.array(num1)],
                        'input_2': [np.array(batch_images_2), np.array(batch_paths_2), np.array(num2)]
                    }, np.array(batch_labels)


# Read a results object class tensor and spit out a tensor of fixed length which represents frequency of each class
def class_tensor_frequency(class_tensor, length = 7):
    class_tensor = class_tensor.numpy()
    frequency = {i: 0 for i in range(length)}  # Initialize frequencies for each class label
    for cls in class_tensor:
        frequency[cls] += 1
    # Convert the frequency dictionary values to a tensor
    frequency_tensor = tf.constant([frequency[i] for i in range(length)], dtype=tf.int32)
    return frequency_tensor

def custom_fit(siamese_model, history, yolo_model, num_epochs=10, steps_per_epoch=100, batch_size = 64, patience = 3):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}/{num_epochs}")
        per_epoch_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}  # Create an empty history dictionary
        train_data_gen = data_generator(train_image_paths_1, train_image_paths_2, train_labels, batch_size)
        val_data_gen = data_generator(test_image_paths_1, test_image_paths_2, test_labels, batch_size)
        #Training loop
        for step in range(steps_per_epoch):
            loading_bar(steps_per_epoch, step)
            inputs, labels = next(train_data_gen)
            
            input_1_images, input_1_paths, input_1_summary = inputs['input_1']
            input_2_images, input_2_paths, input_2_summary = inputs['input_2']

            
            # Train the siamese model
            with tf.GradientTape() as tape:
                labels = [[i] for i in labels]
                if MODE == 0:
                    outputs = siamese_model([input_1_images, input_2_images, input_1_summary, input_2_summary])
                if MODE == 1:
                    outputs = siamese_model([input_1_images, input_2_images])
                if MODE == 2:
                     outputs = siamese_model([input_1_summary, input_2_summary])
                loss = loss_fn(labels, outputs)
                # Calculate accuracy
                predicted_labels = [1 if output >= 0.5 else 0 for output in outputs]
                true_labels = [label[0] for label in labels]
                accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
                # Compute gradients and update weights
                gradients = tape.gradient(loss, siamese_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, siamese_model.trainable_variables))
                per_epoch_history['loss'].append(loss)
                per_epoch_history['accuracy'].append(accuracy)
 
        # Validation loop, I want to see each step, so to make the graph smooth I will recaulte batch size and hold the number of val_steps+per_epoch constant
        val_steps_per_epoch = len(test_image_paths_1) //  batch_size # Adjust for your validation data length
        print(f"Validation Epoch {epoch + 1}/{num_epochs}")
        for val_step in range(val_steps_per_epoch):
            loading_bar(val_steps_per_epoch, val_step)
            val_inputs, val_labels = next(val_data_gen)
            input_1_images, input_1_paths, input_1_summary = val_inputs['input_1']
            input_2_images, input_2_paths, input_2_summary = val_inputs['input_2']
            val_labels = [[i] for i in val_labels]
            if MODE == 0:
                outputs = siamese_model([input_1_images, input_2_images, input_1_summary, input_2_summary])
            if MODE == 1:
                outputs = siamese_model([input_1_images, input_2_images])
            if MODE == 2:
                    outputs = siamese_model([input_1_summary, input_2_summary])
            val_loss = loss_fn(val_labels, outputs)
            # Calculate accuracy
            predicted_labels = [1 if output >= 0.5 else 0 for output in outputs]
            true_labels = [label[0] for label in val_labels]
            val_accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
            per_epoch_history['val_loss'].append(val_loss)
            per_epoch_history['val_accuracy'].append(val_accuracy)

        try:
            # Display loss and accuracy
            avg_loss = sum(per_epoch_history['loss']) / len(per_epoch_history['loss'])
            avg_val_loss = sum(per_epoch_history['val_loss']) / len(per_epoch_history['val_loss'])
            avg_acc = sum(per_epoch_history['accuracy']) / len(per_epoch_history['accuracy'])
            avg_val_acc = sum(per_epoch_history['val_accuracy']) / len(per_epoch_history['val_accuracy'])
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {avg_acc:.4f}, Validation Accuracy: {avg_val_acc:.4f}")
            history['loss'].append(avg_loss)
            history['val_loss'].append(avg_val_loss)
            history['accuracy'].append(avg_acc)
            history['val_accuracy'].append(avg_val_acc)
        except:
            print("made a mistake... but we will CONTINUE.")
       # Check for early stopping
        if epoch >= patience:  # Check if current epoch is greater than or equal to patience
            recent_val_losses = history['val_loss'][-(patience + 1):]
            if all(recent_val_losses[i] <= recent_val_losses[i + 1] for i in range(patience)):
                print(f"Stopping early as validation loss didn't improve for {patience} epochs.")
                break
    return history, siamese_model

def create_siamese_model(input_shape):
    
    if MODE == 0:
        input_1 = tf.keras.Input(shape=input_shape)
        input_2 = tf.keras.Input(shape=input_shape)    
        # Load a pre-trained image classification network as the shared subnetwork
        base_network = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape, pooling='max')
        base_network.trainable = False  # Freeze the pre-trained weights
        output_1 = base_network(input_1)
        output_2 = base_network(input_2)
        input_num1 = tf.keras.Input(shape=(7))  # Shape can be adjusted based on your data
        input_num2 = tf.keras.Input(shape=(7))  # Shape can be adjusted based on your data
        # Concatenate the outputs and YOLO detections
        embedding = tf.concat([output_1, output_2, input_num1, input_num2], axis=1)
        # Run this embedding into the MLP
        output_dense = layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))(embedding)  # Adding L2 regularization
        output_dense = layers.Dropout(0.6)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        output_dense = layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))(embedding)  # Adding L2 regularization
        output_dense = layers.Dropout(0.6)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        output_dense = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01))(output_dense)  # Adding L2 regularization
        output_dense = layers.Dropout(0.5)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        outputs = layers.Dense(1, activation='sigmoid')(output_dense)
        siamese_model = tf.keras.Model(inputs=[input_1, input_2, input_num1, input_num2], outputs=outputs)

    if  MODE == 1:
        input_1 = tf.keras.Input(shape=input_shape)
        input_2 = tf.keras.Input(shape=input_shape)    
        # Load a pre-trained image classification network as the shared subnetwork
        base_network = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape, pooling='max')
        base_network.trainable = False  # Freeze the pre-trained weights
        output_1 = base_network(input_1)
        output_2 = base_network(input_2)
        # Concatenate the outputs
        embedding = tf.concat([output_1, output_2], axis=1)
        # Run this embedding into the MLP
        output_dense = layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))(embedding)  # Adding L2 regularization
        output_dense = layers.Dropout(0.6)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        output_dense = layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))(embedding)  # Adding L2 regularization
        output_dense = layers.Dropout(0.6)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        output_dense = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01))(output_dense)  # Adding L2 regularization
        output_dense = layers.Dropout(0.5)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        outputs = layers.Dense(1, activation='sigmoid')(output_dense)
        siamese_model = tf.keras.Model(inputs=[input_1, input_2], outputs=outputs)

    if MODE == 2:
        input_num1 = tf.keras.Input(shape=(7))
        input_num2 = tf.keras.Input(shape=(7))
        embedding = tf.concat([input_num1, input_num2], axis=1)
        output_dense = layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))(embedding)  # Adding L2 regularization
        output_dense = layers.Dropout(0.6)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        output_dense = layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.01))(embedding)  # Adding L2 regularization
        output_dense = layers.Dropout(0.6)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        output_dense = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01))(output_dense)  # Adding L2 regularization
        output_dense = layers.Dropout(0.5)(output_dense)  # Adding Dropout layer with a dropout rate of 0.5
        outputs = layers.Dense(1, activation='sigmoid')(output_dense)
        siamese_model = tf.keras.Model(inputs=[input_num1, input_num2], outputs=outputs)

    return siamese_model

def visualize_activations(model, layer_name, input_image_1, input_image_2):
    # Create a submodel that includes the specified layer
    intermediate_model = models.Model(inputs=model.input,
                                      outputs=model.get_layer(layer_name).output)

    image_1 = load_and_preprocess_image(input_image_1)
    image_2 = load_and_preprocess_image(input_image_2)
            
    # Make YOLO predictions
    yolo_output_1 = yolo_model.predict(input_image_1, save=False, imgsz=640, conf=0.35, verbose=False)
    yolo_output_2 = yolo_model.predict(input_image_2, save=False, imgsz=640, conf=0.35, verbose=False)

    num1 = []
    num2 = []

    # convert the yolo output to a length tensor
    for idx in range(1):
        class_tensor_1 = yolo_output_1[idx].boxes.cls
        class_tensor_2 = yolo_output_2[idx].boxes.cls
                
        num1.append(class_tensor_frequency(class_tensor_1))
        num2.append(class_tensor_frequency(class_tensor_2))

    num1 = np.array(num1)
    num2 = np.array(num2)

    activations = intermediate_model.predict([np.expand_dims(image_1, axis=0), np.expand_dims(image_2, axis=0), num1, num2])
    
    # Squeeze to remove dimensions with size 1
    activations = np.squeeze(activations)

    # Visualize the activations as a line plot
    plt.plot(activations)
    plt.title(f'Activations of Layer {layer_name}')
    plt.xlabel('Activation Index')
    plt.ylabel('Activation Value')
    plt.show()

def compute_saliency_map(siamese_model, input_image_1, input_image_2, num1, num2, layer_name):
    # Create a submodel that includes the specified layer
    intermediate_model = models.Model(inputs=siamese_model.input,
                                      outputs=siamese_model.get_layer(layer_name).output)

    # Convert input images and numerical inputs to TensorFlow tensors
    input_array_1 = tf.convert_to_tensor(np.expand_dims(input_image_1, axis=0))
    input_array_2 = tf.convert_to_tensor(np.expand_dims(input_image_2, axis=0))
    # num1_array = tf.convert_to_tensor(np.expand_dims(num1, axis=0))
    # num2_array = tf.convert_to_tensor(np.expand_dims(num2, axis=0))

    num1_array = num1
    num2_array = num2
    
    # Watch the input tensors
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_array_1)
        tape.watch(input_array_2)
        # tape.watch(num1_array)
        # tape.watch(num2_array)

        # Get the output of the specified layer for both inputs
        layer_output_1 = intermediate_model([input_array_1, input_array_2, num1_array, num2_array])
        layer_output_2 = intermediate_model([input_array_1, input_array_2, num1_array, num2_array])

    # Calculate the gradients of the output with respect to the input images
    gradients_1 = tape.gradient(layer_output_1, [input_array_1, num1_array])
    gradients_2 = tape.gradient(layer_output_2, [input_array_2, num2_array])

    # Extract the gradients of the input images
    gradients_image_1, _ = gradients_1
    gradients_image_2, _ = gradients_2

    # Compute the saliency map using gradients
    saliency_map_image_1 = tf.abs(gradients_image_1.numpy()[0]) / 2.0
    saliency_map_image_2 = tf.abs(gradients_image_2.numpy()[0]) / 2.0
    # saliency_map_num1 = tf.abs(gradients_num1.numpy()[0]) / 2.0
    # saliency_map_num2 = tf.abs(gradients_num2.numpy()[0]) / 2.0

    # Combine saliency maps for images and numerical inputs
    #saliency_map_combined = (saliency_map_image_1 + saliency_map_image_2) / 2.0 # 4.0 # + saliency_map_num1 + saliency_map_num2

    # Reduce to a single channel if the input images are RGB
    if saliency_map_image_1.shape[-1] == 3:
        saliency_map_image_1 = np.mean(saliency_map_image_1, axis=-1)

    if saliency_map_image_2.shape[-1] == 3:
        saliency_map_image_2 = np.mean(saliency_map_image_2, axis=-1)

    # Normalize the saliency map to the range [0, 1]
    #saliency_map_combined = (saliency_map_combined - np.min(saliency_map_combined)) / (np.max(saliency_map_combined) - np.min(saliency_map_combined))
    saliency_map_image_1 = (saliency_map_image_1 - np.min(saliency_map_image_1)) / (np.max(saliency_map_image_1) - np.min(saliency_map_image_1))
    saliency_map_image_2 = (saliency_map_image_2 - np.min(saliency_map_image_2)) / (np.max(saliency_map_image_2) - np.min(saliency_map_image_2))

    return saliency_map_image_1, saliency_map_image_2

def visualize_saliency_map(siamese_model, input_image_1, input_image_2, layer_name, file_name):
    image_1 = load_and_preprocess_image(input_image_1)
    image_2 = load_and_preprocess_image(input_image_2)

            
    # Make YOLO predictions
    yolo_output_1 = yolo_model.predict(input_image_1, save=False, imgsz=640, conf=0.35, verbose=False)
    yolo_output_2 = yolo_model.predict(input_image_2, save=False, imgsz=640, conf=0.35, verbose=False)

    num1 = []
    num2 = []

    # convert the yolo output to a length tensor
    for idx in range(1):
        class_tensor_1 = yolo_output_1[idx].boxes.cls
        class_tensor_2 = yolo_output_2[idx].boxes.cls
                
        num1.append(class_tensor_frequency(class_tensor_1))
        num2.append(class_tensor_frequency(class_tensor_2))

    num1 = tf.convert_to_tensor(num1)
    num2 = tf.convert_to_tensor(num2)
    
    saliency_map_1, saliency_map_2 = compute_saliency_map(siamese_model, image_1, image_2, num1, num2, layer_name)

    # Overlay the saliency map on the input images with increased weights
    overlaid_image_1 = (0.3 * image_1) + (0.7 * np.expand_dims(saliency_map_1, axis=-1))
    overlaid_image_2 = (0.3 * image_2) + (0.7 * np.expand_dims(saliency_map_2, axis=-1))

    # Plot the original images, saliency map, and overlaid images with increased weights
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 6, 1)
    plt.imshow(image_1)
    plt.title(f'Input Image {os.path.basename(input_image_1)}')

    plt.subplot(1, 6, 2)
    plt.imshow(image_2)
    plt.title(f'Input Image {os.path.basename(input_image_2)}')

    plt.subplot(1, 6, 3)
    plt.imshow(saliency_map_1, cmap='plasma')  # Experiment with different colormaps
    plt.title('Saliency Map Im. 1')

    plt.subplot(1, 6, 4)
    plt.imshow(saliency_map_2, cmap='plasma')  # Experiment with different colormaps
    plt.title('Saliency Map Im. 2')

    plt.subplot(1, 6, 5)
    plt.imshow(overlaid_image_1)
    plt.title(f'Overlaid {os.path.basename(input_image_1)} with Saliency Map')

    plt.subplot(1, 6, 6)
    plt.imshow(overlaid_image_2)
    plt.title(f'Overlaid {os.path.basename(input_image_2)} with Saliency Map')

    plt.savefig('saliency_maps/' + file_name)
    plt.close()
    plt.show()
    
def looped_visualization(siamese_model, filenames):
    # Choose a layer to visualize (you can find layer names using siamese_model.summary())
    layers_to_visualize = ['dense', 'dense_1', 'dense_2']
    root_path = './images'
    for i in range(len(filenames)):
        for j in range(len(filenames)):
            if i != j:
                print(i, j)
                visualize_saliency_map(siamese_model, os.path.join(root_path, filenames[i]), os.path.join(root_path, filenames[j]), layers_to_visualize[0], f'{filenames[i][:-4].replace("/", "_")}_{filenames[j][-7:-4]}')

if __name__ == '__main__':
    # Mode is modified by the argument
    try:
        MODE = int(sys.argv[1])
    except Exception:
        MODE = 0
    print('MODE:', MODE)

    # Read the CSV file
    csv_file_path = './data.json'
    data = pd.read_json(csv_file_path)

    test = data['test']
    test_image_paths_1 = test['img_path_1']
    test_image_paths_2 = test['img_path_2']
    test_labels = test['labels']
    if MODE == 0 or MODE == 1 or MODE == 2:
        # Read the CSV file
        csv_file_path = './data.json'
        data = pd.read_json(csv_file_path)

        # Define data_generator inputs
        train = data['train']
        train_image_paths_1 = train['img_path_1']
        train_image_paths_2 = train['img_path_2']
        train_labels = train['labels']

        # Setting up some of the constants
        training_length = len(test_image_paths_1)
        batch_size = 64
        print("training length: ", len(train_image_paths_1), "testing length: ", len(test_image_paths_2))

        # Compile the model
        input_shape = (299, 299, 3)  # Adjust the input shape based on your images

        
        siamese_model = create_siamese_model(input_shape)
        completed = False
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}  # Create an empty history dictionary
        try:
            history, siamese_model = custom_fit(siamese_model, history, yolo_model, num_epochs = 100, steps_per_epoch=training_length//batch_size, batch_size = batch_size)
            completed = True

            #
        except KeyboardInterrupt:
            
            # If KeyboardInterrupt (Ctrl+C) is detected, save the model which is passed by reference not passed by value
            save_model(siamese_model, 'model_interrupted_MODE{}.h5'.format(MODE))
            print("Training interrupted. Model {} saved.".format(MODE))

            # Plotting the training and validation loss
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plotting the training and validation accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.show()


        if completed:
            # save the model when you are done with training it
            save_model(siamese_model, 'temporal_ordering_model_trained_MODE{}.h5'.format(MODE))

            # Plotting the training and validation loss
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plotting the training and validation accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.show()

    if MODE == 3:
        print('We have sucessfuly entered mode 3.')
        
        # Load your siamese model
        siamese_model = create_siamese_model((299, 299, 3))
        siamese_model.load_weights('temporal_ordering_model_trained_MODE0.h5')  # Change the path accordingly
        siamese_model.summary()
        
        # Choose a layer to visualize (you can find layer names using siamese_model.summary())
        layers_to_visualize = ['dense', 'dense_1', 'dense_2']

        # Visualize activations for a pair of sample images C:\\Users\\Himani\Laproscopic-Surgery-Work\
        sample_image_path_1 = 'C:\\Users\\Himani\Laproscopic-Surgery-Work\\Surgery Images - Temporal Ordering\\images\\02142010_192913\\001.jpg'
        sample_image_path_2 = 'C:\\Users\\Himani\Laproscopic-Surgery-Work\\Surgery Images - Temporal Ordering\\images\\02142010_192913\\020.jpg'
        visualize_activations(siamese_model, layers_to_visualize[0], sample_image_path_1, sample_image_path_2)
        visualize_activations(siamese_model, layers_to_visualize[1], sample_image_path_1, sample_image_path_2)
        visualize_activations(siamese_model, layers_to_visualize[2], sample_image_path_1, sample_image_path_2)

    if MODE == 4:
        print('We have sucessfuly entered mode 4.')
        
        # Load your siamese model
        siamese_model = create_siamese_model((299, 299, 3))
        siamese_model.load_weights('temporal_ordering_model_trained_MODE0.h5')  # Change the path accordingly
        siamese_model.summary()

        # Visualize activations for a pair of sample images C:\\Users\\Himani\Laproscopic-Surgery-Work\
        filenames = ['02142010_192913/001.jpg', '02142010_192913/020.jpg', '02142010_204825/007.jpg', '02142010_204825/020.jpg', '09092011_130836/013.jpg']
        looped_visualization(siamese_model, filenames)
    
    if MODE == 5:
        plt.tight_layout()
        plt.show()
        # Generate a random input sample for visualization
        sample_input = tf.random.normal((1, your_input_shape))

        # Extract the activations of the first hidden layer
        activation_model = tf.keras.Model(inputs=siamese_model.input, outputs=siamese_model.layers[0].output)
        activations = activation_model.predict(sample_input)

        plt.bar(range(activations.shape[1]), activations[0])
        plt.title('Activations of the First Hidden Layer')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')
        plt.show()

    if MODE == 6:
        loaded_model = tf.keras.models.load_model(select_file())
        # Assuming you have a function to preprocess and load your test data in batches
        batch_size = 32
        test_data_generator = data_generator(test_image_paths_1, test_image_paths_2, test_labels, batch_size)

        # Initialize an empty list to store predictions
        y_pred_accumulated = []
        number_of_batches = len(test_labels) // batch_size
        test_labels = test_labels[:batch_size * number_of_batches]

        for _ in range(number_of_batches):
            inputs, labels = next(test_data_generator)
            input_1_images, input_1_paths, input_1_summary = inputs['input_1']
            input_2_images, input_2_paths, input_2_summary = inputs['input_2']

            # Iterate through batches and make predictions
            predictions_batch = loaded_model.predict([input_1_images, input_2_images, input_1_summary, input_2_summary])
            y_pred_accumulated.extend(predictions_batch)

        y_pred_binary = np.array(y_pred_accumulated) > 0.5
        cm = confusion_matrix(test_labels, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Image 1', 'Image 2'], yticklabels=['Image 1', 'Image 2'], annot_kws={"size": 16})
        plt.title('Confusion Matrix', fontsize=36)
        plt.xlabel('Predicted Label', fontsize=20)
        plt.ylabel('True Label', fontsize=20)
        plt.show()
