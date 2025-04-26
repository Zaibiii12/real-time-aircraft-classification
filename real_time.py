import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# Define Mish activation function
def mish(x):
    return x * tf.tanh(tf.nn.softplus(x))

# Function to load class names from variants.txt file
def load_class_names(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


# Define dataset directory and the class file path
dataset_dir = r'C:\Users\Zohaib\Downloads\deep learning project\fgvc-aircraft-2013b\data'  # Corrected path
class_file = os.path.join(dataset_dir, 'variants.txt')  # Correctly join the path to variants.txt

# Load class names from the variants.txt file
class_names = load_class_names(class_file)

# Load your trained model (adjust path if needed) with custom_objects to handle the Mish activation function
model = tf.keras.models.load_model("best_model.h5"
                                , custom_objects={'mish': mish})

# Image size used during training
IMG_SIZE = (299, 299)


# Real-time detection function
def process_frame(frame, model):
    # Resize the frame to match the model's input size
    resized = cv2.resize(frame, IMG_SIZE)

    # Normalize the image to the range [0, 1]
    resized = resized / 255.0

    # Add an extra dimension to match the model's input (batch size)
    resized = np.expand_dims(resized, axis=0)

    # Make prediction
    prediction = model.predict(resized)

    # Get the index of the predicted class
    label_idx = np.argmax(prediction, axis=1)[0]

    # Get the class label
    label = class_names[label_idx]

    return label


# Open the webcam for real-time detection
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot access the camera.")

print("Press 'q' to quit the detection.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Process the frame and get the prediction
    label = process_frame(frame, model)

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Aircraft Detection', frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
