import os
import numpy as np
import tensorflow as tf
import cv2

# Paths to necessary files
model_path = r'C:best_model.h5'
class_file = r'C:fgvc-aircraft-2013b/data/variants.txt'


# Load class names from variants.txt
def load_class_names(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


# Preprocessing function for input image
IMG_SIZE = (299, 299)


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found.")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, IMG_SIZE)  # Resize to model's input size
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension


# Load the trained model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'mish': mish})
    return model


# Custom Mish activation (if used in the model)
def mish(x):
    return x * tf.keras.backend.tanh(tf.keras.backend.softplus(x))


# Prediction function
def predict_aircraft(image_path, model, class_names):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index]
    return predicted_label, confidence


# Main prediction script
if __name__ == "__main__":
    # Load model and class names
    class_names = load_class_names(class_file)
    model = load_model(model_path)

    # Input image path
    image_path = input("Enter the path to the aircraft image: ").strip()

    try:
        # Predict aircraft type
        label, confidence = predict_aircraft(image_path, model, class_names)
        print(f"Predicted Aircraft Variant: {label}")
        print(f"Confidence: {confidence:.2f}")

        # Display the image with the prediction
        image = cv2.imread(image_path)
        cv2.putText(image, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Prediction', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
