import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
import cv2
import matplotlib.pyplot as plt

tf.keras.mixed_precision.set_global_policy("float32")


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_logical_device_configuration(
        physical_devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )

dataset_dir = r'C:\Users\Zohaib\Downloads\deep learning project\fgvc-aircraft-2013b\data'
images_dir = os.path.join(dataset_dir, 'images')
variant_file = os.path.join(dataset_dir, 'images_variant_train.txt')
val_file = os.path.join(dataset_dir, 'images_variant_val.txt')
test_file = os.path.join(dataset_dir, 'images_variant_test.txt')
class_file = os.path.join(dataset_dir, 'variants.txt')

def load_class_names(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def load_annotations(file_path, class_names):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            image_name, label = line.strip().split(' ', 1)
            if label not in class_names:
                raise ValueError(f"Label '{label}' not found in class names.")
            label_index = class_names.index(label)
            image_path = os.path.join(images_dir, f"{image_name}.jpg")
            if os.path.exists(image_path):
                annotations.append((image_path, label_index))
    return annotations

class_names = load_class_names(class_file)

num_classes = len(class_names)

train_annotations = load_annotations(variant_file, class_names)
val_annotations = load_annotations(val_file, class_names)
test_annotations = load_annotations(test_file, class_names)

AUTO = tf.data.AUTOTUNE
IMG_SIZE = (299, 299)

def preprocess(file_path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, tf.one_hot(label, num_classes)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

def prepare_dataset(annotations, is_training=True):
    paths, labels = zip(*annotations)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    dataset = dataset.map(preprocess, num_parallel_calls=AUTO)
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(2048).repeat()
    return dataset.batch(batch_size).prefetch(AUTO)

batch_size = 8
train_dataset = prepare_dataset(train_annotations, is_training=True)
val_dataset = prepare_dataset(val_annotations, is_training=False)
test_dataset = prepare_dataset(test_annotations, is_training=False)

# Custom Mish Activation
def mish(x):
    return x * K.tanh(K.softplus(x))

base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
for layer in base_model.layers[-50:]:
    layer.trainable = True

model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation=mish),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation=mish),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax', dtype='float32')  # Keep final layer as float32
])

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=10000, alpha=0.0001
)
optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

def train_model(model, train_dataset, val_dataset, epochs):
    steps_per_epoch = len(train_annotations) // batch_size
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001),
        ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

epochs = 50
print("Training...")
model, history = train_model(model, train_dataset, val_dataset, epochs)
model.save("aircraft_model_finetuned.h5")

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training_history(history)

predictions = model.predict(test_dataset)
true_labels = [label for _, label in test_annotations]
predicted_labels = np.argmax(predictions, axis=1)
print(classification_report(true_labels, predicted_labels, target_names=class_names))

def process_frame(frame, model):
    resized = cv2.resize(frame, IMG_SIZE) / 255.0
    prediction = model.predict(np.expand_dims(resized, axis=0))
    label_idx = np.argmax(prediction, axis=1)[0]
    return class_names[label_idx]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access the camera.")

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    label = process_frame(frame, model)
    cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Aircraft Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
