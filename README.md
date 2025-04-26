# 🛩️ Real-Time Aircraft Detection using FGVC Data (Deep Learning)

This project presents a fine-tuned **DenseNet201** model for **real-time aircraft detection**, trained on the **FGVC Aircraft dataset** using deep learning techniques.

---

## 📚 Dataset
- **Name:** FGVC Aircraft
- **Images:** 10,000+
- **Source:** [FGVC Aircraft Dataset on Kaggle](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)

> Note: The original FGVC Aircraft dataset is available [here](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).

---

## 🏗️ Model Architecture
- **Base Model:** DenseNet201 (Pretrained on ImageNet)
- **Custom Layers:**
  - GlobalAveragePooling2D
  - Dense(1024) + Mish Activation + Batch Normalization + Dropout
  - Dense(512) + Mish Activation + Batch Normalization + Dropout
  - Final Dense layer with Softmax Activation

- **Optimizer:** Adam
- **Learning Rate:** CosineDecay scheduler
- **Batch Size:** 8
- **Epochs:** 50

---

## 📈 Performance
- Achieved **high validation accuracy**.
- Used EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint for optimal performance during training.

---

## 🛠️ Model Download

You can download the trained model here:

➡️ [Download Aircraft Classification Model (DenseNet201) from Hugging Face](https://huggingface.co/zaibiii/aircraft-classification-densenet201/resolve/main/aircraft_model_finetuned.h5)

➡️ [Hugging Face Model Page](https://huggingface.co/zaibiii/aircraft-classification-densenet201)

---

## 🚀 How to Use

```python
import tensorflow as tf

# Define Mish activation
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'mish'

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

get_custom_objects().update({'mish': Mish(mish)})

# Load the model
model = tf.keras.models.load_model('path/to/aircraft_model_finetuned.h5', custom_objects={"mish": mish})

# Predict
prediction = model.predict(preprocessed_image)
