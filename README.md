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

