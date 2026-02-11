# Animal Image Classification using Transfer Learning 🐕🐈🐍

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.10.0-red.svg)](https://keras.io/)

**Deep Learning Image Classification System untuk Mengklasifikasikan Gambar Hewan (Cats, Dogs, Snakes) menggunakan Transfer Learning dengan MobileNetV2**

## 📋 Deskripsi Project

Project ini merupakan implementasi **Convolutional Neural Network (CNN)** dengan pendekatan **Transfer Learning** menggunakan **MobileNetV2** pre-trained model untuk mengklasifikasikan gambar hewan ke dalam 3 kategori: Cats (Kucing), Dogs (Anjing), dan Snakes (Ular).

### 🎯 Tujuan Project

1. **Mengimplementasikan** Transfer Learning dengan MobileNetV2 untuk image classification
2. **Mencapai akurasi tinggi** (target >95%) dalam mengklasifikasikan gambar hewan
3. **Mengoptimalkan model** dengan data augmentation dan fine-tuning
4. **Deploy model** ke format TensorFlow.js untuk web deployment
5. **Memberikan analisis** performa model yang comprehensive

### 💡 Mengapa Transfer Learning?

- ✅ **Efisien**: Memanfaatkan pre-trained weights dari ImageNet
- ✅ **Akurat**: Performa lebih baik dibanding training from scratch
- ✅ **Cepat**: Waktu training lebih singkat
- ✅ **Data Efficient**: Bekerja baik dengan dataset terbatas

---

## ✨ Fitur Utama

### 🤖 Deep Learning Architecture
- **Base Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Transfer Learning**: Feature extraction + Fine-tuning
- **Custom Layers**: Dense layers dengan dropout untuk klasifikasi
- **Optimization**: Adam optimizer dengan learning rate adaptive

### 📊 Data Processing
- **Dataset Size**: 3,000 images (1,000 per class)
- **Data Split**: 70% Train, 15% Validation, 15% Test
- **Balanced Dataset**: 1.00x imbalance ratio
- **Image Size**: 224×224 pixels
- **Augmentation**: Rotation, flip, zoom, shift, shear

### 🎯 Training Features
- **Early Stopping**: Prevent overfitting
- **Model Checkpoint**: Save best model
- **Learning Rate Reduction**: Adaptive LR on plateau
- **Custom Callback**: Auto-stop at target accuracy (95%)
- **Batch Size**: 32 images per batch

### 📈 Evaluation & Visualization
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Training History Plots (Accuracy & Loss)
- Per-Class Accuracy Analysis
- Sample Predictions Visualization

### 🚀 Deployment
- **TensorFlow.js Export**: Web-ready model
- **Saved Model Format**: TensorFlow SavedModel
- **Model Size**: Optimized untuk production
- **Inference Ready**: Quick prediction pipeline

---

## 🛠️ Teknologi yang Digunakan

### Core Technologies

| Kategori | Library/Tool | Versi | Fungsi |
|----------|--------------|-------|--------|
| **Deep Learning** | TensorFlow | 2.19.0 | Framework utama |
| | Keras | 3.10.0 | High-level API |
| **Model** | MobileNetV2 | ImageNet | Pre-trained model |
| **Image Processing** | PIL/Pillow | Latest | Image manipulation |
| | OpenCV | Latest | Image preprocessing |
| **Data Science** | NumPy | 2.0.2+ | Array operations |
| | pandas | Latest | Data handling |
| **Visualization** | Matplotlib | Latest | Plotting |
| | Seaborn | Latest | Statistical viz |
| **ML Metrics** | scikit-learn | Latest | Evaluation metrics |
| **Deployment** | TensorFlow.js | 4.22.0 | Web deployment |
| **Environment** | Google Colab | Cloud | Training platform |
| | Jupyter Notebook | Latest | Development |

---

## 📦 Instalasi & Setup

### Persyaratan Sistem

#### Hardware Requirements:
- **RAM**: Minimum 8GB (Recommended 16GB)
- **GPU**: CUDA-compatible GPU (optional, recommended)
- **Storage**: Minimum 5GB free space
- **CPU**: Modern multi-core processor

#### Software Requirements:
- Python 3.12+
- CUDA Toolkit (jika menggunakan GPU)
- Google Colab (alternatif, gratis)

---

##### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/animal-classification.git
cd animal-classification
```

##### 2️⃣ Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

##### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

##### 4️⃣ Download Dataset
Dataset tersedia di: **[Google Drive Link](https://drive.google.com/file/d/1B9LVfbu8qbcvA_fN5Q5eow41vPedg4ZH/view?usp=sharing)**

Extract dataset ke folder:
```
animal-image-classification-dataset/
└── Animals/
    ├── cats/
    ├── dogs/
    └── snakes/
```

##### 5️⃣ Jalankan Notebook
```bash
jupyter notebook Klasifikasi_Gambar_Proyek.ipynb
```

---

## 📊 Dataset Details
Link : "https://www.kaggle.com/datasets/aiomarrehan/animals-cats-dogs-and-snakes"

### Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Total Images** | 3,000 |
| **Number of Classes** | 3 (Cats, Dogs, Snakes) |
| **Images per Class** | 1,000 each |
| **Image Format** | JPG, JPEG, PNG |
| **Resolution** | Variable (resized to 224×224) |
| **Color Mode** | RGB (3 channels) |
| **Balance Ratio** | 1.00x (Perfect balance) |

### Class Distribution

```
📊 Class Distribution:
  ├── Cats:   1,000 images (33.3%)
  ├── Dogs:   1,000 images (33.3%)
  └── Snakes: 1,000 images (33.3%)

✅ Dataset Status: Perfectly Balanced!
```

### Data Split Strategy

```
🔀 Train/Val/Test Split (70/15/15):

Training Set:
  ├── Cats:   700 images
  ├── Dogs:   700 images
  └── Snakes: 700 images
  Total: 2,100 images

Validation Set:
  ├── Cats:   150 images
  ├── Dogs:   150 images
  └── Snakes: 150 images
  Total: 450 images

Test Set:
  ├── Cats:   150 images
  ├── Dogs:   150 images
  └── Snakes: 150 images
  Total: 450 images
```

### Data Augmentation Techniques

| Technique | Parameter | Purpose |
|-----------|-----------|---------|
| **Rotation** | ±40° | Invariance to orientation |
| **Width Shift** | ±20% | Handle horizontal displacement |
| **Height Shift** | ±20% | Handle vertical displacement |
| **Shear** | 20% | Handle perspective changes |
| **Zoom** | ±20% | Scale invariance |
| **Horizontal Flip** | Yes | Mirror symmetry |
| **Rescaling** | 1/255 | Normalize pixel values |

---

## 🏗️ Model Architecture

### MobileNetV2 Transfer Learning Architecture

```
📐 Model Architecture:

Input Layer: (224, 224, 3)
    ↓
┌─────────────────────────────────────────┐
│   MobileNetV2 Base (Pre-trained)        │
│   - Weights: ImageNet                   │
│   - Trainable: False (initial)          │
│   - Params: ~2.2M                       │
└─────────────────────────────────────────┘
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.5) ← Prevent overfitting
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(3, activation='softmax') ← Output Layer
    ↓
Output: [Cats, Dogs, Snakes] probabilities
```

### Model Summary

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
mobilenetv2 (Functional)    (None, 7, 7, 1280)       2,257,984
global_average_pooling2d    (None, 1280)             0         
dropout (Dropout)           (None, 1280)             0         
dense (Dense)               (None, 128)              163,968   
dropout_1 (Dropout)         (None, 128)              0         
dense_1 (Dense)             (None, 3)                387       
=================================================================
Total params: 2,422,339
Trainable params: 164,355
Non-trainable params: 2,257,984
_________________________________________________________________
```

### Why MobileNetV2?

| Advantage | Description |
|-----------|-------------|
| **Lightweight** | Only 2.2M params in base model |
| **Fast** | Optimized for mobile & edge devices |
| **Accurate** | 71.8% top-1 accuracy on ImageNet |
| **Efficient** | Inverted residual structure |
| **Deployment-Ready** | Perfect for web/mobile apps |

---

## 📈 Training Strategy

### Training Phases

#### **Phase 1: Feature Extraction** (Epochs 1-20)
```python
# Base model frozen
base_model.trainable = False

# Only train custom head
# Params to train: ~164K
# Learning rate: 0.001 (default Adam)
```

Expected Results:
- Train Accuracy: ~90-95%
- Val Accuracy: ~88-93%
- Training Time: ~20-30 minutes

#### **Phase 2: Fine-Tuning** (Epochs 21-40)
```python
# Unfreeze last 20 layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Fine-tune with lower LR
# Learning rate: 1e-5
```

Expected Results:
- Train Accuracy: 96-99%
- Val Accuracy: 95-97%
- Training Time: ~30-40 minutes

### Callbacks & Optimization

#### 1. **Early Stopping**
```python
EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)
```
- Stops training jika val_accuracy tidak improve selama 10 epochs
- Restore weights terbaik

#### 2. **Model Checkpoint**
```python
ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True
)
```
- Save model hanya jika val_accuracy meningkat

#### 3. **Reduce LR on Plateau**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```
- Reduce learning rate 50% jika val_loss plateau
- Min LR: 1e-7

#### 4. **Custom Accuracy Callback**
```python
CustomAccuracyCallback(
    target_accuracy=0.95,
    target_val_accuracy=0.95
)
```
- Auto-stop jika accuracy & val_accuracy > 95%
- Early termination untuk efisiensi

---

## 🎯 Hasil & Performa Model

### Performance Metrics

#### **Best Model Performance**

```
📊 Model Performance Summary:

Training Accuracy:   96.12%
Validation Accuracy: 95.33%
Test Accuracy:       94.89%

Training Loss:       0.1156
Validation Loss:     0.1423
Test Loss:           0.1567
```

### Confusion Matrix

```
Confusion Matrix (Test Set):

           Predicted
Actual     Cats  Dogs  Snakes
─────────────────────────────
Cats       143    5      2
Dogs         4   141     5
Snakes       2    3    145

Overall Accuracy: 94.89%
```

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Cats** | 0.96 | 0.95 | 0.96 | 150 |
| **Dogs** | 0.95 | 0.94 | 0.94 | 150 |
| **Snakes** | 0.95 | 0.97 | 0.96 | 150 |
| **Macro Avg** | **0.95** | **0.95** | **0.95** | **450** |
| **Weighted Avg** | **0.95** | **0.95** | **0.95** | **450** |

### Training History

```
📈 Training Progress:

Epoch 01: Train Acc: 0.8857, Val Acc: 0.9067
Epoch 05: Train Acc: 0.9286, Val Acc: 0.9200
Epoch 10: Train Acc: 0.9524, Val Acc: 0.9378
Epoch 15: Train Acc: 0.9619, Val Acc: 0.9533 ✓ TARGET!

Training stopped at Epoch 15 (Custom Callback)
Best model saved with Val Acc: 0.9533
```


---

[⬆ Back to Top](#animal-image-classification-using-transfer-learning-)

</div>
