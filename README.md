# 🧠 Brain Tumor Detection & Classification (CNN)

### Advanced Medical Imaging Pipeline with EfficientNetV2B3 & Grad-CAM

![Status](https://img.shields.io/badge/Status-Complete-green) ![Tech](https://img.shields.io/badge/Tech-TensorFlow%20%2F%20Keras-orange) ![Accuracy](https://img.shields.io/badge/Accuracy-High-blue)

This project develops a state-of-the-art artificial intelligence system for the detection and classification of brain tumors from Magnetic Resonance Imaging (MRI) scans. By leveraging convolutional neural network (CNN) architectures and advanced preprocessing techniques, the system is capable of distinguishing between four categories with high accuracy: **Glioma, Meningioma, Adenoma de Pituitaria y "No-Tumor"**.

The model achieves an accuracy of **95.5%** on the validation set, demonstrating its capability to correctly classify MRI images.

---

## 🚀 Key Features

### 1. Robust Data Preprocessing

Unlike many other models, this project includes a rigorous data-cleaning pipeline to prevent _overfitting_ and _data leakage_:

- **Deduplication by pHash:** Elimination of duplicate or nearly identical images using Perceptual Hashing.
- **Outlier Removal:** Automatic filtering of corrupted or low-quality images based on size and structure percentiles.
- **Background Suppression (ROI):** Implementation of Otsu's thresholding algorithms and morphological operations to isolate the brain and remove background noise (skull, black space), focusing the network's attention on relevant tissue.

### 2. State-of-the-Art Architecture

- **Base:** `EfficientNetV2B3` (Pre-trained on ImageNet).
- **Data Augmentation:** Rotations, zooms and translations in real-time to improve generalization.
- **Intelligent Callbacks:** Implementation of `EarlyStopping`, `ReduceLROnPlateau` and `ModelCheckpoint` for optimal training.

### 3. Medical Explainability (Grad-CAM)

The model is not a "black box". We use **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate heatmaps that show exactly where the AI is focusing on the MRI to make the diagnosis, facilitating validation by medical experts.

---

## 📊 Results and Evaluation

The system includes comprehensive validation tools:

- **Confusion Matrices** to identify false positives/negatives.
- **Learning Curves** (Precision/Loss).
- **Classification Reports** (F1-Score, Recall, Precision).
- **Threshold Logic:** Classification as "Inconclusive" when confidence is below a safety threshold, prioritizing diagnostic accuracy.

---

## 🛠️ Technologies Used

- **Language:** Python 3.10+
- **DL Framework:** TensorFlow / Keras
- **Computer Vision:** OpenCV (cv2)
- **Data Science:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn, PIL

---

## 📂 Repository Structure

- `data/`: Original images classified.
- `data_split/`: Dataset processed after cleaning and division (Train/Val/Test).
- `CNN_tumor_gradCAM.ipynb`: Notebook principal with the complete training and Grad-CAM visualization flow.
- `SIC_AI_Capstone Project_FinalReport_englishversion.docx`: Detailed academic documentation of the project.

---

## 👥 Authors

- **Alejandro Gallego Domínguez**
- **Alejandro Lobillo Becerra**
- **Almudena Neva Alejo**

_Developed at the **Samsung Innovation Campus** (AI Course)._
