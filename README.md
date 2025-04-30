# 🧬 Breast Cancer Histopathological Image Classification

This project focuses on automating the classification of breast cancer histopathological images from the BreakHis 400x dataset using a hybrid deep learning approach that integrates CNNs, Vision Transformers, and texture-based features.

## 📌 Overview

Breast cancer remains one of the leading causes of death among women. Manual classification of histopathological slides is time-consuming and inconsistent. This project aims to develop a robust automated pipeline to classify breast cancer images into **benign** and **malignant** categories, improving clinical decision support.

## 🚀 Models Used

- **EfficientNet-B0** – Compact and efficient CNN for image classification.
- **ResNet18** – Deep residual CNN with skip connections.
- **Local Binary Pattern (LBP)** – Captures texture-based differences.
- **Vision Transformer (ViT)** – Captures global spatial dependencies.
- **HybridNet** – Custom architecture combining CNN, ViT, and LBP features.

## 🗂️ Dataset

**BreakHis 400x**  
- 1,693 total images (547 benign, 1,146 malignant)  
- RGB images at 400x magnification  
- Resolution: 700×460 pixels  
- Train/Test split: 70/30

| Split | Benign | Malignant | Total |
|-------|--------|-----------|-------|
| Train | 371    | 777       | 1148  |
| Test  | 176    | 369       | 545   |
| **Total** | **547** | **1146** | **1693** |

## 📊 Methodology

1. **EDA** – Class distribution and RGB channel analysis
2. **Feature Extraction** – EfficientNet/ResNet + LBP + ViT
3. **Fusion** – Concatenated features into a unified model
4. **Training** – Using CrossEntropyLoss with class weights and Adam optimizer
5. **Evaluation** – Performance metrics and Grad-CAM visualizations

## 📈 Results

| Model          | Accuracy | F1-score | AUC    | Precision | Recall |
|----------------|----------|----------|--------|-----------|--------|
| ResNet18       | 94.50% ± 0.50% | 96.00% ± 0.39% | 0.9849 | 0.97 | 0.97 |
| EfficientNet-B0| 95.80% ± 0.46% | 95.80% ± 0.46% | 0.9907 | 0.98 | 0.98 |

> ✅ **EfficientNet-B0** slightly outperformed ResNet18 with fewer parameters.

## 🧠 Interpretability

- **Grad-CAM** visualizations were used to interpret decision-making of CNN layers.
- Both models focus activation on important tissue regions in correctly classified cases.
- Misclassifications often occurred when feature signals were diffused or at the edges.

## 📍 Key Insights

- Texture and color patterns in histopathological images are strong indicators of malignancy.
- Combining CNN with ViT and LBP features improves robustness.
- Techniques from lung cancer classification generalized well to breast cancer data.

## ⚠️ Limitations

- No k-fold cross-validation due to compute limitations.
- Fixed train-test split may affect generalization performance.

## 🔗 Repository

👉 [GitHub Repository](https://github.com/bishram-acharya/hybrid_classifier)

---

© 2025 Bishram Acharya
