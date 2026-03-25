# Pediatric Appendicitis Classification via B-Mode Ultrasonography
A PyTorch-based Convolutional Neural Network (ResNet50) engineered for binary classification of pediatric appendicitis from raw, 2D abdominal ultrasound frames.

## 📚 Project Overview
Diagnosing pediatric appendicitis from ultrasonography is highly dependent on measuring the physical diameter and compressibility of the organ. Standard computer vision pipelines frequently fail in this domain due to image distortion and UI artifacts.

This project implements a custom transfer-learning pipeline utilizing a pretrained ResNet50 architecture. The training loop and data transformation pipelines were explicitly engineered to preserve physical tissue scale and prevent artificial feature memorization, achieving an AUC of 0.836 on a small, highly specialized clinical dataset.

## ⚙️ Key Architectural Solutions
1. Spatial Integrity & Aspect-Ratio Preservation
A critical failure point in medical imaging models is the blind squashing of images to fit network input requirements (e.g., standard Resize((224, 224))). Squashing rectangular ultrasounds distorts the aspect ratio, rendering an inflamed 8mm appendix mathematically identical in size to a healthy 4mm appendix.

Solution: This pipeline implements a shortest-edge scaling approach (transforms.Resize(256)) followed by a random/center crop to exactly 224x224. This mathematically preserves the physical scale of the tissue, allowing the CNN to evaluate true organ diameter as a primary feature.

2. Clinical Artifact Removal (The "Clever Hans" Problem)
Raw medical imaging often contains sonographer UI overlays and measurement calipers. Neural networks will frequently bypass anatomical learning and memorize these artificial marks to "cheat" the diagnosis.

Solution: A custom Lambda tensor slice (transforms.Lambda(lambda x: x[:, :-50, :])) was implemented across both training and validation pipelines to systematically shear off UI elements and measurement data, forcing the model to evaluate only the biological tissue.

3. Model Interpretability (Grad-CAM)
Validation loss alone is insufficient to prove clinical reasoning. The pipeline utilizes Gradient-weighted Class Activation Mapping (Grad-CAM) to verify the spatial coordinates of the model's activations.

Result: Visual telemetry confirms the model accurately isolates the "target sign" (the cross-sectional structure of the appendix/bowel) and normal fascial planes, rather than relying on random background noise or image borders.

## 📊 Performance Metrics
Model: ResNet50 (Pretrained, Custom Classification Head)

Loss Function: BCEWithLogitsLoss

Optimizer: AdamW with StepLR Scheduling

Peak AUC: 0.836

Threshold Optimization: The decision boundary was mathematically recalibrated using Youden's J Statistic to optimize the False Negative / False Positive trade-off, resulting in an optimal probability threshold of 0.504.

## 🧠 Dataset
This project utilizes the Regensburg Pediatric Appendicitis Dataset, acquired in a retrospective study from a cohort of pediatric patients admitted with abdominal pain to Children’s Hospital St. Hedwig in Regensburg, Germany.

The study was approved by the Ethics Committee of the University of Regensburg (no. 18-1063-101, 18-1063_1-101, and 18-1063_2-101) and was performed following applicable guidelines and regulations. (Zenodo)

## 💻 Installation & Usage
Clone the repository and install dependencies:

Bash
git clone https://github.com/YourUsername/ped-appendicitis-resnet.git
cd ped-appendicitis-resnet
pip install -r requirements.txt
Dependencies:

Python 3.10+

PyTorch 2.x

torchvision

scikit-learn

PIL / Pillow

matplotlib & seaborn (for Grad-CAM and telemetry visualization)
