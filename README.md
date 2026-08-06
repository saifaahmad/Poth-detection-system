\# Pothole Detection Using Deep Learning



A binary image classification system that detects whether a road image contains a pothole, using transfer learning on a pretrained CNN.



\## Features

\- \*\*Binary image classification\*\* (Pothole / Normal Road) using transfer learning on MobileNetV2 (pretrained on ImageNet)

\- \*\*Custom classification head\*\*: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)

\- \*\*99.03% test accuracy\*\* on a held-out test set

\- \*\*Interactive web app\*\* built with Streamlit for live image upload and prediction with confidence scoring



\## Tech Stack

\- TensorFlow / Keras

\- MobileNetV2 (Transfer Learning)

\- Streamlit (deployment/UI)

\- NumPy, Pillow



\## Dataset

\- Source: Kaggle

\- Classes: Pothole, Normal Road

\- Training: 476 images (230 pothole, 246 normal)

\- Validation: 102 images (49 pothole, 53 normal)

\- Testing: 103 images (50 pothole, 53 normal)



\## Model Details

\- \*\*Base model:\*\* MobileNetV2 (ImageNet weights, frozen convolutional layers)

\- \*\*Input size:\*\* 224×224

\- \*\*Batch size:\*\* 32

\- \*\*Epochs:\*\* 3

\- \*\*Optimizer:\*\* Adam

\- \*\*Loss function:\*\* Binary Cross-Entropy



\## Results

| Metric | Score |

|---|---|

| Training Accuracy | 96.67% |

| Validation Accuracy | 96.08% |

| Test Accuracy | 99.03% |



\## How to Run

```bash

pip install streamlit tensorflow numpy pillow

streamlit run app.py

```

\## How It Works

1\. User uploads a road image

2\. Image is resized to 224×224 and normalized

3\. MobileNetV2 (with custom classification head) predicts a probability score

4\. If score ≥ 0.5 → "Pothole Detected", else → "Normal Road"



\## Why Transfer Learning?

Training a CNN from scratch requires large datasets and heavy compute. MobileNetV2 already learns generic image features from ImageNet, enabling fast training and high accuracy even with a relatively small dataset (476 images).



\## Future Scope

\- Object detection (YOLO) instead of classification, for pothole localization

\- Real-time video processing

\- GPS-based pothole mapping

\- Mobile application integration

