# 🧠 EyeDiseasePred — Retinal Disease Classification System

An AI-powered clinical decision-support system for automated detection of retinal diseases using deep learning, ensemble learning, and an interactive UI with LLM-based medical guidance.

---

## 🚀 Project Overview

Retinal diseases such as **Glaucoma, Cataract, and Diabetic Retinopathy** are leading causes of vision loss worldwide. Early diagnosis is critical but traditional methods rely on manual inspection, which can be slow, subjective, and resource-intensive.

**EyeDiseasePred** provides an end-to-end AI pipeline that:

- Classifies retinal fundus images into 4 categories:
  - 🟢 Normal  
  - 🔴 Glaucoma  
  - 🟡 Cataract  
  - 🟠 Diabetic Retinopathy  
- Uses multiple deep learning CNN models
- Combines predictions via F1-weighted ensemble voting
- Includes a clinical UI with LLM-powered medical advisory

---

## 🧠 Models Used

Transfer learning was applied using three pretrained CNN backbones:

- **VGG16**
- **ResNet50**
- **EfficientNetB0**

Each model was trained with ImageNet weights and a custom classification head.

Saved model checkpoints:

model_vgg.h5
model_resnet.h5
model_effnet.h5


---

## ⚙️ Methodology Pipeline

### 1️⃣ Data Processing

- Dataset organized into class folders
- ImageDataGenerator used for:
  - Rescaling
  - Data augmentation (rotation, zoom, brightness, flipping)
- Train–validation split
- Separate held-out test set

---

### 2️⃣ Model Training Summary

| Model | Epochs | Final Training Accuracy | Best Validation Accuracy |
|------|--------|--------------------------|----------------------------|
| VGG16 | 20 | 89.6% | 81.0% |
| ResNet50 | 10 | 89.1% | 73.3% |
| EfficientNetB0 | 20 | **94.9%** | 79.5% |

Key insights:

- EfficientNet achieved highest training accuracy
- VGG16 showed most stable validation performance
- ResNet showed early validation instability

---

## 🧩 Ensemble Detection Approach

Each test image is processed as follows:

1. Each model outputs softmax probabilities for all 4 classes.
2. Per-class F1 scores are used as ensemble weights.
3. Weighted voting aggregates predictions.

### Voting Formula

vote[class] = Σ (model_probability[class] × model_F1_weight[class])
final_prediction = argmax(vote)


Two ensemble modes implemented:

- Flat weighted voting
- Dynamic class-specific normalized voting

---

## 📊 Per-Class F1 Scores (Ensemble Weights)

| Class | VGG16 | ResNet50 | EfficientNet |
|------|--------|------------|----------------|
| Cataract | 0.73 | 0.75 | 0.78 |
| Diabetic Retinopathy | 0.98 | 0.99 | 0.82 |
| Glaucoma | 0.85 | 0.73 | 0.77 |
| Normal | 0.86 | 0.75 | 0.82 |

---

## 📈 Example Confusion Matrix (Ensemble)

[[48, 1, 12, 3],
[ 0, 64, 0, 0],
[ 2, 1, 57, 4],
[ 0, 0, 1, 65]]


Key observations:

- Strong separability for Diabetic Retinopathy
- Mild confusion between Cataract and Glaucoma
- Normal class highly accurate

---

## 🖥️ Clinical Decision-Support UI

The system includes a user-friendly GUI designed for clinical use.

### 🧾 Input Workflow

The user provides:

- Two retinal images:
  - `left_eye.jpg`
  - `right_eye.jpg`
- A JSON file containing patient metadata:
  - Patient ID
  - Age
  - Symptoms
  - Medical history

---

### 🔍 Prediction Workflow

The UI:

1. Loads patient directory
2. Runs ensemble inference on both eye images
3. Displays:
   - Per-eye predictions
   - Confidence scores
   - Final combined diagnosis

---

### 🤖 LLM-Powered Medical Advisory

The interface includes an interactive questionnaire module.

Based on patient responses, an LLM generates:

- Possible diagnosis explanation
- Medication guidance steps
- Lifestyle recommendations
- Preventive care suggestions

This transforms the system from a pure classifier into a **clinical decision-support assistant**.

---

## 📊 Outputs Generated

The pipeline produces:

- Training accuracy and loss curves
- Confusion matrices
- Classification reports
- Per-image probability breakdowns
- Ensemble vote visualizations

All plots and detailed outputs are available in:

4_class_colab.ipynb


---

## ⚠️ Observations & Limitations

- EfficientNet shows mild overfitting tendencies
- ResNet requires further regularization tuning
- Ensemble depends on reliable F1 weight estimation
- Class imbalance affects Cataract vs Glaucoma separation

---

## 🔮 Future Work

- Larger cross-validated datasets
- Explainable AI integration (Grad-CAM / SHAP)
- Probability calibration
- Meta-classifier fusion models
- Telemedicine deployment

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Matplotlib
- Tkinter (GUI)
- LLM API integration

---

## 👨‍💻 Authors

**Piyush Kumar Patro**  
TARP Project Team

---

## 📜 License

For academic and research purposes only.