# 🫁 Detection and Diagnosis of Latent Tuberculosis (TB) in Patients using Machine Learning and AI
### utilizing Deep Learning & CNN Architecture

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Live_Research-success?style=for-the-badge)

**Author:** Jitendra Prajapat  
**Institution:** Suresh Gyan Vihar University  
**Department:** Computer Science & Engineering  

---

## 🔬 Project Overview
Tuberculosis (TB) remains a leading cause of death worldwide. **Latent TB** is particularly dangerous as it can remain asymptomatic before becoming active. This research project presents an **AI-powered diagnostic tool** capable of detecting Tuberculosis from standard Chest X-Rays (CXRs) with high precision.

Using a custom **Convolutional Neural Network (CNN)**, the model was trained on a dataset of **4,200 chest X-ray images** (3,500 Normal, 700 Tuberculosis). To address the significant class imbalance, the training pipeline employed **Weighted Loss Functions** to ensure the model prioritized TB detection.

### 📊 Key Performance Metrics
| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Training Accuracy** | **99.70%** | After 10 Epochs |
| **Validation Accuracy** | **98.69%** | Tested on unseen data |
| **Model Architecture** | Custom CNN | 3 Convolutional Blocks + 2 Dense Layers |

---

## 🚀 Features
* **Instant Diagnosis:** Upload an X-ray (JPG/PNG) and get a result in <2 seconds.
* **Confidence Scoring:** The AI provides a percentage probability (e.g., "99.4% Confidence").
* **Medical-Grade Interface:** Dark Mode UI designed for low-light clinical environments.
* **Privacy-First:** All processing happens in the cloud; no patient data is permanently stored.

---

## 🛠️ Technology Stack
* **Core Engine:** Python 3.11
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** Pillow (PIL)
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## 💻 Installation (Local)
To run this project on your own machine:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Jitsa-Latent-TB-Detector.git](https://github.com/YOUR_USERNAME/Jitsa-Latent-TB-Detector.git)
    cd Jitsa-Latent-TB-Detector
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## ⚠️ Disclaimer
This tool is developed for **educational and research purposes only**. It is intended to assist medical professionals, not replace them. Always confirm AI diagnoses with clinical tests (e.g., Sputum Smear or GeneXpert).

---

**© 2026 Jitendra Prajapat**
*Developed at Suresh Gyan Vihar University*
