# 💳 Credit Card Fraud Detection System

A robust and interpretable machine learning system for detecting fraudulent credit card transactions in real-time. Built with XGBoost and deployed via Streamlit, this project addresses the challenges of extreme class imbalance and evolving fraud patterns in the financial industry.

## 🚀 Live Demo

🔗 [Click here to try the app](https://credit-card-fraud-detection-siglp9lemjg36vh7rfq6d5.streamlit.app/)  
*(Note: hosted on Streamlit Community Cloud)*

---

## 🧠 Key Features

- **Advanced Preprocessing:** Time-based feature engineering, one-hot encoding, and scaling.
- **Class Imbalance Handling:** SMOTE applied to the training set to balance classes.
- **Powerful Model:** XGBoost classifier outperforms baseline Logistic Regression.
- **Model Explainability:** SHAP values used for interpreting individual predictions.
- **Interactive Interface:** Built with Streamlit for real-time fraud prediction.

---

## 📊 Model Comparison & Results

| Metric                | Logistic Regression | XGBoost Model      | Improvement                                 |
|-----------------------|---------------------|--------------------|---------------------------------------------|
| **Recall (Fraud)**    | 0.90 (90%)          | **0.86 (86%)**     | Slight decrease (missed 4 more frauds)      |
| **Precision (Fraud)** | 0.10 (10%)          | **0.71 (71%)**     | **Massive 7.1x increase!**                  |
| **False Positives**   | 793                 | **34**             | **Dramatic Reduction (~23x fewer FPs!)**    |
| **False Negatives**   | 10                  | 14                 | Slight increase (missed 4 more frauds)      |
| **ROC AUC**           | 0.91                | **0.98**           | Significant improvement                     |
| **Avg Precision**     | 0.7175 (~0.7)       | **0.87**           | **Huge improvement!**                       |

✅ The XGBoost model achieves **a strong balance** between recall and precision, significantly reducing false positives while maintaining high fraud detection.

---

## 🛠️ Methodology Overview

- **Dataset:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Feature Engineering:**
  - Time converted to `Hour_of_Day`
  - One-Hot Encoding of categorical features
  - StandardScaler applied to `Amount`
- **Class Imbalance:** SMOTE applied only to training data.
- **Models:**
  - Baseline: Logistic Regression
  - Final: XGBoost Classifier
- **Evaluation Metrics:** Precision, Recall, F1-Score, ROC AUC, PR AUC
- **Threshold Tuning:** Optimized for Precision-Recall balance.

---

## 🖥️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_PROJECT_REPO.git
cd YOUR_PROJECT_REPO

# 2. Create and activate virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
# Place creditcard.csv in the root directory

# 5. Train model and generate assets
# Run the notebook to train and save model files, scalers, SHAP explainer, and plots

# 6. Run the Streamlit app
streamlit run app.py
