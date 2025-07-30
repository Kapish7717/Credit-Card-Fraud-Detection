import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os

# --- 0. Page Configuration ---
st.set_page_config(
    page_title="Credit Card Fraud Detector 💳", 
    layout="centered", 
    initial_sidebar_state="auto", 
   
)

# --- Define Paths to Saved Files ---
MODEL_PATH = os.path.join('trained_models', 'xgboost_fraud_detection_model.json')
SCALER_PATH = os.path.join('trained_models', 'scaler_amount.joblib')
TRAINED_COLUMNS_PATH = os.path.join('trained_models', 'trained_feature_columns.npy')

# --- Load Model, Scaler, and Trained Column Names (Cached for Efficiency) ---
@st.cache_resource
def load_ml_resources():
    try:
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(MODEL_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        loaded_trained_columns = np.load(TRAINED_COLUMNS_PATH, allow_pickle=True).tolist()
        return loaded_model, loaded_scaler, loaded_trained_columns
    except Exception as e:
        st.error(f"Error loading ML resources: {e}. Please ensure all model files are in 'trained_models' directory.")
        st.stop()

model, scaler, trained_columns = load_ml_resources()

# --- Preprocessing Function for User Input ---
def preprocess_user_input(input_df: pd.DataFrame, scaler_obj, trained_columns_list: list) -> pd.DataFrame:
    # 1. Create Hour_of_Day feature
    input_df['Hour_of_Day'] = (input_df['Time'] / 3600).astype(int) % 24

    # 2. Apply One-Hot Encoding for Hour_of_Day
    all_possible_hour_cols = [f'Hour_{i}' for i in range(24)]
    input_df_encoded_hours = pd.get_dummies(input_df['Hour_of_Day'], prefix='Hour', dtype=int)
    
    processed_df_temp = input_df.copy()
    processed_df_temp = processed_df_temp.drop(['Hour_of_Day'], axis=1)

    for hour_col in all_possible_hour_cols:
        processed_df_temp[hour_col] = 0
    for col in input_df_encoded_hours.columns:
        if col in processed_df_temp.columns:
            processed_df_temp[col] = input_df_encoded_hours[col]
        
    # 3. Apply Scaling to 'Amount'
    numerical_cols_to_scale = ['Amount'] 
    processed_df_temp.loc[:, numerical_cols_to_scale] = scaler_obj.transform(processed_df_temp[numerical_cols_to_scale])
    
    # 4. CRITICAL: Reorder columns to match the training data
    final_input_df_for_prediction = processed_df_temp.reindex(columns=trained_columns_list, fill_value=0)
    
    return final_input_df_for_prediction

# --- 1. App Title and Introduction ---
st.title("💳 Credit Card Fraud Detection App")
st.markdown("""
    This application predicts the likelihood of a credit card transaction being fraudulent 
    based on various transaction details.
    
    *Leveraging Machine Learning (XGBoost) for enhanced security.*
""")
st.markdown("---") 

# --- 2. Input Fields ---
st.header("Input Transaction Details")

col1, col2 = st.columns(2)
with col1:
    time_input = st.number_input("Time (seconds from first dataset transaction)", value=float(0), step=1.0, format="%.0f", help="e.g., 0 for early morning, ~86400 for next day's start.")
with col2:
    amount_input = st.number_input("Amount ($)", value=float(100.0), step=0.01, format="%.2f", help="Transaction amount in USD.")

# --- Placeholder for V-Features ---
st.subheader("PCA Features (V1-V28)")
st.info("For this demo, V-features are set to a default (0.0). In a real system, these would be complex, anonymized transaction features.")

# Prepare the input DataFrame based on user inputs and defaults for V-features
input_data = {
    'Time': [time_input],
    'Amount': [amount_input]
}
num_v_features = 28
for i in range(1, num_v_features + 1):
    input_data[f'V{i}'] = [0.0] # Use 0.0 or a typical mean/median from your training EDA

input_df = pd.DataFrame(input_data)

st.markdown("---") 

# --- 3. Prediction Button & Logic ---
if st.button("Analyze Transaction"):
    if amount_input <= 0:
        st.warning("Please enter a positive amount for prediction.")
    else:
        with st.spinner("Analyzing transaction..."):
            processed_input_df = preprocess_user_input(input_df, scaler, trained_columns)
            
            prediction_proba = model.predict_proba(processed_input_df)[0][1]
            prediction_class = model.predict(processed_input_df)[0]
            
            st.subheader("Prediction Result:")
            if prediction_class == 1:
                st.error(f"🚨 **Potential Fraud Detected!** 🚨")
                st.markdown(f"Confidence Score (Probability of Fraud): **{prediction_proba:.4f}**")
                st.markdown("""
                <div style="background-color:#ffebeb; padding:10px; border-radius:5px; border:1px solid #ff0000;">
                    <p style="color:#ff0000; font-weight:bold;">Immediate Action Advised!</p>
                    <ul style="color:#ff0000;">
                        <li>Flag transaction for review.</li>
                        <li>Contact cardholder for verification.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ **Transaction Appears Legitimate.** ✅")
                st.markdown(f"Confidence Score (Probability of Fraud): **{prediction_proba:.4f}**")
                st.markdown("""
                <div style="background-color:#ebffeb; padding:10px; border-radius:5px; border:1px solid #008000;">
                    <p style="color:#008000;">Low Risk Transaction.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.info("💡 **How it works:** This model (XGBoost Classifier) analyzes patterns in transaction time, amount, and masked PCA features to determine the likelihood of fraud.")

# --- 4. About Section & Disclaimers (in Sidebar) ---
st.sidebar.header("About This App")
st.sidebar.markdown("""
This application is a demonstration of a machine learning model built to detect credit card fraud. 
It utilizes a powerful **XGBoost Classifier** trained on anonymized transaction data.

**Key Features:**
- Real-time prediction based on user inputs.
- Highlights the importance of preprocessing and feature engineering.
- Demonstrates handling of highly imbalanced datasets.

**Developed by:** Your Name
**GitHub Repository:** [Link to your GitHub Repo](https://github.com/your_username/your_fraud_detection_project)

""")

st.sidebar.header("Disclaimer")
st.sidebar.warning("""
This is a **demonstration model** for educational and portfolio purposes only. 
It should **NOT** be used for actual financial decisions. 
Real-world fraud detection systems are far more complex, involve continuous monitoring, 
and incorporate human expertise for final decisions.
""")