import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib  # To save and load trained models

# Load the trained models (ensure you've saved them after training)
logistic_model = joblib.load("logistic_model.pkl")
knn_model = joblib.load("knn_model.pkl")
rf_model = joblib.load("rf_classifier.pkl")

# Title
st.title("Breast Cancer Prediction")

# Input Fields
st.subheader("Enter Tumor Characteristics:")
mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_area = st.number_input("Mean Area")
# Add more inputs as needed for all features

# Model Selection
model_choice = st.selectbox(
    "Select a Model",
    ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"]
)

# Predict Button
if st.button("Predict"):
    input_data = np.array([mean_radius, mean_texture, mean_area]).reshape(1, -1)  # Adjust for all features
    
    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict(input_data)
    elif model_choice == "K-Nearest Neighbors":
        prediction = knn_model.predict(input_data)
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)
    
    if prediction[0] == 0:
        st.success("The breast cancer is Malignant.")
    else:
        st.success("The breast cancer is Benign.")
