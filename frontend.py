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
st.title("Breast Cancer Prediction by Preet and Abhinav")

# Input Fields for 30 Features
st.subheader("Enter Tumor Characteristics:")

# Input fields for mean features
mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_perimeter = st.number_input("Mean Perimeter")
mean_area = st.number_input("Mean Area")
mean_smoothness = st.number_input("Mean Smoothness")
mean_compactness = st.number_input("Mean Compactness")
mean_concavity = st.number_input("Mean Concavity")
mean_concave_points = st.number_input("Mean Concave Points")
mean_symmetry = st.number_input("Mean Symmetry")
mean_fractal_dimension = st.number_input("Mean Fractal Dimension")

# Input fields for error features
radius_error = st.number_input("Radius Error")
texture_error = st.number_input("Texture Error")
perimeter_error = st.number_input("Perimeter Error")
area_error = st.number_input("Area Error")
smoothness_error = st.number_input("Smoothness Error")
compactness_error = st.number_input("Compactness Error")
concavity_error = st.number_input("Concavity Error")
concave_points_error = st.number_input("Concave Points Error")
symmetry_error = st.number_input("Symmetry Error")
fractal_dimension_error = st.number_input("Fractal Dimension Error")

# Input fields for worst features
worst_radius = st.number_input("Worst Radius")
worst_texture = st.number_input("Worst Texture")
worst_perimeter = st.number_input("Worst Perimeter")
worst_area = st.number_input("Worst Area")
worst_smoothness = st.number_input("Worst Smoothness")
worst_compactness = st.number_input("Worst Compactness")
worst_concavity = st.number_input("Worst Concavity")
worst_concave_points = st.number_input("Worst Concave Points")
worst_symmetry = st.number_input("Worst Symmetry")
worst_fractal_dimension = st.number_input("Worst Fractal Dimension")

# Model Selection
model_choice = st.selectbox(
    "Select a Model",
    ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"]
)

# Predict Button
if st.button("Predict"):
    input_data = np.array([
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
        mean_fractal_dimension, radius_error, texture_error, perimeter_error,
        area_error, smoothness_error, compactness_error, concavity_error,
        concave_points_error, symmetry_error, fractal_dimension_error,
        worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
        worst_compactness, worst_concavity, worst_concave_points, worst_symmetry,
        worst_fractal_dimension
    ]).reshape(1, -1)  # Reshape to 2D array for prediction

    # Select the model and make the prediction
    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict(input_data)
    elif model_choice == "K-Nearest Neighbors":
        prediction = knn_model.predict(input_data)
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)

    # Display the result
    if prediction[0] == 0:
        st.success("The breast cancer is Malignant.")
    else:
        st.success("The breast cancer is Benign.")