"""
Naive Bayes Learning System
---------------------------------
This is an interactive web application built using Streamlit
to demonstrate the working of the Naive Bayes algorithm.

Author: Suman Das
"""

# ==============================
# IMPORT LIBRARIES
# ==============================

import streamlit as st                 # For building web UI
import pandas as pd                   # For data handling
import numpy as np                    # For numerical operations

from sklearn.model_selection import train_test_split   # For splitting dataset
from sklearn.naive_bayes import GaussianNB             # Naive Bayes model
from sklearn.metrics import accuracy_score, confusion_matrix  # Evaluation metrics


# ==============================
# TITLE & INTRODUCTION
# ==============================

st.title("📊 Naive Bayes Learning System")

st.markdown("""
### 🧠 About Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem**.

#### Formula:
P(C|X) = (P(X|C) * P(C)) / P(X)

- **P(C)** → Prior probability of class  
- **P(X|C)** → Likelihood  
- Assumes all features are independent  

This app allows users to upload data, train the model, and make predictions.
""")


# ==============================
# DATASET UPLOAD MODULE
# ==============================

# Upload CSV file
file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])

if file:

    # Load dataset into pandas DataFrame
    df = pd.read_csv(file)

    # Display dataset preview
    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    # Show shape (rows, columns)
    st.write("📐 Shape of dataset:", df.shape)


    # ==============================
    # TARGET COLUMN SELECTION
    # ==============================

    # User selects target variable (label)
    target = st.selectbox("🎯 Select Target Column", df.columns)


    # Separate features (X) and target (y)
    X = df.drop(columns=[target])
    y = df[target]


    # ==============================
    # DATA PREPROCESSING
    # ==============================

    st.subheader("⚙️ Preprocessing")

    # Convert categorical variables into numeric using one-hot encoding
    X = pd.get_dummies(X)

    # Show processed data
    st.write("After Encoding:")
    st.write(X.head())


    # ==============================
    # TRAIN-TEST SPLIT
    # ==============================

    st.subheader("📊 Train-Test Split")

    # User controls test size
    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )


    # ==============================
    # MODEL TRAINING
    # ==============================

    st.subheader("🤖 Model Training")

    # Initialize Gaussian Naive Bayes model
    model = GaussianNB()

    # Train model using training data
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)


    # ==============================
    # MODEL EVALUATION
    # ==============================

    st.subheader("📈 Model Evaluation")

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write("✅ Accuracy:", accuracy)

    # Display confusion matrix
    st.write("📉 Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))


    # ==============================
    # PREDICTION MODULE
    # ==============================

    st.subheader("🔮 Try Your Own Prediction")

    st.write("Enter values for each feature:")

    input_data = []

    # Create input fields for each feature
    for col in X.columns:
        val = st.number_input(f"{col}", value=0.0)
        input_data.append(val)

    # Predict button
    if st.button("Predict"):

        # Convert input to array and predict
        pred = model.predict([input_data])

        # Display prediction result
        st.success(f"🎯 Predicted Class: {pred[0]}")
