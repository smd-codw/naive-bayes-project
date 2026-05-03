import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Naive Bayes Learning System")

st.markdown("""
### Naive Bayes Formula

P(C|X) = (P(X|C) * P(C)) / P(X)

- P(C): Prior Probability
- P(X|C): Likelihood
- Assumes features are independent
""")

# Upload Dataset
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)

    # Target selection
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Encoding categorical data
    X = pd.get_dummies(X)

    st.subheader("After Encoding")
    st.write(X.head())

    # Train-Test Split
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train Model
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    # Prediction
    st.subheader("Try Prediction")

    input_data = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        pred = model.predict([input_data])
        st.success(f"Prediction: {pred[0]}")