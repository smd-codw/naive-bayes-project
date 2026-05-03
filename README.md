# Naive Bayes Learning System

## 📌 Project Description
This project is an interactive web-based application that helps users understand and experiment with the Naive Bayes classification algorithm. Users can upload their own dataset, train a model, and visualize results.

## 🚀 Features
- Upload CSV dataset
- Dataset preview (head, shape)
- Automatic preprocessing (encoding)
- Train-test split
- Gaussian Naive Bayes model training
- Model evaluation (Accuracy & Confusion Matrix)
- User input prediction system

## 🧠 Naive Bayes Concept

Naive Bayes is based on Bayes Theorem:

P(C|X) = (P(X|C) * P(C)) / P(X)

- P(C): Prior probability
- P(X|C): Likelihood
- Assumes feature independence

## 🛠 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas

## ▶️ How to Run

```bash
pip install streamlit pandas scikit-learn
streamlit run app.py
