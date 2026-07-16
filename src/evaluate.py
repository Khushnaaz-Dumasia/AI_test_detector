# src/evaluate.py
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("--- Day 9: Model Evaluation (Unit 5) ---")
    
    # 1. Load the test data
    try:
        X_test = np.load('../data/processed/X_test.npy')
        y_test = np.load('../data/processed/y_test.npy')
    except FileNotFoundError:
        print("Error: Could not find test data. Did you run classifiers.py?")
        return

    # 2. Load the saved scaler and models
    print("Loading trained models...\n")
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../models/naive_bayes.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    with open('../models/logistic_regression.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    # 3. Scale the test features (must use the same scaler from training!)
    X_test_scaled = scaler.transform(X_test)

    # 4. Evaluate Naive Bayes
    print("========================================")
    print("       NAIVE BAYES RESULTS              ")
    print("========================================")
    nb_preds = nb_model.predict(X_test_scaled)
    print("Confusion Matrix:\n", confusion_matrix(y_test, nb_preds))
    # classification_report automatically calculates Precision, Recall, and F1
    print("\nClassification Report:\n", classification_report(y_test, nb_preds, target_names=['Student (0)', 'AI (1)']))

    # 5. Evaluate Logistic Regression
    print("\n========================================")
    print("    LOGISTIC REGRESSION RESULTS         ")
    print("========================================")
    lr_preds = lr_model.predict(X_test_scaled)
    print("Confusion Matrix:\n", confusion_matrix(y_test, lr_preds))
    print("\nClassification Report:\n", classification_report(y_test, lr_preds, target_names=['Student (0)', 'AI (1)']))

if __name__ == "__main__":
    main()