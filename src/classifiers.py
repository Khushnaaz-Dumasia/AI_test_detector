# src/classifiers.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def main():
    print("--- Day 8: Classifier Training (Unit 4) ---")
    
    # 1. Define paths
    features_path = '../data/processed/X_features.csv'
    labels_path = '../data/processed/y_labels.npy'
    models_dir = '../models/'
    
    # Create a models directory to save our trained weights for the demo
    os.makedirs(models_dir, exist_ok=True)

    # 2. Load the features and labels
    print("Loading feature matrix and labels...")
    try:
        X = pd.read_csv(features_path)
        y = np.load(labels_path)
    except FileNotFoundError:
        print("Error: Could not find features or labels. Did you run create_feature_matrix.py?")
        return

    # 3. Train-Test Split for Classification
    # We split 80% for training the classifiers, 20% for evaluating tomorrow (Day 9)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)} documents")
    print(f"Testing set size: {len(X_test)} documents")

    # 4. Feature Scaling
    # Logistic Regression requires features to be on a similar scale. 
    # We use GaussianNB because our features are continuous, not discrete word counts.
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 5. Train Naive Bayes (Gaussian)
    print("Training Gaussian Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    
    # 6. Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # 7. Save the Models and Scaler for Day 9 & Day 12
    print("\nSaving models to /models directory...")
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, 'naive_bayes.pkl'), 'wb') as f:
        pickle.dump(nb_model, f)
    with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)
        
    # Save the test set specifically so we can evaluate it tomorrow
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_test.npy', y_test)
    
    print("Success! Models trained and saved. Ready for Unit 5 evaluation.")

if __name__ == "__main__":
    main()