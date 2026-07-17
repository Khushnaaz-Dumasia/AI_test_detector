# transformer_baseline.py
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from transformers import pipeline
import torch

def run_transformer_evaluation():
    print("Initializing Hugging Face Transformer pipeline...")
    # Using the industry standard Hugging Face model repository path
    detector = pipeline(
        "text-classification", 
        model="openai-community/roberta-base-openai-detector",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # 1. Load your test/dev dataset
    print("Loading test data...")
    df = pd.read_csv('../data/raw/LLM.csv') # Using your raw file to pull full text strings
    
    # Map labels safely to align with the dataset mapping (student=0, ai=1)
    df['binary_label'] = df['Label'].map({'student': 0, 'ai': 1})
    
    # Take a small sample slice for evaluation if your dataset is massive
    test_samples = df.sample(n=100, random_state=42) if len(df) > 100 else df
    
    texts = test_samples['Text'].tolist()
    true_labels = test_samples['binary_label'].tolist()
    
    print(f"Running inference over {len(texts)} samples using RoBERTa...")
    predictions = []
    
    for text in texts:
        try:
            # Handle token truncation for texts longer than 512 tokens safely
            truncated_text = " ".join(text.split()[:400])
            result = detector(truncated_text)[0]
            
            # The model labels are 'Real' (Human) or 'Fake' (AI)
            pred_label = 1 if result['label'] == 'Fake' else 0
            predictions.append(pred_label)
        except Exception as e:
            predictions.append(0) # Fallback baseline default
            
    # 2. Performance Comparison Metrics
    print("\n================ ROBERTA BASE AI DETECTOR REPORT ================")
    print(classification_report(true_labels, predictions, target_names=['Human (Student)', 'AI Generated']))
    
    roberta_f1 = f1_score(true_labels, predictions, average='macro')
    print(f"RoBERTa Baseline Macro F1-Score: {roberta_f1:.4f}")
    print("================================================================")
    
if __name__ == "__main__":
    run_transformer_evaluation()