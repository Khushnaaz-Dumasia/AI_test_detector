import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from transformers import pipeline
import torch

def run_transformer_evaluation():
    print("Initializing Hugging Face Transformer pipeline...")
    
    # RECOMMENDED: Use Hello-SimpleAI for modern ChatGPT/LLM text detection
    # Alternatively, keep "openai-community/roberta-base-openai-detector" for GPT-2 baseline
    MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
    
    detector = pipeline(
        "text-classification", 
        model=MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # 1. Load test/dev dataset
    print("Loading test data...")
    df = pd.read_csv('../data/raw/LLM.csv') # Ensure path is correct relative to execution location
    
    # Map labels safely to align with dataset (student=0, ai=1)
    df['binary_label'] = df['Label'].map({'student': 0, 'ai': 1})
    
    # Take a small sample slice for evaluation
    test_samples = df.sample(n=100, random_state=42) if len(df) > 100 else df
    
    texts = test_samples['Text'].tolist()
    true_labels = test_samples['binary_label'].tolist()
    
    print(f"Running inference over {len(texts)} samples using {MODEL_NAME}...")
    predictions = []
    
    for text in texts:
        try:
            # Safely truncate text (RoBERTa context window limit is 512 tokens)
            truncated_text = " ".join(str(text).split()[:300])
            result = detector(truncated_text)[0]
            
            raw_label = str(result['label']).upper()
            
            # Robust label handling across Hugging Face checkpoints
            # AI/Fake indicators: 'LABEL_1', 'FAKE', 'ChatGPT', 'AI'
            if any(ai_tag in raw_label for ai_tag in ['LABEL_1', 'FAKE', 'CHATGPT', 'AI']):
                pred_label = 1
            else:
                pred_label = 0
                
            predictions.append(pred_label)
        except Exception as e:
            predictions.append(0) # Fallback baseline default
            
    # 2. Performance Comparison Metrics
    print("\n================ TRANSFORMER BASELINE REPORT ================")
    print(classification_report(true_labels, predictions, target_names=['Human (Student)', 'AI Generated']))
    
    roberta_f1 = f1_score(true_labels, predictions, average='macro')
    print(f"Transformer Baseline Macro F1-Score: {roberta_f1:.4f}")
    print("================================================================")
    
if __name__ == "__main__":
    run_transformer_evaluation()