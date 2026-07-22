import os
import sys
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

# --- Dynamic Path Resolution ---
# Locates the project root directory regardless of where you execute the terminal command
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add src to sys.path
sys.path.append(SCRIPT_DIR)

from language_model import BigramLanguageModel
from features import calculate_burstiness

# Set exact paths relative to project root
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'LLM.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# --- STEP 1: Load preprocessed data ---
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return word_tokenize(text)

print("Preparing and tokenizing texts...")
df['clean_tokens'] = df['Text'].apply(clean_and_tokenize)

df['Label'] = df['Label'].astype(str).str.lower().str.strip()
df['binary_label'] = df['Label'].map({'student': 0, 'ai': 1})
df = df.dropna(subset=['binary_label'])

# --- STEP 2: Split Data ---
train_df, dev_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['binary_label'])

# --- STEP 3: Train Bigram Model on Human Data ---
print("Training Bigram Language Model on Human data...")
human_train_texts = train_df[train_df['binary_label'] == 0]['clean_tokens'].tolist()

lm = BigramLanguageModel()
lm.train(human_train_texts)
print(f"Model trained successfully! Vocabulary Size: {lm.vocab_size}")

os.makedirs(MODEL_DIR, exist_ok=True)
lm_path = os.path.join(MODEL_DIR, 'bigram_language_model.pkl')
joblib.dump(lm, lm_path)
print(f"✅ Saved Bigram language model to {lm_path}")

# --- STEP 4: Feature Extraction ---
print("\nExtracting features from development dataset...")
features_list = []
labels_list = []

for idx, row in dev_df.iterrows():
    raw_text = row['Text']
    tokens = row['clean_tokens']
    label = row['binary_label']
    
    perplexity = lm.calculate_perplexity(tokens)
    burstiness = calculate_burstiness(raw_text)
    
    features_list.append({
        'perplexity': perplexity,
        'burstiness': burstiness
    })
    labels_list.append(label)

# --- STEP 5: Save Processed Outputs ---
X_features = pd.DataFrame(features_list)
y_features = np.array(labels_list)
X_features.replace([np.inf, -np.inf], 99999, inplace=True) 

os.makedirs(PROCESSED_DIR, exist_ok=True)
X_features.to_csv(os.path.join(PROCESSED_DIR, 'X_features.csv'), index=False)
np.save(os.path.join(PROCESSED_DIR, 'y_labels.npy'), y_features)

print("\n🚀 Success! Features and bigram_language_model.pkl generated and placed in models/!")