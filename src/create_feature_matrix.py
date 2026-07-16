# create_feature_matrix.py
#  It trains the language model on the human portion of your data, 
# computes both features for every document, 
# and compiles them into your final ML-ready feature matrix $(X)$ and label vector $(y)$.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from language_model import BigramLanguageModel
from features import calculate_burstiness

# --- STEP 1: Load preprocessed data ---
df = pd.read_csv('../data/raw/LLM.csv') # Adjust path if needed

# Re-apply cleaning from your preprocessing step
import re
from nltk.tokenize import word_tokenize

def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return word_tokenize(text)

print("Preparing and tokenizing texts...")
df['clean_tokens'] = df['Text'].apply(clean_and_tokenize)

df['Label'] = df['Label'].astype(str).str.lower().str.strip()
df['binary_label'] = df['Label'].map({'student': 0, 'ai': 1})

df = df.dropna(subset=['binary_label'])

# --- STEP 2: Split Data for Training vs Feature Extraction ---
# We split our data into a training set (to train our Bigram language model)
# and a development/test set (where we will extract features and evaluate).
train_df, dev_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['binary_label'])

# --- STEP 3: Train Language Model on Human Texts Only ---
print("Training Bigram Language Model on Human data...")
human_train_texts = train_df[train_df['binary_label'] == 0]['clean_tokens'].tolist()

lm = BigramLanguageModel()
lm.train(human_train_texts)
print(f"Model trained successfully! Vocabulary Size: {lm.vocab_size}")

# --- STEP 4: Feature Extraction on the Dev Dataset ---
print("\nExtracting features from development dataset...")
features_list = []
labels_list = []

for idx, row in dev_df.iterrows():
    raw_text = row['Text']
    tokens = row['clean_tokens']
    label = row['binary_label']
    
    # Feature 1: Perplexity
    perplexity = lm.calculate_perplexity(tokens)
    
    # Feature 2: Burstiness
    burstiness = calculate_burstiness(raw_text)
    
    # Append features as a dictionary
    features_list.append({
        'perplexity': perplexity,
        'burstiness': burstiness
    })
    labels_list.append(label)

# --- STEP 5: Create ML-Ready Feature Matrix ---
X_features = pd.DataFrame(features_list)
y_features = np.array(labels_list)

# Quick clean-up: Handing edge cases where perplexity might return infinity
X_features.replace([np.inf, -np.inf], 99999, inplace=True) 

print("\n--- Day 7 Feature Matrix (X) ---")
print(X_features.head(10))

print("\n--- Target Labels (y) ---")
print(y_features[:10])

# Save these matrices so you are ready to train classifiers tomorrow!
X_features.to_csv('../data/processed/X_features.csv', index=False)
np.save('../data/processed/y_labels.npy', y_features)
print("\nSuccess! Saved features and labels to 'data/processed/'. ready for ML training!")