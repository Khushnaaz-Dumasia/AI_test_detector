import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import os

# Ensure the tokenizer is downloaded
nltk.download('punkt', quiet=True)

def clean_and_tokenize(text):
    """Cleans text by lowercasing, removing punctuation, and tokenizing."""
    text = str(text).lower()
    # Remove everything except lowercase letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    # Join tokens back into a single string separated by spaces for easier storage
    return ' '.join(tokens)

def main():
    print("Starting preprocessing pipeline...")
    
    # 1. Define file paths
    raw_data_path = 'data/raw/LLM.csv' # UPDATE THIS to your exact file name
    processed_data_path = 'data/processed/cleaned_dataset.csv'
    
    # Check if raw data exists
    if not os.path.exists(raw_data_path):
        print(f"Error: Could not find raw data at {raw_data_path}")
        return

    # 2. Load the raw data
    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)
    
    # 3. Map labels to binary values (0 for student, 1 for AI)
    print("Binarizing labels...")
    df['Label'] = df['Label'].map({'student': 0, 'ai': 1})
    
    # Drop any rows where the label couldn't be mapped (just in case)
    df = df.dropna(subset=['Label'])
    
    # 4. Apply cleaning and tokenization
    print("Cleaning and tokenizing text (this may take a moment)...")
    df['Cleaned_Text'] = df['Text'].apply(clean_and_tokenize)
    
    # 5. Save the processed data
    # Create the processed directory if it somehow doesn't exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Save to CSV, keeping only the necessary columns
    df[['Cleaned_Text', 'Label']].to_csv(processed_data_path, index=False)
    print(f"Success! Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    main()