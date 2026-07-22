# app.py
import streamlit as st
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline
import torch

import sys
import os

# Allow Streamlit to find modules inside the src/ folder
sys.path.append(os.path.abspath("src"))

# Now import your language model class so joblib can unpickle it correctly
from language_model import BigramLanguageModel

# Ensure NLTK tokenizers are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- Configuration & Cache Loading ---
st.set_page_config(page_title="AI Text Detector Analytics", layout="wide")

# --- Update inside app.py ---

@st.cache_resource
def load_detection_assets():
    """Loads ML artifacts and pipeline cleanly without calling Streamlit UI elements."""
    scaler = None
    naive_bayes_model = None
    language_model = None
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        naive_bayes_model = joblib.load('models/naive_bayes.pkl')
        language_model = joblib.load('models/bigram_language_model.pkl')
        custom_loaded = True
    except Exception:
        custom_loaded = False

    roberta_detector = pipeline(
        "text-classification", 
        model="Hello-SimpleAI/chatgpt-detector-roberta",
        top_k=None,
        device=-1
    )
    
    return scaler, naive_bayes_model, language_model, roberta_detector, custom_loaded

# --- Main Application Logic ---
feature_scaler, nb_model, custom_lm, transformer_model, is_custom_loaded = load_detection_assets()

# Handle UI feedback safely outside the cached function
if not is_custom_loaded:
    st.warning("⚠️ Custom models not loaded — using heuristic fallback.")


# --- Custom Engineering Calculation Functions ---
def calculate_burstiness(text):
    sentences = nltk.sent_tokenize(str(text))
    if len(sentences) <= 1:
        return 0.0
    lengths = [len(word_tokenize(s)) for s in sentences]
    return float(np.std(lengths))

def calculate_perplexity(text, lm_model):
    if not lm_model:
        return 35.42  # Clean default baseline value
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    if not tokens:
        return float('inf')
    
    padded = ['<s>'] + tokens + ['</s>']
    log_prob_sum = 0.0
    N = len(padded) - 1
    
    for i in range(N):
        w1, w2 = padded[i], padded[i+1]
        count_w1_w2 = lm_model.bigram_counts[w1][w2]
        count_w1 = lm_model.unigram_counts[w1]
        prob = (count_w1_w2 + 1) / (count_w1 + lm_model.vocab_size)
        log_prob_sum += np.log(prob)
        
    return float(np.exp(-(log_prob_sum / N)))

# --- Dashboard UI Architecture ---
st.title("🔬 Hybrid Predictive Asset Intelligence: AI Text Forensic Toolkit")
st.markdown("This dashboard screens student text assignments across classical structural NLP metrics alongside massive fine-tuned contextual Transformer platforms.")

text_input = st.text_area(
    "Paste the essay content text here to check style profiles:", 
    height=280, 
    placeholder="Type or copy text block here (Minimum 2-3 sentences recommended for metrics calculation accuracy)..."
)

if st.button("Execute Forensic Verification Sequence", type="primary"):
    if text_input.strip() == "":
        st.warning("Input area blank. Enter text value.")
    else:
        with st.spinner("Analyzing text properties..."):
            # 1. Structural Engineering calculation metrics 
            pp_score = calculate_perplexity(text_input, custom_lm)
            burst_score = calculate_burstiness(text_input)
            
            # 2. Classifiers Inference processing routines
            if nb_model and feature_scaler:
                features = np.array([[pp_score, burst_score]])
                features_scaled = feature_scaler.transform(features)
                nb_pred = nb_model.predict(features_scaled)[0]
                nb_prob = nb_model.predict_proba(features_scaled)[0][1]
            else:
                st.caption("⚠️ Custom models not loaded — using heuristic fallback.")
                nb_pred = 1 if pp_score > 150 and burst_score < 4.0 else 0
                nb_prob = 0.84 if nb_pred == 1 else 0.12

           # Transformer Pipeline Inference execution
            truncated_inp = " ".join(text_input.split()[:400])
            tf_raw_outputs = transformer_model(truncated_inp)[0]
            
            # Extract distinct AI and Human probabilities safely
            ai_prob = 0.0
            human_prob = 0.0

            for entry in tf_raw_outputs:
                lbl = str(entry['label']).upper()
                score = entry['score']
                if any(tag in lbl for tag in ['CHATGPT', 'LABEL_1', 'FAKE', 'AI']):
                    ai_prob = score
                else:
                    human_prob = score

            # Final verdict determination
            is_ai_generated = ai_prob > human_prob
            
            # --- Results Presentation ---
            st.subheader("Analysis Results Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Hand-Crafted Perplexity (Bigram)", value=f"{pp_score:.2f}")
                st.caption("Lower scores imply human text predictability loops.")
                
            with col2:
                st.metric(label="Hand-Crafted Sentence Burstiness", value=f"{burst_score:.2f}")
                st.caption("Low standard deviation values signal structural uniform AI outputs.")
                
            with col3:
                tf_verdict = f"{'AI Generated' if is_ai_generated else 'Human Written'}"
                # Always displays the exact AI Probability Score
                st.metric(
                    label="RoBERTa AI Probability", 
                    value=f"{ai_prob*100:.1f}%", 
                    delta=tf_verdict,
                    delta_color="inverse" if is_ai_generated else "normal"
                )
            
            st.divider()
            
            # Comparative Engine Analytics View Block
            st.subheader("Model Consensus Matrix")
            m1, m2 = st.columns(2)
            
            with m1:
                st.markdown("### Classical Core (Naive Bayes)")
                if nb_pred == 1:
                    st.error(f"Prediction Verdict: **AI Text Signature Confirmed** ({nb_prob*100:.1f}% confidence probability value)")
                else:
                    st.success(f"Prediction Verdict: **Authentic Student Writer** ({(1-nb_prob)*100:.1f}% confidence probability value)")
                    
            with m2:
                st.markdown("### Modern Deep Learning Baseline (RoBERTa)")
                if is_ai_generated:
                    st.error(f"Prediction Verdict: **Synthetic Generation Footprint Flagged** ({ai_prob*100:.1f}% confidence metric)")
                else:
                    st.success(f"Prediction Verdict: **Human Original Text Content** ({human_prob*100:.1f}% confidence metric)")