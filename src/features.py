# features.py
import numpy as np
import nltk
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)

def calculate_burstiness(raw_text):
    """
    Measures sentence length variation (standard deviation).
    Higher values mean more 'bursty', varied writing (highly human).
    """
    # 1. Split document into sentences
    sentences = nltk.sent_tokenize(str(raw_text))
    
    if len(sentences) <= 1:
        # If there's only 1 sentence, there is no variation (burstiness is 0)
        return 0.0
        
    # 2. Count words per sentence
    sentence_lengths = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        sentence_lengths.append(len(words))
        
    # 3. Calculate standard deviation of sentence lengths
    burstiness = np.std(sentence_lengths)
    return burstiness

