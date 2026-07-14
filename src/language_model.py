# language_model.py
import numpy as np
from collections import Counter, defaultdict

class BigramLanguageModel:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.vocab_size = 0

    def train(self, tokenized_texts):
        """
        Trains the bigram model on a list of tokenized documents.
        Only train this on your HUMAN split!
        """
        for tokens in tokenized_texts:
            # We add start and end tokens to handle sentence/document boundaries
            padded_tokens = ['<s>'] + tokens + ['</s>']
            
            for i in range(len(padded_tokens) - 1):
                w1, w2 = padded_tokens[i], padded_tokens[i+1]
                self.unigram_counts[w1] += 1
                self.bigram_counts[w1][w2] += 1
                self.vocab.add(w1)
                self.vocab.add(w2)
                
        # Count unique words in our vocabulary
        self.vocab_size = len(self.vocab)

    def get_bigram_prob(self, w1, w2):
        """
        Calculates the probability of w2 given w1 using Laplace (Add-1) smoothing.
        P(w2 | w1) = (Count(w1, w2) + 1) / (Count(w1) + V)
        """
        count_w1_w2 = self.bigram_counts[w1][w2]
        count_w1 = self.unigram_counts[w1]
        
        # Laplace Smoothing formula
        prob = (count_w1_w2 + 1) / (count_w1 + self.vocab_size)
        return prob

    def calculate_perplexity(self, tokens):
        """
        Calculates the perplexity of a given list of tokens.
        A lower perplexity means the model finds the text more 'natural' (human-like).
        """
        if not tokens:
            return float('inf')
            
        padded_tokens = ['<s>'] + tokens + ['</s>']
        N = len(padded_tokens) - 1
        log_prob_sum = 0.0

        for i in range(N):
            w1, w2 = padded_tokens[i], padded_tokens[i+1]
            prob = self.get_bigram_prob(w1, w2)
            log_prob_sum += np.log(prob)

        # Perplexity = exp(-1/N * sum(log_P))
        perplexity = np.exp(- (log_prob_sum / N))
        return perplexity