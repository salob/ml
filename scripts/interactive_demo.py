#!/usr/bin/env python3
"""
Interactive Movie Review Sentiment Classifier
Load trained models and classify user-provided reviews in real-time.
"""

import os
import pickle
import re
from html import unescape

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================================
# Data Preprocessing (same as training)
# ============================================================================

def clean_review(text):
    """Clean HTML tags and entities from review text."""
    text = unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================================
# CNN Model Definition (must match training)
# ============================================================================

class IMDBCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, num_filters=100, 
                 filter_sizes=[3, 4, 5], dropout=0.5):
        super(IMDBCNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        
    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


# ============================================================================
# Transformer Model Definition (must match training)
# ============================================================================

class IMDBTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, 
                 num_layers=2, hidden_dim=256, dropout=0.3):
        super(IMDBTransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, text):
        embedded = self.embedding(text)
        seq_len = embedded.size(1)
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        
        padding_mask = (text == 0)
        transformed = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        pooled = transformed.mean(dim=1)
        dropped = self.dropout(pooled)
        return self.fc(dropped)


# ============================================================================
# Model Loaders
# ============================================================================

class LogRegClassifier:
    """Wrapper for Logistic Regression model."""
    
    def __init__(self, model_path='models/logreg_model.pkl', 
                 vectorizer_path='models/logreg_vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        
    def load(self):
        """Load the trained model and vectorizer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found: {self.vectorizer_path}")
            
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print("✅ Logistic Regression model loaded")
        
    def predict(self, review):
        """Predict sentiment for a single review."""
        cleaned = clean_review(review)
        features = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'confidence': probability[1] if prediction == 1 else probability[0],
            'probabilities': {
                'negative': probability[0],
                'positive': probability[1]
            }
        }


class CNNClassifier:
    """Wrapper for CNN model."""
    
    def __init__(self, model_path='models/cnn_model.pt', 
                 vocab_path='models/cnn_vocab.pkl'):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load(self):
        """Load the trained model and vocabulary."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary not found: {self.vocab_path}")
            
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            
        vocab_size = len(self.vocab)
        self.model = IMDBCNNClassifier(vocab_size=vocab_size)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("✅ CNN model loaded")
        
    def text_to_indices(self, text, max_len=500):
        """Convert text to token indices."""
        cleaned = clean_review(text)
        tokens = cleaned.lower().split()
        indices = [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
        
        # Pad or truncate
        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            
        return torch.LongTensor([indices])
    
    def predict(self, review):
        """Predict sentiment for a single review."""
        indices = self.text_to_indices(review).to(self.device)
        
        with torch.no_grad():
            output = self.model(indices)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
            
        return {
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'confidence': probability if prediction == 1 else (1 - probability),
            'probabilities': {
                'negative': 1 - probability,
                'positive': probability
            }
        }


class TransformerClassifier:
    """Wrapper for Transformer model."""
    
    def __init__(self, model_path='models/transformer_model.pt', 
                 vocab_path='models/transformer_vocab.pkl'):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load(self):
        """Load the trained model and vocabulary."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary not found: {self.vocab_path}")
            
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            
        vocab_size = len(self.vocab)
        self.model = IMDBTransformerClassifier(vocab_size=vocab_size)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("✅ Transformer model loaded")
        
    def text_to_indices(self, text, max_len=500):
        """Convert text to token indices."""
        cleaned = clean_review(text)
        tokens = cleaned.lower().split()
        indices = [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
        
        # Pad or truncate
        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            
        return torch.LongTensor([indices])
    
    def predict(self, review):
        """Predict sentiment for a single review."""
        indices = self.text_to_indices(review).to(self.device)
        
        with torch.no_grad():
            output = self.model(indices)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0
            
        return {
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'confidence': probability if prediction == 1 else (1 - probability),
            'probabilities': {
                'negative': 1 - probability,
                'positive': probability
            }
        }


# ============================================================================
# Interactive Demo
# ============================================================================

def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print(" 🎬 IMDB Sentiment Analysis - Interactive Demo")
    print("="*70)
    print("\nEnter your own movie reviews to see how the models classify them!")
    print("Type 'quit' or 'exit' to end the session.\n")


def print_result(model_name, result):
    """Print prediction result in a nice format."""
    sentiment = result['prediction']
    confidence = result['confidence'] * 100
    
    # Color coding
    if sentiment == 'Positive':
        symbol = "😊"
        color = "\033[92m"  # Green
    else:
        symbol = "😞"
        color = "\033[91m"  # Red
    reset = "\033[0m"
    
    print(f"\n  {model_name}:")
    print(f"    {color}{symbol} {sentiment}{reset} (confidence: {confidence:.1f}%)")
    print(f"    Probabilities: Negative={result['probabilities']['negative']:.3f}, "
          f"Positive={result['probabilities']['positive']:.3f}")


def load_available_models():
    """Load all available trained models."""
    models = {}
    
    # Try to load Logistic Regression
    try:
        logreg = LogRegClassifier()
        logreg.load()
        models['LogReg'] = logreg
    except FileNotFoundError as e:
        print(f"⚠️  Logistic Regression not available: {e}")
    
    # Try to load CNN
    try:
        cnn = CNNClassifier()
        cnn.load()
        models['CNN'] = cnn
    except FileNotFoundError as e:
        print(f"⚠️  CNN not available: {e}")
    
    # Try to load Transformer
    try:
        transformer = TransformerClassifier()
        transformer.load()
        models['Transformer'] = transformer
    except FileNotFoundError as e:
        print(f"⚠️  Transformer not available: {e}")
    
    return models


def main():
    """Main interactive loop."""
    print_banner()
    
    # Load models
    print("Loading models...\n")
    models = load_available_models()
    
    if not models:
        print("\n❌ No trained models found!")
        print("\nTo use this demo, you need to first train the models.")
        print("Run the training scripts in the IMDB/ directory.")
        print("\nNote: Models should be saved in the 'models/' directory with:")
        print("  - LogReg: models/logreg_model.pkl, models/logreg_vectorizer.pkl")
        print("  - CNN: models/cnn_model.pt, models/cnn_vocab.pkl")
        print("  - Transformer: models/transformer_model.pt, models/transformer_vocab.pkl")
        return
    
    print(f"\n✅ Loaded {len(models)} model(s): {', '.join(models.keys())}\n")
    print("-"*70)
    
    # Interactive loop
    while True:
        print("\n" + "="*70)
        review = input("Enter your movie review (or 'quit' to exit):\n> ").strip()
        
        if review.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using the sentiment classifier!")
            break
            
        if not review:
            print("⚠️  Please enter a review.")
            continue
        
        print("\n" + "-"*70)
        print("🔮 Predictions:")
        
        # Get predictions from all available models
        for model_name, model in models.items():
            try:
                result = model.predict(review)
                print_result(model_name, result)
            except Exception as e:
                print(f"\n  {model_name}: ❌ Error: {e}")
        
        print("-"*70)


if __name__ == "__main__":
    main()
