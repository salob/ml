#!/usr/bin/env python3
"""
Web-based Movie Review Sentiment Classifier using Gradio
Deploy this to Azure, Hugging Face Spaces, or run locally for sharing with others.
"""

import os
import pickle
import re
from html import unescape
import warnings

# Suppress PyTorch nested tensor warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.transformer')

import torch
import torch.nn as nn
import gradio as gr


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
# Vocabulary Class (must match training)
# ============================================================================

class Vocabulary:
    """Vocabulary class for mapping words to indices - MUST match imdb_cnn.py"""
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_count = {}
        
    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        if word not in self.word_count:
            self.word_count[word] = 0
        self.word_count[word] += 1
        
    def text_to_sequence(self, text, max_length):
        tokens = text.split()
        sequence = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        if len(sequence) < max_length:
            sequence = sequence + [self.word2idx['<pad>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence


# ============================================================================
# CNN Model Definition (must match training)
# ============================================================================

class TextCNN(nn.Module):
    """CNN model architecture - MUST match imdb_cnn.py training script"""
    def __init__(self, vocab_size, embedding_dim, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, 128, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Change for Conv1d
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# ============================================================================
# Vocabulary Class (must match training)
# ============================================================================

class Vocabulary:
    """Vocabulary class for text tokenization - MUST match imdb_cnn.py"""
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_count = {}
        
    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        if word not in self.word_count:
            self.word_count[word] = 0
        self.word_count[word] += 1
        
    def text_to_sequence(self, text, max_length):
        """Convert text to sequence of indices."""
        tokens = text.split()
        sequence = [self.word2idx.get(word, self.word2idx['<unk>']) for word in tokens]
        
        # Pad or truncate
        if len(sequence) < max_length:
            sequence = sequence + [self.word2idx['<pad>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
            
        return sequence


# ============================================================================
# Transformer Model Definition (must match training)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer - MUST match imdb_transformer.py"""
    def __init__(self, embedding_dim, max_length=5000):
        super().__init__()
        import math
        
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TinyTransformer(nn.Module):
    """Transformer model architecture - MUST match imdb_transformer.py"""
    def __init__(self, vocab_size, embedding_dim=128, num_heads=8, num_layers=3, 
                 hidden_dim=256, max_length=200):
        super().__init__()
        import math
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def create_padding_mask(self, x):
        """Create mask to ignore padding tokens"""
        return (x == 0)  # True for padding tokens
    
    def forward(self, x):
        import math
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Global average pooling (ignoring padding)
        mask = (~padding_mask).float().unsqueeze(-1)
        mask_sum = mask.sum(dim=1, keepdim=True)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Prevent division by zero
        x = (x * mask).sum(dim=1, keepdim=True) / mask_sum
        x = x.squeeze(-1)
        
        # Classification
        x = self.classifier(x)
        return x


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
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
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
                 vocab_path='models/cnn_vocab.pkl',
                 max_length=200,  # Must match training MAX_LENGTH
                 embedding_dim=100):  # Must match training EMBEDDING_DIM
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load(self):
        """Load the trained model and vocabulary."""
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            
        vocab_size = len(self.vocab.word2idx)  # Access word2idx from Vocabulary object
        self.model = TextCNN(vocab_size=vocab_size, 
                            embedding_dim=self.embedding_dim,
                            max_length=self.max_length)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def text_to_indices(self, text):
        """Convert text to token indices using the Vocabulary object."""
        cleaned = clean_review(text)
        # Preprocess the same way as training
        cleaned = cleaned.lower()
        cleaned = re.sub(r'[^a-z\s]', '', cleaned)
        
        # Use vocab's text_to_sequence method
        sequence = self.vocab.text_to_sequence(cleaned, self.max_length)
        return torch.LongTensor([sequence])
    
    def predict(self, review):
        """Predict sentiment for a single review."""
        indices = self.text_to_indices(review).to(self.device)
        
        with torch.no_grad():
            output = self.model(indices)
            probability = output.item()  # Model already applies sigmoid
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
                 vocab_path='models/transformer_vocab.pkl',
                 max_length=200,  # Must match training MAX_LENGTH
                 embedding_dim=128,  # Must match training
                 num_heads=8,
                 num_layers=3,
                 hidden_dim=256):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load(self):
        """Load the trained model and vocabulary."""
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            
        vocab_size = len(self.vocab.word2idx)  # Access word2idx from Vocabulary object
        self.model = TinyTransformer(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            max_length=self.max_length
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def text_to_indices(self, text):
        """Convert text to token indices using the Vocabulary object."""
        cleaned = clean_review(text)
        # Preprocess the same way as training
        cleaned = cleaned.lower()
        cleaned = re.sub(r'[^a-z\s]', '', cleaned)
        
        # Use vocab's text_to_sequence method
        sequence = self.vocab.text_to_sequence(cleaned, self.max_length)
        return torch.LongTensor([sequence])
    
    def predict(self, review):
        """Predict sentiment for a single review."""
        indices = self.text_to_indices(review).to(self.device)
        
        with torch.no_grad():
            output = self.model(indices)
            probability = output.item()  # Model already applies sigmoid
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
# Load Models
# ============================================================================

print("Loading models...")
models = {}

try:
    logreg = LogRegClassifier()
    logreg.load()
    models['Logistic Regression'] = logreg
    print("Logistic Regression loaded")
except Exception as e:
    print(f"LogReg not available: {e}")

try:
    cnn = CNNClassifier()
    cnn.load()
    models['CNN'] = cnn
    print("CNN loaded")
except Exception as e:
    print(f"CNN not available: {e}")

try:
    transformer = TransformerClassifier()
    transformer.load()
    models['Transformer'] = transformer
    print(" Transformer loaded")
except Exception as e:
    print(f"Transformer not available: {e}")


# ============================================================================
# Gradio Interface
# ============================================================================

def predict_sentiment(review_text):
    """
    Predict sentiment using all available models.
    Returns formatted results for Gradio display.
    """
    if not review_text or not review_text.strip():
        return " Please enter a movie review to analyze."
    
    results = []
    results.append("# 🍿 Film Review Sentiment Analysis Results\n")
    
    for model_name, model in models.items():
        try:
            result = model.predict(review_text)
            sentiment = result['prediction']
            confidence = result['confidence'] * 100
            
            # Emoji based on sentiment
            emoji = "🟢" if sentiment == 'Positive' else "🔴"
            
            results.append(f"## {model_name}")
            results.append(f"{emoji} **{sentiment}** (Confidence: {confidence:.1f}%)")
            results.append(f"- Negative: {result['probabilities']['negative']:.3f}")
            results.append(f"- Positive: {result['probabilities']['positive']:.3f}")
            results.append("")
            
        except Exception as e:
            results.append(f"## {model_name}")
            results.append(f" Error: {str(e)}")
            results.append("")
    
    return "\n".join(results)


# Sample reviews for quick testing
examples = [
    ["When I go to the cinema I expect to be entertained and this movie delivered"],
    ["There was a lot of fighting and blood and I'm not really very fond of violence to be honest. The story line was good though"],
    ["This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommended!"],
    ["Terrible movie. Waste of time and money. The plot was confusing and the acting was mediocre at best."],
    ["An okay film. Some good moments but also some boring parts. Not bad but not great either."],
    ["Best film I've seen this year! Incredible cinematography and a touching story that will stay with me forever."],
    ["I fell asleep halfway through. Nothing interesting happens and the characters are bland and forgettable."]
]

# Create Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter your movie review here...",
        label="Movie Review"
    ),
    outputs=gr.Markdown(label="Predictions"),
    title="🎬 IMDB Film Review Sentiment Analysis",
    description="""
    This demo uses three different machine learning models to predict whether a movie review is **positive** or **negative**:
    
    - **Logistic Regression** (Simple baseline, ~91% accuracy)
    - **CNN** (Convolutional Neural Network, moderate complexity, ~85% accuracy)
    - **Transformer** (High complexity deep learning, ~83% accuracy)
    
    All models were trained on 40,000 IMDB movie reviews. Enter your own review to see how they classify it!
    
    ---
    *Part of ML Energy Efficiency Research comparing model complexity vs. carbon emissions*
    """,
    examples=examples,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    if not models:
        print("\n No models loaded! Cannot start web demo.")
        print("\nMake sure trained models are in the 'models/' directory:")
        print("  - LogReg: models/logreg_model.pkl, models/logreg_vectorizer.pkl")
        print("  - CNN: models/cnn_model.pt, models/cnn_vocab.pkl")
        print("  - Transformer: models/transformer_model.pt, models/transformer_vocab.pkl")
    else:
        print(f"\n Starting web demo with {len(models)} model(s)...\n")
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,
            share=True  # Set to True for temporary public link
        )
