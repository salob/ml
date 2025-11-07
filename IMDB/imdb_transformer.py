# imdb_transformer.py
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report
import math

# For reproducibility
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parameters
MAX_WORDS = 100000  # Same as max_features in TF-IDF
MAX_LENGTH = 200    # Max length of each review
EMBEDDING_DIM = 128 # Dimension of word embeddings (slightly larger for transformer)
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformer specific parameters
NUM_HEADS = 8       # Number of attention heads
NUM_LAYERS = 3      # Number of transformer layers
HIDDEN_DIM = 256    # Hidden dimension in feedforward network

def load_imdb_data():
    """Load IMDB dataset from local CSV files"""
    print("Loading IMDb dataset from CSV files...")
    train_df = pd.read_csv('imdb_train.csv')
    val_df = pd.read_csv('imdb_val.csv')
    test_df = pd.read_csv('imdb_test.csv')
    
    return (
        train_df['text'].values, train_df['label'].values,
        val_df['text'].values, val_df['label'].values,
        test_df['text'].values, test_df['label'].values
    )

# Load data from CSV files
print("Loading IMDb dataset...")
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_imdb_data()

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# Text preprocessing
def preprocess_text(text):
    # Only lowercase, matching TF-IDF's preprocessing
    return text.lower()

class Vocabulary:
    def __init__(self, max_size=MAX_WORDS):
        self.max_size = max_size
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = ['<pad>', '<unk>']
        
    def build_vocab(self, texts):
        words = [word for text in texts for word in text.split()]
        word_counts = Counter(words).most_common(self.max_size - 2)
        for word, _ in word_counts:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
    
    def text_to_sequence(self, text, max_length):
        words = text.split()
        sequence = [self.word2idx.get(word, 1) for word in words[:max_length]]
        if len(sequence) < max_length:
            sequence += [0] * (max_length - len(sequence))
        return sequence[:max_length]

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = [preprocess_text(text) for text in texts]
        self.labels = np.array(labels)  # Convert to numpy array
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        sequence = self.vocab.text_to_sequence(text, self.max_length)
        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super().__init__()
        
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

# Tiny Transformer Model
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_length):
        super().__init__()
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
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Global average pooling (ignoring padding)
        mask = (~padding_mask).float().unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Classification
        x = self.classifier(x)
        return x

# Prepare data
print("Preparing data...")
vocab = Vocabulary(max_size=MAX_WORDS)
train_texts = [preprocess_text(text) for text in train_texts]
vocab.build_vocab(train_texts)

train_dataset = IMDBDataset(train_texts, train_labels, vocab, MAX_LENGTH)
val_dataset = IMDBDataset(val_texts, val_labels, vocab, MAX_LENGTH)
test_dataset = IMDBDataset(test_texts, test_labels, vocab, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model
model = TinyTransformer(
    vocab_size=len(vocab.word2idx),
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    max_length=MAX_LENGTH
).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Training
print("Training Tiny Transformer model...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    print(f'Epoch: {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')

# Evaluation function
def evaluate(data_loader, name="set"):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data).squeeze()
            pred = (output > 0.5).float()
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro")
    print(f"\n{name} Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(targets, predictions, target_names=["neg", "pos"]))
    return acc, f1

# Evaluate
print("\nEvaluating model...")
val_acc, val_f1 = evaluate(val_loader, name="Validation")
test_acc, test_f1 = evaluate(test_loader, name="Test")

print(f"\nModel Summary:")
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Layers: {NUM_LAYERS} transformer layers")
print(f"Attention heads: {NUM_HEADS}")
print(f"Embedding dimension: {EMBEDDING_DIM}")