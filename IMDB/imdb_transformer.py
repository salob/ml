# imdb_transformer.py
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2025, help='Random seed')
args = parser.parse_args()

SEED = args.seed
os.environ["PYTHONHASHSEED"] = str(SEED)
# Enable MPS fallback to CPU for unsupported ops (nested tensors in Transformer)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print(f"Using seed: {SEED}")

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
from codecarbon import EmissionsTracker
from carbontracker.tracker import CarbonTracker
import csv
from datetime import datetime

# For reproducibility
# SEED = 2025

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parameters
MAX_WORDS = 100000  # Same as max_features in TF-IDF
MAX_LENGTH = 200    # Max length of each review
EMBEDDING_DIM = 128 # Dimension of word embeddings (slightly larger for transformer)
BATCH_SIZE = 32
NUM_EPOCHS = 10     # Increased max epochs for early stopping
PATIENCE = 2        # Stop if no improvement for 3 epochs
MIN_EPOCHS = 3      # Minimum training epochs

# Device selection: CUDA > MPS > CPU
# Note: MPS with CPU fallback for operations not yet supported (nested tensors)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using Apple Silicon MPS GPU")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")

# Transformer specific parameters
NUM_HEADS = 8       # Number of attention heads
NUM_LAYERS = 3      # Number of transformer layers
HIDDEN_DIM = 256    # Hidden dimension in feedforward network

# Start emissions tracking
primary_tracker = EmissionsTracker(project_name="IMDB_Transformer", 
                                   pue=1,
                                   experiment_id="c3685c4f-39d8-4c14-b23b-ff1ab159ec74")
secondary_tracker = CarbonTracker(epochs=NUM_EPOCHS,# only for deep learning
                                  update_interval=1,
                                  epochs_before_pred=1,
                                  api_keys={"electricitymaps": "u92FLSYBzQ9ciRIIhtSC"},
                                  log_dir="./carbontracker_logs/",
                                  log_file_prefix="ct_imdb_transformer_")

def load_imdb_data():
    """Load IMDB dataset from local CSV files"""
    print("Loading IMDb dataset from CSV files...")
    train_df = pd.read_csv('data/imdb_train.csv')
    val_df = pd.read_csv('data/imdb_val.csv')
    test_df = pd.read_csv('data/imdb_test.csv')

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
    if pd.isna(text) or not text.strip():
        return "unknown"  # Fallback for empty/NaN texts
    return text.lower().strip()

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
        if not words:  # Handle empty text
            words = ['<unk>']  # Use unknown token for empty text
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
        
        # Debug: Check for completely empty sequences
        non_padding_counts = (~padding_mask).sum(dim=1)
        if (non_padding_counts == 0).any():
            print(f"Warning: Found {(non_padding_counts == 0).sum()} completely padded sequences")
            # Replace completely empty sequences with a single unknown token
            empty_mask = (non_padding_counts == 0)
            x[empty_mask, 0] = 1  # Set first token to <unk>
            padding_mask = self.create_padding_mask(x)  # Recalculate mask
        
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
        x = x.squeeze(-1)  # Remove the keepdim dimension
        
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

# Evaluation function (defined before training to use during epochs)
def evaluate(data_loader, name="set", return_loss=False):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data).squeeze()
            if return_loss:
                loss = criterion(output, target)
                total_loss += loss.item()
            pred = (output > 0.5).float()
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Handle edge case where predictions or targets might be empty
    if len(predictions) == 0 or len(targets) == 0:
        print(f"Warning: Empty predictions or targets in {name}")
        return 0.0, 0.0, float('inf') if return_loss else (0.0, 0.0)
    
    acc = accuracy_score(targets, predictions)
    # Handle potential empty slice issues in f1_score
    try:
        f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    except:
        f1 = 0.0
    
    print(f"\n{name} Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Handle classification report errors
    try:
        print(classification_report(targets, predictions, target_names=["neg", "pos"], zero_division=0))
    except Exception as e:
        print(f"Classification report warning: {e}")
    
    if return_loss:
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
        return acc, f1, avg_loss
    return acc, f1

# Training with early stopping
print("Training Tiny Transformer model with early stopping...")
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
early_stop_epoch = NUM_EPOCHS

# Track epoch-by-epoch metrics for convergence analysis
epoch_history = []
# Start code carbon emissions tracking
primary_tracker.start()
for epoch in range(NUM_EPOCHS):
    secondary_tracker.epoch_start()

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
    
    # Evaluate after each epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch+1}, Average Training Loss: {avg_loss:.4f}')
    
    # Validation evaluation with loss
    val_acc_epoch, val_f1_epoch, val_loss = evaluate(val_loader, name="Validation", return_loss=True)
    print(f'Epoch: {epoch+1}, Val Acc: {val_acc_epoch:.4f}, Val F1: {val_f1_epoch:.4f}, Val Loss: {val_loss:.4f}')
    
    # Store epoch metrics
    epoch_metrics = {
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'val_loss': val_loss,
        'val_accuracy': val_acc_epoch,
        'val_f1': val_f1_epoch,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    epoch_history.append(epoch_metrics)
    
    # Early stopping logic
    print(f'Current val_loss: {val_loss:.6f}, Best val_loss: {best_val_loss:.6f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model state
        best_model_state = model.state_dict().copy()
        print(f'New best validation loss: {best_val_loss:.4f}')
    else:
        patience_counter += 1
        print(f'No improvement for {patience_counter} epochs')
        
        # Check for early stopping
        if patience_counter >= PATIENCE and epoch >= MIN_EPOCHS - 1:
            print(f'Early stopping at epoch {epoch+1}')
            early_stop_epoch = epoch + 1
            secondary_tracker.epoch_end()
            break
    
    secondary_tracker.epoch_end()
    print("-" * 50)

# Load best model weights
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f'Loaded best model from validation loss: {best_val_loss:.4f}')

# Stop emissions tracking
primary_tracker.stop()
print("\nEmissions tracking complete. Check emissions.csv for results.")

print(f'Training completed at epoch {early_stop_epoch} (early stopping)')
secondary_tracker.stop()

# Final Evaluation
print("\nFinal evaluation...")
val_acc, val_f1 = evaluate(val_loader, name="Validation")
test_acc, test_f1 = evaluate(test_loader, name="Test")

print(f"\nModel Summary:")
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Layers: {NUM_LAYERS} transformer layers")
print(f"Attention heads: {NUM_HEADS}")
print(f"Embedding dimension: {EMBEDDING_DIM}")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Count model parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Collect metrics
metrics = {
    'timestamp': datetime.now().isoformat(),
    'model_type': 'TinyTransformer',
    'seed': SEED,
    'val_accuracy': val_acc,
    'val_f1': val_f1,
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'total_parameters': total_params,
    'embedding_dim': EMBEDDING_DIM,
    'num_heads': NUM_HEADS,
    'num_layers': NUM_LAYERS,
    'hidden_dim': HIDDEN_DIM,
    'max_length': MAX_LENGTH,
    'batch_size': BATCH_SIZE,
    'max_epochs': NUM_EPOCHS,
    'actual_epochs': early_stop_epoch,
    'best_val_loss': best_val_loss,
    'early_stopped': early_stop_epoch < NUM_EPOCHS,
    'max_words': MAX_WORDS,
    'device': str(DEVICE)
}

# Save summary metrics to CSV
csv_file = "logs/imdb_transformer_metrics.csv"
file_exists = os.path.exists(csv_file)

with open(csv_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=metrics.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)

print(f"Metrics saved to {csv_file}")

# Save epoch-by-epoch training history
history_file = f"logs/imdb_transformer_history_seed{SEED}.csv"
with open(history_file, 'w', newline='') as f:
    if epoch_history:
        writer = csv.DictWriter(f, fieldnames=epoch_history[0].keys())
        writer.writeheader()
        writer.writerows(epoch_history)
        print(f"Training history saved to {history_file}")

# Save model and vocabulary for interactive demo
os.makedirs("models", exist_ok=True)
model_path = "models/transformer_model.pt"
vocab_path = "models/transformer_vocab.pkl"

# Load best model state
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), model_path)

import pickle
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)

print(f"Model saved to {model_path}")
print(f"Vocabulary saved to {vocab_path}")

