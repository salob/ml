# imdb_cnn.py
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
from codecarbon import EmissionsTracker

# For reproducibility
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# (EmissionsTracker will be started/stopped per run to record each run separately)

# Parameters
MAX_WORDS = 100000  # Same as max_features in TF-IDF
MAX_LENGTH = 200    # Max length of each review
EMBEDDING_DIM = 100 # Dimension of word embeddings
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# CNN Model
class TextCNN(nn.Module):
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
model = TextCNN(len(vocab.word2idx), EMBEDDING_DIM, MAX_LENGTH).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

def evaluate(data_loader, name="set", compute_loss=False):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data).squeeze()
            if compute_loss:
                total_loss += criterion(output, target).item() * data.size(0)
            pred = (output > 0.5).float()
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro")
    if compute_loss:
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"\n{name} Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(classification_report(targets, predictions, target_names=["neg", "pos"]))
        return acc, f1, avg_loss
    else:
        print(f"\n{name} Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(classification_report(targets, predictions, target_names=["neg", "pos"]))
        return acc, f1


def run_experiment(run_idx, seed):
    print(f"\n=== Run {run_idx+1} / {N_RUNS} (seed={seed}) ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tracker = EmissionsTracker(project_name="IMDB_CNN")
    tracker.start()

    # reinitialize model and optimizer for each run
    model_local = TextCNN(len(vocab.word2idx), EMBEDDING_DIM, MAX_LENGTH).to(DEVICE)
    criterion_local = nn.BCELoss()
    optimizer_local = optim.Adam(model_local.parameters())

    # temporarily override global model/criterion/optimizer for evaluation calls
    global model, criterion, optimizer
    model, criterion, optimizer = model_local, criterion_local, optimizer_local

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        print(f'Epoch: {epoch+1}, Average Train Loss: {avg_train_loss:.4f}')

        val_acc, val_f1, val_loss = evaluate(val_loader, name="Validation", compute_loss=True)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

    print("\nFinal evaluation on test set...")
    test_acc, test_f1 = evaluate(test_loader, name="Test")

    tracker.stop()
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    return {"test_acc": test_acc, "test_f1": test_f1, "history": history}


# Run multiple repeats
N_RUNS = 10
all_results = []
for run_idx in range(N_RUNS):
    res = run_experiment(run_idx, SEED + run_idx)
    all_results.append(res)

# Summary
import numpy as _np
test_accs = _np.array([r["test_acc"] for r in all_results])
test_f1s = _np.array([r["test_f1"] for r in all_results])
print(f"\nSummary over {N_RUNS} runs:")
print(f"Test acc mean±std: {test_accs.mean():.4f} ± {test_accs.std():.4f}")
print(f"Test F1  mean±std: {test_f1s.mean():.4f} ± {test_f1s.std():.4f}")
