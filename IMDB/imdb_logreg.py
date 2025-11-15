# imdb_logreg.py
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from codecarbon import EmissionsTracker
from carbontracker.tracker import CarbonTracker
import os
import csv
from datetime import datetime

# Make things a bit more random
import argparse
import os

# Add seed argument
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2025, help='Random seed')
args = parser.parse_args()

SEED = args.seed
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"Using seed: {SEED}")

# For reproducibility
# SEED = 2025

random.seed(SEED)
np.random.seed(SEED)

# Start emissions tracking
primary_tracker = EmissionsTracker(project_name="IMDB_LogReg",pue=1.0,
                                   experiment_id="logreg_experiment_001",
                                   experiment_name="IMDB Logistic Regression"
) # pue=1.58 match CarbonTracker's default PUE
# secondary tracker to validate codecarbon results
secondary_tracker = CarbonTracker(epochs=1,# only for deep learning
                                  update_interval=1,
                                  epochs_before_pred=0,
                                  log_dir="./logs/",
                                  log_file_prefix="ct_imdb_logreg_")

primary_tracker.start()
secondary_tracker.epoch_start()
# 1. Load IMDb dataset from CSV files
print("Loading IMDb dataset...")
train_df = pd.read_csv('imdb_train.csv')
val_df = pd.read_csv('imdb_val.csv')
test_df = pd.read_csv('imdb_test.csv')

# Extract texts and labels
train_texts = train_df['text'].values
train_labels = train_df['label'].values
val_texts = val_df['text'].values
val_labels = val_df['label'].values
test_texts = test_df['text'].values
test_labels = test_df['label'].values

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=100_000,
    lowercase=True,
)
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

# 4. Logistic Regression classifier
clf = LogisticRegression(
    solver="liblinear",
    C=1.0,              # regularization strength
    random_state=SEED,
    max_iter=1000,
    n_jobs=-1,
)
print("Training Logistic Regression...")
clf.fit(X_train, train_labels)

# 5. Evaluation
def evaluate(X, y, name="set"):
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    print(f"\n{name} Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(y, preds, target_names=["neg", "pos"]))
    return acc, f1

val_acc, val_f1 = evaluate(X_val, val_labels, name="Validation")
test_acc, test_f1 = evaluate(X_test, test_labels, name="Test")

# Stop emissions tracking
primary_tracker.stop()
secondary_tracker.epoch_end()
secondary_tracker.stop()
print("\nEmissions tracking complete. Check emissions.csv for results.")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Collect metrics
metrics = {
    'timestamp': datetime.now().isoformat(),
    'model_type': 'LogisticRegression',
    'seed': SEED,
    'val_accuracy': val_acc,
    'val_f1': val_f1,
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'max_features': 100000,
    'ngram_range': '(1,2)',
    'solver': 'liblinear',
    'C': 1.0,
    'max_iter': 1000
}

# Save to CSV
csv_file = "logs/imdb_logreg_metrics.csv"
file_exists = os.path.exists(csv_file)

with open(csv_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=metrics.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)

print(f"Metrics saved to {csv_file}")
