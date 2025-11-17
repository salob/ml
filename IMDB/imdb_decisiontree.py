# imdb_decisiontree.py
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from codecarbon import EmissionsTracker
from carbontracker.tracker import CarbonTracker
import os
import csv
from datetime import datetime
import pickle

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
primary_tracker = EmissionsTracker(project_name="IMDB_DecisionTree",pue=1.0,
                                   experiment_id="8f74f3ac-7ddf-48a1-8053-f976e6c5cb1e",
) # pue=1.58 match CarbonTracker's default PUE
# secondary tracker to validate codecarbon results
secondary_tracker = CarbonTracker(epochs=1,# only for deep learning
                                  update_interval=1,
                                  epochs_before_pred=0,
                                  log_dir="./logs/",
                                  log_file_prefix="ct_imdb_decisiontree_")

primary_tracker.start()
secondary_tracker.epoch_start()
# 1. Load IMDb dataset from CSV files
print("Loading IMDb dataset...")
train_df = pd.read_csv('data/imdb_train.csv')
val_df = pd.read_csv('data/imdb_val.csv')
test_df = pd.read_csv('data/imdb_test.csv')

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

# 4. Decision Tree classifier
clf = DecisionTreeClassifier(
    random_state=SEED,
    max_depth=None,      # Unlimited depth (will fully grow the tree)
    min_samples_split=2, # Default: minimum samples to split an internal node
    min_samples_leaf=1,  # Default: minimum samples at a leaf node
)
print("Training Decision Tree...")
clf.fit(X_train, train_labels)

# Report tree depth and node count
print(f"Tree depth: {clf.get_depth()}")
print(f"Number of leaves: {clf.get_n_leaves()}")

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
    'model_type': 'DecisionTree',
    'seed': SEED,
    'val_accuracy': val_acc,
    'val_f1': val_f1,
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'tree_depth': int(clf.get_depth()),
    'num_leaves': int(clf.get_n_leaves()),
    'max_features': 100000,
    'ngram_range': '(1,2)',
    'max_depth': str(clf.max_depth),
    'min_samples_split': clf.min_samples_split,
    'min_samples_leaf': clf.min_samples_leaf
}

# Save to CSV
csv_file = "logs/imdb_decisiontree_metrics.csv"
file_exists = os.path.exists(csv_file)

with open(csv_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=metrics.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)

print(f"Metrics saved to {csv_file}")

# Save model and vectorizer for interactive demo
os.makedirs("models", exist_ok=True)
model_path = "models/decisiontree_model.pkl"
vectorizer_path = "models/decisiontree_vectorizer.pkl"

with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Model saved to {model_path}")
print(f"Vectorizer saved to {vectorizer_path}")
