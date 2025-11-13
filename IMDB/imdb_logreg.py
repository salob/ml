# imdb_logreg.py
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from codecarbon import EmissionsTracker

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

# Start emissions tracking
tracker = EmissionsTracker(project_name="IMDB_LogReg")
tracker.start()

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
tracker.stop()
print("\nEmissions tracking complete. Check emissions.csv for results.")
