#!/usr/bin/env python3
"""
Export vocabularies and vectorizer features to CSV.

Outputs (if artifacts exist):
 - reports/cnn_vocab.csv (index,token)
 - reports/transformer_vocab.csv (index,token)
 - reports/logreg_vectorizer_features.csv (index,token,ngram_len)

Usage: python scripts/export_vocab.py
"""

import os
import csv
import pickle
from pathlib import Path


MODELS_DIR = Path('models')
REPORTS_DIR = Path('reports')

# Minimal placeholder class so pickle can resolve __main__.Vocabulary
class Vocabulary:  # noqa: N801 - match saved class name
    def __init__(self, *args, **kwargs):
        pass


def export_nn_vocab(pickle_path: Path, out_csv: Path) -> bool:
    """Load a pickled Vocabulary and export index/token mapping to CSV.

    Returns True on success, False if file missing or load failed.
    """
    if not pickle_path.exists():
        print(f"[skip] Not found: {pickle_path}")
        return False

    try:
        with open(pickle_path, 'rb') as f:
            vocab = pickle.load(f)
    except Exception as e:
        print(f"[error] Failed to load {pickle_path.name}: {e}")
        return False

    idx2word = getattr(vocab, 'idx2word', None)
    if not isinstance(idx2word, list):
        print(f"[error] {pickle_path.name} has no idx2word list; can't export")
        return False

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'token'])
        for i, tok in enumerate(idx2word):
            writer.writerow([i, tok])

    print(f"[ok] Wrote {out_csv} ({len(idx2word):,} rows)")
    return True


def export_vectorizer(vectorizer_path: Path, out_csv: Path) -> bool:
    """Load a pickled TfidfVectorizer and export index/token (+ngram_len)."""
    if not vectorizer_path.exists():
        print(f"[skip] Not found: {vectorizer_path}")
        return False

    try:
        with open(vectorizer_path, 'rb') as f:
            vec = pickle.load(f)
    except Exception as e:
        print(f"[error] Failed to load {vectorizer_path.name}: {e}")
        return False

    vocab = getattr(vec, 'vocabulary_', None)
    if not isinstance(vocab, dict):
        print(f"[error] {vectorizer_path.name} has no vocabulary_ dict; can't export")
        return False

    # Invert to index -> token for ordered export
    inv = {idx: tok for tok, idx in vocab.items()}
    max_idx = max(inv.keys()) if inv else -1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'token', 'ngram_len'])
        for i in range(max_idx + 1):
            tok = inv.get(i)
            if tok is None:
                # Some vectorizers can have sparse indices; skip gaps safely
                continue
            ngram_len = tok.count(' ') + 1  # crude but effective for (1,2)
            writer.writerow([i, tok, ngram_len])

    print(f"[ok] Wrote {out_csv} ({len(inv):,} rows)")
    return True


def main():
    targets = [
        (MODELS_DIR / 'cnn_vocab.pkl', REPORTS_DIR / 'cnn_vocab.csv', export_nn_vocab),
        (MODELS_DIR / 'transformer_vocab.pkl', REPORTS_DIR / 'transformer_vocab.csv', export_nn_vocab),
        (MODELS_DIR / 'logreg_vectorizer.pkl', REPORTS_DIR / 'logreg_vectorizer_features.csv', export_vectorizer),
    ]

    any_done = False
    for src, dst, fn in targets:
        if fn(src, dst):
            any_done = True

    if not any_done:
        print("No outputs were generated. Ensure models/ artifacts exist.")


if __name__ == '__main__':
    main()
