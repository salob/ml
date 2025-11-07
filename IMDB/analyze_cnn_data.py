import pandas as pd
import torch
from imdb_cnn import Vocabulary, preprocess_text, MAX_LENGTH, MAX_WORDS, EMBEDDING_DIM

# 1. Load a few examples
print("1. Loading raw examples...")
train_df = pd.read_csv('imdb_train.csv')
examples = train_df['text'].values[:3]  # Get first 3 reviews
labels = train_df['label'].values[:3]

print("\nOriginal text examples:")
for i, (text, label) in enumerate(zip(examples, labels)):
    print(f"\nExample {i+1} (Label: {label}):")
    print(f"Length: {len(text.split())} words")
    print(text[:200] + "...")  # First 200 characters

# 2. Preprocessing
print("\n2. After preprocessing:")
processed_texts = [preprocess_text(text) for text in examples]
for i, text in enumerate(processed_texts):
    print(f"\nExample {i+1} processed:")
    print(f"Length: {len(text.split())} words")
    print(text[:200] + "...")

# 3. Vocabulary Building
print("\n3. Vocabulary statistics:")
vocab = Vocabulary(max_size=MAX_WORDS)
vocab.build_vocab(processed_texts)
print(f"Vocabulary size: {len(vocab.idx2word)}")
print("\nFirst 10 words in vocabulary:")
for i, word in enumerate(vocab.idx2word[:10]):
    print(f"{i}: {word}")

# 4. Converting to sequences
print("\n4. Sequences (word indices):")
for i, text in enumerate(processed_texts):
    sequence = vocab.text_to_sequence(text, MAX_LENGTH)
    print(f"\nExample {i+1} as sequence:")
    print(f"Sequence length: {len(sequence)}")
    print(f"First 20 indices: {sequence[:20]}")
    # Show what these indices represent
    words = [vocab.idx2word[idx] if idx < len(vocab.idx2word) else '<unk>' for idx in sequence[:20]]
    print(f"As words: {' '.join(words)}")

# 5. Final tensor shape
print("\n5. Final tensor shapes:")
sequences = [vocab.text_to_sequence(text, MAX_LENGTH) for text in processed_texts]
batch = torch.tensor(sequences, dtype=torch.long)
print(f"Batch shape: {batch.shape}")  # Should be [3, MAX_LENGTH]

# Show padding statistics
padding_counts = [(sequence == 0).sum() for sequence in sequences]
print("\nPadding statistics:")
for i, count in enumerate(padding_counts):
    print(f"Example {i+1}: {count} padding tokens ({count/MAX_LENGTH*100:.1f}% of sequence)")