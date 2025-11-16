"""
Download and prepare the IMDB dataset for sentiment analysis.

Steps performed by this script:
1. Check if the raw IMDB dataset folder (`aclImdb/`) exists locally.
   - If not, download the official dataset archive from Stanford 
     (http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).
   - Extract it into a folder named `aclImdb/`.

2. Read all movie review text files from both train and test directories:
   - `aclImdb/train/pos`, `aclImdb/train/neg` (25,000 reviews)
   - `aclImdb/test/pos`, `aclImdb/test/neg` (25,000 reviews)
   - Total: 50,000 reviews
   - Assign label = 1 for positive reviews, label = 0 for negative reviews.

3. Build a Pandas DataFrame with two columns:
   - "text" (the review content)
   - "label" (0 or 1 sentiment class)

4. Split the combined 50k dataset into:
   - 90% training set (~45,000 reviews)
   - 5% validation set (~2,500 reviews)
   - 5% test set (~2,500 reviews)
   (Stratified to maintain positive/negative balance.)

5. Save the splits into CSV files:
   - data/imdb_train.csv
   - data/imdb_val.csv
   - data/imdb_test.csv

This script only needs to be run once. After that, the CSV files
can be used directly by other programs (e.g., logistic regression,
CNN, or transformer training scripts).
"""

import os
import requests
import tarfile
from io import BytesIO
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import re

# Set random seed for reproducibility
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

def clean_text(text):
    """Remove HTML tags and clean up text while preserving sentiment-relevant content"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Keep important contractions and apostrophes by protecting them temporarily
    text = re.sub(r"n't", " not", text)  # don't -> do not, can't -> can not
    text = re.sub(r"'re", " are", text)  # you're -> you are
    text = re.sub(r"'ve", " have", text) # I've -> I have
    text = re.sub(r"'ll", " will", text) # I'll -> I will
    text = re.sub(r"'d", " would", text) # I'd -> I would
    text = re.sub(r"'m", " am", text)    # I'm -> I am
    text = re.sub(r"'s", " is", text)    # it's -> it is, that's -> that is
    
    # Remove standalone numbers (but keep words with numbers like "great2see")
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Remove most punctuation but keep emoticon-like patterns and sentence structure
    # Keep: letters, spaces, basic punctuation that affects sentiment
    text = re.sub(r'[^\w\s!?.]', ' ', text)
    
    # Normalize repeated punctuation (but preserve intensity for sentiment)
    text = re.sub(r'\.{2,}', '.', text)  # multiple dots -> single dot
    
    # Remove standalone single characters except 'i' and 'a'
    text = re.sub(r'\b[b-hj-z]\b', ' ', text)  # Remove single letters except i and a
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def download_and_extract_imdb():
    """Download the IMDB dataset if not already present"""
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    if not os.path.exists("data/aclImdb"):
        print("Downloading IMDB dataset...")
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=BytesIO(response.content), mode="r:gz")
        file.extractall()
        file.close()
        print("Download complete and extracted")
    else:
        print("IMDB dataset already downloaded")

def create_dataset():
    """Create pandas DataFrames from the raw IMDB data"""
    texts = []
    labels = []
    
    # Load both training and test data (50k total reviews)
    print("Processing all IMDB data (train + test)...")
    for data_split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            path = f'data/aclImdb/{data_split}/{sentiment}'
            label = 1 if sentiment == 'pos' else 0
            for filename in os.listdir(path):
                if filename.endswith('.txt'):
                    with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                        text = clean_text(f.read())
                        texts.append(text)
                        labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({'text': texts, 'label': labels})
    print(f"Total reviews loaded: {len(df)}")
    
    # Split into train, validation, and test sets
    # First split: 90% train, 10% temp (which will be split into val and test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.1, 
        stratify=df['label'],
        random_state=SEED
    )
    
    # Split temp into validation and test (50% each, resulting in 5% of total each)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=SEED
    )
    
    print(f"Dataset splits: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Train: {len(train_df)/len(df)*100:.1f}%, Val: {len(val_df)/len(df)*100:.1f}%, Test: {len(test_df)/len(df)*100:.1f}%")
    
    # Save to CSV files
    print("Saving datasets to CSV...")
    train_df.to_csv('data/imdb_train.csv', index=False)
    val_df.to_csv('data/imdb_val.csv', index=False)
    test_df.to_csv('data/imdb_test.csv', index=False)
    print("Done! Dataset is ready for use")

if __name__ == "__main__":
    download_and_extract_imdb()
    create_dataset()