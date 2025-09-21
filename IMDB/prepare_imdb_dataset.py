"""
Download and prepare the IMDB dataset for sentiment analysis.
This script only needs to be run once to create the dataset files.

Download and prepare the IMDB dataset for sentiment analysis.

Steps performed by this script:
1. Check if the raw IMDB dataset folder (`aclImdb/`) exists locally.
   - If not, download the official dataset archive from Stanford 
     (http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).
   - Extract it into a folder named `aclImdb/`.

2. Read all movie review text files from `aclImdb/train/pos` and `aclImdb/train/neg`.
   - Assign label = 1 for positive reviews, label = 0 for negative reviews.

3. Build a Pandas DataFrame with two columns:
   - "text" (the review content)
   - "label" (0 or 1 sentiment class)

4. Split the dataset into:
   - 90% training set
   - 5% validation set
   - 5% test set
   (Stratified to maintain positive/negative balance.)

5. Save the splits into CSV files:
   - imdb_train.csv
   - imdb_val.csv
   - imdb_test.csv

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

# Set random seed for reproducibility
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

def download_and_extract_imdb():
    """Download the IMDB dataset if not already present"""
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    if not os.path.exists("aclImdb"):
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
    
    # Load training data
    print("Processing training data...")
    for sentiment in ['pos', 'neg']:
        path = f'aclImdb/train/{sentiment}'
        label = 1 if sentiment == 'pos' else 0
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({'text': texts, 'label': labels})
    
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
    
    # Save to CSV files
    print("Saving datasets to CSV...")
    train_df.to_csv('imdb_train.csv', index=False)
    val_df.to_csv('imdb_val.csv', index=False)
    test_df.to_csv('imdb_test.csv', index=False)
    print("Done! Dataset is ready for use")

if __name__ == "__main__":
    download_and_extract_imdb()
    create_dataset()