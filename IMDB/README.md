# IMDB Sentiment Analysis

This project implements and compares two different approaches for sentiment analysis on the IMDB movie reviews dataset:
1. Logistic Regression with TF-IDF
2. Convolutional Neural Network (CNN) with Word Embeddings

## Project Structure
- `prepare_imdb_dataset.py`: Downloads and prepares the IMDB dataset, creating CSV files
- `imdb_logreg.py`: Logistic regression implementation
- `imdb_cnn.py`: CNN implementation

## Setting Up the Environment with uv

This project uses `uv` for faster package management. If you haven't installed uv yet, you can install it using Homebrew:
```bash
brew install uv
```

### Creating a New Virtual Environment
```bash
# Create and activate a new virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Managing Dependencies
- Add a new package:
  ```bash
  uv pip install package_name
  ```
- Update requirements.txt:
  ```bash
  uv pip freeze > requirements.txt
  ```
- Remove a package:
  ```bash
  uv pip uninstall package_name
  ```

### Key Dependencies
- PyTorch: Deep learning framework (CNN model)
- scikit-learn: Machine learning tools (Logistic Regression, TF-IDF)
- pandas: Data manipulation
- numpy: Numerical computations

## Running the Models

1. First, prepare the dataset:
   ```bash
   python prepare_imdb_dataset.py
   ```

2. Run the logistic regression model:
   ```bash
   python imdb_logreg.py
   ```

3. Run the CNN model:
   ```bash
   python imdb_cnn.py
   ```

Each model will print its training progress and final evaluation metrics for comparison.

## Performance Notes
- Logistic Regression: Fast to train, good baseline performance
- CNN: More complex, potentially better accuracy, longer training time