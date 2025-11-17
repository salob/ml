# ML Energy Efficiency Research

A comparative study of machine learning model complexity versus energy consumption and carbon emissions for sentiment analysis on the IMDB dataset.

## Project Overview

This project implements and compares three different machine learning approaches for sentiment analysis, with a focus on measuring their energy efficiency and carbon footprint:

1. **Logistic Regression with TF-IDF** - Simple baseline model
2. **Convolutional Neural Network (CNN)** - Moderate complexity deep learning
3. **Transformer** - High complexity deep learning

The project uses **CodeCarbon** as the primary carbon tracking tool, with **CarbonTracker** serving as a validation control to ensure measurement consistency. This dual-tracking approach provides confidence in the relative emissions patterns across models, even though the absolute values differ due to methodology variations (PUE coefficients, carbon intensity calculations, and per-epoch vs. overall averaging).

## Key Findings

- **Best Accuracy**: Logistic Regression (88.9%) outperformed deep learning models despite being the simplest
- **Energy Consumption**: Transformer models consumed significantly more energy (~4-6 Wh) compared to LogReg (~0.02 Wh)
- **Measurement Validation**: CarbonTracker served as a control to validate CodeCarbon's consistency. While CarbonTracker reported ~2.5x higher absolute values (due to PUE=1.58, different carbon intensity, and per-epoch summation vs. overall averaging), both tools showed identical relative patterns across runs, confirming CodeCarbon's measurements are reliable and not anomalous
- **Early Stopping**: Implemented across all models to prevent overfitting and reduce unnecessary computation

## Repository Structure

```
ml/
├── IMDB/                          # Model implementations
│   ├── imdb_logreg.py            # Logistic Regression with TF-IDF
│   ├── imdb_cnn.py               # Convolutional Neural Network
│   └── imdb_transformer.py       # Transformer model
│
├── scripts/                       # Utility and automation scripts
│   ├── prepare_imdb_dataset.py   # Dataset preparation and CSV creation
│   ├── run_multiple_experiments.py # Automated experiment runner
│   ├── parse_carbontracker_logs.py # Parse CarbonTracker log files
│   ├── convert_emissions.py      # Convert CodeCarbon data to readable units
│   ├── generate_emissions_report.py # Create interactive HTML reports
│   ├── helper_calculate_emissions.py # Summary statistics per model
│   └── helper_plot_convergence.py # Plot per-epoch training curves
│
├── data/                          # Dataset files (CSV format)
│   ├── imdb_train.csv
│   ├── imdb_val.csv
│   └── imdb_test.csv
│
├── logs/                          # Carbon tracking logs and outputs
│   ├── emissions_readable.csv    # Processed CodeCarbon data
│   └── carbontracker_readable.csv # Processed CarbonTracker data
│
├── reports/                       # Generated HTML reports
│   ├── emissions_report.html     # Interactive emissions visualization
│   └── convergence_analysis.png  # Training convergence plots
│
├── docs/                          # Additional documentation
│   ├── README.md                 # Detailed setup instructions
│   └── SETUP.md
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Getting Started

### Prerequisites

- Python 3.13+
- `uv` package manager (recommended) or `pip`

### Installation

1. **Install uv** (if not already installed):
   ```bash
   brew install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ml
   ```

3. **Create and activate virtual environment**:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

### Key Dependencies

- **PyTorch** (>=2.8.0) - Deep learning framework
- **scikit-learn** (>=1.7.0) - Traditional ML algorithms
- **pandas** (>=2.3.0) - Data manipulation
- **CodeCarbon** (>=3.0.0) - Carbon emissions tracking
- **CarbonTracker** (>=2.4.2) - Alternative carbon tracking
- **matplotlib** (>=3.10.0) & **seaborn** (>=0.13.0) - Visualization

## Running Experiments

### Standard Workflow (Recommended Order)

Run the following scripts in sequence for a complete experimental workflow:

#### 1. Prepare the Dataset
```bash
python scripts/prepare_imdb_dataset.py
```
**Purpose**: Downloads the IMDB dataset and creates train/validation/test CSV splits (80/10/10).  
**Output**: `data/imdb_train.csv`, `data/imdb_val.csv`, `data/imdb_test.csv`

#### 2. Run Multiple Experiments
```bash
python scripts/run_multiple_experiments.py
```
**Purpose**: Executes 5 runs of each model (LogReg, CNN, Transformer) with carbon tracking enabled.  
**Output**: 
- CodeCarbon logs in `logs/emissions.csv`
- CarbonTracker logs in `logs/ct_*_carbontracker_output.log`
- Training metrics and model checkpoints

**Note**: This is the most time-consuming step. Total runtime:
- LogReg: ~10 seconds × 5 = ~50 seconds
- CNN: ~3-4 minutes × 5 = ~15-20 minutes  
- Transformer: ~20-30 minutes × 5 = ~1.5-2.5 hours

#### 3. Parse CarbonTracker Logs
```bash
python scripts/parse_carbontracker_logs.py
```
**Purpose**: Extracts energy and emissions data from CarbonTracker log files, applies PUE adjustments.  
**Output**: `logs/carbontracker_readable.csv`

#### 4. Convert CodeCarbon Emissions
```bash
python scripts/convert_emissions.py
```
**Purpose**: Converts CodeCarbon data from kg/kWh to g/Wh for easier comparison.  
**Output**: `logs/emissions_readable.csv`

#### 5. Generate Emissions Report
```bash
python scripts/generate_emissions_report.py
```
**Purpose**: Creates an interactive HTML report comparing both carbon tracking tools.  
**Output**: `reports/emissions_report.html`

**View the report**:
```bash
open reports/emissions_report.html
```

### Helper Scripts

These scripts provide additional analysis but are not part of the main workflow:

#### Calculate Emissions Summary
```bash
python scripts/helper_calculate_emissions.py
```
**Purpose**: Prints summary statistics of emissions per model to the console.  
**Use Case**: Quick overview of total energy consumption and CO₂ emissions.

#### Plot Convergence Analysis
```bash
python scripts/helper_plot_convergence.py
```
**Purpose**: Generates per-epoch loss and accuracy curves to analyze overfitting and convergence.  
**Output**: `reports/convergence_analysis.png`  
**Use Case**: Understanding training dynamics and validating early stopping effectiveness.

## Individual Model Execution

You can also run individual models directly:

```bash
# Logistic Regression (~10 seconds)
python IMDB/imdb_logreg.py

# CNN (~3-4 minutes)
python IMDB/imdb_cnn.py

# Transformer (~20-30 minutes, varies with early stopping)
python IMDB/imdb_transformer.py
```

**Note**: Individual runs will still log emissions data to the tracking systems.

## Interactive Demo

After training your models, you can test them with your own movie reviews!

### Command-Line Demo

```bash
python scripts/interactive_demo.py
```

### Web Demo (Share with Classmates!)

```bash
python scripts/web_demo.py
```

This launches a web interface powered by Gradio where you can:
- Enter film reviews and get instant predictions
- See results from all three models
- Try example reviews with one click
- Share a public link with classmates (set `share=True` in the code, this feature is provided by Gradio out of the box!)

**Deployment Options:**
- **Quick Share**: Temporary 72-hour public link (set `share=True`)
- **Hugging Face Spaces**: Free permanent hosting ([guide](docs/DEPLOYMENT.md), haven't tried or implemented this as of yet)

### Demo Features

Both demos will:
- Load all available trained models (LogReg, CNN, Transformer)
- Let you enter custom film reviews
- Show predictions from all models along with confidence scores
- Display sentiment (Positive/Negative) with probabilities

**Requirements**: You must first train the models using the individual model scripts or `run_multiple_experiments.py`. The demo looks for model files in the `models/` directory.

## Understanding the Reports

### Emissions Report (`emissions_report.html`)

coming soon

**Key Metrics**:
- **Energy (Wh)**: Watt-hours consumed during training
- **CO₂ Emissions (g)**: Grams of CO₂ equivalent emitted
- **PUE-adjusted values**: CarbonTracker values normalised to PUE=1.0 for fair comparison

### Convergence Analysis (`convergence_analysis.png`)

Shows per-epoch training and validation metrics:
- **Loss curves**: Training vs validation loss over epochs
- **Accuracy curves**: Training vs validation accuracy
- **Early stopping markers**: Visual indication of when training stopped

## Carbon Tracking Methodology

This project uses **CodeCarbon as the primary carbon tracking tool**, with **CarbonTracker as a validation control** to ensure the emissions patterns are consistent and CodeCarbon measurements are reliable.

### CodeCarbon (Primary Tool)
- **PUE**: 1.0 (assumes local machine, no data center overhead)
- **Carbon Intensity**: ~290.8 gCO₂/kWh (Ireland grid)
- **Tracking**: CPU, GPU, and RAM energy consumption
- **Calculation**: Overall average energy consumption across entire training run

### CarbonTracker (Validation Control)
- **PUE**: 1.58 (default, assumes data center infrastructure)
- **Carbon Intensity**: 279.7 gCO₂/kWh (Ireland grid)
- **Tracking**: Actual consumption from output logs
- **Calculation**: Sums energy per epoch rather than overall averaging

**Why the difference?**  
CarbonTracker reports ~2.5x higher raw emissions primarily due to:
1. **PUE multiplier** (1.58 vs 1.0) - accounts for most of the difference
2. **Different carbon intensity values** (~290.8 vs 279.7 gCO₂/kWh)
3. **Calculation methodology** - per-epoch summation vs. overall averaging

**Validation Result**: Despite the numerical differences, both tools show **consistent relative patterns** across models and runs, confirming that CodeCarbon's measurements accurately reflect the comparative energy efficiency of different model architectures. CarbonTracker serves as a sanity check to ensure CodeCarbon data isn't anomalous or inconsistent.

## Experimental Configuration

- **Dataset Size**: 50,000 reviews (combined train+test from original IMDB dataset)
- **Training Set**: 40,000 reviews (80%)
- **Validation Set**: 5,000 reviews (10%)
- **Test Set**: 5,000 reviews (10%)
- **Early Stopping**: 
  - Max epochs: 25
  - Patience: 3
  - Min epochs: 5
- **Runs per Model**: 5

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- **IMDB Dataset**: Large Movie Review Dataset (Maas et al., 2011)
- **CodeCarbon**: Open-source tool for tracking CO₂ emissions
- **CarbonTracker**: Energy and carbon footprint tracker for deep learning ML training
- **Chart.js**: Interactive visualisations in the emissions report
- **Gradio**: Quick deploy web ui for demoing the models
- **scikit-learn and PyTorch**: Python libraries used for building, training, testing and deploying the models
