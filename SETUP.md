# Python Environment Setup Guide

This guide explains how to set up a Python virtual environment and install the required dependencies for this machine learning project using **uv** - a fast Python package installer written in Rust.

## Prerequisites

- Python 3.8 or higher installed on your system
- uv package installer (see installation below)

## Installing uv

If you don't have uv installed yet, use one of these methods that **don't pollute your system Python**:

**On macOS/Linux (Recommended):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This installs uv as a standalone binary in `~/.cargo/bin/` (no Python packages in your system).

**Or using Homebrew:**
```bash
brew install uv
```

This installs uv as a system tool, keeping your Python environment clean.

**Note:** Avoid `pip install uv` as it defeats the purpose of keeping your system Python clean! The curl and Homebrew methods install uv as a standalone tool that doesn't touch your Python installation.

## Setup Steps (Using uv)

### 1. Create a Virtual Environment

Navigate to the project directory and create a virtual environment using uv:

```bash
uv venv
```

This creates a `.venv` directory in your project.

### 2. Activate the Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

You should see `(.venv)` prefix in your terminal prompt, indicating the virtual environment is active.

### 3. Install Required Packages

With uv, you can install all dependencies much faster:

```bash
uv pip install -r requirements.txt
```

This will install:
- **numpy** - Numerical computing library
- **pandas** - Data manipulation and analysis
- **seaborn** - Statistical data visualization
- **matplotlib** - Plotting and visualization
- **scikit-learn** - Machine learning algorithms and tools
- **codecarbon** - Carbon emissions tracking for ML models

**Why uv?** It's 10-100x faster than pip, written in Rust, and fully compatible with pip.

### 4. Verify Installation

Check that all packages are installed correctly:

```bash
uv pip list
```

You should see all the required packages listed with their versions.

## Alternative: Traditional pip Setup

If you prefer using pip instead of uv:

### 1. Create Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure VS Code Python Interpreter

1. Open the Command Palette (`Cmd+Shift+P` on macOS)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your `.venv` folder:
   - Should show something like `./.venv/bin/python`

## Working with the Virtual Environment

### Activating the Environment

Always activate the virtual environment before working on the project:

```bash
source .venv/bin/activate  # macOS/Linux
```

### Deactivating the Environment

When you're done working:

```bash
deactivate
```

### Adding New Packages with uv

If you install new packages, update the `requirements.txt`:

**Using uv:**
```bash
uv pip install <package-name>
uv pip freeze > requirements.txt
```

**Using pip:**
```bash
pip install <package-name>
pip freeze > requirements.txt
```

## Troubleshooting

### Import Errors

If you see "Import could not be resolved" errors in VS Code:

1. Ensure the virtual environment is activated
2. Verify the correct Python interpreter is selected in VS Code
3. Reload the VS Code window (`Cmd+Shift+P` → "Developer: Reload Window")

### Permission Errors

If you encounter permission errors during installation:

```bash
pip install --user -r requirements.txt
```

### Upgrading pip

If you see warnings about an outdated pip version:

**Using uv:**
```bash
uv pip install --upgrade pip
```

**Using pip:**
```bash
pip install --upgrade pip
```

## uv-specific Features

### Sync Dependencies

Keep your environment in sync with `requirements.txt`:

```bash
uv pip sync requirements.txt
```

This removes packages not in requirements.txt and installs missing ones.

### Compile Requirements

For reproducible builds, create a lock file:

```bash
uv pip compile requirements.txt -o requirements.lock
uv pip sync requirements.lock
```

### Speed Comparison

uv is significantly faster than pip:
- **10-100x faster** installation
- **Written in Rust** for performance
- **Drop-in replacement** for pip commands
- **Parallel downloads** and installations

## Project Structure

```
ml/
├── .venv/                         # Virtual environment (not tracked in git)
├── requirements.txt               # Python dependencies
├── HousingPricePredictor/        # Housing price prediction project
├── IMDB/                         # IMDB sentiment analysis projects
└── README.md                     # Main project documentation
```

## CodeCarbon Configuration

This project uses CodeCarbon to track carbon emissions from ML model training. The configuration is stored in `.codecarbon.config` and emissions data is saved to `emissions.csv`.

**Note:** CodeCarbon may request your password on macOS to access PowerMetrics for accurate CPU/GPU power tracking.

## Best Practices

1. **Use uv for faster installations** - Especially beneficial for large projects with many dependencies
2. **Always use the virtual environment** - This ensures consistent dependencies across different machines
3. **Don't commit the .venv folder** - Add `.venv/` or `venv/` to your `.gitignore`
4. **Keep requirements.txt updated** - When adding new packages, update the requirements file
5. **Use specific versions** - For production, pin exact versions in `requirements.txt`
6. **Consider using uv pip compile** - Creates a lock file for fully reproducible environments

## Quick Reference Commands

| Task | uv Command | pip Command |
|------|------------|-------------|
| Create venv | `uv venv` | `python3 -m venv venv` |
| Install packages | `uv pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Install single package | `uv pip install <package>` | `pip install <package>` |
| List packages | `uv pip list` | `pip list` |
| Freeze requirements | `uv pip freeze > requirements.txt` | `pip freeze > requirements.txt` |
| Sync environment | `uv pip sync requirements.txt` | N/A |
| Compile lock file | `uv pip compile requirements.txt` | N/A |

## Next Steps

Once your environment is set up:

1. Navigate to a project directory (e.g., `HousingPricePredictor/`)
2. Run the Python scripts:
   ```bash
   python HousePricePredictor.py
   ```
3. View the generated visualizations and outputs

## Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [pip documentation](https://pip.pypa.io/en/stable/)
- [VS Code Python environments](https://code.visualstudio.com/docs/python/environments)
- [Astral (uv creators)](https://astral.sh/)
