# AQI Prediction — Comparative ML Analysis

Source code for the DS340W capstone paper *"Comparative Analysis of Machine Learning Models for Air Quality Index Prediction: Classical, Ensemble, and Deep Learning Approaches."*

This notebook trains and compares four regression models on the Indian Cities Air Quality dataset:

- **Linear Regression** (classical)
- **Random Forest Regressor** (ensemble — bagging)
- **Gradient Boosting Regressor** (ensemble — boosting)
- **Multi-Layer Perceptron** (deep learning)

Each model is evaluated using both regression metrics (MAE, RMSE, R²) and bucket-level classification metrics (Accuracy, Precision, Recall, F1) derived via post-hoc bucketization of continuous predictions into India's six AQI tiers.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Dataset Setup](#3-dataset-setup)
4. [Running the Notebook](#4-running-the-notebook)
5. [Expected Runtime and Outputs](#5-expected-runtime-and-outputs)
6. [Reproducibility Notes](#6-reproducibility-notes)
7. [Troubleshooting](#7-troubleshooting)
8. [Project Structure](#8-project-structure)

---

## 1. Requirements

- **Python 3.9 or newer** (Python 3.10 or 3.11 recommended; TensorFlow does not yet support Python 3.13 reliably)
- **~2 GB free disk space** (TensorFlow alone is roughly 500 MB)
- **No GPU required** — the entire notebook runs on CPU in 2–3 minutes on a modern laptop

### Python libraries used

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical arrays |
| scikit-learn | Linear Regression, Random Forest, Gradient Boosting, metrics, grid search, scaling |
| tensorflow / keras | Multi-Layer Perceptron |
| matplotlib | Plotting |
| seaborn | Heatmaps and styled plots |
| jupyter | Notebook interface |

---

## 2. Installation

Pick **one** of the three paths below. Option A (virtual environment) is recommended because it isolates dependencies from your system Python.

### Option A — Virtual environment (recommended)

**macOS / Linux:**
```bash
# Clone or download the repo, then cd into it
cd aqi-prediction

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip (avoids many resolution errors)
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
cd aqi-prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow jupyter
```

### Option B — Conda environment

```bash
conda create -n aqi python=3.11 -y
conda activate aqi
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow jupyter
```

> `pip install tensorflow` inside a conda env is generally more reliable than `conda install tensorflow`, which can pull older builds.

### Option C — Google Colab (zero install)

1. Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload notebook → select `Code.ipynb`.
2. Upload `city_day.csv` via the Files panel on the left sidebar (drag and drop, or click the upload icon).
3. Run all cells. All dependencies except `imbalanced-learn` are pre-installed; nothing else needs to be added.

---

## 3. Dataset Setup

The notebook expects a file named **`city_day.csv`** in the same directory as `Code.ipynb`.

**Source:** [Air Quality Data in India (Kaggle)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

The specific file referenced in the notebook is available from:
```
https://www.kaggle.com/code/anjusunilkumar/air-quality-index-prediction?select=city_day.csv
```

**Steps:**
1. Download `city_day.csv` from Kaggle (requires a free Kaggle account).
2. Place it in the same folder as `Code.ipynb`.

**Expected dataset:** 29,531 rows × 16 columns. After dropping rows missing the AQI target, approximately 24,850 rows remain.

---

## 4. Running the Notebook

### Launching Jupyter

From the directory containing `Code.ipynb` with your environment activated:
```bash
jupyter notebook
```
This opens Jupyter in your browser. Click `Code.ipynb` to open it.

Alternatively, if you prefer JupyterLab:
```bash
pip install jupyterlab
jupyter lab
```

### Execution order

**Run cells top-to-bottom.** The notebook assumes linear execution; later cells depend on variables defined earlier.

In Jupyter: `Cell → Run All`, or use `Shift + Enter` on each cell sequentially.

### What the notebook does, stage by stage

| Stage | Cells (approx.) | Output |
|---|---|---|
| Imports and data loading | 1–6 | Raw dataframe preview |
| Preprocessing (nulls, drop leakage columns) | 7–13 | Cleaned dataframe |
| EDA (correlation heatmap) | 14 | Figure: Feature Correlation Heatmap |
| Train-test split and scaling | 15–18 | `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test` |
| Baseline training (LR, RF, GB with defaults) | 19–23 | Baseline metrics table and bar chart |
| Feature importance (LR, RF, GB) | 24–27 | Three feature importance plots |
| `evaluate_classification()` function definition | 28 | (no visual output) |
| Tuned Linear Regression | 30–32 | Metrics, actual-vs-predicted plot, confusion matrix |
| Tuned Random Forest (GridSearchCV) | 33–36 | Best params, metrics, actual-vs-predicted plot, confusion matrix |
| Tuned Gradient Boosting (GridSearchCV) | 37–40 | Best params, metrics, actual-vs-predicted plot, confusion matrix |
| MLP (Keras Sequential) | 41–45 | Training curves, metrics, actual-vs-predicted plot, confusion matrix |
| Consolidated comparison table and viz | 46–49 | Final 4-model table + regression/classification bar charts |

---

## 5. Expected Runtime and Outputs

### Runtime (approximate, on a modern consumer laptop, CPU only)

| Stage | Time |
|---|---|
| Data loading and preprocessing | < 5 seconds |
| Baseline model training | ~15 seconds |
| Random Forest `GridSearchCV` | ~60 seconds |
| Gradient Boosting `GridSearchCV` | ~90 seconds |
| MLP training | ~30 seconds |
| Everything else | < 10 seconds |
| **Total** | **~3 minutes** |

### Expected final output

The last cell prints the consolidated comparison table. Reference values from a clean run with seed 42:

```
================================================================================
  Final Model Comparison (Tuned)
================================================================================
                         MAE    RMSE     R²  Accuracy  Precision  Recall     F1
Linear Regression     31.203  59.108  0.809     0.686      0.693   0.686  0.678
Random Forest         20.549  40.216  0.912     0.797      0.800   0.797  0.796
Gradient Boosting     21.627  40.376  0.911     0.773      0.776   0.773  0.769
Neural Network (MLP)  26.084  50.319  0.862     0.725      0.718   0.725  0.705
================================================================================
```

If your numbers differ, see [Section 6](#6-reproducibility-notes).

---

## 6. Reproducibility Notes

The notebook sets random seeds throughout to make results reproducible:

- `train_test_split(..., random_state=42)` — fixes the 80/20 split
- `RandomForestRegressor(random_state=42)` and `GradientBoostingRegressor(random_state=42)`
- `np.random.seed(42)` and `tf.random.set_seed(42)` before MLP construction

**Sklearn results should match exactly.** Linear Regression, Random Forest, and Gradient Boosting metrics will be bit-for-bit identical across runs on the same machine.

**MLP results may vary slightly** across platforms or TensorFlow versions because of:
- Non-deterministic CUDA kernels (not applicable if running CPU-only)
- Non-deterministic CPU thread scheduling for certain TensorFlow operations
- Floating-point ordering differences between BLAS backends

These variations are typically in the 1–2% range on any given metric. The MLP ranking (fourth among the four models) is stable.

---

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'tensorflow'`

TensorFlow is not installed in the currently active Python environment. Run:
```bash
pip install tensorflow
```
Then restart the Jupyter kernel: `Kernel → Restart`.

### `ModuleNotFoundError: No module named 'sklearn'`

Install scikit-learn:
```bash
pip install scikit-learn
```
Note the package name on PyPI is `scikit-learn`, but it is imported as `sklearn`.

### `FileNotFoundError: [Errno 2] No such file or directory: 'city_day.csv'`

The dataset is missing from the notebook's working directory. Download it from Kaggle (see [Section 3](#3-dataset-setup)) and place it next to `Code.ipynb`. You can verify the working directory by running `!pwd` (macOS/Linux) or `!cd` (Windows) in a notebook cell.

### TensorFlow prints CUDA / GPU warnings at import

Example messages:
```
Could not load dynamic library 'libcudart.so.11.0'
Unable to register cuDNN factory...
```
These are **informational only** and can be ignored. They mean TensorFlow looked for GPU libraries, did not find them, and silently fell back to CPU. The MLP trains fine on CPU for this dataset size.

### TensorFlow installation fails on Python 3.13

TensorFlow's support for Python 3.13 is still rolling out. Use Python 3.10 or 3.11. With `pyenv`:
```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install tensorflow
```

### Apple Silicon (M1/M2/M3) installation issues

The standard `pip install tensorflow` works on Apple Silicon in 2025+. If you hit issues:
```bash
pip install tensorflow-macos
```
(Note: the older `tensorflow-metal` add-on for GPU acceleration is not required — CPU is fast enough for this notebook.)

### `Kernel died, restarting` during MLP training

Usually caused by insufficient RAM. Close other applications, then restart the kernel. If it continues, reduce the MLP batch size from 32 to 16 in the MLP training cell.

### Different tuned hyperparameters from the printed best-params

GridSearchCV selects the configuration with the highest mean CV R². If scikit-learn's scoring or CV implementation changes across versions, the selected config can shift. The final test-set metrics should still be close to the reference values in Section 5. To pin scikit-learn:
```bash
pip install scikit-learn==1.5.0
```

### Plots not showing in Jupyter

Add `%matplotlib inline` to the top of the imports cell, then restart the kernel.

---

## 8. Project Structure

```
aqi-prediction/
├── Code.ipynb             # Main notebook (run this)
├── city_day.csv           # Dataset (download separately from Kaggle)
├── README.md              # This file
└── requirements.txt       # Optional: pinned dependency list
```

### Suggested `requirements.txt`

Create a file named `requirements.txt` next to the notebook with the following contents:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
tensorflow>=2.15
jupyter>=1.0
```

Install everything at once with:
```bash
pip install -r requirements.txt
```

---

## Author

Kartik Gupta — College of Engineering, The Pennsylvania State University
kfg5401@psu.edu
