<div align="center">

# 🪨 Facies Classification using Machine Learning

### Automated Lithofacies Prediction from Well Log Data | End-to-End ML Pipeline for Subsurface Reservoir Characterization

<br>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC4E20?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

<br>

> **Bridging geoscience domain knowledge with modern machine learning to automate geological facies prediction from wireline well log data — enabling faster, reproducible, and data-driven reservoir characterization.**

<br>

[🚀 Quick Start](#-installation) · [📓 Notebooks](#-notebooks) · [📊 Results](#-results--model-performance) · [🗂️ Structure](#️-repository-structure) · [👨‍💻 Author](#-author)

---

</div>

## 📌 Project Overview

Facies classification is a foundational step in **subsurface reservoir characterization** — the process of identifying rock types (lithofacies) from indirect geophysical measurements acquired in boreholes. Traditionally, this involves time-intensive manual interpretation by petrophysicists.

This project replaces manual facies identification with a **fully automated, ML-driven pipeline** trained on multi-log well data. By learning petrophysical patterns in Gamma Ray, Resistivity, Density, Neutron Porosity, and Sonic logs, the models predict lithofacies labels with high accuracy — even in wells where facies labels are missing.

### What makes this project unique:
- 🔁 **Dual workflow** — both supervised classification and unsupervised clustering approaches implemented
- 🧪 **Missing PE log imputation** — handles the common industry problem of incomplete log suites
- 📐 **Modular `src/` architecture** — reusable utility functions for classification and data preparation
- 🎯 **Validation on blind wells** — tested on `validation_data_nofacies.csv` with no labels for realistic evaluation

---

## 🎯 Objectives

- Preprocess and clean multi-well log data, handling missing values (including PE log imputation)
- Engineer domain-informed features to improve ML model discrimination between facies
- Train, tune, and compare **Random Forest**, **SVM**, and **XGBoost** classifiers
- Apply unsupervised clustering as an independent facies grouping strategy
- Evaluate models rigorously using cross-validation, F1-score, and confusion matrices
- Predict facies on unlabeled well intervals and visualize results as geologic log tracks

---

## 🗃️ Dataset Description

The dataset is sourced from the **[SEG 2016 Machine Learning Contest](https://github.com/seg/2016-ml-contest)** — a landmark open benchmark in geophysical ML — and consists of wireline log measurements from multiple wells in a clastic reservoir system.

### Well Log Features

| Log | Symbol | Range | Geological Significance |
|-----|--------|-------|------------------------|
| Gamma Ray | GR | 0–300 API | Shale volume estimation; separates sands from shales |
| Resistivity | ILD | 0.2–2000 Ω·m | Hydrocarbon vs. brine saturation; fluid typing |
| Bulk Density | RHOB | 1.65–2.85 g/cc | Porosity estimation; primary lithology indicator |
| Neutron Porosity | NPHI | -0.15–0.60 v/v | Porosity; gas-effect detection when crossed with RHOB |
| Sonic | DT | 40–190 µs/ft | Porosity; mechanical properties; seismic-to-well tie |
| Photoelectric Factor | PE | 1.0–6.5 b/e⁻ | Mineralogy discrimination (partial — imputed in pipeline) |
| Non-Marine Indicator | NM_M | Binary | Depositional environment flag |
| Relative Position | RELPOS | 0–1 | Stratigraphic position within formation |

### Facies Classes

| Code | Facies | Abbreviation |
|------|--------|--------------|
| 1 | Nonmarine sandstone | SS |
| 2 | Nonmarine coarse siltstone | CSiS |
| 3 | Nonmarine fine siltstone | FSiS |
| 4 | Marine siltstone and shale | SiSh |
| 5 | Mudstone (limestone) | MS |
| 6 | Wackestone (limestone) | WS |
| 7 | Dolomite | D |
| 8 | Packstone-grainstone (limestone) | PS |
| 9 | Phylloid-algal bafflestone | BS |

### Dataset Files

| File | Description |
|------|-------------|
| `data/input_pilot_data.csv` | Raw multi-well log data with facies labels |
| `data/final-data.csv` | Cleaned, feature-engineered dataset |
| `data/predicted_missing_pe.csv` | PE log values imputed via regression |
| `data/train_set_df.pickled` | Serialized training set |
| `data/test_set_df.pickled` | Serialized test set |
| `data/validation_data_nofacies.csv` | Blind validation well — no labels |
| `data/podu_data.xlsx` | Supplementary formation data |

---

## ⚙️ Machine Learning Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW WELL LOG DATA                           │
│              input_pilot_data.csv + podu_data.xlsx              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
          ╔══════════════════════════╗
          ║   1. DATA PREPARATION    ║   src/data_prep.py
          ║  ─────────────────────  ║
          ║  • Null value analysis   ║
          ║  • PE log imputation     ║   → predicted_missing_pe.csv
          ║  • Outlier handling      ║
          ║  • Normalization         ║
          ╚══════════════╦═══════════╝
                         │
                         ▼
          ╔══════════════════════════╗
          ║  2. FEATURE ENGINEERING  ║   notebooks/preprocessing.ipynb
          ║  ─────────────────────  ║
          ║  • Depth gradient logs   ║
          ║  • Cross-plot ratios     ║
          ║    (RHOB/NPHI, GR/ILD)   ║
          ║  • Adjacency features    ║
          ╚══════════════╦═══════════╝
                         │
               ┌─────────┴──────────┐
               ▼                    ▼
   ╔═══════════════════╗  ╔═══════════════════════╗
   ║  3A. SUPERVISED   ║  ║  3B. UNSUPERVISED      ║
   ║  CLASSIFICATION   ║  ║  CLUSTERING            ║
   ║  ─────────────── ║  ║  ─────────────────────║
   ║  • Random Forest  ║  ║  • K-Means             ║
   ║  • SVM (RBF)      ║  ║  • Agglomerative       ║
   ║  • XGBoost        ║  ║  • DBSCAN              ║
   ║                   ║  ║                        ║
   ║  notebooks/       ║  ║  notebooks/            ║
   ║  supervised/      ║  ║  unsupervised/         ║
   ╚═════════╦═════════╝  ╚═══════════╦════════════╝
             │                        │
             └─────────┬──────────────┘
                       ▼
          ╔══════════════════════════╗
          ║    4. EVALUATION         ║   src/classification_utilities.py
          ║  ─────────────────────  ║
          ║  • Accuracy & F1-score   ║
          ║  • Confusion matrix      ║
          ║  • Cross-validation      ║
          ║  • Adjacent score        ║
          ╚══════════════╦═══════════╝
                         │
                         ▼
          ╔══════════════════════════╗
          ║  5. PREDICTION &         ║
          ║     VISUALIZATION        ║
          ║  ─────────────────────  ║
          ║  • Blind well prediction ║
          ║  • Facies log tracks     ║
          ║  • Feature importance    ║
          ╚══════════════════════════╝
                         │
                         ▼
                   results/ + figures/
```

---

## 📓 Notebooks

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `facies-classification.ipynb` | Main end-to-end pipeline | Full workflow, model comparison |
| `ml-classification.ipynb` | Extended ML experiments | Hyperparameter tuning, ablation |
| `notebooks/preprocessing.ipynb` | Data cleaning & PE imputation | `predicted_missing_pe.csv`, `final-data.csv` |
| `notebooks/test-pe-plot.ipynb` | PE log imputation validation | Imputation accuracy plots |
| `notebooks/supervised/` | Supervised model notebooks | Per-model training & evaluation |
| `notebooks/unsupervised/` | Clustering experiments | K-Means, hierarchical clustering |

---

## 🛠️ Technologies Used

<table>
<tr>
<td><b>Category</b></td>
<td><b>Technology</b></td>
<td><b>Version</b></td>
<td><b>Use</b></td>
</tr>
<tr>
<td>Language</td>
<td>Python</td>
<td>3.8+</td>
<td>Core development</td>
</tr>
<tr>
<td rowspan="3">Machine Learning</td>
<td>Scikit-learn</td>
<td>1.3+</td>
<td>RF, SVM, preprocessing, metrics</td>
</tr>
<tr>
<td>XGBoost</td>
<td>1.7+</td>
<td>Gradient boosted classification</td>
</tr>
<tr>
<td>SciPy</td>
<td>1.10+</td>
<td>Statistical analysis</td>
</tr>
<tr>
<td rowspan="2">Data Handling</td>
<td>Pandas</td>
<td>2.0+</td>
<td>Dataframe operations, I/O</td>
</tr>
<tr>
<td>NumPy</td>
<td>1.24+</td>
<td>Numerical computing, arrays</td>
</tr>
<tr>
<td rowspan="2">Visualization</td>
<td>Matplotlib</td>
<td>3.7+</td>
<td>Well log plots, facies tracks</td>
</tr>
<tr>
<td>Seaborn</td>
<td>0.12+</td>
<td>Confusion matrices, heatmaps</td>
</tr>
<tr>
<td>Environment</td>
<td>Jupyter Notebook</td>
<td>7.0+</td>
<td>Interactive development</td>
</tr>
</table>

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Ashraf-ISM/Facies-classification-using-ml.git
cd Facies-classification-using-ml
```

### Step 2 — Set Up Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 Manual install (if requirements.txt unavailable)</summary>

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter scipy openpyxl
```
</details>

---

## 🚀 Usage

### Option 1: Run the Main Notebook (Recommended)

```bash
jupyter notebook facies-classification.ipynb
```

### Option 2: Run Preprocessing First, Then Classification

```bash
# Step 1 — Preprocess data and impute missing PE log
jupyter notebook notebooks/preprocessing.ipynb

# Step 2 — Run supervised classification
jupyter notebook notebooks/supervised/

# Step 3 — Run unsupervised clustering
jupyter notebook notebooks/unsupervised/
```

### Option 3: Use Utility Modules Directly

```python
from src.data_prep import load_and_clean
from src.classification_utilities import plot_confusion_matrix, adjacent_score

# Load data
df = load_and_clean('data/final-data.csv')

# Evaluate predictions
adjacent_score(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred, facies_labels)
```

---



### Key Visualizations Produced

| Output | Description | Location |
|--------|-------------|----------|
| Facies log tracks | Predicted vs. actual facies side-by-side | `figures/` |
| Confusion matrix | Per-class classification heatmap | `figures/` |
| Feature importance | Top log contributors to prediction | `figures/` |
| PE imputation plot | Measured vs. predicted PE values | `figures/` |
| Cross-plot analysis | RHOB–NPHI and GR–ILD lithology separation | `figures/` |

---

## 🗂️ Repository Structure

```
Facies-classification-using-ml/
│
├── 📁 data/                              # All datasets
│   ├── input_pilot_data.csv              # Raw multi-well log data (labeled)
│   ├── final-data.csv                    # Cleaned & feature-engineered data
│   ├── predicted_missing_pe.csv          # PE log imputation output
│   ├── train_set_df.pickled              # Serialized training split
│   ├── test_set_df.pickled               # Serialized test split
│   ├── validation_data_nofacies.csv      # Blind well — no facies labels
│   ├── df.pickled                        # Full processed dataframe
│   ├── df1.pickled                       # Alternate processed version
│   └── podu_data.xlsx                    # Supplementary formation data
│
├── 📁 notebooks/                         # Jupyter analysis notebooks
│   ├── preprocessing.ipynb               # Data cleaning & PE imputation
│   ├── test-pe-plot.ipynb                # PE imputation validation plots
│   ├── supervised/                       # Supervised classification notebooks
│   └── unsupervised/                     # Clustering experiment notebooks
│
├── 📁 src/                               # Reusable Python modules
│   ├── data_prep.py                      # Data loading & preprocessing functions
│   └── classification_utilities.py       # Metrics, scoring, plotting utilities
│
├── 📁 figures/                           # Output plots and visualizations
├── 📁 results/                           # Model outputs and predictions
├── 📁 assets/                            # Images for README / documentation
├── 📁 Materials/                         # Reference papers and datasets
│
├── facies-classification.ipynb           # 🔑 Main end-to-end notebook
├── ml-classification.ipynb               # Extended ML experiments
├── git-push.sh                           # Git automation script
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
└── LICENSE                               # MIT License
```

---

## 🔮 Future Improvements

- [ ] **Deep Learning (1D CNN / LSTM)** — capture depth-sequential log patterns that tree-based models miss
- [ ] **Bayesian optimization** — automated hyperparameter tuning for RF and XGBoost
- [ ] **Multi-well generalization** — train on N-1 wells, predict on held-out well to test cross-well robustness
- [ ] **Probabilistic outputs** — report prediction confidence per depth sample alongside hard labels
- [ ] **LAS file pipeline** — direct ingestion via `lasio` for industry-standard `.las` format
- [ ] **Streamlit dashboard** — interactive upload-and-predict app for petrophysicists
- [ ] **SeismoForge integration** — plug facies output directly into the seismic interpretation GUI

---

## 📚 References

- [SEG 2016 Machine Learning Contest](https://github.com/seg/2016-ml-contest) — Hall, B. (2016). *Facies classification using machine learning*. The Leading Edge, 35(10), 906–909.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.

---

## 👨‍💻 Author

**Md Ashraf Ali**
M.Sc(Tech) Applied Geophysics | IIT (ISM) Dhanbad
DST-INSPIRE Scholar | ONGC Intern | IISc Summer Research Fellow '24

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Md%20Ashraf%20Ali-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ashraf-iit-ism)
[![Portfolio](https://img.shields.io/badge/Portfolio-ash--geophysics.netlify.app-FF5722?style=flat-square&logo=google-chrome&logoColor=white)](https://ash-geophysics.netlify.app)
[![Email](https://img.shields.io/badge/Email-mdashraf6209%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mdashraf6209@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Ashraf--ISM-181717?style=flat-square&logo=github)](https://github.com/Ashraf-ISM)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

---

<div align="center">

**⭐ If this project helped your research or learning, consider starring the repo! ⭐**

*Built with Python, well logs, and a genuine passion for data-driven geoscience.*

![Visitor Count](https://komarev.com/ghpvc/?username=Ashraf-ISM&label=Profile+Views&color=0e75b6&style=flat)

</div>
