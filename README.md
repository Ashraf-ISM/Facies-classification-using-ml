# 🪨 Facies Classification using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EC4E20?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> **Automated geological facies prediction from well log data using supervised machine learning — bridging geoscience domain knowledge with modern ML workflows.**

---

## 📌 Project Overview

Facies classification is a critical step in subsurface reservoir characterization. Traditionally, it relies on time-consuming manual interpretation by petrophysicists. This project automates that process using supervised machine learning models trained on well log data.

By learning patterns in petrophysical measurements (GR, Resistivity, Density, Neutron Porosity, Sonic), the models can predict lithofacies labels across unlabeled intervals — enabling faster, more consistent reservoir zonation at scale.

This project demonstrates an end-to-end ML pipeline applicable to real-world E&P (Exploration & Production) workflows, covering everything from raw data preprocessing to model evaluation and facies visualization.

---

## 🎯 Objectives

- Preprocess and engineer features from multi-log well data for ML readiness
- Train and compare multiple supervised classification algorithms for facies prediction
- Evaluate model performance using cross-validation and classification metrics
- Visualize predicted vs. actual facies logs to assess geological consistency
- Build a reproducible, modular workflow that can be applied to new wells

---

## 🗃️ Dataset Description

The dataset consists of **well log measurements** commonly acquired during wireline logging operations in oil & gas exploration.

| Log | Symbol | Geological Significance |
|-----|--------|------------------------|
| Gamma Ray | GR | Shale volume estimation; distinguishes sands from shales |
| Resistivity | RT / ILD | Hydrocarbon vs. brine-saturated zones; fluid typing |
| Bulk Density | RHOB | Porosity estimation; lithology indicator |
| Neutron Porosity | NPHI | Porosity; gas effect detection when paired with RHOB |
| Sonic | DT | Porosity; mechanical properties; seismic-to-well tie |

**Target Variable:** Discrete facies labels (e.g., Sandstone, Shale, Limestone, Dolomite, etc.)

> 📁 Dataset source: *(Add source — e.g., SEG 2016 ML Contest, or your own well data)*
> 📊 Number of samples: *(Add row count)*
> 🔢 Number of facies classes: *(Add number)*

---

## ⚙️ Machine Learning Workflow

```
Raw Well Log Data (.csv / .las)
        │
        ▼
┌─────────────────────┐
│  1. Data Loading &  │
│     Exploration     │  ← Missing value analysis, log statistics, class distribution
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Preprocessing   │  ← Null handling, normalization/standardization, outlier removal
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Feature         │  ← Interaction features, lag features, depth gradients
│     Engineering     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Model Training  │  ← Random Forest · SVM · XGBoost
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Evaluation      │  ← Accuracy, F1-score, Confusion Matrix, Cross-Validation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  6. Visualization   │  ← Predicted vs Actual facies logs, feature importance plots
└─────────────────────┘
```

---

## 🛠️ Technologies Used

| Category | Library / Tool | Purpose |
|----------|---------------|---------|
| Language | Python 3.8+ | Core development |
| ML Framework | Scikit-learn | Random Forest, SVM, preprocessing, evaluation |
| Boosting | XGBoost | Gradient boosted facies classification |
| Data Handling | Pandas, NumPy | Data wrangling, array operations |
| Visualization | Matplotlib, Seaborn | Log plots, confusion matrices, feature importance |
| Environment | Jupyter Notebook | Interactive development and presentation |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ashraf-ISM/Facies-classification-using-ml.git
cd Facies-classification-using-ml
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate           # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

---

## 🚀 Usage

### Run in Jupyter Notebook

```bash
jupyter notebook
```

Open the main notebook:

```
notebooks/facies_classification.ipynb
```

### Run as Python Script

```bash
python src/train.py --data data/well_logs.csv --model random_forest
```

> ⚠️ *Update file paths in the notebook/script to match your local dataset location.*

---

## 📊 Results & Outputs

### Model Performance Comparison

| Model | Accuracy | F1-Score (Weighted) | Cross-Val Score |
|-------|----------|---------------------|-----------------|
| Random Forest | *(add %)* | *(add)* | *(add)* |
| Support Vector Machine | *(add %)* | *(add)* | *(add)* |
| XGBoost | *(add %)* | *(add)* | *(add)* |

> 💡 *Fill in your actual scores — even approximate values make this section significantly more impactful to recruiters and researchers.*

### Sample Outputs

- 📉 **Facies log plots** — side-by-side comparison of predicted vs. actual facies tracks
- 🔥 **Confusion matrix heatmaps** — per-class classification performance
- 📊 **Feature importance charts** — which logs drive facies prediction most
- 📈 **Cross-validation score distributions** — model stability across folds

---

## 🗂️ Project Structure

```
Facies-classification-using-ml/
│
├── data/
│   ├── raw/                    # Original well log data (CSV/LAS)
│   └── processed/              # Cleaned, feature-engineered data
│
├── notebooks/
│   └── facies_classification.ipynb   # Main analysis notebook
│
├── src/
│   ├── preprocess.py           # Data loading and preprocessing functions
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # Model training and cross-validation
│   ├── evaluate.py             # Metrics, confusion matrix, scoring
│   └── visualize.py            # Log plots and result visualization
│
├── outputs/
│   ├── figures/                # Saved plots and charts
│   └── models/                 # Serialized trained models (.pkl)
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```

> 📝 *Update this tree to match your actual repository structure if it differs.*

---

## 🔮 Future Improvements

- [ ] **Deep Learning integration** — implement 1D CNN or LSTM for sequential log data to capture depth-dependent patterns
- [ ] **Unsupervised clustering** — add K-Means / DBSCAN for facies exploration without labeled data
- [ ] **Multi-well generalization** — train across multiple wells and test cross-well prediction robustness
- [ ] **LAS file support** — integrate `lasio` for direct ingestion of industry-standard LAS well log files
- [ ] **Interactive dashboard** — build a Streamlit or PyQt5 GUI for non-technical users to upload logs and get predictions
- [ ] **Uncertainty quantification** — add probabilistic outputs (e.g., prediction confidence per depth sample)

---

## 👨‍💻 Author

**Md Ashraf Ali**
M.Sc(Tech) Applied Geophysics | IIT (ISM) Dhanbad
DST-INSPIRE Scholar | ONGC Intern | IISc Summer Research Fellow '24

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/ashraf-iit-ism)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF5722?style=flat&logo=google-chrome&logoColor=white)](https://ash-geophysics.netlify.app)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail&logoColor=white)](mailto:mdashraf6209@gmail.com)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If this project helped you, consider giving it a star!** ⭐

*Built with curiosity, well logs, and too much coffee.*

</div>
