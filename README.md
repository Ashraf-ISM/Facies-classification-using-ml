# 🛢️ Facies Classification Using Machine Learning  

> 🔬 A machine learning pipeline for **electrofacies classification** from subsurface well log data.  
> Developed as part of a **Case Study at IIT(ISM) Dhanbad**, this project combines **supervised** and **unsupervised ML methods** to generate and validate facies profiles.  

---

## 🏷️ Project Badges  

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)  
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter)](https://jupyter.org/)  
[![Scikit-learn](https://img.shields.io/badge/ML-ScikitLearn-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)  
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)  
[![Matplotlib](https://img.shields.io/badge/Viz-Matplotlib-004C99.svg?logo=plotly&logoColor=white)](https://matplotlib.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)  
[![Stars](https://img.shields.io/github/stars/ashraf-iit-ism/Facies-classification-using-ml?style=social)](https://github.com/ashraf-iit-ism/Facies-classification-using-ml/stargazers)  

---

## 📖 Overview  

Facies classification is critical in petroleum exploration and reservoir characterization.  
This project builds a **continuous electrofacies profile** from wireline logs, using:  

- **Supervised ML models** → Random Forest, SVM, Gradient Boosting  
- **Unsupervised ML methods** → K-Means clustering  
- **Visualization** → facies logs, confusion matrices, cross-plots  

The final goal is to evaluate ML-driven facies predictions against **interpreted lithofacies** for improved geological insight.  

---

## 📄 Problem Statement  

The dataset comes from a University of Kansas exercise (Bohling & Dubois, 2003; Dubois et al., 2007).  
It includes wireline logs from **nine wells** in the Hugoton and Panoma gas fields (North America).  
Each depth sample is labeled with a **facies type** derived from core observations.  

### 🎯 Objectives  
- Explore & analyze the dataset (distributions, cross-plots, histograms)  
- Prepare logs for modeling (missing data, PE log synthesis, scaling)  
- Train & evaluate supervised ML classifiers  
- Apply K-Means clustering for electrofacies generation  
- Compare predictions/clusters with lithofacies  
- Visualize results for interpretation  
- Document workflow & findings 

---

## 🛠 Workflow  

**Step 1 – Data Exploration**  
📊 Load dataset → generate histograms, distributions, cross-plots  

**Step 2 – Data Preparation**  
🧹 Handle missing values → log generation → scaling  

**Step 3 – Model Training**  
🤖 Supervised ML → RandomForest, SVM, GradientBoosting  
🔍 Unsupervised ML → K-Means clustering  

**Step 4 – Visualization**  
📈 Confusion matrices, facies logs, predicted vs. actual plots  

**Step 5 – Reporting**  
📝 Prepare presentation summarizing methods, results, and insights  

---

## 📂 Repository Structure  

| File / Folder | Type | Description |
|---------------|------|-------------|
| 📓 [**facies-classification.ipynb**](./facies-classification.ipynb) | Notebook | Main ML workflow |
| 📑 [**input_pilot_data.csv**](./input_pilot_data.csv) | Dataset | Well log dataset with facies labels |
| 📘 [**README.md**](./README.md) | Markdown | Documentation & usage guide |
| ⚙️ [**requirements.txt**](./requirements.txt) | Config | Python dependencies |
| 📂 [**notebooks/**](./notebooks) | Folder | Additional Jupyter notebooks |
| 📂 [**src/**](./src) | Folder | Source code & helper scripts |
| 📂 [**figures/**](./figures) | Folder | Visualizations & plots |
| 📂 [**results/**](./results) | Folder | Outputs & predictions |

---

## 🚀 Getting Started  

### 🔧 Installation  

Clone the repository:  
```bash
git clone https://github.com/ashraf-iit-ism/Facies-classification-using-ml.git
cd Facies-classification-using-ml
