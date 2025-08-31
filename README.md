# Facies Classification Using Machine Learning

This repository contains my work for the  Case Study at IIT(ISM) Dhanbad.  
The objective is to build a **continuous electrofacies profile** from subsurface wireline logs using both **supervised** and **unsupervised machine learning approaches** and validate the results against interpreted lithofacies.

---

## 📄 Problem Statement

The dataset is taken from a University of Kansas class exercise on Neural Networks and Fuzzy Systems (Bohling & Dubois, 2003; Dubois et al., 2007).  
It contains wireline logs from **nine wells** in the Hugoton and Panoma gas fields (North America), each depth sample labeled with a **facies type** based on core observations.

The main tasks are:

1. **Explore and analyze** the dataset – histograms, distributions, cross-plots, and outlier detection.
2. **Prepare** the wireline logs for modeling – handle missing data, generate missing logs (e.g., PE), and perform scaling.
3. **Implement supervised classification models** – Random Forest, SVM, Gradient Boosting, etc.
4. **Implement unsupervised clustering** – K-Means for electrofacies generation.
5. **Compare** predictions and clusters with lithofacies.
6. **Visualize** electrofacies logs, confusion matrices, and cross-plots.
7. **Present findings** in a 12–15 slide presentation by **12 April 2025**.

---

## 🛠 Workflow

**Step 1 – Data Exploration**
- Load `input_pilot_data.csv`
- Plot histograms, cross-plots, and statistical summaries
- Identify key features (GR, resistivity, density, porosity, PE, etc.)

**Step 2 – Data Preparation**
- Handle missing values
- Feature selection and scaling
- Dimensionality reduction if required

**Step 3 – Model Training**
- Supervised: Train multiple ML models using scikit-learn
- Unsupervised: Apply K-Means clustering

**Step 4 – Visualization**
- Confusion matrices
- Electrofacies logs for each well
- Cross-plots of predicted vs. actual facies

**Step 5 – Presentation**
- Compile workflow, results, and interpretations into group slides

---

## 📂 Repository Structure
Facies-classification-using-ml/
│
├── facies-classification.ipynb # Main Jupyter Notebook with ML workflow
├── facies_vectors.csv # Well log dataset with facies labels
├── README.md # Project documentation
└── requirements.txt # Python dependencies



---

## 📚 References
- Bohling, G. C., & Dubois, M. K. (2003). *An integrated application of neural network and fuzzy logic to lithofacies classification*.  
- Dubois, M. K., Bohling, G. C., & Chakrabarti, S. (2007). *Comparison of four approaches to a rock facies classification problem*. Computers & Geosciences, 33(5), 599–617.  
- [scikit-learn Documentation](https://scikit-learn.org/stable/index.html)  
- [Anaconda Python Distribution](https://www.anaconda.com/products/individual)

---

## 👨‍💻 Author
**Md Ashraf**  
M.Sc.(Tech) Applied Geophysics, IIT(ISM) Dhanbad  
[GitHub Profile](https://github.com/Ashraf8434)

