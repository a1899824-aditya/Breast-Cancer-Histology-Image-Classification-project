# 🧬 Breast Cancer Classification – Machine Learning Pipeline

This project develops a machine learning model to classify breast tumors as benign or malignant using histological image features. Created for the COMP SCI 7317 – Machine Learning Tools course, this end-to-end project meets strict medical diagnostic requirements including sensitivity and false-positive rate constraints.

---

## 📁 Dataset

- `assignment2_data_2024.csv`  
- Modified version of the UCI Breast Cancer Wisconsin dataset  
- 220 samples × 20 numerical features  
- Binary target: `benign`, `malignant`

---

## 🎯 Project Goals

- Clean, explore, and preprocess real-world biomedical data  
- Build a machine learning pipeline using `scikit-learn`  
- Compare multiple classifiers and optimize using GridSearchCV  
- Meet client constraints:
  - Sensitivity (Recall) ≥ 90%  
  - False Positive Rate ≤ 20%

---

## 🛠 Tools & Libraries

- Python, Jupyter Notebook  
- `pandas`, `numpy`, `matplotlib`, `seaborn`  
- `scikit-learn`:  
  - `Pipeline`, `GridSearchCV`, `SGDClassifier`, `SVC`, `RandomForestClassifier`  
  - `LabelEncoder`, `SimpleImputer`, `StandardScaler`

---

## 🔄 Workflow

### 🔍 Data Cleaning
- Corrected typos in target labels (`maligant` → `malignant`)  
- Recasted data types  
- Detected and replaced outliers with NaN using IQR  
- Verified missing data handling  

### 🧪 Baseline Modeling
- Dummy (random) classifier: Accuracy ~48%  
- SGD Classifier (untuned): Accuracy ~91%, Recall = 77%

### 🔧 Pipeline & Optimization
- Built scikit-learn pipelines for:
  - Imputation (median)  
  - Scaling (StandardScaler)  
- Used GridSearchCV for:
  - SGDClassifier  
  - SVC (RBF, Poly, Linear kernels)  
  - RandomForestClassifier  

### 🏆 Final Model: Random Forest

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | 93.18%    |
| Balanced Acc. | 88.46%    |
| Precision     | 1.00      |
| Recall        | 0.77 ❌   |
| F1 Score      | 0.87      |
| AUC           | 0.88      |
| FNR           | 23%       |

> ❗ Model does **not** meet client’s recall requirement (short by 0.13)

---

## 📊 Feature Importance & Visualization

- Used T-score to rank top 4 discriminative features:
  - Mean Concave Points
  - Mean Concavity
  - Mean Perimeter
  - Mean Radius

- Plotted pairwise decision boundaries using top feature pairs

---

## ✅ Key Takeaways

- End-to-end pipeline including preprocessing, modeling, evaluation, and interpretation  
- Balanced model accuracy with medical risk constraints  
- Proper use of cross-validation, ROC/AUC, and confusion matrix for diagnostics

---

## 📁 Files

- `a1899824_assignment2_umlt.ipynb`: Full notebook  
- `assignment2_data_2024.csv`: Raw dataset  
- `Assignment2_umlt.pdf`: Final report (with summary, visuals, discussion)

---

## 👤 Author

Aditya Venugopalan Nediyirippil  
GitHub: [a1899824-aditya](https://github.com/a1899824-aditya)
