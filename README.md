# Breast Cancer Detection

A simple machine learning project that predicts whether a tumor is **benign** or **malignant** using the Breast Cancer dataset.

## 📌 Overview

* Uses the Breast Cancer dataset (from sklearn or CSV).
* Trains a classification model (Random Forest / Logistic Regression).
* Predicts tumor type: **Benign** or **Malignant**.
* Shows accuracy and confusion matrix.

## 📁 Project Structure

```
project/
│-- breast_cancer_detection.py
│-- breast_cancer.csv (optional if not using sklearn dataset)
│-- README.md
│-- results/
│     ├── accuracy.txt
│     └── confusion_matrix.png
```

## ▶️ How to Run

1. Add the dataset (if using a CSV) OR rely on sklearn's built-in dataset.
2. Run:

```
python breast_cancer_detection.py
```

3. Check the `results/` folder for generated outputs.

## 🔧 Requirements

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib

Install:

```
pip install pandas numpy scikit-learn matplotlib
```

## 📊 Output

* Accuracy score
* Confusion matrix plot for predictions

## 🙌 Author

Palak Rai
# breast-cancer-risk-detector
Breast cancer risk prediction using ML, DL, and hybrid models with greedy feature selection. Includes preprocessing, model training, evaluation, and comparison on a 100K medical dataset. Complete end-to-end pipeline for research and healthcare analytics.
