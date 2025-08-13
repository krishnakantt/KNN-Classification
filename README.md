# Iris Flower Classification using KNN
This project implements a K-Nearest Neighbors (KNN) classifier to classify iris flowers into three species — Setosa, Versicolor, and Virginica — based on their sepal and petal measurements. It includes data loading, visualization, model training, prediction, and evaluation.

---

## 📂 Project Structure
```bash
Iris-KNN-Classification/
│
├── KNN.py              # Main Python script
├── requirements.txt    # Required dependencies
└── README.md           # Project documentation
```

---

## 📊 Dataset
- Source: sklearn.datasets.load_iris()
- Features:
    - Sepal length (cm)
    - Sepal width (cm)
    - Petal length (cm)
    - Petal width (cm)
- Target Classes:
    - 0 → Setosa
    - 1 → Versicolor
    - 2 → Virginica
 
---

## 🚀 Features
- Loads the Iris dataset from Scikit-learn
- Creates a pandas DataFrame with features and target labels
- Visualizes Sepal Length vs. Sepal Width for Setosa and Versicolor
- Splits the dataset into training and testing sets
- Trains a KNN model (k=3)
- Predicts on the test set
- Evaluates performance using accuracy, classification report, and confusion matrix

---

## 📌 Installation
1. Clone this repository:
```bash
git clone https://github.com/krishnakantt/KNN-Classification
cd Iris-KNN-Classification
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Script:
```bash
python KNN.py
```

---

## 📈 Example Output
```bash
Model accuracy: 0.97
Classification report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00         9
           1       0.95      1.00      0.97        10
           2       1.00      0.90      0.95        11
```

---

