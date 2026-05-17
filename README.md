# Breast-Cancer-Predictor.
# 🎗️ Breast Cancer Survival Prediction

A machine learning project to predict breast cancer patient survival status using a Gradient Boosting model, with an interactive web app built with Streamlit.

---

## 📁 Project Structure

```
├── canser.ipynb               # Notebook - EDA, preprocessing, model training
├── br.py                      # Streamlit web app
├── breast_cancer_model.sav    # Saved trained pipeline
├── Breast_Cancer.csv          # Dataset
└── README.md
```

---

## 📊 Dataset

- **Source:** Breast Cancer dataset (4024 records, 16 features)
- **Target:** `Status` — Alive (0) or Dead (1)
- **Features:** Age, Race, Marital Status, T Stage, N Stage, 6th Stage, Differentiate, Grade, A Stage, Tumor Size, Estrogen Status, Progesterone Status, Regional Node Examined, Reginol Node Positive, Survival Months

---

## ⚙️ Methodology

### 1. Data Cleaning
- Removed 1 duplicate row

### 2. Exploratory Data Analysis (EDA)
- Value counts for categorical features
- Distribution plots and count plots
- Correlation heatmap

### 3. Preprocessing
- Label Encoding for categorical columns
- Train/Test split (80/20) with `random_state=42`

### 4. Model Pipeline
Built using `imblearn.Pipeline` to prevent **data leakage**:
```
StandardScaler → SMOTE → GradientBoostingClassifier
```

### 5. Hyperparameter Tuning
- Used `GridSearchCV` with `scoring='recall'` and `cv=5`
- Tuned: `n_estimators`, `learning_rate`, `max_depth`, `subsample`

---

## 📈 Results

| Metric | Class 0 (Alive) | Class 1 (Dead) |
|--------|----------------|----------------|
| Precision | 0.95 | 0.54 |
| Recall | 0.88 | 0.73 |
| F1-Score | 0.91 | 0.62 |
| **Accuracy** | **85.8%** | |

---

## 🌐 Web App

Built with **Streamlit** — allows users to input patient data and get a prediction instantly.

### Run the app:
```bash
streamlit run br.py
```

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn streamlit
```

---

## 💡 Key Decisions

- **SMOTE inside Pipeline** to avoid data leakage on test data
- **Recall as scoring metric** because in medical diagnosis, missing a positive case (False Negative) is more dangerous than a false alarm
