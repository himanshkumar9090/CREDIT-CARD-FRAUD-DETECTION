# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions in a
highly imbalanced dataset (~0.17% fraud). It compares three classifiers using
imbalance-aware metrics and tunes the decision threshold of the best model.

## Dataset

- **Name:** Credit Card Fraud Detection (ULB Machine Learning Group)
- **Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** 284,807 transactions made by European cardholders in September 2013 (two days), 492 of them fraudulent.
- **Features:** `Time`, `Amount`, `V1`-`V28` (PCA-anonymised components) and the label `Class` (1 = fraud, 0 = legitimate).

Download `creditcard.csv` and place it in the project folder.
If the file is missing, the script runs on **synthetic demo data** so the pipeline can still be tested
(results on synthetic data are not meaningful).

## Technologies Used

Python 3.9+, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib.

## Project Files

| File | Purpose |
|------|---------|
| `credit_card_fraud_detection.py` | Complete project code (EDA, training, evaluation, model export) |
| `requirements.txt` | Python dependencies |
| `Credit_Card_Fraud_Detection_Documentation.docx` | Full project documentation |
| `README.md` | This overview |

## Setup & Run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. place creditcard.csv in this folder, then run
python credit_card_fraud_detection.py --data creditcard.csv
```

To use it as a notebook, copy each section (marked by the `# ----` headers) into separate Jupyter cells,
or run `jupytext --to notebook credit_card_fraud_detection.py`.

## Method Summary

1. **EDA** - class balance, amount distribution, fraud by hour, feature correlations.
2. **Preprocessing** - duplicates removed; `Amount`/`Time` scaled with `RobustScaler`; stratified 75/25 split.
3. **Imbalance handling** - class weights plus random undersampling (10:1) on the **training set only**.
4. **Models** - Logistic Regression, Random Forest, Gradient Boosting.
5. **Evaluation** - Precision, Recall, F1, ROC-AUC and **PR-AUC** (accuracy is misleading at 0.17% fraud).
6. **Threshold tuning** - choose the probability cut-off that maximises F1 on the precision-recall curve.

## Outputs (created in `./outputs/`)

`01_eda.png`, `02_correlations.png`, `03_roc_pr_curves.png`, `04_confusion_default.png`,
`04_confusion_tuned.png`, `05_feature_importance.png`, `model_comparison.csv`, `fraud_model.joblib`.

## Key Information

- The best model is selected by PR-AUC, the most informative metric for rare-event detection.
- Resampling is applied only to training data to avoid leaking information into the test set.
- Threshold tuning here uses the test set for simplicity; in production, tune on a separate validation set.
- Exact scores depend on your scikit-learn version and random seed (`RANDOM_STATE = 42`).

## Loading the saved model

```python
import joblib
bundle = joblib.load("outputs/fraud_model.joblib")
model, thr = bundle["model"], bundle["threshold"]
# X_new must contain bundle["features"] in the same order, with Amount/Time scaled
is_fraud = model.predict_proba(X_new)[:, 1] >= thr
```

## Disclaimer

Educational project. Not a production fraud-prevention system.
