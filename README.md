# ApexBank Bharat AI Credit Line Underwriting & Risk Decisioning Platform (INR / ₹)

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Currency](https://img.shields.io/badge/Currency-INR%20(%E2%82%B9)-emerald.svg)]()
[![Regulatory Compliance](https://img.shields.io/badge/Compliance-RBI%20Master%20Directions-blue.svg)]()

An institutional-grade Machine Learning system designed for Indian commercial banks and NBFCs (e.g. HDFC Bank, ICICI Bank, SBI, Axis Bank, Bajaj Finance) to automate **unsecured credit card & revolving credit line underwriting**, **credit limit sanctioning in Indian Rupees (₹)**, and **Reserve Bank of India (RBI) Fair Practices Code adverse action disclosures**.

---

## Underwriting Architecture: Dual-Stage ML Pipeline

```
                       [Indian Applicant Credit Profile]
                                       │
                                       ▼
                  [Feature Engineering & Financial Ratios]
             (FOIR/DTI, Revolving Burden, CIBIL Derogatory Density)
                                       │
                                       ▼
             ┌───────────────────────────────────────────────────┐
             │   STAGE 1: Approval Classification & PD Scoring   │
             │   ├── Logistic Regression (Basel II Scorecard)    │
             │   └── XGBoost Classifier (Champion Ensemble)      │
             └─────────────────────────┬─────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
      [PD <= 0.20 & Model=1]                    [PD > 0.35 or RBI Knockout]
                │                                             │
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │   STAGE 2: Credit Line    │                 │   STAGE 3: Explainability │
  │   Limit Sizing Regressor  │                 │   (RBI Fair Practices)    │
  │   (XGBoost Regressor)     │                 │   Principal Reason Codes  │
  │   (₹25,000 to ₹18,00,000) │                 │   (CIBIL, DPD, FOIR, NPA) │
  └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                │                                             │
                ▼                                             ▼
        [SANCTIONED / APPROVED]                       [REJECTED / DECLINED]
   Credit Limit : ₹3,50,000                      Reasons:
   Risk Tier    : Prime (CIBIL 745)              1. High FOIR / DTI (≥ 50%)
   Indicative APR: 19.5% - 24.0% p.a.            2. Recent DPD Delinquencies
```

---

## Key Capabilities

1. **Dual Machine Learning Engine**:
   - **Logistic Regression (Scorecard Benchmark)**: Traditional interpretable Basel II / institutional banking scorecard model with linear log-odds probabilities.
   - **XGBoost Classifier (Champion Ensemble)**: High-performance gradient boosted decision tree capturing complex non-linear borrower interactions, utilization spikes, and cross-feature thresholds.
   - **XGBoost Regressor (Credit Limit Sizing)**: Predicts risk-adjusted credit limit capacity from ₹25,000 up to ₹18,00,000 (18 Lakhs) based on monthly disposable cash flow and CIBIL tiering.
2. **Indian Banking Economics (INR / ₹ & CIBIL Scores)**:
   - All income, existing credit card balances, and sanctioned credit lines are denominated in **Indian Rupees (₹ / Lakhs)**.
   - Credit bureau scoring is built around **TransUnion CIBIL scores (300 to 900)**.
   - Fixed Obligation to Income Ratio (FOIR / DTI) underwriting caps adhering to Indian retail lending norms.
3. **RBI Fair Practices Code (FPC) Compliance**:
   - Automated adverse action reason generation when credit is rejected or capped:
     - `RBI_NPA_01`: Suit Filed / Written-off / NPA Status on CIBIL.
     - `RBI_FOIR_01`: Excessive Fixed Obligation to Income Ratio (FOIR ≥ 50%).
     - `RBI_CIBIL_01`: CIBIL Bureau score below minimum prime guideline (< 620).
     - `RBI_UTL_01`: High credit card revolving utilization (≥ 75%).
     - `RBI_DPD_01`: Multiple 30-90+ Days Past Due (DPD) delinquencies within 24 months.
4. **Loan Officer Interactive Portal**:
   - Real-time underwriting simulator with instant toggle between **XGBoost** and **Logistic Regression**.
   - Preloaded Indian customer archetypes (Senior Tech Lead, Salaried Consultant, MSME Trader, Defaulted Borrower).
   - Cohort batch stress-testing simulator in INR.
   - Live Chart.js ROC-AUC curves and XGBoost feature gain bar charts.

---

## Model Benchmark Performance (Tested on 3,200 Indian Applications)

### Stage 1: Approval Classification Benchmark

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Underwriting Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Scorecard)** | **93.72%** | **94.08%** | **93.91%** | **0.9399** | **0.9882** | Regulatory Basel Baseline |
| **XGBoost Classifier (Champion)** | **92.81%** | **93.81%** | **92.36%** | **0.9308** | **0.9853** | Production Champion Ensemble |

### Stage 2: Credit Limit Sizing (XGBoost Regressor on Approved Portfolio)

- **Mean Absolute Error (MAE)**: **₹11,340.77** (on credit lines up to ₹18,00,000)
- **Root Mean Squared Error (RMSE)**: **₹18,027.18**
- **Coefficient of Determination ($R^2$)**: **0.9873**

---

## Project Structure

```
bank-credit-line-approval/
├── data/
│   ├── raw/
│   │   └── credit_applications.csv          # 16,000 Indian credit applications (INR)
│   └── processed/
│       ├── train.csv                        # 80% Stratified training set (12,800 rows)
│       └── test.csv                         # 20% Held-out testing set (3,200 rows)
├── models/
│   ├── approval_classifier.joblib           # Trained XGBoost Classifier
│   ├── logistic_regression_scorecard.joblib # Trained Logistic Regression Scorecard
│   ├── credit_limit_regressor.joblib        # Trained XGBoost Regressor (INR Sizing)
│   ├── preprocessor.joblib                  # Column transformer (scaling & encoding)
│   └── model_metadata.json                  # Performance metrics, ROC curve & feature importances
├── src/
│   ├── data_generator.py                    # Indian banking data generator (INR & CIBIL)
│   ├── data_preprocessing.py                # FOIR / DTI financial feature engineering
│   ├── train_models.py                      # XGBoost & Logistic Regression training & benchmarking
│   ├── explainability.py                    # RBI Fair Practices Code & Adverse Action reasons
│   ├── inference_engine.py                  # End-to-end multi-stage inference engine (INR)
│   └── web_server.py                        # Zero-dependency HTTP server & REST API
├── web/
│   ├── index.html                           # FinTech underwriting dashboard UI
│   ├── styles.css                           # Glassmorphic decision cards & glowing statuses
│   └── app.js                               # Real-time algorithm switcher & Chart.js charts
├── tests/
│   └── test_pipeline.py                     # Automated unit and integration test suite
├── notebooks/
│   └── eda_and_model_evaluation.py          # Portfolio EDA and CIBIL tier audit
├── run_server.bat                           # 1-click Windows server launcher
├── run_training.bat                         # 1-click Windows retraining script
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## Quickstart Guide

### 1. Requirements & Dependencies
Make sure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Portal
Run:
```bash
python src/web_server.py 8050
```
*(Or double-click `run_server.bat` on Windows).*  
Open your browser at: **[http://localhost:8050](http://localhost:8050)**

### 3. Run Automated Tests
```bash
python tests/test_pipeline.py
```

### 4. Retrain Models
To regenerate synthetic Indian banking data and retrain both XGBoost and Logistic Regression:
```bash
python src/data_generator.py
python src/train_models.py
```
*(Or double-click `run_training.bat` on Windows).*

### 5. Run Portfolio EDA Audit
```bash
python notebooks/eda_and_model_evaluation.py
```

---

## REST API Specification

### Underwrite Single Applicant in INR (₹)
`POST /api/underwrite`
```json
{
  "model_choice": "xgboost",
  "annual_income": 1450000,
  "credit_score": 765,
  "debt_to_income": 0.22,
  "revolving_utilization": 0.25,
  "revolving_balance": 65000,
  "credit_history_years": 9.0,
  "age": 34,
  "employment_years": 7.0,
  "delinquencies_2yrs": 0,
  "num_derogatory_records": 0,
  "num_credit_inquiries_6m": 1,
  "home_ownership": "PARENTAL_OWNED",
  "loan_purpose": "CREDIT_CARD_CONSOLIDATION",
  "bankruptcy_history": 0
}
```

**Response Example:**
```json
{
  "applicant_id": "IN-APP-MUMBAI-01",
  "decision": "APPROVED",
  "decision_code": "AUTO_APPROVE",
  "model_used": "XGBoost Classifier",
  "probability_of_default": 0.005,
  "risk_tier": "Prime (Tier 2 - CIBIL 710-779)",
  "recommended_credit_line_inr": 260000,
  "formatted_credit_line": "₹2,60,000",
  "suggested_pricing_apr": "19.50% - 24.00% p.a. (Standard Revolving)",
  "model_comparison": {
    "xgboost_approval_prob": 0.9974,
    "logistic_regression_approval_prob": 0.9992
  },
  "key_strengths": [
    {
      "title": "Strong Prime CIBIL Profile",
      "detail": "CIBIL score of 765 exceeds standard retail banking benchmark (700+)."
    }
  ]
}
```
