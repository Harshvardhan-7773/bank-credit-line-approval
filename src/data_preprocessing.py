"""
Data Preprocessing and Feature Engineering Pipeline for Indian Credit Line Underwriting (INR).
Handles financial ratios, feature engineering, missing value imputation,
categorical one-hot encoding, and scaling.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

NUMERICAL_BASE_FEATURES = [
    'age',
    'annual_income',
    'employment_years',
    'credit_score', # CIBIL Score (300-900)
    'credit_history_years',
    'debt_to_income',
    'revolving_utilization',
    'revolving_balance',
    'num_open_accounts',
    'num_credit_inquiries_6m',
    'delinquencies_2yrs',
    'num_derogatory_records',
    'bankruptcy_history'
]

CATEGORICAL_FEATURES = [
    'home_ownership',
    'loan_purpose'
]

ENGINEERED_FEATURES = [
    'monthly_income',
    'disposable_income',
    'revolving_burden_ratio',
    'inquiries_per_account',
    'derogatory_density'
]

ALL_FEATURE_COLS = NUMERICAL_BASE_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific credit underwriting risk ratios and metrics in INR (₹).
    """
    df = df.copy()
    
    # 1. Monthly income & Monthly disposable cash flow (₹)
    df['monthly_income'] = df['annual_income'] / 12.0
    df['disposable_income'] = np.maximum(0, df['monthly_income'] * (1.0 - df['debt_to_income'].clip(0, 0.99)))
    
    # 2. Revolving debt burden relative to annual earnings
    df['revolving_burden_ratio'] = df['revolving_balance'] / np.maximum(df['annual_income'], 50000.0)
    
    # 3. Credit shopping intensity: Recent inquiries per existing active line
    df['inquiries_per_account'] = df['num_credit_inquiries_6m'] / np.maximum(df['num_open_accounts'], 1.0)
    
    # 4. Derogatory event frequency per year of credit file history
    df['derogatory_density'] = (df['delinquencies_2yrs'] + df['num_derogatory_records'] * 2.0) / np.maximum(df['credit_history_years'], 0.5)
    
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    numeric_features = NUMERICAL_BASE_FEATURES + ENGINEERED_FEATURES
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor


def prepare_train_test_data(
    raw_data_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads raw credit data, adds engineered features, performs stratified train-test split,
    and saves processed splits to disk.
    """
    print(f"Loading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Add domain features
    df = add_engineered_features(df)
    
    # Stratified split on approval_status
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['approval_status']
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved training dataset ({len(train_df)} rows) to: {train_path}")
    print(f"Saved testing dataset ({len(test_df)} rows) to: {test_path}")
    
    return train_df, test_df


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'credit_applications.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    prepare_train_test_data(raw_path, processed_dir)
