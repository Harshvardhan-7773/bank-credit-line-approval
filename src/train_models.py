"""
Machine Learning Training and Model Benchmarking Pipeline for Indian Bank Credit Line Approval.
Specifically focuses on:
  1. Logistic Regression (Basel II Scorecard Benchmark)
  2. XGBoost Classifier (Champion Gradient Boosted Decision Tree)
  3. XGBoost Regressor (Credit Line Limit Sizing in INR / ₹)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb

from data_preprocessing import (
    build_preprocessor, prepare_train_test_data,
    NUMERICAL_BASE_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES
)


def train_and_evaluate_models(base_dir: str):
    """
    Trains Logistic Regression and XGBoost models, benchmarks their underwriting metrics,
    and exports champion models and metadata.
    """
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'credit_applications.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Prepare data splits
    train_df, test_df = prepare_train_test_data(raw_data_path, processed_dir)
    
    feature_cols = NUMERICAL_BASE_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
    
    X_train_raw = train_df[feature_cols]
    y_train_class = train_df['approval_status']
    
    X_test_raw = test_df[feature_cols]
    y_test_class = test_df['approval_status']
    
    # 2. Fit preprocessor
    print("\n--- Fitting Feature Preprocessor ---")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    # Extract feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    numeric_feature_names = NUMERICAL_BASE_FEATURES + ENGINEERED_FEATURES
    all_transformed_feature_names = numeric_feature_names + cat_feature_names
    
    # 3. Benchmark Logistic Regression vs XGBoost Classifier
    print("\n==================================================================")
    print(" STAGE 1: APPROVAL CLASSIFICATION BENCHMARK")
    print(" Comparing: Logistic Regression Scorecard vs XGBoost Ensemble")
    print("==================================================================")
    
    classifiers = {
        'Logistic Regression (Basel Scorecard)': LogisticRegression(
            max_iter=1000, 
            C=1.0, 
            solver='lbfgs', 
            random_state=42
        ),
        'XGBoost Classifier (Champion)': xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric='logloss',
            random_state=42
        )
    }
    
    comparison_results = {}
    fitted_models = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_proc, y_train_class)
        fitted_models[name] = clf
        
        y_pred = clf.predict(X_test_proc)
        y_prob = clf.predict_proba(X_test_proc)[:, 1]
        
        acc = float(accuracy_score(y_test_class, y_pred))
        prec = float(precision_score(y_test_class, y_pred))
        rec = float(recall_score(y_test_class, y_pred))
        f1 = float(f1_score(y_test_class, y_pred))
        auc = float(roc_auc_score(y_test_class, y_prob))
        
        print(f"  -> Accuracy : {acc * 100:.2f}%")
        print(f"  -> Precision: {prec * 100:.2f}%")
        print(f"  -> Recall   : {rec * 100:.2f}%")
        print(f"  -> F1-Score : {f1:.4f}")
        print(f"  -> ROC-AUC  : {auc:.4f}")
        
        comparison_results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4)
        }
        
    # Select XGBoost as the Champion Classifier
    champion_name = 'XGBoost Classifier (Champion)'
    champion_clf = fitted_models[champion_name]
    
    # Calculate detailed performance metrics for champion
    best_y_pred = champion_clf.predict(X_test_proc)
    best_y_prob = champion_clf.predict_proba(X_test_proc)[:, 1]
    
    cm = confusion_matrix(y_test_class, best_y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test_class, best_y_prob)
    idx = np.linspace(0, len(fpr) - 1, 50).astype(int)
    roc_points = [{'fpr': round(float(fpr[i]), 4), 'tpr': round(float(tpr[i]), 4)} for i in idx]
    
    # Feature Importances from XGBoost
    xgb_importances = champion_clf.feature_importances_
    feature_importances = []
    for feat_name, imp in zip(all_transformed_feature_names, xgb_importances):
        feature_importances.append({
            'feature': feat_name,
            'importance': round(float(imp), 4)
        })
    feature_importances.sort(key=lambda x: x['importance'], reverse=True)
    
    # 4. Stage 2: Credit Limit Sizing with XGBoost Regressor (INR / ₹)
    print("\n==================================================================")
    print(" STAGE 2: CREDIT LINE SIZING REGRESSOR (XGBRegressor)")
    print(" Predicting Credit Line Ceiling in Indian Rupees (₹)")
    print("==================================================================")
    
    approved_train_mask = (y_train_class == 1)
    approved_test_mask = (y_test_class == 1)
    
    X_train_reg = X_train_proc[approved_train_mask]
    y_train_reg = train_df.loc[approved_train_mask, 'approved_credit_line'].values
    
    X_test_reg = X_test_proc[approved_test_mask]
    y_test_reg = test_df.loc[approved_test_mask, 'approved_credit_line'].values
    
    xgb_limit_regressor = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    xgb_limit_regressor.fit(X_train_reg, y_train_reg)
    y_reg_pred = xgb_limit_regressor.predict(X_test_reg)
    
    reg_mae = float(mean_absolute_error(y_test_reg, y_reg_pred))
    reg_rmse = float(np.sqrt(mean_squared_error(y_test_reg, y_reg_pred)))
    reg_r2 = float(r2_score(y_test_reg, y_reg_pred))
    
    print(f"XGBoost Regressor Performance on Approved Portfolio (INR):")
    print(f"  -> Mean Absolute Error (MAE): Rs. {reg_mae:,.2f}")
    print(f"  -> Root Mean Squared Error (RMSE): Rs. {reg_rmse:,.2f}")
    print(f"  -> R-squared (R2 Score): {reg_r2:.4f}")
    
    # 5. Save Models and Metadata
    print("\n--- Serializing Models & Metadata ---")
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
    joblib.dump(champion_clf, os.path.join(models_dir, 'approval_classifier.joblib'))
    joblib.dump(fitted_models['Logistic Regression (Basel Scorecard)'], os.path.join(models_dir, 'logistic_regression_scorecard.joblib'))
    joblib.dump(xgb_limit_regressor, os.path.join(models_dir, 'credit_limit_regressor.joblib'))
    
    metadata = {
        'currency': 'INR',
        'currency_symbol': '₹',
        'champion_classifier': champion_name,
        'model_benchmarks': comparison_results,
        'classifier_metrics': {
            'confusion_matrix': cm,
            'roc_auc': comparison_results[champion_name]['roc_auc'],
            'accuracy': comparison_results[champion_name]['accuracy'],
            'precision': comparison_results[champion_name]['precision'],
            'recall': comparison_results[champion_name]['recall'],
            'f1_score': comparison_results[champion_name]['f1_score'],
            'roc_curve': roc_points
        },
        'regressor_metrics': {
            'model': 'XGBoost Regressor',
            'mae_inr': round(reg_mae, 2),
            'rmse_inr': round(reg_rmse, 2),
            'r2_score': round(reg_r2, 4)
        },
        'top_feature_importances': feature_importances[:12],
        'underwriting_thresholds': {
            'auto_approve_max_pd': 0.15,
            'review_max_pd': 0.28,
            'min_cibil_score': 600,
            'max_foir': 0.50
        }
    }
    
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    print(f"Models and model_metadata.json successfully exported to {models_dir}")
    return metadata


if __name__ == '__main__':
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_and_evaluate_models(base_path)
