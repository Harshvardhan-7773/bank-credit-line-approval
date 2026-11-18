"""
Core Indian Credit Underwriting and Risk Inference Engine (INR / ₹).
Executes Stage 1 (Approval Classification / PD Scoring via XGBoost & Logistic Regression),
Stage 2 (Credit Line Limit Sizing via XGBoost Regressor in INR / ₹), and
Stage 3 (RBI Fair Practices Explainability & CIBIL Risk Tiering).
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data_preprocessing import (
    add_engineered_features, NUMERICAL_BASE_FEATURES,
    CATEGORICAL_FEATURES, ENGINEERED_FEATURES
)
from explainability import generate_adverse_action_reasons, generate_approval_strengths, format_inr


class CreditUnderwritingEngine:
    """
    Production-grade inference engine for evaluating Indian credit line applications in INR (₹).
    Powered by XGBoost and Logistic Regression.
    """
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
            
        self.models_dir = models_dir
        self.preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        self.xgb_classifier = joblib.load(os.path.join(models_dir, 'approval_classifier.joblib'))
        self.logreg_scorecard = joblib.load(os.path.join(models_dir, 'logistic_regression_scorecard.joblib'))
        self.xgb_regressor = joblib.load(os.path.join(models_dir, 'credit_limit_regressor.joblib'))
        
        with open(os.path.join(models_dir, 'model_metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        self.feature_cols = NUMERICAL_BASE_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES

    def _determine_risk_tier(self, cibil_score: int, pd_prob: float) -> str:
        if cibil_score >= 780 and pd_prob < 0.08:
            return "Super Prime (Tier 1 - CIBIL 780+)"
        elif cibil_score >= 710 and pd_prob < 0.20:
            return "Prime (Tier 2 - CIBIL 710-779)"
        elif cibil_score >= 630 and pd_prob < 0.35:
            return "Near Prime (Tier 3 - CIBIL 630-709)"
        else:
            return "Subprime (Tier 4 - CIBIL < 630)"

    def underwrite_applicant(self, applicant: Dict[str, Any], model_choice: str = 'xgboost') -> Dict[str, Any]:
        """
        Evaluates a single credit applicant record through the multi-stage pipeline.
        
        Parameters:
        -----------
        applicant : dict
            Applicant financial and credit parameters in INR.
        model_choice : str
            'xgboost' (champion) or 'logistic_regression' (scorecard benchmark).
        """
        record = {
            'applicant_id': applicant.get('applicant_id', 'IN-APP-LIVE'),
            'age': int(applicant.get('age', 32)),
            'annual_income': float(applicant.get('annual_income', 950000)),
            'employment_years': float(applicant.get('employment_years', 5)),
            'home_ownership': str(applicant.get('home_ownership', 'RENTED')).upper(),
            'loan_purpose': str(applicant.get('loan_purpose', 'CREDIT_CARD_CONSOLIDATION')).upper(),
            'credit_score': int(applicant.get('credit_score', 735)), # CIBIL Score (300-900)
            'credit_history_years': float(applicant.get('credit_history_years', 7.5)),
            'debt_to_income': float(applicant.get('debt_to_income', 0.28)),
            'revolving_utilization': float(applicant.get('revolving_utilization', 0.35)),
            'revolving_balance': float(applicant.get('revolving_balance', 45000)),
            'num_open_accounts': int(applicant.get('num_open_accounts', 5)),
            'num_credit_inquiries_6m': int(applicant.get('num_credit_inquiries_6m', 1)),
            'delinquencies_2yrs': int(applicant.get('delinquencies_2yrs', 0)),
            'num_derogatory_records': int(applicant.get('num_derogatory_records', 0)),
            'bankruptcy_history': int(applicant.get('bankruptcy_history', 0)) # NPA / Written-off
        }

        # Format into single-row DataFrame & feature engineer
        df_single = pd.DataFrame([record])
        df_single = add_engineered_features(df_single)
        X_trans = self.preprocessor.transform(df_single[self.feature_cols])
        
        # 1. Stage 1: Approval Classification & PD Scoring
        selected_clf = self.xgb_classifier if model_choice.lower() == 'xgboost' else self.logreg_scorecard
        pred_prob_approval = float(selected_clf.predict_proba(X_trans)[0, 1])
        prob_default = float(np.clip(1.0 - pred_prob_approval, 0.005, 0.995))
        
        # Check both models for comparison
        xgb_prob = float(self.xgb_classifier.predict_proba(X_trans)[0, 1])
        logreg_prob = float(self.logreg_scorecard.predict_proba(X_trans)[0, 1])
        
        # RBI / Institutional Hard Knockout Rules
        hard_knockout_reasons = []
        if record['bankruptcy_history'] == 1 and record['credit_score'] < 680:
            hard_knockout_reasons.append("Reported NPA / Settled trade line on CIBIL with sub-700 score")
        if record['credit_score'] < 550:
            hard_knockout_reasons.append(f"CIBIL score ({record['credit_score']}) is below RBI institutional floor (550)")
        if record['debt_to_income'] > 0.55:
            hard_knockout_reasons.append(f"FOIR / DTI ({record['debt_to_income']*100:.1f}%) exceeds maximum statutory cap (55%)")
            
        is_hard_knockout = len(hard_knockout_reasons) > 0

        # Decision triage
        if is_hard_knockout:
            decision = "DECLINED"
            decision_code = "RBI_HARD_DECLINE"
        elif pred_prob_approval >= 0.70 and prob_default <= 0.20:
            decision = "APPROVED"
            decision_code = "AUTO_APPROVE"
        elif pred_prob_approval >= 0.35:
            decision = "MANUAL_REVIEW"
            decision_code = "REFER_TO_CREDIT_MANAGER"
        else:
            decision = "DECLINED"
            decision_code = "AUTO_DECLINE"
            
        # 2. Stage 2: Credit Limit Sizing via XGBoost Regressor (INR / ₹)
        if decision in ("APPROVED", "MANUAL_REVIEW"):
            predicted_limit = float(self.xgb_regressor.predict(X_trans)[0])
            # Bound within Indian retail credit limits (₹25,000 to ₹18,00,000, rounded to nearest ₹5,000)
            recommended_limit = int(np.clip(np.round(predicted_limit / 5000) * 5000, 25000, 1800000))
            if decision == "MANUAL_REVIEW":
                recommended_limit = int(np.round((recommended_limit * 0.75) / 5000) * 5000)
        else:
            recommended_limit = 0

        # 3. Stage 3: Explainability & Pricing
        adverse_reasons = generate_adverse_action_reasons(record) if decision != "APPROVED" else []
        key_strengths = generate_approval_strengths(record) if decision != "DECLINED" else []
        risk_tier = self._determine_risk_tier(record['credit_score'], prob_default)
        
        # Indian Retail Banking Credit Card / Credit Line APR structure
        apr_map = {
            "Super Prime (Tier 1 - CIBIL 780+)": "14.99% - 17.50% p.a. (Fixed Premium Rate)",
            "Prime (Tier 2 - CIBIL 710-779)": "19.50% - 24.00% p.a. (Standard Revolving)",
            "Near Prime (Tier 3 - CIBIL 630-709)": "28.00% - 36.00% p.a. (Risk-Adjusted)",
            "Subprime (Tier 4 - CIBIL < 630)": "N/A (Credit Declined / Secured FD Line Only)"
        }
        
        return {
            'applicant_id': record['applicant_id'],
            'decision': decision,
            'decision_code': decision_code,
            'model_used': 'XGBoost Classifier' if model_choice.lower() == 'xgboost' else 'Logistic Regression Scorecard',
            'probability_of_default': round(prob_default, 4),
            'approval_confidence': round(pred_prob_approval, 4),
            'model_comparison': {
                'xgboost_approval_prob': round(xgb_prob, 4),
                'logistic_regression_approval_prob': round(logreg_prob, 4)
            },
            'risk_tier': risk_tier,
            'recommended_credit_line_inr': recommended_limit,
            'formatted_credit_line': format_inr(recommended_limit) if recommended_limit > 0 else '₹0',
            'suggested_pricing_apr': apr_map.get(risk_tier, "24.00% p.a."),
            'hard_knockout_triggered': is_hard_knockout,
            'hard_knockout_reasons': hard_knockout_reasons,
            'adverse_action_reasons': adverse_reasons,
            'key_strengths': key_strengths,
            'financial_summary': {
                'annual_income': format_inr(record['annual_income']),
                'monthly_disposable': format_inr(df_single['disposable_income'].iloc[0]),
                'foir_dti_pct': f"{record['debt_to_income']*100:.1f}%",
                'cibil_score': record['credit_score'],
                'revolving_utilization_pct': f"{record['revolving_utilization']*100:.1f}%",
                'revolving_balance': format_inr(record['revolving_balance'])
            }
        }

    def batch_underwrite(self, applicants: List[Dict[str, Any]], model_choice: str = 'xgboost') -> Dict[str, Any]:
        """
        Evaluates a cohort of Indian credit applicants and returns aggregated risk book metrics.
        """
        results = [self.underwrite_applicant(app, model_choice) for app in applicants]
        
        decisions = [r['decision'] for r in results]
        total = len(results)
        approved_count = decisions.count('APPROVED')
        review_count = decisions.count('MANUAL_REVIEW')
        declined_count = decisions.count('DECLINED')
        
        approved_lines = [r['recommended_credit_line_inr'] for r in results if r['decision'] in ('APPROVED', 'MANUAL_REVIEW')]
        total_credit_extended = sum(approved_lines)
        avg_credit_line = (total_credit_extended / len(approved_lines)) if approved_lines else 0
        
        return {
            'summary': {
                'total_applications': total,
                'model_used': 'XGBoost Classifier' if model_choice.lower() == 'xgboost' else 'Logistic Regression Scorecard',
                'approved_count': approved_count,
                'approval_rate_pct': round((approved_count / total) * 100, 2) if total else 0,
                'manual_review_count': review_count,
                'review_rate_pct': round((review_count / total) * 100, 2) if total else 0,
                'declined_count': declined_count,
                'decline_rate_pct': round((declined_count / total) * 100, 2) if total else 0,
                'total_credit_line_extended_inr': total_credit_extended,
                'formatted_total_capital': format_inr(total_credit_extended),
                'average_credit_line_inr': round(avg_credit_line, 2),
                'formatted_avg_line': format_inr(avg_credit_line)
            },
            'applications': results
        }


if __name__ == '__main__':
    engine = CreditUnderwritingEngine()
    sample_app = {
        'applicant_id': 'IN-APP-MUMBAI-01',
        'age': 34,
        'annual_income': 1450000, # ₹14.5 Lakhs
        'employment_years': 7.0,
        'home_ownership': 'PARENTAL_OWNED',
        'loan_purpose': 'CREDIT_CARD_CONSOLIDATION',
        'credit_score': 765, # CIBIL 765
        'credit_history_years': 9.0,
        'debt_to_income': 0.22,
        'revolving_utilization': 0.25,
        'revolving_balance': 65000,
        'num_open_accounts': 6,
        'num_credit_inquiries_6m': 1,
        'delinquencies_2yrs': 0,
        'num_derogatory_records': 0,
        'bankruptcy_history': 0
    }
    res = engine.underwrite_applicant(sample_app, model_choice='xgboost')
    print("\nUnderwriting Decision for Prime Indian Applicant (XGBoost):")
    print(json.dumps(res, indent=2, ensure_ascii=False))
