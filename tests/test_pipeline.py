"""
Unit and Integration Tests for Indian Bank Credit Line Approval System (INR / ₹).
Validates XGBoost Classifier, Logistic Regression Scorecard, and XGBoost Regressor.
"""

import os
import sys
import unittest

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from inference_engine import CreditUnderwritingEngine


class TestIndianCreditUnderwriting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CreditUnderwritingEngine()

    def test_super_prime_approval_inr(self):
        """A top corporate executive should be APPROVED with generous INR limit."""
        app = {
            'applicant_id': 'TEST-PRIME-INR-01',
            'age': 42,
            'annual_income': 2800000, # ₹28 Lakhs
            'employment_years': 14.0,
            'home_ownership': 'SELF_OWNED',
            'loan_purpose': 'HOME_RENOVATION',
            'credit_score': 825, # CIBIL 825
            'credit_history_years': 16.0,
            'debt_to_income': 0.16,
            'revolving_utilization': 0.14,
            'revolving_balance': 35000,
            'num_open_accounts': 8,
            'num_credit_inquiries_6m': 0,
            'delinquencies_2yrs': 0,
            'num_derogatory_records': 0,
            'bankruptcy_history': 0
        }
        res = self.engine.underwrite_applicant(app, model_choice='xgboost')
        self.assertEqual(res['decision'], 'APPROVED')
        self.assertGreater(res['recommended_credit_line_inr'], 300000) # At least ₹3 Lakhs
        self.assertLess(res['probability_of_default'], 0.10)
        self.assertIn('Super Prime', res['risk_tier'])
        self.assertGreater(len(res['key_strengths']), 0)

    def test_subprime_npa_decline_with_rbi_reasons(self):
        """A subprime applicant with NPA/written-off record and high DTI must be DECLINED."""
        app = {
            'applicant_id': 'TEST-NPA-INR-01',
            'age': 30,
            'annual_income': 360000, # ₹3.6 Lakhs
            'employment_years': 1.5,
            'home_ownership': 'RENTED',
            'loan_purpose': 'CREDIT_CARD_CONSOLIDATION',
            'credit_score': 520, # CIBIL 520
            'credit_history_years': 3.0,
            'debt_to_income': 0.58,
            'revolving_utilization': 0.92,
            'revolving_balance': 180000,
            'num_open_accounts': 4,
            'num_credit_inquiries_6m': 5,
            'delinquencies_2yrs': 3,
            'num_derogatory_records': 2,
            'bankruptcy_history': 1
        }
        res = self.engine.underwrite_applicant(app, model_choice='xgboost')
        self.assertEqual(res['decision'], 'DECLINED')
        self.assertEqual(res['recommended_credit_line_inr'], 0)
        self.assertGreater(res['probability_of_default'], 0.50)
        self.assertGreater(len(res['adverse_action_reasons']), 0)
        
        codes = [r['code'] for r in res['adverse_action_reasons']]
        self.assertTrue('RBI_NPA_01' in codes or 'RBI_CIBIL_01' in codes or 'RBI_FOIR_01' in codes)

    def test_model_selection_comparison(self):
        """Verify both XGBoost and Logistic Regression pipelines execute."""
        app = {
            'applicant_id': 'TEST-COMPARE',
            'annual_income': 1200000,
            'credit_score': 740,
            'debt_to_income': 0.25,
            'revolving_utilization': 0.30,
            'revolving_balance': 50000
        }
        res_xgb = self.engine.underwrite_applicant(app, model_choice='xgboost')
        res_lr = self.engine.underwrite_applicant(app, model_choice='logistic_regression')
        
        self.assertEqual(res_xgb['model_used'], 'XGBoost Classifier')
        self.assertEqual(res_lr['model_used'], 'Logistic Regression Scorecard')
        self.assertIn('xgboost_approval_prob', res_xgb['model_comparison'])
        self.assertIn('logistic_regression_approval_prob', res_xgb['model_comparison'])

    def test_batch_underwrite_inr(self):
        """Verify batch portfolio aggregation with INR totals."""
        apps = [
            {'applicant_id': 'B1', 'annual_income': 1800000, 'credit_score': 780, 'debt_to_income': 0.20, 'revolving_utilization': 0.20},
            {'applicant_id': 'B2', 'annual_income': 320000, 'credit_score': 510, 'debt_to_income': 0.60, 'revolving_utilization': 0.85, 'bankruptcy_history': 1}
        ]
        batch_res = self.engine.batch_underwrite(apps, model_choice='xgboost')
        self.assertEqual(batch_res['summary']['total_applications'], 2)
        self.assertIn('formatted_total_capital', batch_res['summary'])
        self.assertIn('₹', batch_res['summary']['formatted_total_capital'])


if __name__ == '__main__':
    unittest.main()
