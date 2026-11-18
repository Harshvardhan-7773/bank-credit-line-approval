"""
Exploratory Data Analysis and Model Governance Audit Script (Indian Banking Portfolio in INR / ₹).
Analyzes CIBIL score distributions, Indian income brackets, FOIR correlations, and approval rates.
"""

import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_eda_audit(base_dir: str):
    raw_csv = os.path.join(base_dir, 'data', 'raw', 'credit_applications.csv')
    df = pd.read_csv(raw_csv)
    
    print("=" * 75)
    print(" APEXBANK BHARAT CREDIT PORTFOLIO: EXPLORATORY DATA ANALYSIS (INR / ₹)")
    print("=" * 75)
    print(f"Total Applications Analyzed: {len(df):,}")
    print(f"Overall Institutional Approval Rate: {df['approval_status'].mean() * 100:.2f}%")
    approved_df = df[df['approval_status'] == 1]
    print(f"Mean Sanctioned Credit Line Extended: Rs. {approved_df['approved_credit_line'].mean():,.2f}")
    print(f"Median Sanctioned Credit Line: Rs. {approved_df['approved_credit_line'].median():,.2f}")
    
    print("\n--- TransUnion CIBIL Score Tier Breakdown ---")
    bins = [300, 600, 700, 780, 850, 900]
    labels = ['Subprime (<600)', 'Near Prime (600-699)', 'Prime (700-779)', 'Super Prime (780-849)', 'Exceptional (850+)']
    df['cibil_tier'] = pd.cut(df['credit_score'], bins=bins, labels=labels, right=False)
    
    tier_summary = df.groupby('cibil_tier', observed=False).agg(
        total_apps=('applicant_id', 'count'),
        approval_rate=('approval_status', lambda x: f"{x.mean()*100:.1f}%"),
        avg_income=('annual_income', lambda x: f"Rs. {x.mean():,.0f}"),
        avg_foir_dti=('debt_to_income', lambda x: f"{x.mean()*100:.1f}%"),
        avg_credit_line=('approved_credit_line', lambda x: f"Rs. {x[x > 0].mean():,.0f}" if len(x[x > 0]) > 0 else "Rs. 0")
    )
    print(tier_summary.to_string())
    
    print("\n--- Residential Status vs Approval Rate ---")
    home_summary = df.groupby('home_ownership').agg(
        total=('applicant_id', 'count'),
        approval_rate=('approval_status', lambda x: f"{x.mean()*100:.1f}%"),
        avg_limit=('approved_credit_line', lambda x: f"Rs. {x[x > 0].mean():,.0f}" if len(x[x > 0]) > 0 else "Rs. 0")
    )
    print(home_summary.to_string())

    print("\n--- Correlation Matrix with Approval Status ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()['approval_status'].sort_values(ascending=False)
    print(corr.to_string())
    print("=" * 75)


if __name__ == '__main__':
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    run_eda_audit(base_path)
