"""
Synthetic Indian Banking Credit Line Application Dataset Generator (INR / ₹).
Generates realistic financial, behavioral, and credit bureau data adhering to
Reserve Bank of India (RBI) underwriting norms and TransUnion CIBIL score distributions.
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def generate_credit_dataset(
    n_samples: int = 16000,
    random_state: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generates a realistic credit application dataset in Indian Rupee (₹) terms.
    
    Parameters:
    -----------
    n_samples : int
        Number of customer credit applications to generate.
    random_state : int
        Random seed for reproducibility.
    output_path : str, optional
        File path to save the generated CSV.
        
    Returns:
    --------
    pd.DataFrame : Generated dataset
    """
    np.random.seed(random_state)
    
    # 1. Applicant demographics & employment
    age = np.random.triangular(21, 34, 68, size=n_samples).astype(int)
    max_emp_years = np.maximum(0, age - 21)
    employment_years = np.round(np.random.uniform(0, 1, size=n_samples) * np.minimum(max_emp_years, 32), 1)
    
    # Annual income in INR (₹) (Log-normal distribution centered around ~₹9.5 Lakhs, range ₹2.4 Lakhs - ₹60 Lakhs)
    log_income = np.random.normal(13.75, 0.58, size=n_samples)
    annual_income = np.clip(np.exp(log_income), 240000, 6000000)
    # Round to nearest ₹5,000
    annual_income = np.round(annual_income / 5000) * 5000
    
    home_ownership_choices = ['RENTED', 'PARENTAL_OWNED', 'MORTGAGED', 'SELF_OWNED']
    home_probs = []
    for inc, a in zip(annual_income, age):
        if inc > 1500000 or a > 38:
            home_probs.append([0.22, 0.15, 0.42, 0.21])
        else:
            home_probs.append([0.55, 0.28, 0.12, 0.05])
    
    home_ownership = [np.random.choice(home_ownership_choices, p=p) for p in home_probs]
    
    loan_purpose_choices = [
        'CREDIT_CARD_CONSOLIDATION', 
        'HOME_RENOVATION', 
        'MEDICAL_EMERGENCY', 
        'WEDDING_EXPENSE', 
        'BUSINESS_WORKING_CAPITAL', 
        'CONSUMER_DURABLES'
    ]
    loan_purpose_probs = [0.36, 0.22, 0.16, 0.12, 0.09, 0.05]
    loan_purpose = np.random.choice(loan_purpose_choices, size=n_samples, p=loan_purpose_probs)
    
    # 2. TransUnion CIBIL Credit Bureau History (Score range: 300 - 900 in India)
    base_cibil = np.random.normal(725, 70, size=n_samples)
    # Adjust CIBIL with age and income stability
    cibil_adj = (age - 32) * 1.0 + np.log(annual_income / 500000) * 18
    credit_score = np.clip(base_cibil + cibil_adj, 300, 900).astype(int)
    
    # Credit history length in years (CIBIL vintage)
    credit_history_years = np.clip(
        np.round((age - 21) * np.random.uniform(0.35, 0.85, size=n_samples), 1),
        0.5, 40.0
    )
    
    # Debt-To-Income (DTI / FOIR - Fixed Obligation to Income Ratio) (0.05 to 0.70)
    dti_mean = 0.32 - (credit_score - 720) * 0.0004
    dti_mean = np.clip(dti_mean, 0.15, 0.50)
    debt_to_income = np.clip(np.random.normal(dti_mean, 0.09), 0.05, 0.70)
    debt_to_income = np.round(debt_to_income, 3)
    
    # Revolving Utilization Rate (Credit card limit utilization: 0.01 to 1.15)
    util_mean = 0.42 - (credit_score - 720) * 0.0011
    util_mean = np.clip(util_mean, 0.10, 0.85)
    revolving_utilization = np.clip(np.random.normal(util_mean, 0.18), 0.01, 1.15)
    revolving_utilization = np.round(revolving_utilization, 3)
    
    # Active credit lines/cards (1 to 25)
    num_open_accounts = np.clip(np.random.poisson(6 + (credit_history_years / 3.5)), 1, 30).astype(int)
    
    # Credit inquiries in past 6 months (Hard bureau pulls)
    inquiry_mean = np.clip(1.2 + (0.5 - (credit_score - 680) * 0.003), 0.2, 4.5)
    num_credit_inquiries_6m = np.clip(np.random.poisson(inquiry_mean), 0, 12).astype(int)
    
    # Delinquencies in past 2 years (DPD: 30-90+ days past due)
    delinq_prob = np.clip(0.32 - (credit_score - 640) * 0.0015, 0.02, 0.75)
    delinquencies_2yrs = (np.random.uniform(0, 1, size=n_samples) < delinq_prob).astype(int) * np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.65, 0.20, 0.10, 0.05])
    
    # Derogatory / Wilful default / written-off trade lines
    derog_prob = np.clip(0.18 - (credit_score - 660) * 0.001, 0.01, 0.50)
    num_derogatory_records = (np.random.uniform(0, 1, size=n_samples) < derog_prob).astype(int) * np.random.choice([1, 2, 3], size=n_samples, p=[0.75, 0.18, 0.07])
    
    # Bankruptcy / NPA (Non-Performing Asset / Written-off) history flag
    npa_prob = np.where(credit_score < 580, 0.28, np.where(credit_score < 650, 0.08, 0.01))
    bankruptcy_history = (np.random.uniform(0, 1, size=n_samples) < npa_prob).astype(int)
    
    # Existing revolving credit balance in INR (₹)
    monthly_income = annual_income / 12.0
    approx_existing_limit = monthly_income * np.random.uniform(1.5, 4.0, size=n_samples)
    revolving_balance = np.round(np.clip(approx_existing_limit * revolving_utilization, 0, 1500000), -2)
    
    # 3. Latent Credit Risk Score / Probability of Default (PD) Modeling
    z_score = (
        - 3.20
        + 0.016 * (720 - credit_score)
        + 4.10 * debt_to_income
        + 2.50 * revolving_utilization
        + 0.55 * delinquencies_2yrs
        + 1.10 * num_derogatory_records
        + 1.70 * bankruptcy_history
        + 0.18 * num_credit_inquiries_6m
        - 0.35 * np.log(annual_income / 800000)
        - 0.030 * employment_years
        - 0.020 * credit_history_years
        + np.random.normal(0, 0.35, size=n_samples)
    )
    prob_default = 1.0 / (1.0 + np.exp(-z_score))
    
    # RBI / Institutional Hard Knock-out Underwriting Policies:
    # 1. NPA / Written-off / Bankruptcy record with CIBIL < 680 -> Decline
    # 2. CIBIL < 550 -> Auto-decline
    # 3. DTI (FOIR) > 0.55 -> Auto-decline
    # 4. Severe derogatory records (>= 3) -> Auto-decline
    hard_knockout = (
        ((bankruptcy_history == 1) & (credit_score < 680)) |
        (credit_score < 550) |
        (debt_to_income > 0.55) |
        (num_derogatory_records >= 3)
    )
    
    # Approval decision
    approval_status = np.where(hard_knockout, 0, np.where(prob_default <= 0.22, 1, 0))
    
    # 4. Approved Credit Line Sizing in INR (₹)
    # Disposable Monthly Cash Flow = Monthly Income * (1 - DTI)
    disposable_monthly = monthly_income * (1.0 - np.minimum(debt_to_income, 0.60))
    
    # CIBIL tier multiplier for credit line limit
    cibil_factor = np.where(
        credit_score >= 800, 4.0,
        np.where(credit_score >= 750, 3.0,
        np.where(credit_score >= 700, 2.2, 1.4))
    )
    
    risk_discount = np.clip(1.0 - (prob_default * 1.4), 0.35, 1.0)
    raw_credit_line = disposable_monthly * cibil_factor * risk_discount
    raw_credit_line *= np.random.uniform(0.92, 1.08, size=n_samples)
    
    # Indian credit card / line bounds: ₹25,000 to ₹18,00,000, rounded to nearest ₹5,000
    capped_line = np.clip(np.round(raw_credit_line / 5000) * 5000, 25000, 1800000)
    
    approved_credit_line = np.where(approval_status == 1, capped_line, 0.0).astype(int)
    
    # Build DataFrame
    df = pd.DataFrame({
        'applicant_id': [f'IN-APP-{10000 + i}' for i in range(n_samples)],
        'age': age,
        'annual_income': annual_income,
        'employment_years': employment_years,
        'home_ownership': home_ownership,
        'loan_purpose': loan_purpose,
        'credit_score': credit_score, # CIBIL Score (300-900)
        'credit_history_years': credit_history_years,
        'debt_to_income': debt_to_income,
        'revolving_utilization': revolving_utilization,
        'revolving_balance': revolving_balance,
        'num_open_accounts': num_open_accounts,
        'num_credit_inquiries_6m': num_credit_inquiries_6m,
        'delinquencies_2yrs': delinquencies_2yrs,
        'num_derogatory_records': num_derogatory_records,
        'bankruptcy_history': bankruptcy_history,
        'probability_of_default': np.round(prob_default, 4),
        'approval_status': approval_status,
        'approved_credit_line': approved_credit_line
    })
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {n_samples} Indian credit applications and saved to: {output_path}")
        print(f"Institutional Approval Rate: {df['approval_status'].mean() * 100:.2f}%")
        approved_lines = df[df['approval_status'] == 1]['approved_credit_line']
        print(f"Average Approved Credit Line: ₹{approved_lines.mean():,.2f} (Median: ₹{approved_lines.median():,.2f})")
    
    return df


if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    csv_file = os.path.join(data_dir, 'credit_applications.csv')
    generate_credit_dataset(n_samples=16000, random_state=42, output_path=csv_file)
