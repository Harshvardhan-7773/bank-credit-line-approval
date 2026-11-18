"""
Banking Regulatory Explainability Engine (RBI Fair Practices Code & Adverse Action Reasons).
Provides regulatory reason codes under RBI Credit Card Directions and
CIBIL credit score diagnostics in Indian Rupee (₹) terms.
"""

from typing import Dict, List, Any


def format_inr(val: float) -> str:
    """Formats numbers into standard Indian Rupee notation (e.g. ₹12,50,000)."""
    s = f"{int(round(val)):d}"
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    chunks = []
    while len(remaining) > 2:
        chunks.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        chunks.insert(0, remaining)
    return f"₹{','.join(chunks)},{last_three}"


def generate_adverse_action_reasons(applicant_record: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Evaluates applicant credit attributes against RBI underwriting risk thresholds
    to generate prioritized principal Adverse Action Reasons.
    """
    reasons = []
    
    # Check 1: Written-off / NPA / Bankruptcy History
    if applicant_record.get('bankruptcy_history', 0) == 1:
        reasons.append({
            'code': 'RBI_NPA_01',
            'severity': 100,
            'title': 'Suit Filed / NPA / Written-Off Status on CIBIL',
            'detail': 'Applicant has a delinquent loan settlement, NPA, or written-off trade line reported on bureau file.'
        })
        
    # Check 2: Severe Delinquencies (Days Past Due - DPD)
    delinq = applicant_record.get('delinquencies_2yrs', 0)
    if delinq >= 2:
        reasons.append({
            'code': 'RBI_DPD_01',
            'severity': 90,
            'title': 'Multiple 30-90+ Days Past Due (DPD) Delinquencies',
            'detail': f'{delinq} past-due derogatory repayment defaults recorded within past 24 months.'
        })
    elif delinq == 1:
        reasons.append({
            'code': 'RBI_DPD_02',
            'severity': 65,
            'title': 'Recent Past-Due Payment Delinquency (DPD)',
            'detail': 'Applicant had 1 past-due loan or credit card delinquency in the past 24 months.'
        })
        
    # Check 3: Derogatory / Legal Records
    derog = applicant_record.get('num_derogatory_records', 0)
    if derog >= 2:
        reasons.append({
            'code': 'RBI_DRG_01',
            'severity': 88,
            'title': 'Multiple Derogatory Bureau Records',
            'detail': f'{derog} recovery agent records, cheque bounces, or collection filings on file.'
        })
    elif derog == 1:
        reasons.append({
            'code': 'RBI_DRG_02',
            'severity': 60,
            'title': 'Derogatory Record Reported to Credit Bureau',
            'detail': '1 derogatory public record or third-party collection filing reported.'
        })
        
    # Check 4: TransUnion CIBIL Score (Scale 300 - 900)
    cibil = applicant_record.get('credit_score', 700)
    if cibil < 620:
        reasons.append({
            'code': 'RBI_CIBIL_01',
            'severity': 85,
            'title': 'CIBIL Bureau Score Below Underwriting Guideline',
            'detail': f'CIBIL score of {cibil} is below institutional prime threshold (minimum 700+ recommended by RBI guidelines).'
        })
    elif cibil < 690:
        reasons.append({
            'code': 'RBI_CIBIL_02',
            'severity': 55,
            'title': 'Subprime / Borderline CIBIL Score Tier',
            'detail': f'CIBIL score of {cibil} indicates elevated probability of default under bank credit policy.'
        })
        
    # Check 5: Debt-To-Income (DTI / FOIR)
    dti = applicant_record.get('debt_to_income', 0.3)
    if dti >= 0.50:
        reasons.append({
            'code': 'RBI_FOIR_01',
            'severity': 80,
            'title': 'Excessive Fixed Obligation to Income Ratio (FOIR)',
            'detail': f'FOIR / DTI of {dti*100:.1f}% exceeds institutional underwriting threshold (45%-50% cap).'
        })
    elif dti >= 0.40:
        reasons.append({
            'code': 'RBI_FOIR_02',
            'severity': 50,
            'title': 'Elevated Existing Debt Service Burden (FOIR)',
            'detail': f'Debt-to-income ratio of {dti*100:.1f}% leaves insufficient discretionary buffer for new credit.'
        })
        
    # Check 6: Revolving Line Utilization
    util = applicant_record.get('revolving_utilization', 0.4)
    if util >= 0.75:
        reasons.append({
            'code': 'RBI_UTL_01',
            'severity': 75,
            'title': 'High Credit Card Revolving Utilization',
            'detail': f'Current credit card utilization of {util*100:.1f}% indicates high dependence on revolving debt.'
        })
    elif util >= 0.55:
        reasons.append({
            'code': 'RBI_UTL_02',
            'severity': 45,
            'title': 'Elevated Credit Card Utilization',
            'detail': f'Revolving balance utilization of {util*100:.1f}% exceeds optimal benchmark (<30%).'
        })
        
    # Check 7: Hard Credit Inquiries
    inquiries = applicant_record.get('num_credit_inquiries_6m', 0)
    if inquiries >= 4:
        reasons.append({
            'code': 'RBI_INQ_01',
            'severity': 60,
            'title': 'Excessive Recent Credit Inquiries',
            'detail': f'{inquiries} hard bureau inquiries in past 6 months reflects credit-hungry behavior.'
        })
        
    # Check 8: Credit Vintage
    vintage = applicant_record.get('credit_history_years', 5.0)
    if vintage < 2.0:
        reasons.append({
            'code': 'RBI_VIN_01',
            'severity': 50,
            'title': 'Thin Credit File / Short Credit History',
            'detail': f'Credit file vintage of {vintage:.1f} years provides insufficient historical repayment track.'
        })
        
    reasons.sort(key=lambda x: x['severity'], reverse=True)
    return reasons[:4]


def generate_approval_strengths(applicant_record: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Identifies key positive credit characteristics for approved applications in INR.
    """
    strengths = []
    
    cibil = applicant_record.get('credit_score', 700)
    if cibil >= 790:
        strengths.append({
            'title': 'Super-Prime CIBIL Score (790+)',
            'detail': f'Exceptional CIBIL score of {cibil} qualifies for top-tier institutional pricing and highest credit limit.'
        })
    elif cibil >= 720:
        strengths.append({
            'title': 'Strong Prime CIBIL Profile',
            'detail': f'CIBIL score of {cibil} exceeds standard retail banking benchmark (700+).'
        })
        
    dti = applicant_record.get('debt_to_income', 0.3)
    if dti <= 0.25:
        strengths.append({
            'title': 'Comfortable Debt Obligation (FOIR < 25%)',
            'detail': f'Low debt-to-income of {dti*100:.1f}% provides substantial monthly surplus cash flow.'
        })
        
    util = applicant_record.get('revolving_utilization', 0.4)
    if util <= 0.25:
        strengths.append({
            'title': 'Disciplined Credit Card Utilization',
            'detail': f'Revolving utilization is {util*100:.1f}%, well below the industry 30% guideline.'
        })
        
    income = applicant_record.get('annual_income', 600000)
    if income >= 1800000:
        strengths.append({
            'title': 'High Verifiable Annual Income',
            'detail': f'Annual verified income of {format_inr(income)} provides superior debt absorption capacity.'
        })
    elif income >= 1000000:
        strengths.append({
            'title': 'Stable Salaried Cash Flows',
            'detail': f'Annual verified income of {format_inr(income)} satisfies prime credit line guidelines.'
        })
        
    vintage = applicant_record.get('credit_history_years', 5.0)
    if vintage >= 6.0:
        strengths.append({
            'title': 'Seasoned CIBIL Credit Vintage',
            'detail': f'{vintage:.1f} years of established bureau credit behavior.'
        })
        
    if applicant_record.get('delinquencies_2yrs', 0) == 0 and applicant_record.get('num_derogatory_records', 0) == 0:
        strengths.append({
            'title': 'Flawless Repayment Track Record',
            'detail': 'Zero DPD delinquencies and zero written-off records across all credit facilities.'
        })
        
    return strengths[:4]
