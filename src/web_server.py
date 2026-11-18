"""
Production Indian Credit Underwriting HTTP Server and REST API (INR / ₹).
Powered by XGBoost and Logistic Regression.
Zero external server dependencies (pure standard library http.server).
"""

import os
import sys
import json
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

from inference_engine import CreditUnderwritingEngine

ENGINE = CreditUnderwritingEngine()
BASE_DIR = os.path.abspath(os.path.join(CURR_DIR, '..'))
WEB_DIR = os.path.join(BASE_DIR, 'web')

SAMPLE_PRESETS = [
    {
        'id': 'tech_lead_bangalore',
        'name': '1. Senior Tech Lead (Super Prime)',
        'description': 'High income (₹32 Lakhs), CIBIL 825, low DTI/FOIR (16%), spotless repayment.',
        'data': {
            'applicant_id': 'IN-APP-BLR-825',
            'age': 38,
            'annual_income': 3200000,
            'employment_years': 12.0,
            'home_ownership': 'SELF_OWNED',
            'loan_purpose': 'HOME_RENOVATION',
            'credit_score': 825,
            'credit_history_years': 15.0,
            'debt_to_income': 0.16,
            'revolving_utilization': 0.14,
            'revolving_balance': 45000,
            'num_open_accounts': 9,
            'num_credit_inquiries_6m': 0,
            'delinquencies_2yrs': 0,
            'num_derogatory_records': 0,
            'bankruptcy_history': 0
        }
    },
    {
        'id': 'salaried_consultant_delhi',
        'name': '2. Corporate Consultant (Prime Builder)',
        'description': 'Stable MNC salary (₹13.5 Lakhs), CIBIL 745, utilization 30%, clean CIBIL track.',
        'data': {
            'applicant_id': 'IN-APP-DEL-745',
            'age': 29,
            'annual_income': 1350000,
            'employment_years': 5.5,
            'home_ownership': 'RENTED',
            'loan_purpose': 'CREDIT_CARD_CONSOLIDATION',
            'credit_score': 745,
            'credit_history_years': 7.0,
            'debt_to_income': 0.26,
            'revolving_utilization': 0.30,
            'revolving_balance': 75000,
            'num_open_accounts': 6,
            'num_credit_inquiries_6m': 1,
            'delinquencies_2yrs': 0,
            'num_derogatory_records': 0,
            'bankruptcy_history': 0
        }
    },
    {
        'id': 'trader_review_mumbai',
        'name': '3. MSME Trader (Manual Credit Review)',
        'description': 'Annual income (₹7.2 Lakhs), CIBIL 665, elevated DTI (44%), 1 past delinquency.',
        'data': {
            'applicant_id': 'IN-APP-BOM-665',
            'age': 35,
            'annual_income': 720000,
            'employment_years': 4.0,
            'home_ownership': 'PARENTAL_OWNED',
            'loan_purpose': 'BUSINESS_WORKING_CAPITAL',
            'credit_score': 665,
            'credit_history_years': 5.5,
            'debt_to_income': 0.44,
            'revolving_utilization': 0.65,
            'revolving_balance': 160000,
            'num_open_accounts': 5,
            'num_credit_inquiries_6m': 3,
            'delinquencies_2yrs': 1,
            'num_derogatory_records': 0,
            'bankruptcy_history': 0
        }
    },
    {
        'id': 'distressed_npa_decline',
        'name': '4. Defaulted Borrower (RBI Adverse Decline)',
        'description': 'Income (₹3.6 Lakhs), CIBIL 520, severe DTI (58%), NPA written-off trade line.',
        'data': {
            'applicant_id': 'IN-APP-DECL-520',
            'age': 31,
            'annual_income': 360000,
            'employment_years': 1.5,
            'home_ownership': 'RENTED',
            'loan_purpose': 'MEDICAL_EMERGENCY',
            'credit_score': 520,
            'credit_history_years': 3.0,
            'debt_to_income': 0.58,
            'revolving_utilization': 0.94,
            'revolving_balance': 190000,
            'num_open_accounts': 4,
            'num_credit_inquiries_6m': 5,
            'delinquencies_2yrs': 3,
            'num_derogatory_records': 2,
            'bankruptcy_history': 1
        }
    }
]


class IndianCreditPortalHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type='application/json; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ('/', '/index.html'):
            file_path = os.path.join(WEB_DIR, 'index.html')
            self._serve_file(file_path, 'text/html; charset=utf-8')
            return

        if path.startswith('/static/') or path.startswith('/web/'):
            clean_rel = path.replace('/static/', '').replace('/web/', '')
            file_path = os.path.join(WEB_DIR, clean_rel)
            mime_type, _ = mimetypes.guess_type(file_path)
            self._serve_file(file_path, mime_type or 'application/octet-stream')
            return

        direct_web_file = os.path.join(WEB_DIR, path.lstrip('/'))
        if os.path.exists(direct_web_file) and os.path.isfile(direct_web_file):
            mime_type, _ = mimetypes.guess_type(direct_web_file)
            self._serve_file(direct_web_file, mime_type or 'text/plain')
            return

        if path == '/api/metadata':
            self._set_headers(200, 'application/json; charset=utf-8')
            self.wfile.write(json.dumps(ENGINE.metadata, ensure_ascii=False).encode('utf-8'))
            return

        if path == '/api/presets':
            self._set_headers(200, 'application/json; charset=utf-8')
            self.wfile.write(json.dumps(SAMPLE_PRESETS, ensure_ascii=False).encode('utf-8'))
            return

        self._set_headers(404, 'application/json; charset=utf-8')
        self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode('utf-8'))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception as e:
            self._set_headers(400, 'application/json; charset=utf-8')
            self.wfile.write(json.dumps({'error': f'Invalid JSON payload: {str(e)}'}).encode('utf-8'))
            return

        model_choice = body.get('model_choice', 'xgboost')

        if path == '/api/underwrite':
            result = ENGINE.underwrite_applicant(body, model_choice=model_choice)
            self._set_headers(200, 'application/json; charset=utf-8')
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
            return

        if path == '/api/batch':
            applicants = body.get('applicants', []) if isinstance(body, dict) else body
            result = ENGINE.batch_underwrite(applicants, model_choice=model_choice)
            self._set_headers(200, 'application/json; charset=utf-8')
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
            return

        self._set_headers(404, 'application/json; charset=utf-8')
        self.wfile.write(json.dumps({'error': 'Unknown POST endpoint'}).encode('utf-8'))

    def _serve_file(self, file_path, content_type):
        if not os.path.exists(file_path):
            self._set_headers(404, 'text/plain')
            self.wfile.write(b"404 Not Found")
            return
        with open(file_path, 'rb') as f:
            content = f.read()
        self._set_headers(200, content_type)
        self.wfile.write(content)


def run_server(port=8050):
    server_address = ('', port)
    httpd = HTTPServer(server_address, IndianCreditPortalHandler)
    print(f"\n==================================================================")
    print(f"  INDIAN BANK CREDIT LINE UNDERWRITING PORTAL (INR / Rs.)")
    print(f"  Powered by XGBoost & Logistic Regression")
    print(f"  Serving at: http://localhost:{port}")
    print(f"  Press Ctrl+C to terminate.")
    print(f"==================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
