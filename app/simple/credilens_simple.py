import gradio as gr
import pandas as pd
import joblib
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.logger import logger

def load_system():
    try:
        base_path = Path(__file__).resolve().parent.parent.parent
        model_path = base_path / 'models' / 'saved_models' / 'xgb_model.pkl'
        prep_path = base_path / 'models' / 'saved_models' / 'preprocessor.pkl'
        
        if model_path.exists() and prep_path.exists():
            model = joblib.load(model_path)
            preprocessor = joblib.load(prep_path)
            return model, preprocessor
        return None, None
    except Exception as e:
         logger.error(f"Error loading system: {e}")
         return None, None

def analyze_risk(loan_amnt, term, int_rate, installment, emp_length, home_ownership, annual_inc, dti, fico, revol_bal, revol_util, pub_rec_bankruptcies):
    model, preprocessor = load_system()
    if model is None:
        return "System Error: Model artifacts not found. Please run the training pipeline first.", ""

    # Mock features necessary to match training schema closely
    # The actual LendingClub schema had many strings we stripped / transformed.
    input_data = pd.DataFrame([{
        'loan_amnt': loan_amnt,
        'term': float(term.split(' ')[0]),
        'int_rate': int_rate,
        'installment': installment,
        'grade': 'B', # Dummy placeholder just to pass transform
        'sub_grade': 'B1', 
        'emp_title': 'General', 
        'emp_length': float(emp_length.replace(' years', '').replace('+','').replace('< 1','0')),
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': 'Verified',
        'issue_d': pd.to_datetime('2015-12-01'),
        'purpose': 'debt_consolidation',
        'title': 'Debt consolidation',
        'zip_code': '000xx',
        'addr_state': 'CA',
        'dti': dti,
        'earliest_cr_line': pd.to_datetime('2000-08-01'),
        'fico_range_low': fico,
        'fico_range_high': fico + 4,
        'open_acc': 5,
        'pub_rec': 0,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': 10,
        'application_type': 'Individual',
        'mort_acc': 1 if home_ownership == 'MORTGAGE' else 0,
        'pub_rec_bankruptcies': pub_rec_bankruptcies
    }])
    
    try:
        processed = preprocessor.transform(input_data)
        prob = model.predict_proba(processed)[0][1]
        pred = model.predict(processed)[0]
        
        decision = "DENIED" if pred == 1 else "APPROVED"
        color = "red" if pred == 1 else "green"
        
        results_html = f"""
        <h2 style="color:{color};text-align:center;">Decision: {decision}</h2>
        <p style="text-align:center;font-size:18px;">Risk Score (Probability of Default): {prob*100:.1f}%</p>
        """
        
        suggestions = "Suggestions for approval will appear here once XAI counterfactuals are integrated." if pred == 1 else "Applicant is suitable for the loan."
        return results_html, suggestions
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return f"Error computing prediction: {e}", ""

def create_simple_interface():
    """Build the basic user-friendly Gradio UI."""
    with gr.Blocks() as app:
        gr.Markdown("# 🔍 CrediLens - Simple Credit Risk Analysis")
        gr.Markdown("Applicant Assessment Tool. Enter the applicant profile below.")
        
        with gr.Row():
            with gr.Column(scale=1):
                loan_amnt = gr.Number(label="Loan Amount ($)", value=10000)
                term = gr.Dropdown(label="Term", choices=["36 months", "60 months"], value="36 months")
                int_rate = gr.Number(label="Interest Rate (%)", value=10.5)
                installment = gr.Number(label="Monthly Installment ($)", value=325.0)
                emp_length = gr.Dropdown(label="Employment Length", choices=["< 1 year", "1 years", "2 years", "3 years", "5 years", "10+ years"], value="5 years")
                home_ownership = gr.Dropdown(label="Home Ownership", choices=["RENT", "OWN", "MORTGAGE", "ANY"], value="RENT")
                
            with gr.Column(scale=1):
                annual_inc = gr.Number(label="Annual Income ($)", value=60000)
                dti = gr.Number(label="Debt-to-Income Ratio", value=15.0)
                fico = gr.Number(label="FICO Score (Low End)", value=700)
                revol_bal = gr.Number(label="Revolving Balance ($)", value=5000)
                revol_util = gr.Number(label="Revolving Utilization (%)", value=30.0)
                pub_rec_bankruptcies = gr.Number(label="Public Bankruptcies", value=0)

        analyze_btn = gr.Button("Analyze Risk", variant="primary")
        
        gr.Markdown("---")
        
        with gr.Row():
            results = gr.HTML(label="Results")
        with gr.Row():
            suggestions = gr.Textbox(label="Actionable Steps (Recourse)", interactive=False)
            
        analyze_btn.click(
            fn=analyze_risk,
            inputs=[loan_amnt, term, int_rate, installment, emp_length, home_ownership, annual_inc, dti, fico, revol_bal, revol_util, pub_rec_bankruptcies],
            outputs=[results, suggestions]
        )
        
    return app

if __name__ == "__main__":
    app = create_simple_interface()
    app.launch(share=False, theme=gr.themes.Soft())
