import datetime
import pandas as pd
import numpy as np

class AuditReportGenerator:
    """Consolidated report generator for Model Governance and Risk."""
    
    def __init__(self, model_name="XGBoost Credit Scorer"):
        self.model_name = model_name
        
    def generate_markdown_audit(self, performance_metrics: dict, fairness_metrics: dict, risk_summary: dict):
        """Build a formatted technical audit report."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# CrediLens Technical Audit Report
**Model:** {self.model_name}
**Generated:** {now}

---

## 1. Executive Summary
The model was audited for predictive performance, demographic fairness, and financial risk. 

---

## 2. Model Performance
Current performance on the evaluation set:
- **Accuracy:** {performance_metrics.get('accuracy', 0):.2%}
- **ROC-AUC:** {performance_metrics.get('auc', 0):.4f}
- **Precision (Approved):** {performance_metrics.get('precision', 0):.2%}

---

## 3. Fairness & Compliance Audit
Targeting the **'home_ownership'** protected attribute:
- **Demographic Parity Difference:** {fairness_metrics.get('parity_diff', 0):.4f}
- **Equalized Odds Difference:** {fairness_metrics.get('odds_diff', 0):.4f}
- **Adverse Impact Ratio (AIR):** {fairness_metrics.get('air', 0):.4f}
  *Compliance Note: AIR values below 0.80 may indicate potential bias under the Four-Fifths Rule.*

---

## 4. Financial Risk Exposure
Aggregated portfolio impact based on the current data batch:
- **Portfolio Volume:** ${risk_summary.get('total_volume', 0):,.0f}
- **Expected Loss (EL):** ${risk_summary.get('expected_loss', 0):,.0f}
- **Portfolio Default Rate:** {risk_summary.get('avg_default_rate', 0):.2%}
- **Net Expected Interest:** ${risk_summary.get('expected_interest', 0):,.0f}

---

## 5. Decision Stability
- **Global Robustness Score:** {risk_summary.get('global_stability', 0):.2%} 
  *(Calculated via stochastic perturbation of numeric inputs)*

---
*End of Report*
"""
        return report

    def save_report(self, markdown_content, filepath):
        with open(filepath, 'w') as f:
            f.write(markdown_content)
        return filepath
