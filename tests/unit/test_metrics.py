import pytest
import pandas as pd
from src.fairness.metrics import FairnessMetrics

def test_fairness_metrics():
    # 0 = Approved, 1 = Denied
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    
    # Sensitive attribute: 1 = Privileged, 0 = Unprivileged
    sensitive_attrs = [1, 1, 1, 0, 0, 0]
    
    metrics_calculator = FairnessMetrics()
    results = metrics_calculator.calculate_all_metrics(y_true, y_pred, sensitive_attrs, 'group')
    
    assert 'demographic_parity_difference' in results
    assert 'equalized_odds_difference' in results
    assert 'disparate_impact_ratio' in results
    
    # Basic logic check:
    # Preds for 1 (priv): [0, 0, 0] -> Approval rate = 100% (pred 0)
    # Preds for 0 (unpriv): [1, 1, 1] -> Approval rate = 0% (pred 1)
    # Demographic parity diff should be 1.0
    # Disparate impact ratio should be 0.0 (or close to it with smoothing)
    assert results['demographic_parity_difference'] == 1.0
