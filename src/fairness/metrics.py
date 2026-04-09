from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class FairnessMetrics:
    """Calculate comprehensive fairness metrics."""
    def calculate_all_metrics(self, y_true, y_pred, sensitive_attrs, sensitive_feature_name):
        """
        Calculate demographic parity, equalized odds, and disparate impact.
        """
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_attrs)
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_attrs)
        di_ratio = self._calculate_disparate_impact(y_pred, sensitive_attrs)
        
        return {
            'demographic_parity_difference': dp_diff,
            'equalized_odds_difference': eo_diff,
            'disparate_impact_ratio': di_ratio,
            'sensitive_feature': sensitive_feature_name
        }
        
    def _calculate_disparate_impact(self, y_pred, sensitive_attrs):
        """
        Disparate Impact = P(Y=1|A=0) / P(Y=1|A=1)
        Here we assume A=1 is privileged, A=0 is unprivileged. Or simply output the ratio for the two largest groups.
        """
        df = pd.DataFrame({'pred': y_pred, 'attr': sensitive_attrs})
        rates = df.groupby('attr')['pred'].mean()
        
        if len(rates) < 2: return 1.0 # only 1 group
        
        # Simple ratio of min / max approval (treating 0 as approval here, so actually 1-pred)
        # Using 1-pred assuming 0 is 'Approved' success
        rates_approval = 1.0 - rates
        val_min = rates_approval.min()
        val_max = rates_approval.max()
        
        return val_min / max(val_max, 1e-6)
