import numpy as np
import pandas as pd

class BiasMitigation:
    """Implement bias mitigation strategies."""
    
    def apply_reweighting(self, y, sensitive_attrs):
        """
        Calculate sample weights to counteract bias using the Reweighing algorithm.
        Logic: weight = (P(y) * P(A)) / P(y, A)
        
        Args:
            y: Target labels (Series or array)
            sensitive_attrs: Protected attribute values (Series or array)
        Returns:
            np.array: Sample weights for each instance.
        """
        df = pd.DataFrame({'y': y, 'a': sensitive_attrs})
        n = len(df)
        
        # 1. P(y)
        p_y0 = len(df[df['y'] == 0]) / n
        p_y1 = len(df[df['y'] == 1]) / n
        
        # 2. P(A)
        unique_a = df['a'].unique()
        p_a = {a: len(df[df['a'] == a]) / n for a in unique_a}
        
        # 3. P(y, A)
        weights = np.ones(n)
        
        for y_val in [0, 1]:
            for a_val in unique_a:
                # Count for (y_val, a_val)
                p_y_a = len(df[(df['y'] == y_val) & (df['a'] == a_val)]) / n
                
                # Expected if independent: P(y) * P(A)
                p_exp = (p_y0 if y_val == 0 else p_y1) * p_a[a_val]
                
                if p_y_a > 0:
                    weight_val = p_exp / p_y_a
                    # Apply weight to these instances
                    idx = df[(df['y'] == y_val) & (df['a'] == a_val)].index
                    weights[idx] = weight_val
                    
        return weights
        
    def apply_post_processing(self, y_pred_proba, sensitive_attrs):
        """
        Stub for threshold adjustment (Future expansion).
        """
        preds = (np.array(y_pred_proba) >= 0.5).astype(int)
        return preds
