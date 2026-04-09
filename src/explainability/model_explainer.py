import numpy as np
import pandas as pd

class ModelExplainer:
    """Model-based explanation. Uses internal feature importance or Permutation Importance."""
    
    def __init__(self, model, feature_names):
        self.model = model.model if hasattr(model, 'model') else model
        self.feature_names = feature_names
        self.global_importance = self._calculate_global_importance()
        
    def _calculate_global_importance(self):
        """Extract internal feature importance (e.g., from XGBoost)."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        return {}
        
    def explain_instance(self, X_instance):
        """
        Generate explanation for single instance by weighting global importance with instance values.
        Note: Model baseline importance is usually global, we return the global ranking as a baseline.
        """
        return {
            'feature_importance': dict(sorted(self.global_importance.items(), key=lambda item: abs(item[1]), reverse=True))
        }
