import shap
import numpy as np

class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) implementation."""
    
    def __init__(self, model, X_background):
        # We use TreeExplainer for XGBoost to be fast and exact
        self.explainer = shap.TreeExplainer(model.model) if hasattr(model, 'model') else shap.TreeExplainer(model)
        # Sample background for memory efficiency on Global explanations if needed
        self.X_background = shap.sample(X_background, min(100, len(X_background)))
        
    def explain_instance(self, X_instance):
        """
        Generate SHAP values for single instance.
        """
        shap_values = self.explainer.shap_values(X_instance)
        
        # If output is list (multiclass), grab the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[-1] # Grabbing probability of positive class
            
        feature_importance = self._rank_features(shap_values, X_instance.columns)
        
        return {
            'shap_values': shap_values, # Array of shap values per feature
            'base_value': base_value,
            'feature_importance': feature_importance
        }
        
    def _rank_features(self, shap_values, feature_names):
        """Ranks features by absolute impact."""
        if len(shap_values.shape) > 1:
             # Just taking the first instance assuming explain_instance is passed a single row
             values = shap_values[0]
        else:
             values = shap_values
             
        importance = {name: val for name, val in zip(feature_names, values)}
        # Sort by absolute magnitude
        return dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))

    def explain_global(self, X_sample):
        """Generate global explanations (to be implemented more fully in visualization layer)."""
        return self.explainer.shap_values(X_sample)
