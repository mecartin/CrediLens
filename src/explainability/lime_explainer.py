import lime
import lime.lime_tabular
import numpy as np

class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations)."""
    
    def __init__(self, model_predict_proba_fn, X_train, feature_names, class_names=['Approved', 'Denied']):
        self.predict_proba_fn = model_predict_proba_fn
        self.feature_names = list(feature_names)
        
        # Initialize LIME Tabular Explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
        
    def explain_instance(self, X_instance, num_features=10):
        """
        Generate LIME explanation.
        """
        # Ensure it's a 1D array/series for LIME
        if hasattr(X_instance, 'iloc'):
            X_inst_values = X_instance.iloc[0].values if len(X_instance.shape) > 1 else X_instance.values
        else:
            X_inst_values = X_instance[0] if len(X_instance.shape) > 1 else X_instance
            
        explanation = self.explainer.explain_instance(
            X_inst_values,
            self.predict_proba_fn,
            num_features=num_features,
            num_samples=5000 # Samples for local model
        )
        
        # Convert explanation list back to feature map format
        feature_weights = {self.feature_names[ft_idx]: weight for ft_idx, weight in explanation.local_exp[1]}
        
        return {
            'feature_weights': feature_weights,
            'score': explanation.score,
            'local_pred': explanation.local_pred,
            'intercept': explanation.intercept[1]
        }
