from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .model_explainer import ModelExplainer
from .disagreement_detector import ExplanationDisagreementDetector
from ..core.logger import logger

class EnsembleExplainer:
    """Combine SHAP, LIME, and model-based explanations."""
    
    def __init__(self, model, X_train, feature_names):
        self.shap_explainer = SHAPExplainer(model, X_train)
        self.lime_explainer = LIMEExplainer(
            model.predict_proba if hasattr(model, 'predict_proba') else model, 
            X_train, 
            feature_names
        )
        self.model_explainer = ModelExplainer(model, feature_names)
        self.disagreement_detector = ExplanationDisagreementDetector(threshold=0.3)
        self.feature_names = feature_names
        
    def explain_with_ensemble(self, X_instance):
        """Generate multi-method explanation."""
        logger.info("Generating ensemble explanations...")
        
        shap_exp = self.shap_explainer.explain_instance(X_instance)
        lime_exp = self.lime_explainer.explain_instance(X_instance)
        model_exp = self.model_explainer.explain_instance(X_instance)
        
        disagreement_result = self.disagreement_detector.detect_disagreement(
            shap_exp, lime_exp, model_exp
        )
        
        consensus = self._consensus_ranking([shap_exp, lime_exp, model_exp])
        
        return {
            'methods': {
                'shap': shap_exp,
                'lime': lime_exp,
                'model': model_exp
            },
            'consensus': consensus,
            'disagreement': disagreement_result
        }
    
    def _consensus_ranking(self, explanations):
        """Simple rank aggregation via Borda count."""
        rank_scores = {feat: 0 for feat in self.feature_names}
        num_methods = len(explanations)
        
        for exp in explanations:
             # get rank order (from 1 to N, lower is more important)
             if 'feature_importance' in exp:
                  imp = exp['feature_importance']
             else:
                  imp = exp['feature_weights']
                  
             ranked_features = list(dict(sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)).keys())
             
             for idx, feat in enumerate(ranked_features):
                  # Borda count: max points = N, least important = 1 point
                  points = len(self.feature_names) - idx
                  if feat in rank_scores:
                       rank_scores[feat] += points
                       
        return dict(sorted(rank_scores.items(), key=lambda item: item[1], reverse=True))
