import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ..core.logger import logger

class ExplanationQualityAssessment:
    """Evaluate quality of explanations using multiple criteria."""
    
    def __init__(self, model, X_background):
        self.model = model
        self.X_background = X_background
        
    def assess_fidelity(self, explanation, instance, perturbation_samples=100):
        """Fidelity: How well does explanation match model behavior?"""
        return {'fidelity_score': 0.85, 'is_high_quality': True} # Placeholder logic
    
    def assess_stability(self, instance, n_perturbations=100, epsilon=0.01):
        """Stability: Explanations robust to small changes?"""
        return {'stability_score': 0.90, 'is_stable': True} # Placeholder logic
    
    def assess_compactness(self, explanation, threshold=0.01):
        """Compactness: How many features are really needed?"""
        if 'feature_importance' in explanation:
             imp = explanation['feature_importance']
        elif 'feature_weights' in explanation:
             imp = explanation['feature_weights']
        else:
             imp = {}
        
        important_features = [f for f, val in imp.items() if abs(val) > threshold]
        compactness = 1.0 / (1.0 + len(important_features) / 10)
        
        return {
            'compactness_score': compactness,
            'n_important_features': len(important_features),
            'important_features': important_features,
            'is_compact': len(important_features) <= 5
        }
    
    def comprehensive_assessment(self, explanation, instance):
        """Run all quality assessments."""
        report = {
            'fidelity': self.assess_fidelity(explanation, instance),
            'stability': self.assess_stability(instance),
            'compactness': self.assess_compactness(explanation)
        }
        
        weights = {'fidelity': 0.5, 'stability': 0.3, 'compactness': 0.2}
        overall = sum(
            report[metric][f'{metric}_score'] * weight
            for metric, weight in weights.items()
        )
        
        report['overall_quality'] = overall
        report['quality_grade'] = self._assign_grade(overall)
        return report
        
    def _assign_grade(self, score):
        if score >= 0.9: return 'A'
        if score >= 0.8: return 'B'
        if score >= 0.7: return 'C'
        if score >= 0.6: return 'D'
        return 'F'
