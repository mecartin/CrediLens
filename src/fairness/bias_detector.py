from .metrics import FairnessMetrics

class BiasDetector:
    """Detect bias in model predictions."""
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.metrics_calculator = FairnessMetrics()
        
    def detect_bias(self, y_true, y_pred, sensitive_attrs, sensitive_feature_name):
        metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred, sensitive_attrs, sensitive_feature_name)
        
        warnings = []
        if metrics['demographic_parity_difference'] > self.thresholds.get('demographic_parity', 0.1):
            warnings.append("Demographic Parity exceeded threshold.")
            
        if metrics['equalized_odds_difference'] > self.thresholds.get('equalized_odds', 0.1):
            warnings.append("Equalized Odds difference exceeded threshold.")
            
        di_range = self.thresholds.get('disparate_impact', [0.8, 1.25])
        if not (di_range[0] <= metrics['disparate_impact_ratio'] <= di_range[1]):
            warnings.append("Disparate Impact ratio is outside 80% rule (4/5ths rule).")
            
        return {
            'metrics': metrics,
            'bias_detected': len(warnings) > 0,
            'warnings': warnings
        }
