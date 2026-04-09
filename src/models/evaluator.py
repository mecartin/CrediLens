import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from ..core.logger import logger

class Evaluator:
    """Comprehensive evaluation metrics for the risk model."""
    
    def evaluate(self, model, X_test, y_test) -> dict:
        """Calculate complete suite of performance metrics."""
        logger.info("Evaluating model performance on test set.")
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, probs),
            'f1': f1_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'confusion_matrix': confusion_matrix(y_test, preds).tolist()
        }
        
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        
        return metrics
        
    def tune_threshold(self, model, X_val, y_val):
        """
        Optimize decision threshold for F1-score as requested by the config.
        """
        logger.info("Tuning prediction threshold...")
        probs = model.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                
        logger.info(f"Best threshold found: {best_threshold:.2f} with F1: {best_f1:.4f}")
        return best_threshold
