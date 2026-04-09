import xgboost as xgb
import numpy as np

from ..core.logger import logger
from ..core.exceptions import ModelNotTrainedError

class XGBoostModel:
    """Wrapper around XGBoost classifier with custom thresholding."""
    def __init__(self, params: dict = None, threshold: float = 0.5):
        self.model = None
        self.params = params or {}
        self.threshold = threshold
        self.is_trained = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("Training XGBoost Model...")
        
        # GPU detection and configuration
        gpu_params = {
            'tree_method': 'hist',
            'device': 'cuda'
        }
        
        final_params = {**self.params, **gpu_params}
        
        if X_val is not None and y_val is not None:
             self.model = xgb.XGBClassifier(**final_params, early_stopping_rounds=20)
             eval_set = [(X_val, y_val)]
             self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
             self.model = xgb.XGBClassifier(**final_params)
             self.model.fit(X_train, y_train)
             
        self.is_trained = True
        logger.info("Model training complete.")
        
    def set_threshold(self, threshold: float):
        self.threshold = threshold
        
    def predict_proba(self, X):
        if not self.is_trained: raise ModelNotTrainedError("Model not trained.")
        return self.model.predict_proba(X)
        
    def predict(self, X):
        if not self.is_trained: raise ModelNotTrainedError("Model not trained.")
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
