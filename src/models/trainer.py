import os
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path

from ..core.config import ConfigManager
from ..core.logger import logger
from ..core.utils import timer

from ..data.loader import DataLoader
from ..data.preprocessor import AdvancedPreprocessor
from .xgboost_model import XGBoostModel
from .optimizer import OptunaOptimizer
from .evaluator import Evaluator

class ModelTrainer:
    """Orchestrates complete training workflow."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.data_loader = DataLoader(self.config_manager)
        self.preprocessor = AdvancedPreprocessor(self.config_manager)
        self.optimizer = OptunaOptimizer(self.config_manager)
        self.evaluator = Evaluator()
        
    @timer
    def train_full_pipeline(self, filepath: str, sample_size: int = None, save_path: str = 'models/saved_models'):
        """End to end training process."""
        logger.info("Initializing Full Pipeline Training...")
        
        # 1. Load Data
        df, report_quality = self.data_loader.run_pipeline(filepath, sample_size)
        if report_quality['quality_score'] < 50:
             logger.warning("Low data quality score. Proceed at your own risk.")
             
        # Separate Features & Target
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Train / Validation / Test Split
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val)
        
        # 2. Preprocess & Feature Engineering via AdvancedPreprocessor
        logger.info("Fitting and transforming Preprocessor on training set...")
        X_tr_proc, y_tr_proc = self.preprocessor.fit_transform(X_train, y_train)
        
        logger.info("Transforming validation and test sets...")
        X_va_proc = self.preprocessor.transform(X_val)
        X_te_proc = self.preprocessor.transform(X_test)
        
        # 3. Hyperparameter Tuning (Optuna)
        if self.config_manager.get('model_config', 'optimization.method') == 'optuna':
            best_params = self.optimizer.optimize(X_tr_proc, y_tr_proc)
        else:
            best_params = self.config_manager.get('model_config', 'model.hyperparameters', {})
            
        # 4. Final Model Training
        # We re-train on both Train and Val for maximum signal, using test for evaluation.
        final_model = XGBoostModel(params=best_params)
        final_model.train(X_tr_proc, y_tr_proc, X_va_proc, y_val) # Actually training XGB wrapper
        
        # 5. Threshold Tuning
        threshold = self.evaluator.tune_threshold(final_model, X_va_proc, y_val)
        final_model.set_threshold(threshold)
        
        # 6. Evaluate
        metrics = self.evaluator.evaluate(final_model, X_te_proc, y_test)
        
        # 7. Persistence
        self._save_artifacts(final_model, self.preprocessor, save_path)
        
        logger.info("Training Pipeline Completed Successfully.")
        return final_model, metrics

    def _save_artifacts(self, model, preprocessor, save_path: str):
        """Save pipeline."""
        logger.info("Saving trained models and preprocessor.")
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(model, Path(save_path) / 'xgb_model.pkl')
        joblib.dump(preprocessor, Path(save_path) / 'preprocessor.pkl')
        logger.info(f"Artifacts saved in {save_path}")
