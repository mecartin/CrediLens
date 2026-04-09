import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from ..core.config import ConfigManager
from ..core.logger import logger

class OptunaOptimizer:
    """Advanced hyperparameter optimization using Optuna for XGBoost."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.opt_config = self.config_manager.get('model_config', 'optimization')
        
        self.n_trials = self.opt_config.get('n_trials', 50)
        self.timeout = self.opt_config.get('timeout', 3600)
        self.metric = self.opt_config.get('metric', 'roc_auc')
        
        # Read bounds from config
        self.search_space = self.opt_config.get('search_space', {})
        
    def _objective(self, trial, X_train, y_train):
        """Optuna objective function for K-Fold CV."""
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', *self.search_space.get('n_estimators', [100, 500])),
            'max_depth': trial.suggest_int('max_depth', *self.search_space.get('max_depth', [3, 10])),
            'learning_rate': trial.suggest_float('learning_rate', *self.search_space.get('learning_rate', [0.01, 0.3]), log=True),
            'subsample': trial.suggest_float('subsample', *self.search_space.get('subsample', [0.5, 1.0])),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *self.search_space.get('colsample_bytree', [0.5, 1.0])),
            'tree_method': 'hist', # Speedup
            # Optional parameters not explicitly in bounds but useful
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        }
        
        # Determine scale pos weight dynamically based on training distribution
        class_counts = np.bincount(y_train)
        base_weight = class_counts[0] / class_counts[1]
        
        # Check if SMOTE was used by seeing if it's near balanced
        use_smote = self.config_manager.get('model_config', 'preprocessing.smote.enabled', False)
        balance_ratio = class_counts[0] / class_counts[1]
        
        if use_smote and (0.8 < balance_ratio < 1.2):
            params['scale_pos_weight'] = 1.0 + (base_weight - 1.0) * 0.2
        else:
            params['scale_pos_weight'] = base_weight
            
        cv_scores = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params, early_stopping_rounds=20)
            
            # Fit
            eval_set = [(X_va, y_va)]
            model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
            
            # Predict
            preds = model.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, preds)
            cv_scores.append(auc)
            
            # Report and prune intermediate steps
            trial.report(auc, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(cv_scores)

    def optimize(self, X_train, y_train):
        """Run optimization."""
        logger.info(f"Starting Optuna Hyperparameter optimization for {self.n_trials} trials.")
        study = optuna.create_study(
            direction=self.opt_config.get('direction', 'maximize'),
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10)
        )
        
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False
        )
        
        logger.info(f"Best trial value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
