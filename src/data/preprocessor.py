import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from ..core.config import ConfigManager
from ..core.logger import logger
from .feature_engineering import FeatureEngineering

class AdvancedPreprocessor:
    """Sophisticated preprocessing pipeline."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.preprocessor_config = self.config_manager.get('model_config', 'preprocessing')
        
        self.feature_engineering = FeatureEngineering()
        self.pipeline = None
        self.preprocessor = None
        self.smote = None
        
        # Determine SMOTE
        smote_cfg = self.preprocessor_config.get('smote', {})
        self.use_smote = smote_cfg.get('enabled', False)
        
        if self.use_smote:
            # Note: strategy 0.8 means minority reaches 80% size of majority
            self.smote = SMOTE(
                sampling_strategy=smote_cfg.get('strategy', 0.8),
                k_neighbors=smote_cfg.get('k_neighbors', 5),
                random_state=self.config_manager.get('model_config', 'model.hyperparameters.random_state', 42)
            )
            
        self.numeric_features = []
        self.categorical_features = []
        
        # Save features names
        self.feature_names_out = None
        
    def _build_transformers(self, df: pd.DataFrame):
        """Construct the column transformer."""
        # Find numeric and categorical columns
        self.numeric_features = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Found {len(self.numeric_features)} numeric features and {len(self.categorical_features)} categorical features.")
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), # Default median imputation
            ('scaler', StandardScaler() if self.preprocessor_config.get('scaling', 'standard') == 'standard' else MinMaxScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply full preprocessing pipeline including feature engineering and smote."""
        logger.info("Starting Fit Transform pipeline")
        
        # 1. Feature Engineering (must occur before ColumnTransformer)
        X_fe = self.feature_engineering.create_features(X)
        
        # Drop dates for processing
        dates_cols = X_fe.select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist()
        X_fe = X_fe.drop(columns=dates_cols, errors='ignore')
        
        # 2. Build and fit preprocessor
        self._build_transformers(X_fe)
        X_transformed_array = self.preprocessor.fit_transform(X_fe)
        
        # Capture feature names
        num_names = self.numeric_features
        cat_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features).tolist()
        self.feature_names_out = num_names + cat_names
        
        # Rebuild df
        X_processed = pd.DataFrame(X_transformed_array, columns=self.feature_names_out)
        
        # 3. Apply SMOTE if enabled
        if self.use_smote and y is not None:
             logger.info(f"Class distribution before SMOTE: {y.value_counts().to_dict()}")
             X_resampled, y_resampled = self.smote.fit_resample(X_processed, y)
             logger.info(f"Class distribution after SMOTE: {y_resampled.value_counts().to_dict()}")
             return X_resampled, y_resampled
             
        return X_processed, y
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data or incoming production data."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted.")
            
        X_fe = self.feature_engineering.create_features(X)
        
        # Drop dates
        dates_cols = X_fe.select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist()
        X_fe = X_fe.drop(columns=dates_cols, errors='ignore')
        
        X_transformed_array = self.preprocessor.transform(X_fe)
        X_processed = pd.DataFrame(X_transformed_array, columns=self.feature_names_out)
        
        return X_processed
        
    def get_feature_names_out(self):
        """Return the names of the engineered and processed features."""
        if self.feature_names_out is None:
            raise ValueError("Preprocessor has not been fitted.")
        return self.feature_names_out
