import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import AdvancedPreprocessor
from src.core.config import ConfigManager

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'loan_amnt': [1000, 5000, 10000],
        'annual_inc': [50000, 60000, 120000],
        'installment': [50, 150, 300],
        'home_ownership': ['RENT', 'OWN', 'MORTGAGE'],
        'emp_length': [1, 5, 10],
        # Feature Engineering dependencies
        'revol_bal': [1000, 2000, 5000],
        'revol_util': [10.0, 20.0, 50.0],
    })

@pytest.fixture
def sample_target():
    return pd.Series([0, 0, 1])

def test_preprocessor_fit_transform(sample_data, sample_target):
    # Mock config to disable SMOTE for basic test
    class MockConfig(ConfigManager):
        def get(self, file_name, key_path, default=None):
            if key_path == 'preprocessing.smote.enabled': return False
            if key_path == 'preprocessing': return {'smote': {'enabled': False}}
            return super().get(file_name, key_path, default)
            
    preprocessor = AdvancedPreprocessor(MockConfig())
    X_processed, y_processed = preprocessor.fit_transform(sample_data, sample_target)
    
    # Ensure properties exist
    assert hasattr(preprocessor, 'preprocessor')
    assert preprocessor.feature_names_out is not None
    
    # Ensure transformed shape matches expectations (feature engineering adds cols, one-hot encodes cats)
    assert len(X_processed) == len(sample_data)
    assert len(y_processed) == len(sample_target)
    
    # Basic check for no nans
    assert X_processed.isna().sum().sum() == 0

def test_preprocessor_transform(sample_data, sample_target):
    # Same as above but test transform alone
    preprocessor = AdvancedPreprocessor()
    preprocessor.use_smote = False
    
    X_processed, _ = preprocessor.fit_transform(sample_data, sample_target)
    
    # Create new instance data
    new_data = pd.DataFrame({
         'loan_amnt': [2000],
         'annual_inc': [70000],
         'installment': [100],
         'home_ownership': ['RENT'],
         'emp_length': [2],
         'revol_bal': [1500],
         'revol_util': [15.0]
    })
    
    transformed = preprocessor.transform(new_data)
    assert len(transformed) == 1
    assert list(transformed.columns) == preprocessor.feature_names_out
