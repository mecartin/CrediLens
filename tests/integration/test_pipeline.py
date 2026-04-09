import pytest
import pandas as pd
import numpy as np
import os
from src.models.trainer import ModelTrainer
from src.core.config import ConfigManager

@pytest.fixture
def mock_lending_club_data(tmp_path):
    # Create a small mock dataset in the LendingClub format
    data = []
    
    status_map = {0: 'Fully Paid', 1: 'Charged Off'}
    
    np.random.seed(42)
    for i in range(100):
        target = np.random.choice([0, 1])
        status = status_map[target]
        row = {
            'loan_amnt': np.random.randint(1000, 40000),
            'term': np.random.choice([' 36 months', ' 60 months']),
            'int_rate': f"{np.random.uniform(5, 25):.2f}%",
            'installment': np.random.uniform(50, 1000),
            'grade': np.random.choice(['A', 'B', 'C', 'D']),
            'sub_grade': np.random.choice(['A1', 'B2', 'C3']),
            'emp_title': 'Test Engineer',
            'emp_length': np.random.choice(['10+ years', '2 years', '< 1 year', np.nan]),
            'home_ownership': np.random.choice(['MORTGAGE', 'RENT', 'OWN']),
            'annual_inc': np.random.uniform(30000, 200000),
            'verification_status': 'Verified',
            'issue_d': 'Dec-2015',
            'loan_status': status,
            'purpose': 'debt_consolidation',
            'title': 'Debt consolidation',
            'zip_code': '902xx',
            'addr_state': 'CA',
            'dti': np.random.uniform(5, 35),
            'earliest_cr_line': 'Aug-2003',
            'fico_range_low': np.random.randint(600, 850),
            'fico_range_high': np.random.randint(604, 854),
            'open_acc': np.random.randint(2, 20),
            'pub_rec': np.random.choice([0, 1]),
            'revol_bal': np.random.uniform(0, 50000),
            'revol_util': f"{np.random.uniform(0, 100):.1f}%",
            'total_acc': np.random.randint(5, 50),
            'application_type': 'Individual',
            'mort_acc': np.random.randint(0, 5),
            'pub_rec_bankruptcies': np.random.choice([0, 1])
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    file_path = tmp_path / "mock_lending_club.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

class MockConfig(ConfigManager):
    def get(self, file_name, key_path, default=None):
        # Fast, basic params for integration test
        if key_path == 'model.algorithm': return 'xgboost'
        if key_path == 'optimization.method': return 'none' # Skip slow optuna in test
        if key_path == 'preprocessing': return {'smote': {'enabled': False}}
        if key_path == 'preprocessing.smote.enabled': return False
        if key_path == 'model.hyperparameters': return {'n_estimators': 5, 'max_depth': 3}
        return super().get(file_name, key_path, default)

def test_full_pipeline_training(mock_lending_club_data, tmp_path):
    trainer = ModelTrainer(MockConfig())
    model, metrics = trainer.train_full_pipeline(
        filepath=mock_lending_club_data, 
        sample_size=100,
        save_path=str(tmp_path / "models")
    )
    
    assert model is not None
    assert 'roc_auc' in metrics
    assert os.path.exists(tmp_path / "models" / "xgb_model.pkl")
    assert os.path.exists(tmp_path / "models" / "preprocessor.pkl")
