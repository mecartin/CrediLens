import pytest
import time
import pandas as pd
from src.data.loader import DataLoader
from src.core.config import ConfigManager

def test_data_loader_performance(tmp_path):
    # Create mock dataset
    df = pd.DataFrame({
        'loan_amnt': [1000] * 10000,
        'installment': [50] * 10000,
        'loan_status': ['Fully Paid'] * 10000,
        'int_rate': ['10.5%'] * 10000,
        'term': [' 36 months'] * 10000,
    })
    
    file_path = tmp_path / "large_mock_data.csv"
    df.to_csv(file_path, index=False)
    
    loader = DataLoader(ConfigManager())
    loader.features_to_keep = list(df.columns)
    
    start_time = time.time()
    df_loaded, report = loader.run_pipeline(str(file_path))
    end_time = time.time()
    
    # Assert loading and parsing 10,000 rows takes less than 2 seconds
    assert (end_time - start_time) < 2.0
    assert len(df_loaded) == 10000
