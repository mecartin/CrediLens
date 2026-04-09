import pytest
import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.core.config import ConfigManager

@pytest.fixture
def mock_config():
    return ConfigManager()

def test_data_loader_initialization(mock_config):
    loader = DataLoader(mock_config)
    assert loader is not None
    assert loader.config_manager == mock_config

def test_data_loader_target_parsing():
    # Lending club formats
    df = pd.DataFrame({
        'loan_status': ['Fully Paid', 'Charged Off', 'Current', 'Default', 'Does not meet the credit policy. Status:Fully Paid']
    })
    
    loader = DataLoader()
    parsed_df = loader.prepare_target(df)
    
    # Fully Paid -> 0
    # Charged Off -> 1
    # Current -> NaN (dropped eventually)
    # Default -> 1
    assert parsed_df.loc[0, 'target'] == 0
    assert parsed_df.loc[1, 'target'] == 1
    assert parsed_df.loc[3, 'target'] == 1
