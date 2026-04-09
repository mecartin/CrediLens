import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from .validators import DataValidators
from ..core.config import ConfigManager
from ..core.logger import logger
from ..core.exceptions import DataValidationError
from ..core.utils import timer

class DataLoader:
    """Advanced data loading with comprehensive validation specifically for LendingClub."""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.validators = DataValidators()
        
        # Specific LendingClub columns to keep to prevent OOM
        self.features_to_keep = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
            'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
            'issue_d', 'loan_status', 'purpose', 'title', 'zip_code', 'addr_state', 'dti',
            'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'application_type',
            'mort_acc', 'pub_rec_bankruptcies'
        ]
        
    @timer
    def load_lending_club(self, filepath: str, sample_size: int = None) -> pd.DataFrame:
        """Load LendingClub dataset, parsing specific columns to save memory."""
        logger.info(f"Loading Lending Club data from {filepath}")
        
        # Define types roughly to save memory
        dtype_spec = {
            'loan_amnt': 'float32',
            'installment': 'float32',
            'annual_inc': 'float32',
            'dti': 'float32',
            'fico_range_low': 'float32',
            'fico_range_high': 'float32',
            'open_acc': 'float32',
            'pub_rec': 'float32',
            'revol_bal': 'float32',
            'total_acc': 'float32',
            'mort_acc': 'float32',
            'pub_rec_bankruptcies': 'float32'
        }
        
        try:
            # Load subset of columns if the filepath exists
            usecols = [c for c in self.features_to_keep]
            
            # Read CSV in chunks if we want to sample, or simply read exactly what we need
            if sample_size:
                logger.info(f"Reading sample of exactly {sample_size} rows.")
                df = pd.read_csv(filepath, usecols=usecols, dtype=dtype_spec, nrows=sample_size)
            else:
                df = pd.read_csv(filepath, usecols=usecols, dtype=dtype_spec)
                
            logger.info(f"Loaded DataFrame with shape: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
            
    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert LendingClub `loan_status` to binary classification target.
        Target 1: Denied / Charged Off / Default
        Target 0: Approved / Fully Paid / Current
        """
        logger.info("Preparing binary target variable from loan_status")
        
        # Filter out rows missing loan_status
        df = df.dropna(subset=['loan_status']).copy()
        
        # Map statuses
        # We classify bad loans as 1 (denied/defaulted) and good loans as 0 (approved/paid)
        bad_indicators = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
        good_indicators = ['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid']
        
        def map_target(status):
            if status in bad_indicators: return 1
            elif status in good_indicators: return 0
            else: return np.nan
            
        df['target'] = df['loan_status'].apply(map_target)
        
        # Drop rows where we couldn't determine target
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
        
        # We can drop loan_status now
        df = df.drop(columns=['loan_status'])
        
        logger.info(f"Remaining records after target mapping: {len(df)}")
        return df
        
    def enrich_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced metadata parsing (e.g., date conversion)."""
        logger.info("Enriching metadata and parsing types")
        
        # Parse Dates
        if 'issue_d' in df.columns:
            df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
        if 'earliest_cr_line' in df.columns:
            df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
            
        # Parse term (e.g. " 36 months" -> 36)
        if 'term' in df.columns:
            df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
            
        # Parse emp_length (e.g. "10+ years" -> 10, "< 1 year" -> 0)
        if 'emp_length' in df.columns:
            df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
            df['emp_length'] = df['emp_length'].fillna(0) # Assume 0 if missing
            
        # Parse percentage strings
        if 'int_rate' in df.columns:
            df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '').astype(float)
            
        if 'revol_util' in df.columns:
            df['revol_util'] = df['revol_util'].astype(str).str.replace('%', '').astype(float)
            
        return df

    def run_pipeline(self, filepath: str, sample_size: int = None) -> Tuple[pd.DataFrame, Dict]:
        """Run complete loading pipeline."""
        df = self.load_lending_club(filepath, sample_size)
        df = self.prepare_target(df)
        df = self.enrich_metadata(df)
        
        report = self.validators.quality_report(df, target_col='target')
        return df, report
