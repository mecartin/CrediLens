import pandas as pd
import numpy as np
from ..core.logger import logger

class FeatureEngineering:
    """Domain-specific feature engineering for credit risk."""
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features relevant to Lending Club"""
        logger.debug("Starting domain-specific feature engineering")
        df = df.copy()
        
        # Credit Utilization Ratio
        if 'revol_bal' in df.columns and 'revol_util' in df.columns:
            # We can compute estimated total revolving limit
            total_limit = df['revol_bal'] / (df['revol_util'] / 100.0)
            df['est_total_limit'] = total_limit.replace([np.inf, -np.inf], np.nan).fillna(0)
            
        # Debt-to-Income is already provided loosely, let's create Income to Installment
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['installment_income_ratio'] = (df['installment'] * 12) / df['annual_inc'].clip(lower=1)
            
        # Time since first credit line
        if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
            df['credit_history_length'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 30.0 # roughly months
            
        # FICO average
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0
            
        # Replace inf with nan for generic numerical features
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"Added {len(df.columns)} new engineered columns (incl combinations)")
        return df
