import pandas as pd
import numpy as np
from typing import Dict, Any, List

from ..core.exceptions import DataValidationError
from ..core.logger import logger

class DataValidators:
    """Advanced data validation for CrediLens."""
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Ensure dataframe contains all required columns."""
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
        return True
        
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Return percentage of missing values per column."""
        missing_pct = (df.isnull().sum() / len(df)) * 100
        return missing_pct[missing_pct > 0].to_dict()
        
    def check_class_balance(self, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """Check the balance of the target class."""
        if target_col not in df.columns:
            raise DataValidationError(f"Target column {target_col} not found in dataframe.")
            
        counts = df[target_col].value_counts(normalize=True) * 100
        logger.info(f"Class distribution for {target_col}: {counts.to_dict()}")
        return counts.to_dict()
        
    def quality_report(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Comprehensive quality report."""
        report = {
            'shape': df.shape,
            'missing_values': self.check_missing_values(df),
            'class_balance': self.check_class_balance(df, target_col) if target_col in df.columns else {},
            'duplicates': df.duplicated().sum(),
        }
        
        # Calculate a basic quality score (0-100)
        # Deduct points for high missing values and duplicates
        score = 100
        total_missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        score -= min(30, total_missing_pct) # Max 30 points deducted for missingness
        
        dup_pct = (report['duplicates'] / len(df)) * 100
        score -= min(20, dup_pct * 2) # Deduct 2 points for each 1% duplicate
        
        report['quality_score'] = max(0, score)
        logger.info(f"Data Quality Score: {report['quality_score']:.2f}/100")
        
        return report
