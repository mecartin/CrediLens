import numpy as np
import pandas as pd
from ..core.logger import logger

class PortfolioStressTester:
    """Simulate economic shifts and project financial impacts on the loan portfolio."""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        
    def simulate_recession(self, df: pd.DataFrame, income_multiplier=1.0, dti_multiplier=1.0, interest_rate_multiplier=1.0):
        """
        Apply global multipliers to simulate economic shifts.
        income_multiplier: e.g., 0.8 for 20% drop in portfolio income.
        dti_multiplier: e.g., 1.2 for 20% increase in debt-to-income ratios.
        interest_rate_multiplier: e.g., 1.5 for 50% increase in base interest rates.
        """
        sim_df = df.copy()
        
        if 'annual_inc' in sim_df.columns:
            sim_df['annual_inc'] = sim_df['annual_inc'] * income_multiplier
            
        if 'dti' in sim_df.columns:
            sim_df['dti'] = sim_df['dti'] * dti_multiplier
            
        if 'int_rate' in sim_df.columns:
            sim_df['int_rate'] = sim_df['int_rate'] * interest_rate_multiplier
            
        return sim_df

    def calculate_portfolio_metrics(self, df: pd.DataFrame, probas: np.ndarray, inflation_multiplier=1.0):
        """
        Calculate aggregate financial risk metrics.
        probas: Probability of default (Target=1)
        inflation_multiplier: scalar shift to PD (proxy for reduced repayment capacity)
        """
        # Apply inflation proxy: higher inflation = higher PD (clipped to [0,1])
        adjusted_probas = np.clip(probas * inflation_multiplier, 0, 1)
        
        # Assumptions for financial modeling
        RECOVERY_RATE = 0.10 # Assume 10% recovery if they default
        
        loan_amounts = df['loan_amnt'].values if 'loan_amnt' in df.columns else np.full(len(df), 10000)
        int_rates = df['int_rate'].values / 100.0 if 'int_rate' in df.columns else np.full(len(df), 0.10)
        
        # 1. Total Portfolio Value
        total_value = np.sum(loan_amounts)
        
        # 2. Expected Loss (EL) = Sum(PD * EAD * LGD)
        expected_loss = np.sum(adjusted_probas * loan_amounts * (1 - RECOVERY_RATE))
        
        # 3. Expected Interest Income
        # Income = (1 - PD) * loan_amnt * int_rate
        expected_interest = np.sum((1 - adjusted_probas) * loan_amounts * int_rates)
        
        # 4. Net Position (Simplified)
        net_position = expected_interest - expected_loss
        
        return {
            'total_volume': float(total_value),
            'expected_loss': float(expected_loss),
            'expected_interest': float(expected_interest),
            'net_position': float(net_position),
            'avg_default_rate': float(np.mean(adjusted_probas))
        }

    def run_stress_test(self, df: pd.DataFrame, income_shift=1.0, dti_shift=1.0, interest_rate_shift=1.0, inflation_shift=1.0):
        """Shift the data and rerun predictions to find Delta Loss."""
        baseline_df = df.copy()
        
        # 1. Baseline Predictions
        if self.preprocessor:
            baseline_processed = self.preprocessor.transform(baseline_df)
        else:
            baseline_processed = baseline_df
            
        baseline_probas = self.model.predict_proba(baseline_processed)[:, 1]
        baseline_metrics = self.calculate_portfolio_metrics(baseline_df, baseline_probas)
        
        # 2. Stressed Predictions
        # Inflation shift is handled in metric calculation as a proxy for risk
        stressed_df = self.simulate_recession(df, income_shift, dti_shift, interest_rate_shift)
        if self.preprocessor:
            stressed_processed = self.preprocessor.transform(stressed_df)
        else:
            stressed_processed = stressed_df
            
        stressed_probas = self.model.predict_proba(stressed_processed)[:, 1]
        stressed_metrics = self.calculate_portfolio_metrics(stressed_df, stressed_probas, inflation_shift)
        
        return {
            'baseline': baseline_metrics,
            'stressed': stressed_metrics,
            'delta_loss': stressed_metrics['expected_loss'] - baseline_metrics['expected_loss'],
            'loss_spike_pct': (stressed_metrics['avg_default_rate'] / baseline_metrics['avg_default_rate'] - 1) if baseline_metrics['avg_default_rate'] > 0 else 0
        }
