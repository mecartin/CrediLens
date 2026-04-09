import numpy as np
import pandas as pd
from ..core.logger import logger

class DecisionStabilityTester:
    """Analyze the robustness of individual credit decisions against small input perturbations."""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        
    def calculate_stability(self, applicant_state: dict, n_samples=100, noise_level=0.02):
        """
        Measure how often the decision flips when small noise is added.
        noise_level: Std dev of Gaussian noise as a fraction of the feature value (e.g., 2%).
        """
        # Select numeric features to perturb
        numeric_features = [
            'annual_inc', 'dti', 'loan_amnt', 'fico_range_low', 
            'fico_range_high', 'revol_bal', 'tot_cur_bal'
        ]
        
        # 1. Get baseline decision
        base_df = pd.DataFrame([applicant_state])
        if self.preprocessor:
             base_proc = self.preprocessor.transform(base_df)
        else:
             base_proc = base_df
        
        base_pred = self.model.predict(base_proc)[0] # 0 = Approved, 1 = Denied
        base_proba = self.model.predict_proba(base_proc)[0, 1]
        
        # 2. Generate Perturbed Samples
        # Create a batch of samples
        perturbed_states = []
        for _ in range(n_samples):
            state_copy = dict(applicant_state)
            for feat in numeric_features:
                if feat in state_copy and state_copy[feat] is not None:
                    # Multiplicative Gaussian noise
                    noise = 1.0 + np.random.normal(0, noise_level)
                    state_copy[feat] = state_copy[feat] * noise
            perturbed_states.append(state_copy)
            
        perturbed_df = pd.DataFrame.from_records(perturbed_states)
        
        # 3. Batch Predict
        if self.preprocessor:
             perturbed_proc = self.preprocessor.transform(perturbed_df)
        else:
             perturbed_proc = perturbed_df
             
        perturbed_preds = self.model.predict(perturbed_proc)
        
        # 4. Calculate flip rate
        # If baseline was Approved (0), how many became Denied (1)?
        # If baseline was Denied (1), how many became Approved (0)?
        flips = np.sum(perturbed_preds != base_pred)
        flip_rate = flips / n_samples
        stability_index = 1.0 - flip_rate
        
        # "Margin of Safety": Distance from the 0.5 threshold
        margin = abs(0.5 - base_proba) * 2.0 # 0 to 1
        
        return {
            'stability_index': float(stability_index),
            'flip_rate': float(flip_rate),
            'flips_detected': int(flips),
            'margin_of_safety': float(margin),
            'baseline_proba': float(base_proba),
            'baseline_decision': "Approved" if base_pred == 0 else "Denied"
        }
