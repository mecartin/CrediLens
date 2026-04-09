import numpy as np
from pymoo.core.problem import Problem
from ..core.logger import logger

class CounterfactualProblem(Problem):
    """Multi-objective counterfactual generation using NSGA-II."""
    
    def __init__(self, model, original_instance, feature_info, constraints):
        n_features = len(original_instance)
        n_objectives = 5
        
        # Determine constraints counts
        self.immutables = constraints.get('immutable_indices', [])
        self.monotonic_inc = constraints.get('monotonic_increase_indices', [])
        
        # Total inequalities:
        # 1. Prediction constraint (must flip class)
        # 2. Immutable constraints (one per immutable feature)
        # 3. Monotonic constraints (one per monotonic feature)
        n_constraints = 1 + len(self.immutables) + len(self.monotonic_inc)
        
        # Bounds handling
        xl = feature_info['lower_bounds']
        xu = feature_info['upper_bounds']
        
        super().__init__(
            n_var=n_features,
            n_obj=n_objectives,
            n_ieq_constr=n_constraints,
            xl=xl,
            xu=xu
        )
        
        self.model = model
        self.original = original_instance
        self.feature_info = feature_info
        self.constraints = constraints
        
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population."""
        n = len(X)
        F = np.zeros((n, self.n_obj))
        G = np.zeros((n, self.n_ieq_constr))
        
        for i, candidate in enumerate(X):
            # Objective 1: Proximity (L1 distance)
            F[i, 0] = np.sum(np.abs(candidate - self.original))
            
            # Objective 2: Sparsity (count of changes)
            # using a small epsilon since X is continuous
            F[i, 1] = np.sum(np.abs(candidate - self.original) > 1e-4)
            
            # Objective 3: Plausibility (Mahalanobis distance approx)
            F[i, 2] = self._calculate_plausibility(candidate)
            
            # Objective 4: Actionability (cost)
            F[i, 3] = self._calculate_cost(candidate)
            
            # Objective 5: Diversity
            F[i, 4] = 0.0 # updated in post-processing usually
            
            # Constraints
            G[i, :] = self._evaluate_constraints(candidate)
            
        out["F"] = F
        out["G"] = G
        
    def _evaluate_constraints(self, candidate):
        """Evaluate inequalities."""
        constraints_vals = []
        
        # 1. Prediction must be Approved (0 class)
        try:
             # Want P(Denied) < threshold for Approval => P(Denied) - threshold <= 0
             threshold = getattr(self.model, 'threshold', 0.5)
             probas = self.model.predict_proba([candidate])[0]
             
             # Some models return 1D [P(0), P(1)], others might return 2D
             p_denied = probas[1] if len(probas) > 1 else probas[0]
             
             constraints_vals.append(p_denied - threshold) 
        except Exception as e:
             logger.warning(f"Constraint evaluation failed: {e}")
             constraints_vals.append(1.0) # Mark as violated if prediction fails
             
        # 2. Immutable features
        for idx in self.immutables:
            diff = abs(candidate[idx] - self.original[idx])
            constraints_vals.append(diff - 1e-6)
            
        # 3. Monotonic increasing (e.g. age, emp_length)
        for idx in self.monotonic_inc:
            change = candidate[idx] - self.original[idx]
            # can't decrease => change >= 0 => -change <= 0
            constraints_vals.append(-change)
            
        return np.array(constraints_vals)
        
    def _calculate_plausibility(self, candidate):
        """Approx Mahalanobis or Isolation forest."""
        return np.sum(np.square(candidate - self.original)) # stub
        
    def _calculate_cost(self, candidate):
        """Domain specific cost model."""
        return np.sum(np.abs(candidate - self.original)) * 1.5 # stub
