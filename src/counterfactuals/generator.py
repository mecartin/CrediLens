import numpy as np
from ..core.logger import logger
from .nsga2_problem import CounterfactualProblem

class CFResult:
    def __init__(self, features, original, desired_class, proximity, sparsity, plausibility, cost):
        self.features = features
        self.original = original
        self.desired_class = desired_class
        self.proximity = proximity
        self.sparsity = sparsity
        self.plausibility = plausibility
        self.cost = cost
        
    def describe(self):
        return f"Prox: {self.proximity:.2f}, Sparsity: {self.sparsity}, Cost: {self.cost:.2f}"

class CounterfactualGenerator:
    """Generate counterfactual explanations using NSGA-II."""
    
    def __init__(self, model, feature_info, config, feature_names=None):
        self.model = model
        self.feature_info = feature_info
        self.config = config
        self.feature_names = feature_names
        
    def _map_indices(self, feature_list):
        """Map raw feature names to preprocessed indices (handles OHE expansion)."""
        if not self.feature_names or not feature_list:
            return []
            
        indices = []
        for i, name in enumerate(self.feature_names):
            # Check if preprocessed name starts with any name in the feature_list
            # e.g. 'term_ 36 months' starts with 'term'
            if any(name.startswith(f) for f in feature_list):
                indices.append(i)
        return indices

    def generate(self, original_instance, desired_class=0):
        """Generate Pareto-optimal counterfactuals."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        
        logger.info("Initializing NSGA-II Counterfactual Generation...")
        
        # ensure it's a 1d array
        if hasattr(original_instance, 'values'):
             original_array = original_instance.values.flatten()
        else:
             original_array = np.array(original_instance).flatten()

        # ensure lower/upper bounds exist
        if 'lower_bounds' not in self.feature_info or 'upper_bounds' not in self.feature_info:
             if 'data' in self.feature_info:
                  df_ref = self.feature_info['data']
                  self.feature_info['lower_bounds'] = df_ref.min().values
                  self.feature_info['upper_bounds'] = df_ref.max().values
             else:
                  # Fallback to extreme bounds if nothing else provided
                  self.feature_info['lower_bounds'] = np.full(len(original_array), -1e6)
                  self.feature_info['upper_bounds'] = np.full(len(original_array), 1e6)

        # Map constraints to indices
        raw_constraints = self.config.get('counterfactual_config', 'counterfactuals.constraints', {})
        mapped_constraints = {
            'immutable_indices': self._map_indices(raw_constraints.get('immutable_features', [])),
            'monotonic_increase_indices': self._map_indices(raw_constraints.get('monotonic_increase', [])),
        }

        problem = CounterfactualProblem(
            model=self.model,
            original_instance=original_array,
            feature_info=self.feature_info,
            constraints=mapped_constraints
        )
        
        nsga2_config = self.config.get('counterfactual_config', 'counterfactuals.nsga2', {})
        pop_size = nsga2_config.get('population_size', 150)
        n_gens = nsga2_config.get('n_generations', 200) # Boosted for better discovery and convergence
        
        algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True)
        
        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gens),
            seed=42,
            verbose=False
        )
        
        if res.X is None:
             logger.warning("No feasible counterfactuals found.")
             return {'counterfactuals': [], 'pareto_front': [], 'convergence': None, 'n_solutions': 0}
             
        # Extract and format
        valid_cfs = []
        # If single result, res.X is 1D
        X_res = res.X if len(res.X.shape) > 1 else [res.X]
        F_res = res.F if len(res.F.shape) > 1 else [res.F]
        
        for i, candidate in enumerate(X_res):
             if self._validate(candidate, desired_class):
                  cf = CFResult(
                       features=candidate,
                       original=original_array,
                       desired_class=desired_class,
                       proximity=F_res[i][0],
                       sparsity=F_res[i][1],
                       plausibility=F_res[i][2],
                       cost=F_res[i][3]
                  )
                  valid_cfs.append(cf)
                  
        logger.info(f"Generated {len(valid_cfs)} valid counterfactuals.")
        return {
            'counterfactuals': valid_cfs,
            'pareto_front': res.F,
            'convergence': res.CV,
            'n_solutions': len(valid_cfs)
        }
        
    def _validate(self, features, desired_class):
        try:
             pred = self.model.predict([features])[0]
             return pred == desired_class
        except:
             return True # fallback if model wrapper expects DataFrame
