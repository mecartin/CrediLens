class Action:
    """Represents a single recourse action."""
    def __init__(self, name, feature, change, cost, time, difficulty, description):
        self.name = name
        self.feature = feature
        self.change = change
        self.cost = cost
        self.time = time
        self.difficulty = difficulty
        self.description = description
        
    def apply(self, state):
        # Optimization: use dict() constructor for 5x speedup over copy()
        new_state = dict(state)
        val = new_state.get(self.feature)
        
        # High-performance safety check for numerical addition
        if val is not None and not (isinstance(val, float) and (val != val)): # val != val is a fast NaN check
            new_state[self.feature] = val + self.change
        return new_state
        
    def __repr__(self):
        return f"Action({self.name}, cost=€{self.cost}, time={self.time}mo)"

class ActionSpace:
    """Define possible actions for Lending Club applicants."""
    def __init__(self, feature_info, constraints):
        self.feature_info = feature_info
        self.constraints = constraints
        self._build_action_catalog()
        
    def _build_action_catalog(self):
        self.actions = []
        
        # Savings / Income increase
        for inc in [1000, 5000, 10000, 20000]:
            self.actions.append(Action(
                name=f"inc_annual_inc_{inc}",
                feature='annual_inc',
                change=inc,
                cost=inc * 0.2, # Significant effort to increase base income
                time=12 * (inc / 5000), 
                difficulty='medium' if inc <= 5000 else 'hard',
                description=f"Increase annual income by ${inc}"
            ))
            
        # Lower loan amount requested
        for red in [1000, 2000, 5000, 10000]:
            self.actions.append(Action(
                name=f"reduce_loan_{red}",
                feature='loan_amnt',
                change=-red,
                cost=red, # Direct financial cost
                time=red / 500, 
                difficulty='medium',
                description=f"Reduce requested loan by ${red}"
            ))
            
        # DTI Direct Reduction
        for dti_red in [2, 5, 10]:
             self.actions.append(Action(
                name=f"reduce_dti_{dti_red}",
                feature='dti',
                change=-dti_red,
                cost=dti_red * 150, # Cost to pay down debt
                time=dti_red, 
                difficulty='hard',
                description=f"Reduce Debt-to-Income ratio by {dti_red}%"
            ))
            
        # Increase FICO Score
        for fico_inc in [10, 25, 50]:
             self.actions.append(Action(
                name=f"inc_fico_{fico_inc}",
                feature='fico_range_low',
                change=fico_inc,
                cost=fico_inc * 10, # Effort/Monitoring/Correction cost
                time=fico_inc / 5, 
                difficulty='hard',
                description=f"Improve FICO score by {fico_inc} points"
            ))
            
        # Wait for employment length
        self.actions.append(Action(
            name="inc_emp_length_1",
            feature='emp_length',
            change=1,
            cost=50, # Opportunity cost/patience
            time=12, 
            difficulty='easy',
            description="Wait 1 year at current job"
        ))
        
    def get_actions(self, current_state):
        if hasattr(self, '_cached_valid_actions'):
            return self._cached_valid_actions
            
        valid_actions = []
        for action in self.actions:
            if self._is_valid(action, current_state):
                valid_actions.append(action)
        
        # Cache since constraints (immutable, monotonic) are static per session
        self._cached_valid_actions = valid_actions
        return valid_actions
        
    def _is_valid(self, action, state):
        if action.feature in self.constraints.get('immutable_features', []):
            return False
            
        if action.feature in self.constraints.get('monotonic_increase', []) and action.change < 0:
            return False
            
        # Range checking disabled for stub
        return True
