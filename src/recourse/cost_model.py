class RecurseCostModel:
    """Calculate costs for recourse actions."""
    
    def __init__(self, config):
        self.config = config
        
    def calculate_cost(self, action):
        """Monetary cost in $"""
        return action.cost
        
    def calculate_time(self, action):
        """Time in months"""
        return action.time
        
    def calculate_effort(self, action):
        """Difficulty score 1-10"""
        return {'easy': 2, 'medium': 5, 'hard': 8, 'very_hard': 10}.get(action.difficulty, 5)
        
    def calculate_total_path_cost(self, path):
        """Calculate total multi-objective cost for a path."""
        total_money = sum(self.calculate_cost(a) for a in path)
        total_time = max(self.calculate_time(a) for a in path) if path else 0
        total_effort = sum(self.calculate_effort(a) for a in path)
        
        return {
            'monetary_cost': total_money,
            'time_cost': total_time,
            'effort_cost': total_effort,
            'n_steps': len(path)
        }
