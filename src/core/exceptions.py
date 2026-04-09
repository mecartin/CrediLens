class CrediLensError(Exception):
    """Base exception for CrediLens system."""
    pass

class ConfigurationError(CrediLensError):
    """Raised when there is an issue with configuration."""
    pass

class DataValidationError(CrediLensError):
    """Raised when data fails validation checks."""
    pass

class ModelNotTrainedError(CrediLensError):
    """Raised when trying to use a model that hasn't been trained."""
    pass

class OptimizationError(CrediLensError):
    """Raised when hyperparameter optimization fails."""
    pass

class CausalInferenceError(CrediLensError):
    """Raised when causal inference operations fail."""
    pass
