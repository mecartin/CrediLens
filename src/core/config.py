import os
import yaml
from pathlib import Path
from typing import Any, Dict

from .exceptions import ConfigurationError

class ConfigManager:
    """Manages system configuration loaded from YAML files."""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Default to the config directory relative to this file
            base_path = Path(__file__).resolve().parent.parent.parent
            self.config_dir = base_path / 'config'
        else:
            self.config_dir = Path(config_dir)
            
        self._configs: Dict[str, Any] = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a specific configuration file."""
        if config_name in self._configs:
            return self._configs[config_name]
            
        file_path = self.config_dir / f"{config_name}.yaml"
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._configs[config_name] = config_data
                return config_data
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config {file_path}: {e}")
            
    def get(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """Get a configuration value."""
        config = self.load_config(config_name)
        if key is None:
            return config
            
        keys = key.split('.')
        curr = config
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr
