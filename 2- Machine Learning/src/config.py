"""Configuration management for fraud detection pipeline."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for the fraud detection pipeline."""

    def __init__(self, config_path: str = None):
        """
        Initialize config with default values or from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        if config_path is None:
            # Default to config file in the same directory
            config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"

        self._config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Dict containing the configuration.
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config.get("model", {})

    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get("data", {})

    @property
    def threshold_config(self) -> Dict[str, Any]:
        """Get threshold configuration."""
        return self._config.get("threshold", {})

    @property
    def mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self._config.get("mlflow", {})


# Global config instance
config = Config()
