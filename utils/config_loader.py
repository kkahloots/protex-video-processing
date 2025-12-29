#!/usr/bin/env python
"""
Configuration loader for Protex AI Computer Vision Pipeline.
Loads YAML configuration with default fallbacks and variable substitution.
"""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Loads and provides access to configuration values with defaults."""

    def __init__(self, config_path: str = "config.yaml", validate: bool = True):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._substitute_variables()

        if validate:
            self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with error handling."""
        if not self.config_path.exists():
            print(f"[WARN] Config file not found: {self.config_path}. Using defaults.")
            return {}

        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to load config: {e}. Using defaults.")
            return {}

    def _substitute_variables(self) -> None:
        """Substitute ${var} references with values from config or environment."""

        def substitute(obj: Any, context: Dict[str, Any]) -> Any:
            if isinstance(obj, str):
                # Replace ${var} with context[var] or env var
                pattern = r"\$\{([^}]+)\}"
                matches = re.findall(pattern, obj)
                for var in matches:
                    value = context.get(var) or os.environ.get(var, "")
                    obj = obj.replace(f"${{{var}}}", str(value))
                return obj
            elif isinstance(obj, dict):
                return {k: substitute(v, context) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute(item, context) for item in obj]
            return obj

        # Build context from defaults
        context = self._config.get("defaults", {})
        self._config = substitute(self._config, context)

    def _validate_config(self) -> None:
        """Validate configuration if validator is available."""
        try:
            from config_schema import ConfigValidator

            mode = self.get("defaults.mode", "balanced")
            verbose = self.get("defaults.verbose", True)
            ConfigValidator.validate_all(self._config, mode, verbose)
        except ImportError:
            pass  # Validation is optional

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'preprocessing.target_width')."""
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_mode_config(
        self, stage: str, mode: str, key: str, default: Any = None
    ) -> Any:
        """Get mode-specific configuration value with fallback to stage default."""
        # Try mode-specific value first
        mode_value = self.get(f"{stage}.modes.{mode}.{key}")
        if mode_value is not None:
            return mode_value
        # Fallback to stage-level default
        stage_value = self.get(f"{stage}.{key}")
        if stage_value is not None:
            return stage_value
        return default


# Global config instance
_config = ConfigLoader()


def get_config(key_path: str, default: Any = None) -> Any:
    """Get configuration value using dot notation."""
    return _config.get(key_path, default)


def get_mode_config(stage: str, mode: str, key: str, default: Any = None) -> Any:
    """Get mode-specific configuration value."""
    return _config.get_mode_config(stage, mode, key, default)
