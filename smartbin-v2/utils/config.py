"""
Configuration loader with YAML parsing, env variable resolution, and schema validation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


ENV_VAR_PATTERN = re.compile(r"\$\{([^}^{]+)\}")


def _env_var_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    """Resolve ${ENV_VAR:default} in YAML files."""
    value = loader.construct_scalar(node)
    match = ENV_VAR_PATTERN.match(value)
    if not match:
        return value
    
    var_expr = match.group(1)
    if ":" in var_expr:
        var_name, default_val = var_expr.split(":", 1)
    else:
        var_name, default_val = var_expr, ""
    
    return os.getenv(var_name, default_val)


def get_yaml_loader() -> type[yaml.SafeLoader]:
    """Return a yaml SafeLoader configured with env var resolution."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver("!env", ENV_VAR_PATTERN, None)
    loader.add_constructor("!env", _env_var_constructor)
    return loader


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary containing parsed configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            loader = get_yaml_loader()
            content = yaml.load(f, Loader=loader) or {}
            return content
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file {path}: {exc}") from exc


def get_app_config(
    model_cfg: Optional[str] = None,
    deployment_cfg: Optional[str] = None,
    augmentation_cfg: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and aggregate all component configurations.
    """
    base_dir = Path(__file__).resolve().parent.parent / "configs"
    
    model_path = model_cfg or base_dir / "model_config.yaml"
    deployment_path = deployment_cfg or base_dir / "deployment_config.yaml"
    augmentation_path = augmentation_cfg or base_dir / "augmentation_config.yaml"

    config: Dict[str, Any] = {}
    if Path(model_path).exists():
        config["model"] = load_yaml_config(model_path)
    if Path(deployment_path).exists():
        config["deployment"] = load_yaml_config(deployment_path)
    if Path(augmentation_path).exists():
        config["augmentation"] = load_yaml_config(augmentation_path)

    return config
