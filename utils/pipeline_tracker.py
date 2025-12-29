#!/usr/bin/env python
"""
Pipeline Configuration Tracker

Utility to save pipeline parameters at each stage for report generation.
Enables traceability: "Which parameters created this dataset?"
"""

import json
from pathlib import Path


def save_stage_config(stage: str, config: dict, output_dir: str = "traceables"):
    """
    Save stage configuration to pipeline_config.json.

    Args:
        stage: Stage name (preprocessing, pretagging, cleanup)
        config: Dictionary of stage parameters
        output_dir: Output directory for config file
    """
    output_path = Path(output_dir) / "pipeline_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config if present
    pipeline_config = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            pipeline_config = json.load(f)

    # Update with new stage config
    pipeline_config[stage] = config

    # Write back
    with open(output_path, "w") as f:
        json.dump(pipeline_config, f, indent=2)


def load_pipeline_config(output_dir: str = "traceables"):
    """Load pipeline configuration if available."""
    config_path = Path(output_dir) / "pipeline_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}
