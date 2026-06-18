#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
import yaml

from PynamicMesh.core.pipelines import run_pipeline

def main():

    parser = argparse.ArgumentParser(
        description="Run the PynamicMesh pipeline using a YAML configuration file."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        type=str,
        help="Path to the config.yaml file."
    )
    
    args = parser.parse_args()
    config_path = Path(args.config)


    if not config_path.exists():
        print(f"Error: Configuration file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)


    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading or parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    data_cfg = config.get("Data", {})
    fm_cfg = config.get("Functional_Map", {})
    rg_cfg = config.get("Reeb_Graph", {})


    path_str = data_cfg.get("path_str")
    if not path_str:
        print("Error: 'path_str' is missing under the 'Data' section in the config file.", file=sys.stderr)
        sys.exit(1)


    pipeline_kwargs = {
        "path_str": path_str,
        
        # Functional Map Settings
        "matrix_tranformation": fm_cfg.get("matrix_tranformation", True),
        "diagonal_analysis": fm_cfg.get("diagonal_analysis", True),
        "isometric_analysis": fm_cfg.get("isometric_analysis", True),
        "k_eigenfunc": fm_cfg.get("k_eigenfunc", 100),
        "descriptor": fm_cfg.get("descriptor", "WKS+HKS"),
        "landmarks": fm_cfg.get("landmarks", None),
        "compute_physic_fields": fm_cfg.get("compute_physic_fields", True),
        
        # Reeb Graph Settings
        "compute_reeb": rg_cfg.get("compute_reeb", True),
        "time_graph_analysis": rg_cfg.get("time_graph_analysis", True),
        "reeb_scalar": rg_cfg.get("reeb_scalar", "geodesic"),
        "bins": rg_cfg.get("bins", 30),
    }

    scalar_args = rg_cfg.get("scalar_args", {})
    if isinstance(scalar_args, dict):
        pipeline_kwargs.update(scalar_args)

    print(f"Executing pipeline with configuration from: {config_path}")
    try:
        run_pipeline(**pipeline_kwargs)
        print("Pipeline execution completed successfully.")
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()