#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
import yaml
from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.utils.batch import run_batch
from PynamicMesh.utils.tools import (
    extract_kwargs,
    extract_yaml
)



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

    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Indicate if a batch analysis is performed (separate configs per scene folder)"
    )
    
    args = parser.parse_args()
    config_path = args.config

    config = extract_yaml(config_path)

    data_cfg = config.get("Data", {})
    path_str = data_cfg.get("path_str")

    if not path_str:
        print("Error: 'path_str' is missing under the 'Data' section in the config file.", file=sys.stderr)
        sys.exit(1)

    print(f"Executing pipeline with configuration from: {config_path}")
    
    try:
        if args.batch:
            run_batch(config,path_str)
        else:
            
            fm_cfg = config.get("Functional_Map", {})
            rg_cfg = config.get("Reeb_Graph", {})
            bg_cfg = config.get("Basic_Geometry", {})
            global_kwargs = extract_kwargs(fm_cfg, rg_cfg,bg_cfg)
            run_pipeline(path_str=path_str, is_batch=False, **global_kwargs)
        print("Pipeline execution completed successfully.")
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()