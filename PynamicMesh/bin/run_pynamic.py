#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line execution of the PynamicMesh pipeline from a YAML configuration.

    python run_pynamic.py -c config.yaml                       # one configuration for every scene
    python run_pynamic.py -c config_batch.yaml --batch         # one configuration per scene folder
    python run_pynamic.py -c config.yaml --gif --frame-interval 30 --devices 0,1 --workers 6
    python run_pynamic.py -c config_batch.yaml --batch --parallel-scenes on      # one scene per GPU (cluster)

Every section of the YAML (Basic_Geometry, Functional_Map, Reeb_Graph, Graph_similarity, MS_Complex,
Parametrization, Trajectories, Graph_Animation, Dynamic_Analysis, Frame_Times, GIF) maps to the run_pipeline
arguments of execution_pipeline.py; the command-line options below override the YAML.
"""

import argparse
import sys

from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.utils.batch import run_batch, scene_kwargs
from PynamicMesh.utils.tools import extract_yaml


def main():
    parser = argparse.ArgumentParser(description="Run the PynamicMesh pipeline using a YAML configuration file.")
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to the config.yaml file.")
    parser.add_argument("--batch", action="store_true", default=False,
                        help="Indicate if a batch analysis is performed (separate configs per scene folder)")
    parser.add_argument("--gif", action="store_true", default=None,
                        help="Write the GIF animations of every stage (overrides GIF: gif in the YAML)")
    parser.add_argument("--frame-interval", type=float, default=None,
                        help="Time between consecutive frames (uniform time-lapse; overrides Frame_Times)")
    parser.add_argument("--frame-times", type=str, default=None,
                        help="File with the per-frame times (relative to each scene folder or absolute)")
    parser.add_argument("--time-unit", type=str, default=None, help="Unit of the frame times (default 's')")
    parser.add_argument("--devices", type=str, default=None,
                        help="GPUs to use: 'all' (default), 'cpu' or a list of ids such as '0,1'")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for the per-frame work (default: CPU cores - 1, at most 8)")
    parser.add_argument("--parallel-scenes", choices=("auto", "on", "off"), default=None,
                        help="Process several scenes at the same time, one per GPU ('auto' = when >1 GPU and >1 scene)")
    args = parser.parse_args()
    config_path = args.config

    config = extract_yaml(config_path)

    data_cfg = config.get("Data", {}) or {}
    path_str = data_cfg.get("path_str")
    if not path_str:
        print("Error: 'path_str' is missing under the 'Data' section in the config file.", file=sys.stderr)
        sys.exit(1)
    # compute options may also be given in the Data section (devices / workers)
    devices = args.devices if args.devices is not None else data_cfg.get("devices")
    workers = args.workers if args.workers is not None else data_cfg.get("workers")
    ps = args.parallel_scenes if args.parallel_scenes is not None else data_cfg.get("parallel_scenes", "auto")
    parallel_scenes = {"on": True, "off": False, True: True, False: False}.get(ps, "auto")
    compare = data_cfg.get("compare_scenes", "auto")          # comparison of the scenes (Results/_comparison)

    overrides = {}
    if args.gif:
        overrides["gif"] = True
    tp = {k: v for k, v in (("frame_interval", args.frame_interval), ("frame_times", args.frame_times),
                            ("time_unit", args.time_unit)) if v is not None}
    if tp:
        overrides["time_params"] = tp

    print(f"Executing pipeline with configuration from: {config_path}")

    try:
        if args.batch:
            run_batch(config, path_str, overrides, compute_devices=devices, n_workers=workers,
                      parallel_scenes=parallel_scenes, compare_scenes=compare)
        else:
            # Same flattening as the batch mode (every section, nested parameter dictionaries kept).
            global_kwargs = scene_kwargs(config, overrides)
            run_pipeline(path_str=path_str, is_batch=False, compute_devices=devices, n_workers=workers,
                         parallel_scenes=parallel_scenes, compare_scenes_flag=compare, **global_kwargs)
        print("Pipeline execution completed successfully.")
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
