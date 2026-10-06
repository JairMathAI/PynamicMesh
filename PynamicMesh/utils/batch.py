import copy
import sys

from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.utils.tools import kwargs_from_config, CONFIG_SECTIONS, STAGE_SECTIONS

# Top-level sections of a batch configuration that are DEFAULTS for every scene (e.g. GIF, Frame_Times, or a full
# Trajectories section shared by all scenes); a scene section with the same name overrides them key by key.
SHARED_SECTIONS = set(CONFIG_SECTIONS) | set(STAGE_SECTIONS)


def _deep_merge(base, over):
    """Dictionary `base` updated recursively with `over` (scene values win over the shared defaults)."""
    out = copy.deepcopy(base) if isinstance(base, dict) else {}
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def scene_kwargs(scene_cfg, overrides=None):
    """
    Flattens the sections of one scene configuration into run_pipeline keyword arguments:
    Functional_Map, Reeb_Graph, Basic_Geometry, Graph_similarity, MS_Complex (advanced options nested
    under fm_params / ms_tracker_params / ms_protrusion_params or flat) and the sections of the continuous
    models and outputs: Parametrization, Trajectories, Graph_Animation, Dynamic_Analysis, GIF, Frame_Times
    (parameters nested under param_params / traj_params / graph_animation_params / dyn_analysis_params /
    gif_params / time_params, or flat). `overrides` (e.g. command-line options) are merged last.
    """
    kwargs = kwargs_from_config(scene_cfg or {})
    for k, v in (overrides or {}).items():
        kwargs[k] = _deep_merge(kwargs.get(k, {}), v) if isinstance(v, dict) else v
    return kwargs


def run_batch(config, path_str, overrides=None, compute_devices=None, n_workers=None, parallel_scenes='auto',
              compare_scenes='auto'):
    """One run_pipeline call for the whole folder with a configuration per scene (keys other than Data and the
    shared sections are scene names); shared sections at the top level are defaults for every scene."""
    shared = {k: v for k, v in config.items() if k in SHARED_SECTIONS}
    batch_kwargs = {}
    for key, value in config.items():
        if key == "Data" or key in SHARED_SECTIONS:
            continue
        if not isinstance(value, dict):
            print(f"Warning: '{key}' is not a scene configuration (mapping expected) - ignored.", file=sys.stderr)
            continue
        batch_kwargs[key] = scene_kwargs(_deep_merge(shared, value), overrides)

    if not batch_kwargs:
        print("Error: Batch mode enabled, but no scene configurations found in YAML.", file=sys.stderr)
        sys.exit(1)

    run_pipeline(path_str=path_str, is_batch=True, batch_kwargs=batch_kwargs,
                 compute_devices=compute_devices, n_workers=n_workers, parallel_scenes=parallel_scenes,
                 compare_scenes_flag=compare_scenes)
