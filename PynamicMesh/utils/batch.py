import sys
from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.utils.tools import kwargs_from_config


def scene_kwargs(scene_cfg):
    """
    Flattens the sections of one scene configuration (Functional_Map, Reeb_Graph, Basic_Geometry,
    Graph_similarity, MS_Complex) into run_pipeline keyword arguments.
    The Functional_Map section may contain the advanced options either nested under `fm_params`
    or flat, and the MS_Complex section the tracker options nested under `ms_tracker_params` or
    flat; both forms are accepted (they are normalized by run_pipeline).
    """
    return kwargs_from_config(scene_cfg or {})


def run_batch(config, path_str):

    batch_kwargs = {}
    for key, value in config.items():
        if key == "Data":
            continue
        batch_kwargs[key] = scene_kwargs(value)

    if not batch_kwargs:
        print("Error: Batch mode enabled, but no scene configurations found in YAML.", file=sys.stderr)
        sys.exit(1)

    run_pipeline(path_str=path_str, is_batch=True, batch_kwargs=batch_kwargs)