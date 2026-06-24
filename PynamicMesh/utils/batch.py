from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.utils.tools import extract_kwargs

def run_batch(config,path_str):

    batch_kwargs = {}
    for key, value in config.items():
        if key == "Data":
            continue

        fm_cfg = value.get("Functional_Map", {})
        rg_cfg = value.get("Reeb_Graph", {})
        bg_cfg = value.get("Basic_Geometry", {})
        batch_kwargs[key] = extract_kwargs(fm_cfg, rg_cfg, bg_cfg)
            
    if not batch_kwargs:
        print("Error: Batch mode enabled, but no scene configurations found in YAML.", file=sys.stderr)
        sys.exit(1)
        
    run_pipeline(path_str=path_str, is_batch=True, batch_kwargs=batch_kwargs)
