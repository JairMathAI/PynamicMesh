from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.utils.tools import extract_kwargs

def run_batch(config,path_str):

    batch_kwargs = {}
    for key, value in config.items():
        if key == "Data":
            continue
        # Assuming any other top-level key that is a dict is a scene name
        if isinstance(value, dict) and ("Functional_Map" in value or "Reeb_Graph" in value):
            fm_cfg = value.get("Functional_Map", {})
            rg_cfg = value.get("Reeb_Graph", {})
            batch_kwargs[key] = extract_kwargs(fm_cfg, rg_cfg)
            
    if not batch_kwargs:
        print("Error: Batch mode enabled, but no scene configurations found in YAML.", file=sys.stderr)
        sys.exit(1)
        
    run_pipeline(path_str=path_str, is_batch=True, batch_kwargs=batch_kwargs)
