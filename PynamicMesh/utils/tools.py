import numpy as np
import pickle
import ast
import sys
import yaml
import h5py
from pathlib import Path
import numpy as np
import pyvista as pv
from pyFM.mesh import TriMesh
import re

def natural_sort_key(path_obj):
    """Moved outside the class to allow shared access for batch processing."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', path_obj.name)]

def mesh_mat2object(filepath):

    filepath = Path(filepath)

    if '.obj' in filepath.suffix:
        mesh = TriMesh(str(filepath))
    if '.mat' in filepath.suffix:
        with h5py.File(filepath, 'r') as f:
            vertices = np.array(f['surface/vertices']).T
            faces = np.array(f['surface/faces'], dtype=int).T
            if np.min(faces) == 1:
                faces -= 1   
            mesh = TriMesh(vertices,faces)  
           
    return mesh

def load_aligned_mesh(filepath):
    """
    Loads the mesh and applies the display alignment rotations natively.
    Creates a new TriMesh instance to safely bypass read-only property constraints.
    """

    tm_raw = mesh_mat2object(filepath)
    
    mesh_pv = pv.PolyData(tm_raw.vertices)
    mesh_pv.rotate_x(90, inplace=True)
    mesh_pv.rotate_z(90, inplace=True)
    
    tm_aligned = TriMesh(mesh_pv.points, tm_raw.faces)
    
    return tm_aligned

def extract_yaml(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Configuration file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading or parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return config


# Sections of a (scene) configuration and the extract_kwargs argument they map to.
CONFIG_SECTIONS = {"Functional_Map": "fm_cfg", "Reeb_Graph": "rg_cfg", "Basic_Geometry": "bg_cfg",
                   "Graph_similarity": "gs_cfg", "MS_Complex": "ms_cfg"}
# Advanced options of the Morse–Smale region tracker (RegionTracker); accepted nested in
# `ms_tracker_params` or flat inside the MS_Complex section.
MS_TRACKER_KEYS = ("iou_threshold", "dominance", "fate_radius", "growth_tolerance", "ghost_horizon")


def kwargs_from_config(cfg):
    """
    run_pipeline keyword arguments from a whole configuration mapping (the yaml of one scene or the
    single-scene yaml): every known section (Functional_Map, Reeb_Graph, Basic_Geometry,
    Graph_similarity, MS_Complex) is passed to extract_kwargs; the `Data` section is ignored.
    Nested mappings (fm_params, ms_tracker_params) are kept as dictionaries.
    """
    cfg = cfg or {}
    sections = {arg: dict(cfg.get(name, {}) or {}) for name, arg in CONFIG_SECTIONS.items()}
    fm_params = sections["fm_cfg"].pop("fm_params", None)
    ms_tracker = sections["ms_cfg"].pop("ms_tracker_params", None)
    kwargs = extract_kwargs(**sections)
    if fm_params:
        kwargs["fm_params"] = dict(fm_params)
    if ms_tracker:
        kwargs["ms_tracker_params"] = dict(ms_tracker)
    return kwargs


def extract_kwargs(fm_cfg, rg_cfg, bg_cfg, gs_cfg=None, ms_cfg=None):
    """
    Helper function to extract arguments from the config dictionaries.

    Known keys receive defaults; every other key of the sections is forwarded unchanged, so the
    advanced functional-map options (nested `fm_params` or flat `symmetry_mode`, `landmark_params`,
    `descr_params`, `fit_params`, `n_descr`, `subsample_step`, `refine`, `dt`, `verbose`), the
    scalar-field options of the Reeb graph and the Morse–Smale options (`MS_Complex` section:
    compute_mscomplex, ms_scalar, ms_persistence, ms_min_region_area, mscomplex_graph, ms_graph_type,
    ms_track_regions, ms_graph_metrics and `ms_tracker_params` nested or flat) reach run_pipeline
    (which normalizes them).  ms_cfg is optional so that older callers keep working; prefer
    kwargs_from_config(config) which handles every section at once.
    """
    fm_cfg, rg_cfg, bg_cfg, gs_cfg, ms_cfg = (dict(c or {}) for c in (fm_cfg, rg_cfg, bg_cfg, gs_cfg, ms_cfg))

    k_eigen = fm_cfg.get("k_eigenfunctions", (10, 10))
    
    if isinstance(k_eigen, str):
        try:
            parsed = ast.literal_eval(k_eigen.strip())
            if isinstance(parsed, (tuple, list)):
                k_eigen = tuple(int(v) for v in parsed)
            elif isinstance(parsed, int):
                k_eigen = (parsed, parsed)
            else:
                k_eigen = (10, 10)
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse k_eigenfunctions '{k_eigen}'. Falling back to default (10, 10).", file=sys.stderr)
            k_eigen = (10, 10)
    elif isinstance(k_eigen, list):
        k_eigen = tuple(int(v) for v in k_eigen)
    elif isinstance(k_eigen, int):
        k_eigen = (k_eigen, k_eigen)

    

    kwargs = {
        # Functional Map Settings
        "matrix_tranformation": fm_cfg.get("matrix_tranformation", True),
        "diagonal_analysis": fm_cfg.get("diagonal_analysis", True),
        "isometric_analysis": fm_cfg.get("isometric_analysis", True),
        "k_eigenfunctions": k_eigen,
        "k_eigenvalues": fm_cfg.get("k_eigenvalues", 100),
        "descriptor": fm_cfg.get("descriptor", "WKS+HKS"),
        "landmarks": fm_cfg.get("landmarks", None),
        "compute_physic_fields": fm_cfg.get("compute_physic_fields", True),
        
        # Reeb Graph Settings
        "compute_reeb": rg_cfg.get("compute_reeb", True),
        "time_graph_analysis": rg_cfg.get("time_graph_analysis", True),
        "reeb_scalar": rg_cfg.get("reeb_scalar", "geodesic"),
        "bins": rg_cfg.get("bins", 30),

        # Simple Geometry Settings
        "compute_basicGeo": bg_cfg.get("compute_basicGeo", True),
        "plot_basicGeo": bg_cfg.get("plot_basicGeo", True),
        "metrics": bg_cfg.get("metrics", 'all'),

        # Graph Similarity Settings
        "graph_sim": gs_cfg.get("graph_sim", True),
        "graph_metrics": gs_cfg.get("graph_metrics",'all'),

        # Morse–Smale Complex Settings (results in Results/<scene>/MSComplexAnalysis)
        "compute_mscomplex": ms_cfg.get("compute_mscomplex", False),
        "ms_scalar": ms_cfg.get("ms_scalar", None),              # None -> the Reeb scalar field (reused)
        "ms_persistence": ms_cfg.get("ms_persistence", 0.05),
        "ms_min_region_area": ms_cfg.get("ms_min_region_area", 0.0),
        "mscomplex_graph": ms_cfg.get("mscomplex_graph", False),
        "ms_graph_type": ms_cfg.get("ms_graph_type", "star"),
        "ms_track_regions": ms_cfg.get("ms_track_regions", True),
        "ms_graph_metrics": ms_cfg.get("ms_graph_metrics", gs_cfg.get("graph_metrics", 'all')),
    }

    # Tracker options: nested `ms_tracker_params` and/or flat keys of the MS_Complex section.
    ms_tracker = dict(ms_cfg.get("ms_tracker_params", {}) or {})
    for key in MS_TRACKER_KEYS:
        if key in ms_cfg and key not in ms_tracker:
            ms_tracker[key] = ms_cfg[key]
    if ms_tracker:
        kwargs["ms_tracker_params"] = ms_tracker

    # Scalar-field options: shared by the Reeb graph and the Morse–Smale complex (same get_scalar_field);
    # options given under MS_Complex/scalar_args complete (do not override) the Reeb ones.
    scalar_args = rg_cfg.get("scalar_args", {})
    if isinstance(scalar_args, dict):
        kwargs.update(scalar_args)
    ms_scalar_args = ms_cfg.get("scalar_args", {})
    if isinstance(ms_scalar_args, dict):
        for key, value in ms_scalar_args.items():
            kwargs.setdefault(key, value)

    # Pass-through of every other key (fm_params, flat FM options, extra scalar-field options, ...).
    for cfg in (fm_cfg, rg_cfg, bg_cfg, gs_cfg, ms_cfg):
        for key, value in cfg.items():
            if key in ("scalar_args", "ms_tracker_params") or key in kwargs or key in MS_TRACKER_KEYS:
                continue
            kwargs[key] = value

    return kwargs

LANDMARK_FILES = {
    'FM': 'landmarks.npy',
    'geodesic': 'vert_ref_geo.npy',
    'mass_center_geodesic': 'vert_ref_geo.npy',
    'heat_diffusion': 'sources.npy',
    'matern_kernel': 'sources.npy',
    'harmonic': 'source_sink.npy',
}


def landmark_load(landmarks, target_folder, mood):
    file = LANDMARK_FILES.get(mood)
    if file is None:
        raise ValueError(f"No precomputed selection file is defined for '{mood}'. Options: {list(LANDMARK_FILES)}")
    if mood == 'matern_kernel':
        mood = 'heat_diffusion'
    if mood == 'mass_center_geodesic':
        mood = 'geodesic'

    loaded_landmarks = None
    if isinstance(landmarks, str) and landmarks.lower() == 'precomputed':
        temp = []
        landmarks_file = Path(target_folder) / file
        if landmarks_file.exists():
            try:
                loaded_landmarks = np.load(landmarks_file, allow_pickle=True)
                if mood == 'FM':
                    # One (n, 2) pair array per transition, or None for transitions without selection.
                    # Kept as a list: transitions may have different numbers of pairs.
                    loaded_landmarks = [None if x is None or len(x) == 0 else np.asarray(x, dtype=int)
                                        for x in loaded_landmarks]
                if mood == 'geodesic' or mood == 'heat_diffusion':
                    for element in loaded_landmarks:
                        if isinstance(element, list):
                            temp.append(element)
                        else:
                            temp.append([0])
                    loaded_landmarks = temp
            except Exception as exc:
                if mood != 'FM':
                    raise ValueError(f'Could not read precomputed selection {landmarks_file}: {exc}')
                print(f"[Warning] Could not read precomputed landmarks {landmarks_file} ({exc}); running without landmarks.",
                      file=sys.stderr)
                loaded_landmarks = None
        else:
            print(f"[Warning] Precomputed selection requested but {landmarks_file} does not exist; using defaults.",
                  file=sys.stderr)
            loaded_landmarks = None
    return loaded_landmarks


def landmark_parser(landmarks, loaded_landmarks, i, mood=None):
    current_landmarks = None
    if isinstance(landmarks, str):
        if landmarks.lower() == 'precomputed' and loaded_landmarks is not None:
            if mood == 'FM':
                sel = loaded_landmarks[i-1] if 0 <= i-1 < len(loaded_landmarks) else None
                current_landmarks = None if sel is None else np.asarray(sel, dtype=int)
            if mood == 'matern_kernel':
                mood = 'heat_diffusion'
            if mood == 'geodesic':
                current_landmarks = loaded_landmarks[i]
            if mood == 'heat_diffusion':
                current_landmarks = loaded_landmarks[i][0]
            if mood == 'harmonic':
                current_landmarks = loaded_landmarks[i]
    else:
        if landmarks is not None:
            if isinstance(landmarks, (list, tuple, np.ndarray)) and len(landmarks) > 0:
                first_elem = landmarks[0]
                is_multiple_mappings = False
                if isinstance(first_elem, (list, tuple, np.ndarray)):
                    if len(first_elem) > 0 and isinstance(first_elem[0], (list, tuple, np.ndarray)): 
                        is_multiple_mappings = True
                    elif any(len(x) != 2 for x in landmarks): 
                        is_multiple_mappings = True
                
                if is_multiple_mappings:
                    pair_idx = i - 1
                    if pair_idx < len(landmarks): 
                        current_landmarks = np.array(landmarks[pair_idx])
                    else: 
                        current_landmarks = None
                else: 
                    current_landmarks = np.array(landmarks)
            else: 
                current_landmarks = np.array(landmarks)
        else: 
            current_landmarks = None

    return current_landmarks

def resolve_scalar_args(reeb_scalar, scalar_kwargs, index, num_vertices, loaded_selections=None):
    kwargs_out = scalar_kwargs.copy()
    

    methods_to_check = [reeb_scalar]
    if reeb_scalar == 'multi_pca':
        methods_to_check.extend(kwargs_out.get("fields", []))
    

    if 'geodesic' in methods_to_check or 'mass_center_geodesic' in methods_to_check:
        vertex_ref_index = scalar_kwargs.get("vertex_ref_index", None)
        if vertex_ref_index == 'precomputed':
            selection = landmark_parser(vertex_ref_index, loaded_selections, index, 'geodesic')
            kwargs_out['vertex_ref_index'] = selection
            
        elif vertex_ref_index == 'mass_center' or (isinstance(vertex_ref_index, list) and len(vertex_ref_index) > 0 and vertex_ref_index[0] == 'mass_center'):
            kwargs_out['vertex_ref_index'] = 'mass_center'
            
        else:
            if vertex_ref_index is None:
                kwargs_out['vertex_ref_index'] = [0]
            else:
                is_list_of_lists = isinstance(vertex_ref_index, (list, tuple)) and any(isinstance(x, (list, tuple)) or x is None for x in vertex_ref_index)
                if is_list_of_lists:
                    if index < len(vertex_ref_index) and vertex_ref_index[index] is not None:
                        kwargs_out['vertex_ref_index'] = list(vertex_ref_index[index])
                    else:
                        kwargs_out['vertex_ref_index'] = [0]
                else:
                    if isinstance(vertex_ref_index, (int, float)):
                        kwargs_out['vertex_ref_index'] = [int(vertex_ref_index)]
                    else:
                        kwargs_out['vertex_ref_index'] = list(vertex_ref_index)


    if 'heat_diffusion' in methods_to_check or 'matern_kernel' in methods_to_check:
        source_idx = scalar_kwargs.get("source_idx", None)
        if source_idx == 'precomputed':
            selection = landmark_parser(source_idx, loaded_selections, index, 'heat_diffusion')
            kwargs_out['source_idx'] = selection
            
        elif source_idx == 'mass_center' or (isinstance(source_idx, list) and len(source_idx) > 0 and source_idx[0] == 'mass_center'):
            kwargs_out['source_idx'] = 'mass_center'
            
        else:
            if source_idx is None:
                kwargs_out['source_idx'] = 0
            elif isinstance(source_idx, (list, tuple)):
                if index < len(source_idx) and source_idx[index] is not None:
                    kwargs_out['source_idx'] = int(source_idx[index])
                else:
                    kwargs_out['source_idx'] = 0
            else:
                kwargs_out['source_idx'] = int(source_idx)

    if 'harmonic' in methods_to_check:
        source_idx = scalar_kwargs.get("source_idx", None)
        sink_idx = scalar_kwargs.get("sink_idx", None)
        
        if source_idx == 'precomputed':
            selection = landmark_parser(source_idx, loaded_selections, index, 'harmonic')
            if selection is not None and len(selection) >= 2:
                kwargs_out['source_idx'] = selection[0]
                kwargs_out['sink_idx'] = selection[1]
            else:
                kwargs_out['source_idx'] = 0
                kwargs_out['sink_idx'] = num_vertices - 1
        else:
            default_source = 0
            default_sink = num_vertices - 1
            
            is_list_of_pairs = isinstance(source_idx, (list, tuple)) and any(isinstance(x, (list, tuple)) or x is None for x in source_idx)
            
            if is_list_of_pairs:
                if index < len(source_idx) and source_idx[index] is not None:
                    val = source_idx[index]
                    if len(val) >= 2:
                        kwargs_out['source_idx'] = int(val[0])
                        kwargs_out['sink_idx'] = int(val[1])
                    else:
                        kwargs_out['source_idx'] = default_source
                        kwargs_out['sink_idx'] = default_sink
                else:
                    kwargs_out['source_idx'] = default_source
                    kwargs_out['sink_idx'] = default_sink
            else:
                if isinstance(source_idx, (list, tuple)):
                    kwargs_out['source_idx'] = int(source_idx[0]) if len(source_idx) > 0 else default_source
                    if len(source_idx) >= 2:
                        kwargs_out['sink_idx'] = int(source_idx[1])
                    else:
                        kwargs_out['sink_idx'] = int(sink_idx) if sink_idx is not None else default_sink
                else:
                    kwargs_out['source_idx'] = int(source_idx) if source_idx is not None else default_source
                    kwargs_out['sink_idx'] = int(sink_idx) if sink_idx is not None else default_sink

    return kwargs_out


def optimize_param(meshn_1, meshn):
    """
    Dynamically calculates ZoomOut parameters based on mesh density and resolution mismatch.
    Reduces iterations and increases step size for heavy or highly mismatched meshes 
    to prevent nearest-neighbor search bottlenecks.
    """

    v1 = meshn_1.vertlist.shape[0] if hasattr(meshn_1, 'vertlist') else len(meshn_1.vertices)
    v2 = meshn.vertlist.shape[0] if hasattr(meshn, 'vertlist') else len(meshn.vertices)
    
    max_v = max(v1, v2)
    mismatch_ratio = max_v / min(v1, v2)
    
    if max_v <= 10000 and mismatch_ratio <= 1.5:
        nit = 30
        step = 2
    else:
        return None 
    return (nit, step)