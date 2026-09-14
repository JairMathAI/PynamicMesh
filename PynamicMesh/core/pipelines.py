import os
import re
import numpy as np
import pyvista as pv
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from PynamicMesh.core.custom_fm import CustomFunctionalMapping
from pyFM.mesh import TriMesh
from PIL import Image
import pickle
from PynamicMesh.core.basicGeometry import compute_mesh_geometry, generate_plots_from_csv
from PynamicMesh.core.graph_sim import graph_similarity, plot_graph_similarity
from PynamicMesh.core.physic_model import (
    plot_diagonal,
    generate_tranformation_heatmap,
    Diagonal_metrics,
    compute_heatmap_similarity,
    plot_similarity_metrics,
    computing_fields
)

from PynamicMesh.utils.visualizers import pick_single_mesh
from PynamicMesh.core.reeb_graph import (
    get_scalar_field,
    compute_approx_reeb_graph,
    graph_time_analysis,
    plot_dynamic_graph_analysis
)
from PynamicMesh.core.SMComplex import (
    compute_MS,
    RegionTracker,
    critical_points_report,
    tracking_report,
    ms_graph_analysis
)
from PynamicMesh.utils.tools import (
    landmark_load,
    landmark_parser,
    resolve_scalar_args,
    optimize_param,
    mesh_mat2object,
    load_aligned_mesh
)

# Scalar fields that need the Laplace–Beltrami eigendecomposition / stiffness matrix (mesh.process()).
SPECTRAL_METHODS = {"heat_diffusion", "harmonic", "multi_pca", "matern_kernel"}
# Scalar fields that need the point-to-point map of the functional-map step.
MAP_DEPENDENT_METHODS = {"normal_displacement"}
# Options of CustomFunctionalMapping; accepted nested in `fm_params` or flat at the top level of a config.
FM_PARAM_KEYS = ("symmetry_mode", "landmark_params", "descr_params", "fit_params",
                 "n_descr", "subsample_step", "refine", "dt", "verbose")


def _parse_int_tuple(value):
    """(10, 10) | [10, 10] | 10 | '(10,10)' | '10, 10' | '10'  ->  tuple of ints or int."""
    if isinstance(value, str):
        nums = [int(tok) for tok in re.findall(r"-?\d+", value)]
        if not nums:
            raise ValueError(f"Cannot parse an integer or a pair of integers from '{value}'.")
        return nums[0] if len(nums) == 1 else tuple(nums[:2])
    if isinstance(value, (list, tuple)):
        nums = [int(v) for v in value]
        return nums[0] if len(nums) == 1 else tuple(nums[:2])
    return int(value)


def _none_if_null(value):
    """YAML/CLI strings 'None', 'null', '~', '' -> None."""
    if isinstance(value, str) and value.strip().lower() in ("none", "null", "~", ""):
        return None
    return value


def normalize_params(params):
    """
    Makes a configuration coming from Python, YAML or the command line safe for the pipeline:
    numeric tuples given as strings, textual nulls, and functional-map options given either flat
    (`symmetry_mode: ...` under Functional_Map) or nested (`fm_params: {symmetry_mode: ...}`).
    Returns a new dict.
    """
    p = {k: _none_if_null(v) for k, v in dict(params).items()}
    if "k_eigenfunctions" in p and p["k_eigenfunctions"] is not None:
        p["k_eigenfunctions"] = _parse_int_tuple(p["k_eigenfunctions"])
    if "k_eigenvalues" in p and p["k_eigenvalues"] is not None:
        p["k_eigenvalues"] = int(p["k_eigenvalues"])
    if "bins" in p and p["bins"] is not None:
        p["bins"] = int(p["bins"])

    fm_params = dict(p.get("fm_params") or {})
    for key in FM_PARAM_KEYS:
        if key in p and key not in fm_params:
            fm_params[key] = p[key]
        p.pop(key, None)
    for key in ("landmark_params", "descr_params", "fit_params"):
        if key in fm_params and fm_params[key] is None:
            fm_params[key] = {}
    if "symmetry_mode" in fm_params and fm_params["symmetry_mode"] is None:
        fm_params["symmetry_mode"] = "none"
    if "refine" in fm_params and isinstance(fm_params["refine"], (str, list, tuple)) \
            and fm_params["refine"] not in ("auto", "icp"):
        fm_params["refine"] = _parse_int_tuple(fm_params["refine"])
    p["fm_params"] = fm_params

    # A flat `scalar_args` mapping (YAML style) is expanded into the top-level scalar-field kwargs.
    scalar_args = p.pop("scalar_args", None)
    if isinstance(scalar_args, dict):
        for k, v in scalar_args.items():
            p.setdefault(k, _none_if_null(v))
    return p


def natural_sort_key(path):
    """Sort key so that frame_2 < frame_10 (a plain string sort gives frame_1, frame_10, frame_2, ...)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r'(\d+)', Path(path).name)]


def needs_spectral(reeb_scalar, scalar_kwargs):
    """True if the requested Reeb scalar field (or any sub-field of multi_pca) is spectral."""
    if reeb_scalar in SPECTRAL_METHODS or reeb_scalar.startswith("lb_eigen_"):
        return True
    if reeb_scalar == "multi_pca":
        fields = scalar_kwargs.get("fields", ["z", "mean_curvature", "gaussian_curvature"])
        return any(f in SPECTRAL_METHODS or f.startswith("lb_eigen_") for f in fields)
    return False


def _fm_basis_size(k_eigenfunctions):
    return int(max(k_eigenfunctions)) if isinstance(k_eigenfunctions, (tuple, list)) else int(k_eigenfunctions)


def compute_FM(meshn_1, meshn, i, target_folder, descriptor, current_landmarks, k_eigenvalues, k_eigenfunctions, prev_FM_zo,
               inertia_history, decay_history, cdf_history, similarity_history, heatmap_paths, diagonal_analysis,
               isometric_analysis, fm_params=None):
    """
    Functional map T{i-1} -> T{i}. fm_params (dict) controls the new CustomFunctionalMapping options:
        symmetry_mode, landmark_params, descr_params, fit_params, n_descr, subsample_step, refine.
    """
    fm_params = fm_params or {}
    os.makedirs(target_folder / 'Transform_Matrices', exist_ok=True)
    os.makedirs(target_folder / 'Diagonal_analysis', exist_ok=True)
    os.makedirs(target_folder / 'Landmarks', exist_ok=True)

    # Refinement parameters first: ZoomOut needs k + nit*step eigenpairs and the basis must NOT be
    # recomputed after preprocess (a new eigendecomposition can flip eigenvector signs under the map).
    refine = fm_params.get("refine", "auto")
    params_op = optimize_param(meshn_1, meshn) if refine == "auto" else (refine if refine not in (None, "icp") else None)
    k_fm = _fm_basis_size(k_eigenfunctions)
    k_needed = max(int(k_eigenvalues), k_fm + (params_op[0] * params_op[1] if params_op is not None else 0))
    for mesh in (meshn_1, meshn):
        if mesh.eigenvalues is None or len(mesh.eigenvalues) < k_needed:
            mesh.process(k=k_needed)

    model = CustomFunctionalMapping(meshn_1, meshn)
    model.preprocess(
        n_ev=k_eigenfunctions,
        n_descr=fm_params.get("n_descr", 100),
        descr_type=descriptor,
        landmarks=current_landmarks,
        subsample_step=fm_params.get("subsample_step", 1),
        k_process=k_needed,
        symmetry_mode=fm_params.get("symmetry_mode", "none"),
        landmark_params=fm_params.get("landmark_params"),
        descr_params=fm_params.get("descr_params"),
        verbose=fm_params.get("verbose", False),
    )
    model.fit(**fm_params.get("fit_params", {}))

    if params_op is not None:
        FM_zo = model.zoomout_refine(nit=params_op[0], step=params_op[1])
    else:
        FM_zo = model.icp_refine()
    FM_zo = np.asarray(FM_zo)

    # Point-to-point map mesh2 -> mesh1 (p2p_zo[j] = vertex of T{i-1} matched to vertex j of T{i}),
    # computed directly from the refined map so it does not depend on the pyFM get_p2p signature.
    p2p_zo = model.get_p2p_from_FM(FM_zo)

    if model.landmarks is not None:
        np.save(target_folder / 'Landmarks' / f'landmarks_T{i-1:04d}_T{i:04d}.npy', model.landmarks)

    if diagonal_analysis:
        heatmap_filename = target_folder / 'Diagonal_analysis' / f'FM_{i-1}{i}.png'
        generate_tranformation_heatmap(FM_zo, i, heatmap_filename)
        heatmap_paths.append(heatmap_filename)

        inrt, decy, cdf = Diagonal_metrics(FM_zo)
        inertia_history.append(inrt)
        decay_history.append(decy)
        cdf_history.append(cdf)
        plot_diagonal(inertia_history, decay_history, cdf_history, target_folder / 'Diagonal_analysis' / 'Diagonal_metrics.png')

        make_csv = target_folder / 'Diagonal_analysis' / 'Diagonal_Metrics.csv'
        with open(make_csv, mode='w', encoding='utf-8') as csv_file:
            csv_file.write("Timestep,Map,Inertia_Metric,Decay_Metric\n")
            for idx, (inrt_v, decy_v) in enumerate(zip(inertia_history, decay_history)):
                csv_file.write(f"{idx+1},FM_{idx}{idx+1},{inrt_v:.6f},{decy_v:.6f}\n")

    if prev_FM_zo is not None and isometric_analysis:
        jsd, pearson, spearman, manhattan, euclidean = compute_heatmap_similarity(prev_FM_zo, FM_zo)
        similarity_history.append([jsd, pearson, spearman, manhattan, euclidean])

        plot_similarity_metrics(similarity_history, target_folder / 'Diagonal_analysis' / 'Cross_Heatmap_Similarity.png')

        csv_path = target_folder / 'Diagonal_analysis' / 'Cross_Heatmap_Similarity.csv'
        with open(csv_path, mode='w', encoding='utf-8') as csv_file:
            csv_file.write("Comparison,JSD,Pearson_Corr,Spearman_Corr,Manhattan_Dist,Euclidean_Dist\n")
            for idx, values in enumerate(similarity_history):
                lbl = f"FM_{idx}{idx+1}_vs_FM_{idx+1}{idx+2}"
                csv_file.write(f"{lbl},{values[0]:.6f},{values[1]:.6f},{values[2]:.6f},{values[3]:.6f},{values[4]:.6f}\n")

    # Zero-padded names; computing_fields() reads these (and falls back to the legacy FMV_{i}{i-1}.npy).
    np.save(target_folder / 'Transform_Matrices' / f'FMC_T{i-1:04d}_T{i:04d}.npy', FM_zo)
    np.save(target_folder / 'Transform_Matrices' / f'FMV_T{i:04d}_T{i-1:04d}.npy', p2p_zo)

    return FM_zo, p2p_zo, str(target_folder / 'Transform_Matrices')


def compute_RG(meshn, i, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections, needs_spectral_processing, prev_vertices=None, p2p_zo=None):
    reeb_folder = target_folder / 'Reeb_Graphs'
    os.makedirs(reeb_folder, exist_ok=True)

    tn_kwargs = resolve_scalar_args(reeb_scalar, scalar_kwargs, i, meshn.vertices.shape[0], loaded_selections)
    # get_scalar_field now validates shape/finiteness and warns on constant fields;
    # extra options it understands: equalize_histogram, geodesic_solver, t='auto', signed.
    scalar_tn = get_scalar_field(
        meshn.vertices, meshn.faces, method=reeb_scalar,
        prev_vertices=prev_vertices, p2p=p2p_zo,
        trimesh_obj=meshn if needs_spectral_processing else None,
        **tn_kwargs
    )

    reeb_tn = compute_approx_reeb_graph(meshn.vertices, meshn.faces, scalar_tn, num_bins=bins)

    # Zero-padded frame index so plain string ordering matches time ordering (Reeb_T0002 < Reeb_T0010).
    with open(reeb_folder / f'Reeb_T{i:04d}.pkl', 'wb') as f:
        pickle.dump(reeb_tn, f)
    np.save(reeb_folder / f'Scalar_T{i:04d}.npy', scalar_tn)

    return str(reeb_folder)


def compute_MS_frame(meshn, i, ms_scalar, reeb_scalar, compute_reeb, scalar_kwargs, target_folder, loaded_selections_rg,
                     needs_spectral_processing, ms_persistence, ms_min_region_area, mscomplex_graph, ms_graph_type,
                     prev_vertices=None, p2p_zo=None):
    """
    Morse–Smale complex of frame i (SMComplex.compute_MS). When the requested field is the Reeb field of
    the same frame and it has just been saved, it is reused instead of recomputed.
    """
    scalar_field = None
    reeb_scalar_file = target_folder / 'Reeb_Graphs' / f'Scalar_T{i:04d}.npy'
    if compute_reeb and ms_scalar == reeb_scalar and reeb_scalar_file.exists():
        cached = np.load(reeb_scalar_file)
        if cached.shape[0] == meshn.vertices.shape[0]:
            scalar_field = cached
    selections = loaded_selections_rg if ms_scalar == reeb_scalar else None
    tn_kwargs = resolve_scalar_args(ms_scalar, scalar_kwargs, i, meshn.vertices.shape[0], selections)
    return compute_MS(meshn.vertices, meshn.faces, i, target_folder, scalar_field=scalar_field,
                      scalar_method=ms_scalar, scalar_kwargs=tn_kwargs, persistence=ms_persistence,
                      min_region_area=ms_min_region_area, build_graph=mscomplex_graph, graph_type=ms_graph_type,
                      trimesh_obj=meshn if needs_spectral_processing else None, prev_vertices=prev_vertices, p2p=p2p_zo)


def headmap_gif(heatmap_paths, target_folder):
    frames = [Image.open(img_path) for img_path in heatmap_paths]
    gif_path = target_folder / 'Diagonal_analysis' / 'FM_Heatmap_Animation.gif'
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=700, loop=0)


def process_sequence(folder, path, compute_basicGeo=True, plot_basicGeo=True, metrics='all',
                     matrix_tranformation=True, diagonal_analysis=True, isometric_analysis=True,
                     k_eigenvalues=100, k_eigenfunctions=30, descriptor='WKS', landmarks=None,
                     compute_reeb=True, reeb_scalar='geodesic', bins=20, needs_spectral_processing=True,
                     compute_physic_fields=False, scalar_kwargs=None, fm_params=None,
                     compute_mscomplex=False, ms_scalar=None, ms_persistence=0.05, ms_min_region_area=0.0,
                     mscomplex_graph=False, ms_graph_type='star', ms_track_regions=True, ms_graph_metrics='all',
                     ms_tracker_params=None):
    """
    Morse–Smale options (results under Results/<scene>/MSComplexAnalysis):
        compute_mscomplex  : Morse–Smale complex + protrusion segmentation of every frame, critical point csv/plots
        ms_scalar          : scalar field of the complex (any get_scalar_field method); None = reeb_scalar
        ms_persistence     : persistence simplification threshold (fraction of the field range)
        ms_min_region_area : merge regions smaller than this fraction of the surface into their neighbour
        mscomplex_graph    : build the critical-point graph (star to the centre of mass) and run the
                             Reeb-graph analyses (graph_time_analysis, graph_similarity) on it
        ms_graph_type      : 'star' | 'star+adjacency' (adds edges between adjacent protrusion regions)
        ms_track_regions   : region correspondence / critical point fates through the functional maps
                             (needs matrix_tranformation=True)
        ms_graph_metrics   : metrics for graph_similarity on the critical-point graphs
        ms_tracker_params  : dict for RegionTracker (iou_threshold, dominance, fate_radius, growth_tolerance, ghost_horizon)
    """
    scalar_kwargs = scalar_kwargs or {}
    fm_params = fm_params or {}
    ms_scalar = (ms_scalar or reeb_scalar).lower()
    ms_tracker_params = ms_tracker_params or {}
    itemsfiles = list(folder.iterdir())
    obj_files = sorted([f for f in itemsfiles if f.is_file() and f.suffix in ('.obj', '.mat')], key=natural_sort_key)

    if len(obj_files) < 2:
        return None, None

    out_root = path.parent / 'Results'
    scene_name = obj_files[0].parent.name
    target_folder = out_root / scene_name
    os.makedirs(target_folder, exist_ok=True)

    # The displacement-based scalar field needs the p2p map, which only exists with the FM step.
    if compute_reeb and reeb_scalar in MAP_DEPENDENT_METHODS and not matrix_tranformation:
        print(f"\n[Warning] '{reeb_scalar}' requires matrix_tranformation=True; skipping Reeb graphs for {scene_name}.")
        compute_reeb = False
    if compute_mscomplex and ms_scalar in MAP_DEPENDENT_METHODS and not matrix_tranformation:
        print(f"\n[Warning] '{ms_scalar}' requires matrix_tranformation=True; skipping Morse–Smale complexes for {scene_name}.")
        compute_mscomplex = False
    if compute_mscomplex and ms_track_regions and not matrix_tranformation:
        print(f"\n[Warning] Region tracking needs the functional maps (matrix_tranformation=True); only the segmentation will be computed for {scene_name}.")
        ms_track_regions = False

    # 'auto' landmarks are selected per pair inside CustomFunctionalMapping; the file-based
    # landmark_load/landmark_parser path is only used for precomputed selections.
    auto_landmarks = isinstance(landmarks, str) and landmarks.lower() == 'auto'
    loaded_landmarks_fm = None
    if matrix_tranformation and landmarks is not None and not auto_landmarks:
        loaded_landmarks_fm = landmark_load(landmarks, target_folder, 'FM')

    loaded_selections_rg = None
    if compute_reeb:
        vertex_ref_index = scalar_kwargs.get("vertex_ref_index", None)
        source_idx = scalar_kwargs.get("source_idx", None)
        if reeb_scalar == 'geodesic' and vertex_ref_index == 'precomputed':
            loaded_selections_rg = landmark_load(vertex_ref_index, target_folder, reeb_scalar)
        elif reeb_scalar in ['heat_diffusion', 'matern_kernel', 'harmonic'] and source_idx == 'precomputed':
            loaded_selections_rg = landmark_load(source_idx, target_folder, reeb_scalar)

    meshn_1 = load_aligned_mesh(obj_files[0])

    if needs_spectral_processing:
        meshn_1.process(k=k_eigenvalues)

    # Frame 0: no previous frame, so map-dependent fields would be identically zero (a one-node
    # graph that only pollutes the time series). Start those at frame 1 instead.
    if compute_reeb and reeb_scalar not in MAP_DEPENDENT_METHODS:
        compute_RG(meshn_1, 0, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections_rg, needs_spectral_processing)

    ms_prev = None
    ms_tracker = RegionTracker(target_folder, **ms_tracker_params) if (compute_mscomplex and ms_track_regions) else None
    if compute_mscomplex and ms_scalar not in MAP_DEPENDENT_METHODS:
        ms_prev = compute_MS_frame(meshn_1, 0, ms_scalar, reeb_scalar, compute_reeb, scalar_kwargs, target_folder,
                                   loaded_selections_rg, needs_spectral_processing, ms_persistence, ms_min_region_area,
                                   mscomplex_graph, ms_graph_type)

    computed_metrics = []            # always defined: avoids NameError when compute_basicGeo=False
    geom_path = None
    if compute_basicGeo:
        geom_path = target_folder / 'Basic_Geometry'
        os.makedirs(geom_path, exist_ok=True)
        computed_metrics.append(compute_mesh_geometry(meshn_1, metrics=metrics))

    inertia_history, decay_history, cdf_history = [], [], []
    similarity_history = []
    heatmap_paths = []
    RG_out_path = ''
    FM_out_path = ''
    prev_FM_zo = None

    for i in tqdm(range(1, len(obj_files)), desc=f'processing {scene_name}', leave=False):

        meshn = load_aligned_mesh(obj_files[i])
        p2p_zo = None

        if needs_spectral_processing:
            meshn.process(k=k_eigenvalues)

        if matrix_tranformation:
            if auto_landmarks:
                current_landmarks = 'auto'
            elif landmarks is None:
                current_landmarks = None
            else:
                current_landmarks = landmark_parser(landmarks, loaded_landmarks_fm, i, 'FM')
            FM_zo, p2p_zo, FM_out_path = compute_FM(
                meshn_1, meshn, i, target_folder, descriptor, current_landmarks,
                k_eigenvalues, k_eigenfunctions, prev_FM_zo, inertia_history, decay_history,
                cdf_history, similarity_history, heatmap_paths, diagonal_analysis, isometric_analysis,
                fm_params=fm_params
            )
            prev_FM_zo = FM_zo.copy()

        if compute_reeb:
            prev_verts = meshn_1.vertices if matrix_tranformation else None
            RG_out_path = compute_RG(meshn, i, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections_rg, needs_spectral_processing, prev_verts, p2p_zo)

        if compute_mscomplex:
            prev_verts = meshn_1.vertices if matrix_tranformation else None
            ms_curr = compute_MS_frame(meshn, i, ms_scalar, reeb_scalar, compute_reeb, scalar_kwargs, target_folder,
                                       loaded_selections_rg, needs_spectral_processing, ms_persistence, ms_min_region_area,
                                       mscomplex_graph, ms_graph_type, prev_verts, p2p_zo)
            if ms_tracker is not None and ms_prev is not None and p2p_zo is not None:
                ms_tracker.track_pair(ms_prev, ms_curr, meshn_1.vertices, meshn_1.faces, meshn.vertices, meshn.faces,
                                      p2p_zo, FM_12=FM_zo, mesh_prev=meshn_1, mesh_curr=meshn)
            ms_prev = ms_curr

        if compute_basicGeo:
            computed_metrics.append(compute_mesh_geometry(meshn, metrics=metrics))

        meshn_1 = meshn

    if compute_basicGeo and computed_metrics:
        df_results = pd.DataFrame(computed_metrics)
        csv_path = geom_path / 'features_computed.csv'
        df_results.to_csv(csv_path, index=False)
        if plot_basicGeo:
            generate_plots_from_csv(csv_path)

    if matrix_tranformation and heatmap_paths:
        headmap_gif(heatmap_paths, target_folder)

    if compute_mscomplex:
        critical_points_report(target_folder, single_file=False)
        if ms_tracker is not None and ms_tracker.summary:
            ms_tracker.save()
            tracking_report(target_folder, single_file=False)
        if mscomplex_graph:
            ms_graph_analysis(target_folder, graph_metrics=ms_graph_metrics, single_file=False)

    if compute_physic_fields:
        if not matrix_tranformation:
            print(f"\n[Warning] Cannot calculate fields for {scene_name} without track mappings.")
        else:
            matrix_folder_path = target_folder / 'Transform_Matrices'
            physical_fields_path = target_folder / 'Physical_fields'
            computing_fields(folder, matrix_folder_path, physical_fields_path, single_file=False,
                             dt=fm_params.get('dt', 1.0), loader=load_aligned_mesh)

    return FM_out_path, RG_out_path


def run_pipeline(path_str, is_batch=False, batch_kwargs=None, **kwargs):
    path = Path(path_str)
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{path_str}' is not a valid directory.")
        return

    subdirectories = sorted([f for f in path.iterdir() if f.is_dir()], key=natural_sort_key)
    if not subdirectories:
        print(f'No folders found on {path}')
        return

    process_seq_keys = [
        'plot_basicGeo', 'compute_basicGeo', 'metrics', 'matrix_tranformation', 'diagonal_analysis', 'isometric_analysis',
        'k_eigenfunctions', 'k_eigenvalues', 'descriptor', 'landmarks',
        'compute_reeb', 'reeb_scalar', 'bins', 'compute_physic_fields', 'fm_params',
        'compute_mscomplex', 'ms_scalar', 'ms_persistence', 'ms_min_region_area', 'mscomplex_graph',
        'ms_graph_type', 'ms_track_regions', 'ms_graph_metrics', 'ms_tracker_params'
    ]
    pipeline_only_keys = {'time_graph_analysis', 'graph_sim', 'graph_metrics'}

    for folder in tqdm(subdirectories, desc='processing folder'):
        scene_name = folder.name

        if is_batch:
            if not batch_kwargs or scene_name not in batch_kwargs:
                print(f"\n[Warning] Skipping '{scene_name}': No configuration found in batch config.")
                continue
            current_params = batch_kwargs[scene_name]
        else:
            current_params = kwargs.copy()
        current_params = normalize_params(current_params)

        matrix_tranformation = current_params.get("matrix_tranformation", True)
        compute_reeb = current_params.get("compute_reeb", True)
        reeb_scalar = current_params.get("reeb_scalar", "geodesic").lower()
        time_graph_analysis = current_params.get("time_graph_analysis", True)
        graph_sim = current_params.get("graph_sim", True)
        graph_metrics = current_params.get("graph_metrics", 'all')

        seq_args = {k: current_params[k] for k in process_seq_keys if k in current_params}
        seq_args['reeb_scalar'] = reeb_scalar

        # Everything else is forwarded to get_scalar_field (e.g. vertex_ref_index, source_idx, sink_idx,
        # t, nu, lengthscale, fields, equalize_histogram, geodesic_solver, signed).
        # graph_sim / graph_metrics were previously leaking into scalar_kwargs.
        scalar_kwargs = {
            k: v for k, v in current_params.items()
            if k not in process_seq_keys and k not in pipeline_only_keys
        }

        compute_mscomplex = current_params.get("compute_mscomplex", False)
        ms_scalar = (current_params.get("ms_scalar") or reeb_scalar).lower()
        needs_spectral_processing = (matrix_tranformation or (compute_reeb and needs_spectral(reeb_scalar, scalar_kwargs))
                                     or (compute_mscomplex and needs_spectral(ms_scalar, scalar_kwargs)))

        FM_out_path, RG_out_path = process_sequence(
            folder=folder,
            path=path,
            needs_spectral_processing=needs_spectral_processing,
            scalar_kwargs=scalar_kwargs,
            **seq_args
        )

        # Both graph analyses need Reeb graphs on disk; RG_out_path is '' if none were produced.
        has_reeb = bool(RG_out_path) and Path(RG_out_path).is_dir()

        if time_graph_analysis and has_reeb:
            csv_path = graph_time_analysis(RG_out_path, single_file=False)
            if csv_path:
                plot_dynamic_graph_analysis(csv_path, single_file=False)

        if graph_sim and has_reeb:
            csv_sim_path = graph_similarity(reeb_folder_path=RG_out_path, metrics_list=graph_metrics, single_file=False)
            if csv_sim_path:
                plot_graph_similarity(csv_sim_path, single_file=False)
        elif graph_sim and not has_reeb:
            print(f"\n[Warning] graph_sim skipped for {scene_name}: no Reeb graphs were computed.")