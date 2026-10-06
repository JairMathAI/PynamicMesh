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
    compute_approx_reeb_graph, compute_reeb_graph_controlled, annotate_reeb_graph,
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
# Continuous-description stage: explicit parametrizations (a), vertex trajectories & ARAP interpolation (c) and
# the descriptive analyses derived from them (d).  They run after the frame loop and read the meshes + the
# functional-map p2p maps from disk.  (No dynamical model / differential equation is fitted in this version.)
from PynamicMesh.core.dyn_common import load_frames as dyn_load_frames, resolve_frame_times, FRAME_TIMES_FILE
from PynamicMesh.core import accel
from PynamicMesh.core.parametrization import compute_parametrization as _compute_parametrization
from PynamicMesh.core.trajectories import compute_trajectories as _compute_trajectories
from PynamicMesh.core.graph_animation import compute_graph_animation as _compute_graph_animation
from PynamicMesh.core.dynamic_analysis import compute_dynamic_analysis as _compute_dynamic_analysis
from PynamicMesh.core.motion_analysis import compute_motion_analysis as _compute_motion_analysis, compare_scenes
from PynamicMesh.core.reeb_dynamics import compute_reeb_dynamics as _compute_reeb_dynamics
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots

# Scalar fields that need the Laplace–Beltrami eigendecomposition / stiffness matrix (mesh.process()).
SPECTRAL_METHODS = {"heat_diffusion", "harmonic", "multi_pca", "matern_kernel"}
# Scalar fields that need the point-to-point map of the functional-map step.
MAP_DEPENDENT_METHODS = {"normal_displacement"}
# Options of CustomFunctionalMapping; accepted nested in `fm_params` or flat at the top level of a config.
FM_PARAM_KEYS = ("symmetry_mode", "landmark_params", "descr_params", "fit_params",
                 "n_descr", "subsample_step", "refine", "dt", "verbose",
                 "zoomout_k_final", "zoomout_step", "zoomout_subsample")


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
            and fm_params["refine"] not in ("auto", "icp", "zoomout_auto"):
        fm_params["refine"] = _parse_int_tuple(fm_params["refine"])
    p["fm_params"] = fm_params

    for key in ("param_params", "traj_params", "dyn_analysis_params", "ms_tracker_params", "ms_protrusion_params",
                "time_params", "gif_params", "motion_params", "reeb_dynamics_params",
                "graph_animation_params"):
        if key in p and p[key] is None:
            p[key] = {}

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
    refine: 'auto' (ZoomOut 30 x 2 for pairs <= 10000 vertices and size ratio <= 1.5, else ICP), 'icp', (nit, step), None,
    or 'zoomout_auto' (ZoomOut up to zoomout_k_final for every pair, subsampled for large / mismatched pairs).
    """
    fm_params = fm_params or {}
    os.makedirs(target_folder / 'Transform_Matrices', exist_ok=True)
    os.makedirs(target_folder / 'Diagonal_analysis', exist_ok=True)
    os.makedirs(target_folder / 'Landmarks', exist_ok=True)

    # Refinement parameters first: ZoomOut needs k + nit*step eigenpairs and the basis must NOT be
    # recomputed after preprocess (a new eigendecomposition can flip eigenvector signs under the map).
    refine = fm_params.get("refine", "auto")
    k_fm = _fm_basis_size(k_eigenfunctions)
    zoom_sub = None
    if refine == "zoomout_auto":
        # ZoomOut up to a larger final basis (zoomout_k_final, 100 = the size used by Ovsjanikov et al.) for every pair;
        # large / size-mismatched pairs (where 'auto' falls back to ICP at the initial size) are refined on a subsample
        k_final = int(fm_params.get("zoomout_k_final", 100))
        step = int(fm_params.get("zoomout_step", 2))
        # the number of iterations is set after fit() from the ACTUAL size of the fitted map; here only the number of
        # eigenpairs (computed once, before preprocess, so the basis never changes afterwards)
        params_op = (max(1, (k_final - k_fm) // step), step)
        v1, v2 = len(meshn_1.vertices), len(meshn.vertices)
        sub = fm_params.get("zoomout_subsample", "auto")
        if sub == "auto":
            zoom_sub = 3000 if (max(v1, v2) > 10000 or max(v1, v2) / max(min(v1, v2), 1) > 1.5) else None
        else:
            zoom_sub = int(sub) if sub else None
    else:
        params_op = optimize_param(meshn_1, meshn) if refine == "auto" else (refine if refine not in (None, "icp") else None)
    k_needed = max(int(k_eigenvalues), k_fm + (params_op[0] * params_op[1] if params_op is not None else 0))
    if refine == "zoomout_auto":
        k_needed = max(int(k_eigenvalues), k_final)
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

    if refine == "zoomout_auto":
        k_init = int(np.asarray(model.FM).shape[1])
        k_avail = int(min(len(meshn_1.eigenvalues), len(meshn.eigenvalues)))
        nit = (min(k_final, k_avail) - k_init) // step
        params_op = (nit, step) if nit >= 1 else None
    if params_op is not None:
        if zoom_sub:
            FM_zo = model.zoomout_refine(nit=params_op[0], step=params_op[1], subsample=zoom_sub)
        else:
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


def _auto_references(method, kwargs, meshn, i, folder, prev_vertices=None, p2p=None):
    """Resolves the automatic reference points of a scalar field for frame i, consistently with frame i-1 (state in
    <folder>/auto_reference.json: frame -> parameter -> index, position, criterion)."""
    from PynamicMesh.core.reeb_graph import resolve_auto_references, REFERENCE_PARAMS, _is_auto
    if not any(_is_auto(kwargs.get(p_)) for p_ in REFERENCE_PARAMS.get(str(method).lower(), {})):
        return {k_: v_ for k_, v_ in kwargs.items() if k_ != "reference_transport"}
    import json as _json
    f = Path(folder) / 'auto_reference.json'
    store = {}
    if f.exists() and i > 0:
        try:
            store = _json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            store = {}
    prev = store.get(str(i - 1)) if i > 0 else None
    out, state = resolve_auto_references(method, kwargs, meshn.vertices, meshn.faces, prev_state=prev,
                                         prev_vertices=prev_vertices if prev is not None else None,
                                         p2p=p2p if prev is not None else None)
    store = store if i > 0 else {}
    store[str(i)] = state
    os.makedirs(folder, exist_ok=True)
    f.write_text(_json.dumps(store, indent=1))
    return out


def compute_RG(meshn, i, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections, needs_spectral_processing, prev_vertices=None, p2p_zo=None,
               topology_control=True):
    reeb_folder = target_folder / 'Reeb_Graphs'
    os.makedirs(reeb_folder, exist_ok=True)

    tn_kwargs = resolve_scalar_args(reeb_scalar, scalar_kwargs, i, meshn.vertices.shape[0], loaded_selections)
    # automatic reference points ('auto', 'auto_center', 'auto_extremity'): chosen on the mesh and kept on the same part
    # of the shape through time (transported from the previous frame through the point-to-point map)
    tn_kwargs = _auto_references(reeb_scalar, tn_kwargs, meshn, i, reeb_folder, prev_vertices, p2p_zo)
    # get_scalar_field now validates shape/finiteness and warns on constant fields;
    # extra options it understands: equalize_histogram, geodesic_solver, t='auto', signed.
    scalar_tn = get_scalar_field(
        meshn.vertices, meshn.faces, method=reeb_scalar,
        prev_vertices=prev_vertices, p2p=p2p_zo,
        trimesh_obj=meshn if needs_spectral_processing else None,
        **tn_kwargs
    )

    # the Reeb graph of the pipeline (uniform slabs, as always), annotated with the node types and the genus-loop
    # diagnostic (attributes only, the graph itself is unchanged)
    reeb_tn = annotate_reeb_graph(compute_approx_reeb_graph(meshn.vertices, meshn.faces, scalar_tn, num_bins=bins), meshn.faces)
    if topology_control:
        # complement: the topology-controlled graph (Extended Reeb Graph: slabs hiding handles refined locally) in its
        # own folder, with the scalar field, so it can be viewed / compared; it never replaces the graph above
        ctrl_folder = reeb_folder / 'Topology_Controlled'
        os.makedirs(ctrl_folder, exist_ok=True)
        reeb_ctrl = compute_reeb_graph_controlled(meshn.vertices, meshn.faces, scalar_tn, num_bins=bins)
        with open(ctrl_folder / f'Reeb_T{i:04d}.pkl', 'wb') as f:
            pickle.dump(reeb_ctrl, f)
        np.save(ctrl_folder / f'Scalar_T{i:04d}.npy', scalar_tn)

    # Zero-padded frame index so plain string ordering matches time ordering (Reeb_T0002 < Reeb_T0010).
    with open(reeb_folder / f'Reeb_T{i:04d}.pkl', 'wb') as f:
        pickle.dump(reeb_tn, f)
    np.save(reeb_folder / f'Scalar_T{i:04d}.npy', scalar_tn)

    return str(reeb_folder)


def compute_MS_frame(meshn, i, ms_scalar, reeb_scalar, compute_reeb, scalar_kwargs, target_folder, loaded_selections_rg,
                     needs_spectral_processing, ms_persistence, ms_min_region_area, mscomplex_graph, ms_graph_type,
                     prev_vertices=None, p2p_zo=None, protrusion_params=None, prev_ms=None, prev_faces=None):
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
    if scalar_field is None:                               # own field: its automatic references, consistent in time
        tn_kwargs = _auto_references(ms_scalar, tn_kwargs, meshn, i, target_folder / 'MSComplexAnalysis' / 'MS_Complex',
                                     prev_vertices, p2p_zo)
    return compute_MS(meshn.vertices, meshn.faces, i, target_folder, scalar_field=scalar_field,
                      scalar_method=ms_scalar, scalar_kwargs=tn_kwargs, persistence=ms_persistence,
                      min_region_area=ms_min_region_area, build_graph=mscomplex_graph, graph_type=ms_graph_type,
                      trimesh_obj=meshn if needs_spectral_processing else None, prev_vertices=prev_vertices, p2p=p2p_zo,
                      protrusion_params=protrusion_params, prev_ms=prev_ms, prev_faces=prev_faces,
                      use_disk_prev=False)                  # the previous complex comes from memory, never a stale file


def run_continuous_models(folder, target_folder, matrix_tranformation, compute_parametrization_flag=False,
                          param_params=None, compute_trajectories_flag=False, traj_params=None,
                          compute_dynamic_analysis_flag=False, dyn_analysis_params=None, dt=1.0, verbose=True,
                          compute_graph_animation_flag=False, graph_animation_params=None, times=None,
                          gif=False, gif_params=None, compute_motion_analysis_flag=False, motion_params=None,
                          compute_reeb_dynamics_flag=False, reeb_dynamics_params=None, reeb_scalar="geodesic",
                          scalar_kwargs=None, selections=None):
    """
    Post-loop stage (Results/<scene>/{Parametrization, Trajectories, DynamicAnalysis}).  Order matters:
    parametrization gives the SPHARM coefficients, trajectories give the vertex trajectories and the
    velocity.  Trajectories and the analyses derived from them need the p2p maps of the
    functional-map stage (matrix_tranformation=True, already computed or present on disk).
    """
    param_params = dict(param_params or {}); traj_params = dict(traj_params or {})
    dyn_analysis_params = dict(dyn_analysis_params or {})
    graph_animation_params = dict(graph_animation_params or {})
    scene_name = Path(folder).name
    has_maps = matrix_tranformation or (Path(target_folder) / 'Transform_Matrices').is_dir()
    frames = None
    X = F0 = None
    if compute_parametrization_flag or compute_trajectories_flag or compute_dynamic_analysis_flag or compute_graph_animation_flag:
        frames = dyn_load_frames(folder, loader=load_aligned_mesh)
        if len(frames) < 2:
            print(f"\n[Warning] Continuous models skipped for {scene_name}: fewer than two frames.")
            return
        if len(frames) < 3 and compute_dynamic_analysis_flag:
            print(f"\n[Warning] {scene_name}: only {len(frames)} frames in {folder} - the time analyses need at least 3 frames "
                  "(10 or more to be meaningful); parts of them will be skipped. Parametrization and the single trajectory "
                  )
    if compute_parametrization_flag:
        param_params.setdefault('align_with_p2p', has_maps)
        _compute_parametrization(folder, target_folder, loader=load_aligned_mesh, frames=frames, verbose=verbose, **param_params)
    if compute_trajectories_flag:
        if not has_maps:
            print(f"\n[Warning] Trajectories need the functional maps (matrix_tranformation=True); skipped for {scene_name}.")
        else:
            traj_params.setdefault('dt', dt)
            if times is not None:                          # known acquisition times of the frames
                traj_params.setdefault('times', times)
            res_c = _compute_trajectories(folder, target_folder, loader=load_aligned_mesh, frames=frames, verbose=verbose, **traj_params)
            X, F0 = res_c['X'], res_c['F0']
    if compute_graph_animation_flag:
        # a Reeb / Morse–Smale graph of one frame moved by the trajectory model (needs the trajectories)
        if not (Path(target_folder) / 'Trajectories' / 'trajectories.npy').exists():
            print(f"\n[Warning] Graph animation skipped for {scene_name}: no trajectories (compute_trajectories=True).")
        else:
            _compute_graph_animation(target_folder, frames=frames, verbose=verbose, **graph_animation_params)
    if compute_dynamic_analysis_flag:
        dyn_analysis_params.setdefault('dt', dt)
        if times is not None:
            dyn_analysis_params.setdefault('times', times)
        _compute_dynamic_analysis(target_folder, frames=frames, X=X, F0=F0, verbose=verbose, **dyn_analysis_params)
    if compute_reeb_dynamics_flag:
        # time-varying Reeb graph on the tracked mesh: branches, critical-point paths, birth / death / interchange events
        if not (Path(target_folder) / 'Trajectories' / 'trajectories.npy').exists():
            print(f"\n[Warning] Reeb dynamics skipped for {Path(target_folder).name}: no trajectories (compute_trajectories=True).")
        else:
            _compute_reeb_dynamics(target_folder, reeb_scalar=reeb_scalar, scalar_kwargs=scalar_kwargs, selections=selections,
                                   times=times, spectral=needs_spectral(reeb_scalar, scalar_kwargs or {}), verbose=verbose,
                                   **dict(reeb_dynamics_params or {}))
    if compute_motion_analysis_flag:
        # motion, deformation and events of the tracked surface (uses every result available on disk)
        if not (Path(target_folder) / 'Trajectories' / 'trajectories.npy').exists():
            print(f"\n[Warning] Motion analysis skipped for {Path(target_folder).name}: no trajectories (compute_trajectories=True).")
        else:
            _compute_motion_analysis(target_folder, frames=frames, times=times, verbose=verbose, **dict(motion_params or {}))
    if gif:                                                # GIF animations of the visual results of these stages
        for flag, stage in ((compute_parametrization_flag, 'parametrization'), (compute_trajectories_flag, 'trajectories'),
                            (compute_graph_animation_flag, 'graph_animation'), (compute_dynamic_analysis_flag, 'dynamic_analysis'),
                            (compute_motion_analysis_flag, 'motion_analysis'), (compute_reeb_dynamics_flag, 'reeb_dynamics')):
            if flag:
                _stage_gifs(stage, target_folder, folder, gif_params)


GRAPH_KINDS = ("reeb", "mscomplex", "reeb_controlled")
DEFAULT_GRAPHS = ("reeb", "mscomplex")


def graph_list(graphs, default=DEFAULT_GRAPHS):
    """Normalises a choice of graphs ('reeb', 'mscomplex', 'reeb_controlled'; a name, a list, 'all' or None = default)."""
    if graphs is None:
        return tuple(default)
    if isinstance(graphs, str):
        graphs = GRAPH_KINDS if graphs == "all" else (graphs,)
    out = tuple(g for g in graphs if g in GRAPH_KINDS)
    unknown = [g for g in graphs if g not in GRAPH_KINDS]
    if unknown:
        print(f"\n[Warning] unknown graph kinds {unknown} ignored (choose among {GRAPH_KINDS}).")
    return out


class _StageBar:
    """
    One progress bar (leave=False) for the stages that run after the frame loop of a scene: it counts the stages and
    shows the name of the one running, so the console always tells what the pipeline is doing; the stages show their
    own nested bars below it.
    """

    def __init__(self, desc, steps):
        show = steps and not os.environ.get("PYNAMIC_NO_BARS")       # off in a scene-parallel worker
        self.bar = tqdm(total=len(steps), desc=desc, leave=False, dynamic_ncols=True) if show else None
        self._open = False

    def step(self, name):
        if self.bar is None:
            return
        if self._open:
            self.bar.update(1)
        self.bar.set_postfix_str(name, refresh=True); self._open = True

    def close(self):
        if self.bar is not None:
            if self._open:
                self.bar.update(1)
            self.bar.close(); self.bar = None


def _stage_gifs(stage, target_folder, mesh_folder, gif_params=None):
    """GIF animations of one stage (utils.gif_export) into Results/<scene>/<Stage>/gif; never stops the pipeline."""
    try:
        from PynamicMesh.utils.gif_export import export_stage_gifs
        log = export_stage_gifs(stage, target_folder, mesh_folder, gif_params)
        for label, status, _ in log:
            if status != "ok":
                print(f"\n[Warning] GIF of {label} ({stage}): {status}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[Warning] GIFs of the stage '{stage}' could not be written: {exc}")


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
                     ms_tracker_params=None, ms_protrusion_params=None, compute_parametrization=False, param_params=None,
                     compute_trajectories=False, traj_params=None,
                     compute_dynamic_analysis=False, dyn_analysis_params=None,
                     compute_graph_animation=False, graph_animation_params=None,
                     compute_motion_analysis=False, motion_params=None,
                     reeb_topology_control=True, compute_reeb_dynamics=False, reeb_dynamics_params=None,
                     analysis_graphs=None,
                     time_params=None, gif=False, gif_params=None):
    """
    Acquisition times of the frames (time_params, all optional; nothing given = the previous behaviour, t = k * dt):
        frame_interval : time between two consecutive frames (uniform time-lapse)
        frame_times    : per-frame times (list / array) or a file (csv with a 'time' column, txt one value per line,
                         json list) - irregular intervals allowed; relative paths are looked up in the scene folder
        time_unit      : unit of the times ('s', 'min', ...)
        auto_detect    : use frame_times.csv / frame_times.txt / times.csv / times.txt found in the scene folder
      Known times are written to Results/<scene>/frame_times.csv and used by the physics (per-transition velocity /
      acceleration / rates), the trajectories (real times, irregular intervals), the dynamic analysis (FTLE, change
      points), the basic geometry (time axis, centre-of-mass speed, growth rates), the graph analyses (Time / Dt /
      rates), the Morse–Smale reports (rates, protrusion lifetimes), the parametrization tables and the viewers.
    gif / gif_params : GIF animations of the visual results of every stage (utils.gif_export.GIF_DEFAULTS) in
        Results/<scene>/<Stage>/gif, one GIF per condition of the corresponding viewer.

    Continuous-description options (post-loop stage, see run_continuous_models):
        compute_parametrization :  spherical harmonic map + SPHARM series (genus 0), manifold harmonics,
                                  ARAP planar parametrization (disk)        -> Results/<scene>/Parametrization
        param_params            : dict for parametrization.compute_parametrization (L_max, reg, mh_modes, ...)
        compute_trajectories    :  consistent vertex trajectories through the p2p maps, Hermite / spline /
                                  ARAP interpolation, kinematics, validation   -> Results/<scene>/Trajectories
        traj_params             : dict for trajectories.compute_trajectories (schemes, substeps, validate, ...)
        compute_dynamic_analysis:  Hodge decomposition, FTLE, shape space, reduced coordinates, change points,
                                  persistent homology                          -> Results/<scene>/DynamicAnalysis
        dyn_analysis_params     : dict for dynamic_analysis.compute_dynamic_analysis
        compute_graph_animation :  a Reeb / Morse–Smale graph of one frame moved by the trajectory model (skeleton
                                  animation, edge strain, comparison with the graph of every frame)
                                                                               -> Results/<scene>/GraphAnimation
        graph_animation_params  : dict for graph_animation.compute_graph_animation (graphs, frame, model, export_graphs)

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
        ms_protrusion_params : dict for the convex-hull protrusion fusion of the MS complex (SMComplex.PROTRUSION_DEFAULTS:
                             enabled, beta, alpha, noise_factor, smooth_iters, min_aspect, temporal, ...); None = defaults
    """
    scalar_kwargs = scalar_kwargs or {}
    fm_params = fm_params or {}
    ms_scalar = (ms_scalar or reeb_scalar).lower()
    ms_tracker_params = ms_tracker_params or {}
    ms_protrusion_params = ms_protrusion_params or {}
    itemsfiles = list(folder.iterdir())
    obj_files = sorted([f for f in itemsfiles if f.is_file() and f.suffix in ('.obj', '.mat')], key=natural_sort_key)

    if len(obj_files) < 2:
        return None, None

    out_root = path.parent / 'Results'
    scene_name = obj_files[0].parent.name
    target_folder = out_root / scene_name
    os.makedirs(target_folder, exist_ok=True)

    # acquisition times of the frames: written only when known (every stage then reads Results/<scene>/frame_times.csv);
    # a file left by an earlier run with other settings is removed so unknown times always mean the old behaviour
    time_params = dict(time_params or {})
    frame_times = resolve_frame_times(len(obj_files), frame_interval=time_params.get('frame_interval'),
                                      frame_times=time_params.get('frame_times'), time_unit=time_params.get('time_unit', 's'),
                                      scene_folder=folder, auto_detect=time_params.get('auto_detect', True),
                                      names=[f.name for f in obj_files], legacy_dt=fm_params.get('dt', 1.0))
    if frame_times.known:
        frame_times.save(target_folder, [f.name for f in obj_files])
        print(f"\n[Frame times] {scene_name}: {frame_times.source}; {len(frame_times)} frames over "
              f"{frame_times.t[-1]:.4g} {frame_times.unit}" + ("" if frame_times.uniform else " (irregular intervals)"))
    elif (target_folder / FRAME_TIMES_FILE).exists():
        (target_folder / FRAME_TIMES_FILE).unlink()
    times_arg = frame_times if frame_times.known else None

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

    # meshes loaded + Laplace–Beltrami spectra computed AHEAD by worker processes (independent per frame), consumed in
    # order by the loop below (accel.prefetch: bounded memory; sequential with n_workers = 1)
    mesh_stream = accel.prefetch(accel.load_and_process_mesh,
                                 [(str(f), k_eigenvalues, needs_spectral_processing) for f in obj_files])
    try:
        meshn_1 = next(mesh_stream)
    except BaseException:
        mesh_stream.close(); raise

    # Frame 0: no previous frame, so map-dependent fields would be identically zero (a one-node
    # graph that only pollutes the time series). Start those at frame 1 instead.
    if compute_reeb and reeb_scalar not in MAP_DEPENDENT_METHODS:
        compute_RG(meshn_1, 0, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections_rg, needs_spectral_processing,
                   topology_control=reeb_topology_control)

    ms_prev = None
    ms_tracker = RegionTracker(target_folder, **ms_tracker_params) if (compute_mscomplex and ms_track_regions) else None
    if compute_mscomplex and ms_scalar not in MAP_DEPENDENT_METHODS:
        ms_prev = compute_MS_frame(meshn_1, 0, ms_scalar, reeb_scalar, compute_reeb, scalar_kwargs, target_folder,
                                   loaded_selections_rg, needs_spectral_processing, ms_persistence, ms_min_region_area,
                                   mscomplex_graph, ms_graph_type, protrusion_params=ms_protrusion_params)

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

    try:
        for i in tqdm(range(1, len(obj_files)), desc=f'processing {scene_name}', leave=False):

            meshn = next(mesh_stream)                          # loaded + processed (spectrum) by the workers
            p2p_zo = None

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
                RG_out_path = compute_RG(meshn, i, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections_rg, needs_spectral_processing, prev_verts, p2p_zo,
                                         topology_control=reeb_topology_control)

            if compute_mscomplex:
                prev_verts = meshn_1.vertices if matrix_tranformation else None
                ms_curr = compute_MS_frame(meshn, i, ms_scalar, reeb_scalar, compute_reeb, scalar_kwargs, target_folder,
                                           loaded_selections_rg, needs_spectral_processing, ms_persistence, ms_min_region_area,
                                           mscomplex_graph, ms_graph_type, prev_verts, p2p_zo,
                                           protrusion_params=ms_protrusion_params, prev_ms=ms_prev,
                                           prev_faces=meshn_1.faces if matrix_tranformation else None)
                if ms_tracker is not None and ms_prev is not None and p2p_zo is not None:
                    ms_tracker.track_pair(ms_prev, ms_curr, meshn_1.vertices, meshn_1.faces, meshn.vertices, meshn.faces,
                                          p2p_zo, FM_12=FM_zo, mesh_prev=meshn_1, mesh_curr=meshn)
                ms_prev = ms_curr

            if compute_basicGeo:
                computed_metrics.append(compute_mesh_geometry(meshn, metrics=metrics))

            meshn_1 = meshn
    finally:
        mesh_stream.close()                             # the worker pool never outlives the frame loop

    # post-processing stages of the scene, reported by one bar (each stage shows its own nested bars)
    continuous = (compute_parametrization or compute_trajectories or compute_dynamic_analysis or compute_graph_animation
                  or compute_motion_analysis or compute_reeb_dynamics)
    stage_names = ([("basic geometry")] if compute_basicGeo and computed_metrics else []) \
        + (["GIFs basic geometry"] if gif and compute_basicGeo and computed_metrics else []) \
        + (["FM heatmap GIF"] if matrix_tranformation and heatmap_paths else []) \
        + (["Morse-Smale reports"] if compute_mscomplex else []) + (["GIFs Morse-Smale"] if gif and compute_mscomplex else []) \
        + (["physical fields"] if compute_physic_fields and matrix_tranformation else []) \
        + (["GIFs physical fields"] if gif and compute_physic_fields and matrix_tranformation else []) \
        + (["continuous models" + (" + GIFs" if gif else "")] if continuous else [])
    stages = _StageBar(f"{scene_name}: post-processing", stage_names)

    if compute_basicGeo and computed_metrics:
        stages.step("basic geometry")
        df_results = pd.DataFrame(computed_metrics)
        if 'area' in df_results and len(df_results) and df_results['area'].iloc[0] > 0:
            # size-normalised descriptors (comparable between objects and experiments)
            df_results['size'] = np.sqrt(df_results['area'])
            df_results['area_rel'] = df_results['area'] / df_results['area'].iloc[0]
            if 'volume' in df_results and df_results['volume'].notna().all() and df_results['volume'].iloc[0] > 0:
                df_results['volume_rel'] = df_results['volume'] / df_results['volume'].iloc[0]
            if {'cm_x', 'cm_y', 'cm_z'} <= set(df_results.columns):
                cm = df_results[['cm_x', 'cm_y', 'cm_z']].to_numpy(float)
                df_results['cm_displacement_rel'] = np.linalg.norm(cm - cm[0], axis=1) / df_results['size'].iloc[0]
        if times_arg is not None and len(df_results) == len(frame_times):   # known acquisition times
            df_results.insert(0, 'time_unit', frame_times.unit)
            df_results.insert(0, 'time', frame_times.t)
            df_results.insert(0, 'frame', np.arange(len(df_results)))
        csv_path = geom_path / 'features_computed.csv'
        df_results.to_csv(csv_path, index=False)
        if plot_basicGeo:
            generate_plots_from_csv(csv_path)
        if gif:
            stages.step("GIFs basic geometry")
            _stage_gifs('basic_geometry', target_folder, folder, gif_params)

    if matrix_tranformation and heatmap_paths:
        stages.step("FM heatmap GIF")
        headmap_gif(heatmap_paths, target_folder)

    if compute_mscomplex:
        stages.step("Morse-Smale reports")
        critical_points_report(target_folder, single_file=False)
        if ms_tracker is not None and ms_tracker.summary:
            ms_tracker.save()
            tracking_report(target_folder, single_file=False)
        if mscomplex_graph and 'mscomplex' in graph_list(analysis_graphs):
            ms_graph_analysis(target_folder, graph_metrics=ms_graph_metrics, single_file=False)
        if gif:
            stages.step("GIFs Morse-Smale")
            _stage_gifs('ms_complex', target_folder, folder, gif_params)

    if compute_physic_fields:
        if not matrix_tranformation:
            print(f"\n[Warning] Cannot calculate fields for {scene_name} without track mappings.")
        else:
            stages.step("physical fields")
            matrix_folder_path = target_folder / 'Transform_Matrices'
            physical_fields_path = target_folder / 'Physical_fields'
            computing_fields(folder, matrix_folder_path, physical_fields_path, single_file=False,
                             dt=fm_params.get('dt', 1.0), loader=load_aligned_mesh, times=times_arg)
            if gif:
                stages.step("GIFs physical fields")
                _stage_gifs('physics', target_folder, folder, gif_params)

    if continuous:
        stages.step("continuous models" + (" + GIFs" if gif else ""))
        run_continuous_models(
            folder, target_folder, matrix_tranformation,
            compute_parametrization_flag=compute_parametrization, param_params=param_params,
            compute_trajectories_flag=compute_trajectories, traj_params=traj_params,
            compute_dynamic_analysis_flag=compute_dynamic_analysis, dyn_analysis_params=dyn_analysis_params,
            dt=frame_times.mean_dt if times_arg is not None else fm_params.get('dt', 1.0), verbose=False,
            compute_graph_animation_flag=compute_graph_animation, graph_animation_params=graph_animation_params,
            times=times_arg, gif=gif, gif_params=gif_params,
            compute_motion_analysis_flag=compute_motion_analysis, motion_params=motion_params,
            compute_reeb_dynamics_flag=compute_reeb_dynamics, reeb_dynamics_params=reeb_dynamics_params,
            reeb_scalar=reeb_scalar, scalar_kwargs=scalar_kwargs, selections=loaded_selections_rg,
        )
    stages.close()

    return FM_out_path, RG_out_path


PROCESS_SEQ_KEYS = [
    'plot_basicGeo', 'compute_basicGeo', 'metrics', 'matrix_tranformation', 'diagonal_analysis', 'isometric_analysis',
    'k_eigenfunctions', 'k_eigenvalues', 'descriptor', 'landmarks',
    'compute_reeb', 'reeb_scalar', 'bins', 'compute_physic_fields', 'fm_params',
    'compute_mscomplex', 'ms_scalar', 'ms_persistence', 'ms_min_region_area', 'mscomplex_graph',
    'ms_graph_type', 'ms_track_regions', 'ms_graph_metrics', 'ms_tracker_params', 'ms_protrusion_params',
    'compute_parametrization', 'param_params', 'compute_trajectories', 'traj_params',
    'compute_dynamic_analysis', 'dyn_analysis_params', 'compute_graph_animation', 'graph_animation_params',
    'time_params', 'gif', 'gif_params', 'compute_motion_analysis', 'motion_params',
    'reeb_topology_control', 'compute_reeb_dynamics', 'reeb_dynamics_params', 'analysis_graphs'
]
PIPELINE_ONLY_KEYS = {'time_graph_analysis', 'graph_sim', 'graph_metrics'}


def _run_scene(folder, path, current_params):
    """Everything the pipeline does for ONE scene folder (sequence processing + graph analyses + GIFs)."""
    folder = Path(folder); path = Path(path)
    scene_name = folder.name
    process_seq_keys, pipeline_only_keys = PROCESS_SEQ_KEYS, PIPELINE_ONLY_KEYS
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

    # Graph analyses (time analysis + similarity) of the graphs chosen in `analysis_graphs` ('reeb', 'reeb_controlled';
    # 'mscomplex' is analysed with the Morse–Smale reports); RG_out_path is '' if no Reeb graphs were produced.
    has_reeb = bool(RG_out_path) and Path(RG_out_path).is_dir()
    do_gif = current_params.get('gif', False)
    analysis_graphs = graph_list(current_params.get('analysis_graphs'))
    sources = []
    if has_reeb and 'reeb' in analysis_graphs:
        sources.append(("Reeb", Path(RG_out_path), None, None))
    ctrl = Path(RG_out_path) / 'Topology_Controlled' if has_reeb else None
    if ctrl is not None and 'reeb_controlled' in analysis_graphs and ctrl.is_dir() and any(ctrl.glob('Reeb_T*.pkl')):
        sources.append(("Reeb (topology controlled)", ctrl, Path(RG_out_path).parent / 'Graph_analysis' / 'Topology_Controlled',
                        'ReebControlled'))
    steps = []
    for label, *_ in sources:
        steps += ([f"{label} graph time analysis"] if time_graph_analysis else []) + ([f"{label} graph similarity"] if graph_sim else [])
    graph_stages = _StageBar(f"{scene_name}: graph analyses", steps + (["GIF Reeb graphs"] if do_gif and has_reeb else []))

    for label, gfolder, out_folder, tag in sources:
        if time_graph_analysis:
            graph_stages.step(f"{label} graph time analysis")
            csv_path = graph_time_analysis(gfolder, single_file=False, analysis_folder=out_folder)
            if csv_path:
                plot_dynamic_graph_analysis(csv_path, single_file=False)
        if graph_sim:
            graph_stages.step(f"{label} graph similarity")
            csv_sim_path = graph_similarity(reeb_folder_path=gfolder, metrics_list=graph_metrics, single_file=False,
                                            analysis_folder=out_folder, tag=tag)
            if csv_sim_path:
                plot_graph_similarity(csv_sim_path, single_file=False)
    if graph_sim and not has_reeb:
        print(f"\n[Warning] graph_sim skipped for {scene_name}: no Reeb graphs were computed.")

    if do_gif and has_reeb:                                   # GIFs of the Reeb graph sequences (+ controlled graphs)
        graph_stages.step("GIF Reeb graphs")
        _stage_gifs('reeb', Path(RG_out_path).parent, folder, current_params.get('gif_params'))
    graph_stages.close()


def _scene_task(args):
    """
    Scene-parallel mode: executed in a worker process pinned to one GPU (accel.init_scene_worker). The console output
    of the scene goes to Results/<scene>/pipeline_log.txt. Returns (scene, status, seconds, log path).
    """
    import contextlib, time, traceback
    folder, path, params = args
    folder = Path(folder); path = Path(path)
    log_dir = path.parent / 'Results' / folder.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'pipeline_log.txt'
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        try:
            accel.configure(verbose=True)              # the GPU / workers chosen by init_scene_worker
            _run_scene(folder, path, params)
            status = 'ok'
        except Exception as exc:  # noqa: BLE001 - one failing scene must not stop the others
            traceback.print_exc()
            status = f'failed: {type(exc).__name__}: {exc}'
    return folder.name, status, time.time() - t0, str(log_path)


def run_pipeline(path_str, is_batch=False, batch_kwargs=None, compute_devices=None, n_workers=None,
                 parallel_scenes='auto', compare_scenes_flag='auto', **kwargs):
    """
    Runs the pipeline on every scene folder of path_str.
    compute_devices : GPUs ('all' | 'cpu' | list / '0,1'); n_workers : CPU worker processes (accel.configure)
    parallel_scenes : 'auto' (default) | True | False. Several scenes are processed AT THE SAME TIME, each in its own
        process pinned to ONE GPU (the work inside a scene is a chain of dependent steps, so the way to use several
        GPUs is to give each scene its own). 'auto' = on when more than one GPU is selected and more than one scene
        is processed; True also works on CPU (scenes in parallel processes). The CPU workers are shared among the
        scenes; every scene writes its console output to Results/<scene>/pipeline_log.txt.
    """
    path = Path(path_str)
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{path_str}' is not a valid directory.")
        return

    # compute devices: every visible GPU by default ('cpu' / [0, 2] / '0,1' to choose) and the worker processes of
    # the per-frame work (accel.py); the choice is printed once
    devs, workers = accel.configure(compute_devices, n_workers, verbose=True)
    subdirectories = sorted([f for f in path.iterdir() if f.is_dir()], key=natural_sort_key)
    if not subdirectories:
        print(f'No folders found on {path}')
        return

    scenes = []
    for folder in subdirectories:
        if is_batch:
            if not batch_kwargs or folder.name not in batch_kwargs:
                print(f"\n[Warning] Skipping '{folder.name}': No configuration found in batch config.")
                continue
            scenes.append((folder, batch_kwargs[folder.name]))
        else:
            scenes.append((folder, kwargs.copy()))
    if not scenes:
        return

    if parallel_scenes == 'auto':
        parallel = len(devs) > 1 and len(scenes) > 1
    else:
        parallel = bool(parallel_scenes) and len(scenes) > 1
    if not parallel:
        for folder, params in tqdm(scenes, desc='processing folder'):
            _run_scene(folder, path, params)
        _compare(path, [f.name for f, _ in scenes], compare_scenes_flag)
        return

    # scene-parallel mode: one process per GPU (or per worker slot on CPU), scenes distributed as they finish
    n_slots = min(len(scenes), len(devs) if devs else max(1, workers))
    import os as _os
    total_workers = n_workers if n_workers is not None else max(1, (_os.cpu_count() or 2) - 1)
    per_scene = max(1, int(total_workers) // n_slots)
    print(f"[Compute] scene-parallel mode: {len(scenes)} scenes on {n_slots} "
          f"{'GPUs' if devs else 'processes'} ({per_scene} CPU worker(s) per scene); "
          f"logs in Results/<scene>/pipeline_log.txt")
    from concurrent.futures import as_completed
    results = []
    with accel._no_main_reimport():
        ex = accel.scene_pool(n_slots, per_scene)
        try:
            futs = {ex.submit(_scene_task, (str(f), str(path), p)): f.name for f, p in scenes}
            running = set(futs.values())
            bar = tqdm(total=len(futs), desc='processing folder')
            bar.set_postfix_str(f"running: {', '.join(sorted(running))[:60]}")
            for fut in as_completed(futs):
                name, status, secs, log = fut.result()
                running.discard(name); results.append((name, status, secs, log))
                bar.update(1)
                bar.set_postfix_str(f"running: {', '.join(sorted(running))[:60]}" if running else "done")
                if status != 'ok':
                    tqdm.write(f"[Warning] scene '{name}' {status} (see {log})")
            bar.close()
        finally:
            accel._shutdown_pool(ex)
    for name, status, secs, log in sorted(results):
        print(f"  {name}: {status} in {secs:.0f} s  ->  {log}")
    _compare(path, [n for n, st, _, _ in results if st == 'ok'], compare_scenes_flag)


def _compare(path, names, flag):
    """Comparison of the scenes of the run (Results/_comparison/): 'auto' = when at least two scenes were processed."""
    if flag is False or len(names) < 2:
        return
    try:
        compare_scenes(Path(path).parent / 'Results', names)
    except Exception as exc:  # noqa: BLE001 - never stops the pipeline
        print(f"\n[Warning] scene comparison skipped: {exc}")
