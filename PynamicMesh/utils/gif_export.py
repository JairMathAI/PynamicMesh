"""
gif_export.py  —  GIF animations of the visual results of every pipeline stage
===============================================================================

With run_pipeline(gif=True) every stage that has a viewer writes Results/<scene>/<Stage>/gif/ with ONE GIF PER
CONDITION the viewer can show (modality, page, scheme, graph ...), rendered off-screen with the viewers' own drawing
code, so the animation shows exactly what the corresponding visualizer shows:

    Basic_Geometry/gif        mesh_sequence.gif                               (visualize_obj_sequence)
    Physical_fields/gif       physics_page<p>.gif per page of quantities      (visualize_physics)
    Reeb_Graphs/gif           reeb_graph.gif                                  (visualize_graphs, 'reeb')
    MSComplexAnalysis/gif     ms_<modality>.gif x 4, ms_graph.gif             (visualize_ms_complex, visualize_graphs)
    Parametrization/gif       parametrization_L<degree>.gif                   (visualize_parametrization)
    Trajectories/gif          trajectories_<modality>[_vectors].gif, dynamic_trajectories_<modality>[_vectors].gif,
                              interp_<scheme>.gif                             (the three trajectory viewers)
    DynamicAnalysis/gif       hodge_<component>[_vectors].gif, ftle.gif, shape_modes.gif
    GraphAnimation/gif        <graph>_T<frame>_<model>_<modality>.gif         (visualize_graph_animation)

Timing: with known frame times (Results/<scene>/frame_times.csv) the frame durations follow the real intervals of
the time-lapse (irregular acquisitions look irregular), scaled so the whole GIF lasts n_frames / fps seconds;
without times every frame lasts 1 / fps.
"""
import contextlib
import re
import traceback
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

GIF_DEFAULTS = {
    "fps": 6,              # frames per second (mean rate when real_time)
    "real_time": True,     # frame durations proportional to the acquisition intervals when the times are known
    "max_frames": 240,     # longer sequences are subsampled evenly
    "panel_width": 900,    # pixels per panel (multi-panel viewers are wider)
    "height": 760,
    "max_width": 1800,     # the written GIF is downscaled to this width
    "vectors": True,       # also the vector-field view of the modalities that have one ('_vectors' GIFs)
    "loop": 0,             # 0 = loop forever
}


def slug(text):
    """File-name friendly version of a modality / condition name."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return s[:60] or "view"


def frame_durations(times, n, fps=6, real_time=True):
    """Per-frame durations (ms): uniform 1/fps, or proportional to the intervals of `times` (same total length)."""
    base = 1000.0 / max(float(fps), 0.1)
    if not real_time or times is None or len(times) != n or n < 2:
        return [int(round(base))] * n
    t = np.asarray(times, dtype=np.float64)
    d = np.diff(t)
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        return [int(round(base))] * n
    d = np.r_[d, np.median(d)]                        # the last frame lasts a typical interval
    d = d / d.sum() * base * n
    return [int(max(20, round(x))) for x in d]       # GIF players clamp durations below 20 ms


def write_gif(images, path, durations=None, max_width=1800, loop=0):
    """Writes an animated GIF from RGB(A) arrays (Pillow); returns the path or None."""
    from PIL import Image
    if not images:
        return None
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for img in images:
        im = Image.fromarray(np.asarray(img)[..., :3].astype(np.uint8))
        if im.width > max_width:
            im = im.resize((max_width, int(im.height * max_width / im.width)), Image.LANCZOS)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=255))
    durations = durations or [166] * len(frames)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=list(durations), loop=loop, disposal=2,
                   optimize=False)
    return path


def frame_times_of(results_folder, n):
    """Times of the n observed frames from Results/<scene>/frame_times.csv (None when unknown)."""
    from PynamicMesh.core.dyn_common import find_frame_times
    ft = find_frame_times(results_folder)
    return ft.t if ft is not None and len(ft) == n else None


# ----------------------------------------------------------------------------------------------------------------- #
#  Viewers of utils.visualizers (key-driven closures): drive their own key callbacks off-screen
# ----------------------------------------------------------------------------------------------------------------- #
def _bounds_union(bs):
    bs = [b for b in bs if b is not None and np.all(np.isfinite(b)) and b[1] >= b[0]]
    if not bs:
        return None
    b = np.array(bs)
    return [b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max(), b[:, 4].min(), b[:, 5].max()]


@contextlib.contextmanager
def capture_key_viewer(conditions, n_frames, out_dir, params=None, times=None, step_key="Right", back_key="Left"):
    """
    While active, pyvista's Plotter.show is replaced by a recorder: for every condition (file stem, keys pressed
    before recording) the viewer's own key callbacks step through the n_frames frames (step_key) and every frame is
    captured off-screen; then the viewer is rewound (back_key). Yields the list of written GIF paths.
    """
    import pyvista as pv
    p_ = {**GIF_DEFAULTS, **(params or {})}
    written = []
    orig_show, orig_off = pv.Plotter.show, pv.OFF_SCREEN

    def recorder(plotter, *a, **k):
        cbs = plotter.iren._key_press_event_callbacks

        def press(key, n=1):
            for _ in range(int(n)):
                for f in list(cbs.get(key, [])):
                    f()
        nrow, ncol = plotter.shape
        plotter.window_size = (min(p_["panel_width"] * ncol, 3 * p_["panel_width"]), p_["height"] * max(1, nrow))
        idx = np.unique(np.linspace(0, n_frames - 1, min(n_frames, p_["max_frames"])).round().astype(int))
        for stem, keys in tqdm(conditions, desc="GIF conditions", leave=False):
            for key in keys:
                press(key)
            # camera: fit the union of the bounds of a few frames of this condition in every panel
            samples = np.unique(np.linspace(0, n_frames - 1, min(n_frames, 6)).round().astype(int))
            bnds = {(i, j): [] for i in range(nrow) for j in range(ncol)}
            pos = 0
            for s in samples:
                press(step_key, s - pos); pos = s
                for (i, j) in bnds:
                    plotter.subplot(i, j); bnds[(i, j)].append(plotter.renderer.ComputeVisiblePropBounds())
            press(back_key, pos); pos = 0
            for (i, j), bs in bnds.items():
                ub = _bounds_union(bs)
                plotter.subplot(i, j)
                if ub is not None:
                    plotter.reset_camera(bounds=ub)
            plotter.subplot(0, 0)
            imgs = []
            for f in tqdm(idx, desc=f"GIF {stem}", leave=False):
                press(step_key, f - pos); pos = f
                plotter.render()
                imgs.append(plotter.screenshot(return_img=True))
            press(back_key, pos)
            tsel = None if times is None else np.asarray(times)[idx]
            out = write_gif(imgs, Path(out_dir) / f"{stem}.gif", frame_durations(tsel, len(idx), p_["fps"], p_["real_time"]),
                            p_["max_width"], p_["loop"])
            if out:
                written.append(out)
        plotter.close()

    pv.Plotter.show = recorder; pv.OFF_SCREEN = True
    try:
        yield written
    finally:
        pv.Plotter.show = orig_show; pv.OFF_SCREEN = orig_off


# ----------------------------------------------------------------------------------------------------------------- #
#  Stage exporters (called by the pipeline when gif=True)
# ----------------------------------------------------------------------------------------------------------------- #
def _n_meshes(mesh_folder):
    return len([f for f in Path(mesh_folder).iterdir() if f.is_file() and f.suffix in (".obj", ".mat")])


def _quiet_viewers():
    """No instruction windows / console messages while recording."""
    import PynamicMesh.utils.visualizers as VZ
    import PynamicMesh.utils.dynamics_visualizers as DV
    saved = (VZ.viewer_ui, DV.viewer_ui)
    VZ.viewer_ui = lambda *a, **k: None; DV.viewer_ui = lambda *a, **k: None
    return saved


def _restore_viewers(saved):
    import PynamicMesh.utils.visualizers as VZ
    import PynamicMesh.utils.dynamics_visualizers as DV
    VZ.viewer_ui, DV.viewer_ui = saved


def _safe(label, fn, log):
    import os
    try:
        with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
            res = fn()
        log.append((label, "ok", res))
    except Exception as exc:  # noqa: BLE001 - a GIF must never break the pipeline
        log.append((label, f"failed: {exc}", traceback.format_exc(limit=2)))


def export_stage_gifs(stage, target_folder, mesh_folder, params=None):
    """
    GIFs of one stage: 'basic_geometry' | 'physics' | 'reeb' | 'ms_complex' | 'parametrization' | 'trajectories' |
    'dynamic_analysis' | 'graph_animation'. Returns a list of (label, status, detail).
    """
    import pyvista as pv
    import PynamicMesh.utils.visualizers as VZ
    import PynamicMesh.utils.dynamics_visualizers as DV
    p_ = {**GIF_DEFAULTS, **(params or {})}
    target_folder = Path(target_folder); mesh_folder = Path(mesh_folder)
    n = _n_meshes(mesh_folder)
    t_obs = frame_times_of(target_folder, n)
    log = []
    saved = _quiet_viewers()
    try:
        if stage == "basic_geometry":
            out = target_folder / "Basic_Geometry" / "gif"

            def run():
                with capture_key_viewer([("mesh_sequence", [])], n, out, p_, t_obs) as w:
                    VZ.visualize_obj_sequence(str(mesh_folder))
                return w
            _safe("mesh sequence", run, log)
        elif stage == "physics":
            out = target_folder / "Physical_fields" / "gif"
            npz = sorted((target_folder / "Physical_fields").glob("frame_*.npz"))
            if npz:
                n_pages = _physics_pages(npz[-1])

                def run():
                    conds = [(f"physics_page{p + 1}", [] if p == 0 else ["p"]) for p in range(n_pages)]
                    with capture_key_viewer(conds, len(npz), out, p_, frame_times_of(target_folder, len(npz))) as w:
                        VZ.visualize_physics(str(mesh_folder), str(target_folder / "Transform_Matrices"), on_time=False)
                    return w
                _safe("physics fields", run, log)
        elif stage == "reeb":
            out = target_folder / "Reeb_Graphs" / "gif"
            if (target_folder / "Reeb_Graphs").is_dir():
                def run():
                    with capture_key_viewer([("reeb_graph", [])], n, out, p_, t_obs) as w:
                        VZ.visualize_graphs(str(mesh_folder), str(target_folder / "Reeb_Graphs"), graph="reeb")
                    return w
                _safe("Reeb graphs", run, log)
            ctrl = target_folder / "Reeb_Graphs" / "Topology_Controlled"
            if ctrl.is_dir() and any(ctrl.glob("Reeb_T*.pkl")):            # complementary controlled graphs
                def run_ctrl():
                    with capture_key_viewer([("reeb_graph_topology_controlled", [])], n, out, p_, t_obs) as w:
                        VZ.visualize_graphs(str(mesh_folder), str(ctrl), graph="reeb")
                    return w
                _safe("Reeb graphs (topology controlled)", run_ctrl, log)
        elif stage == "ms_complex":
            out = target_folder / "MSComplexAnalysis" / "gif"
            ms_root = target_folder / "MSComplexAnalysis"
            if (ms_root / "MS_Complex").is_dir():
                n_ms = len(list((ms_root / "MS_Complex").glob("*.pkl"))) or n
                modes = ["segmentation", "correspondence", "fates", "protrusions"]

                def run():
                    conds = [(f"ms_{m}", [] if k == 0 else ["m"]) for k, m in enumerate(modes)]
                    with capture_key_viewer(conds, n_ms, out, p_, frame_times_of(target_folder, n_ms)) as w:
                        VZ.visualize_ms_complex(str(mesh_folder), str(ms_root))
                    return w
                _safe("Morse-Smale complex", run, log)
            if (ms_root / "MS_Graphs").is_dir() and any((ms_root / "MS_Graphs").glob("*.pkl")):
                def run_g():
                    n_g = len(list((ms_root / "MS_Graphs").glob("*.pkl")))
                    with capture_key_viewer([("ms_graph", [])], n_g, out, p_, frame_times_of(target_folder, n_g)) as w:
                        VZ.visualize_graphs(str(mesh_folder), str(ms_root / "MS_Graphs"), graph="mscomplex")
                    return w
                _safe("Morse-Smale graphs", run_g, log)
        else:                                          # viewers of dynamics_visualizers (_FrameViewer)
            stage_dir = {"parametrization": "Parametrization", "trajectories": "Trajectories",
                         "dynamic_analysis": "DynamicAnalysis", "graph_animation": "GraphAnimation",
                         "motion_analysis": "MotionAnalysis", "reeb_dynamics": "ReebDynamics"}[stage]
            out = target_folder / stage_dir / "gif"
            calls = []
            if stage == "parametrization":
                calls = [("parametrization", lambda: DV.visualize_parametrization(str(mesh_folder), str(target_folder)))]
            elif stage == "trajectories":
                calls = [("trajectories", lambda: DV.visualize_trajectories(str(target_folder), mesh_path=str(mesh_folder))),
                         ("dynamic_trajectories", lambda: DV.visualize_dynamic_trajectories(str(target_folder), mesh_path=str(mesh_folder)))]
                interp = target_folder / "Trajectories" / "interpolation"
                for f in sorted(interp.glob("*.npy")) if interp.is_dir() else []:
                    if f.stem != "query_times":
                        calls.append((f"interp_{f.stem}", lambda s_=f.stem: DV.visualize_interpolation(str(target_folder), s_, mesh_path=str(mesh_folder))))
            elif stage == "dynamic_analysis":
                da = target_folder / "DynamicAnalysis"
                if (da / "Hodge").is_dir():
                    calls.append(("hodge", lambda: DV.visualize_hodge(str(target_folder), mesh_path=str(mesh_folder))))
                if (da / "FTLE").is_dir():
                    calls.append(("ftle", lambda: DV.visualize_ftle(str(target_folder), mesh_path=str(mesh_folder))))
                if (da / "ShapeSpace").is_dir():
                    calls.append(("shape_modes", lambda: DV.visualize_shape_modes(str(target_folder))))
            elif stage == "motion_analysis":
                if (target_folder / "MotionAnalysis" / "RigidMotion").is_dir():
                    calls = [("motion_analysis", lambda: DV.visualize_motion_analysis(str(target_folder), mesh_path=str(mesh_folder)))]
                if (target_folder / "Physical_fields" / "from_trajectories" / "comparison").is_dir():
                    # frame-to-frame vs tracked-mesh physical fields: one GIF per field
                    calls.append(("field_comparison", lambda: DV.visualize_field_comparison(str(target_folder), mesh_path=str(mesh_folder))))
            elif stage == "reeb_dynamics":
                if (target_folder / "ReebDynamics" / "paths.csv").exists():
                    calls = [("reeb_dynamics", lambda: DV.visualize_reeb_dynamics(str(target_folder), mesh_path=str(mesh_folder)))]
            elif stage == "graph_animation":
                calls = [("graph_animation", lambda: DV.visualize_graph_animation(str(target_folder), mesh_path=str(mesh_folder)))]
            for prefix, fn in tqdm(calls, desc=f"GIFs {stage}", leave=False):
                def run(prefix=prefix, fn=fn):
                    with DV.record_gifs(out, prefix="" if stage == "graph_animation" else prefix, **p_) as req:
                        fn()
                    return req["written"]
                _safe(prefix, run, log)
    finally:
        _restore_viewers(saved)
        pv.OFF_SCREEN = False
    return log


def _physics_pages(npz_file):
    """Pages of the physics gallery (visualizers.PHYSICS_PAGES with more than one available panel, as the viewer)."""
    try:
        import PynamicMesh.utils.visualizers as VZ
        with np.load(npz_file) as z:
            available = set(z.files) | {"RGB"}
        return max(1, sum(1 for _, panels in VZ.PHYSICS_PAGES if len([p for p in panels if p[0] in available]) > 1))
    except Exception:  # noqa: BLE001
        return 1


# ----------------------------------------------------------------------------------------------------------------- #
#  Stand-alone use: GIFs from results that are already on disk (no recomputation)
# ----------------------------------------------------------------------------------------------------------------- #
STAGES = ("basic_geometry", "physics", "reeb", "ms_complex", "parametrization", "trajectories", "graph_animation",
          "dynamic_analysis", "motion_analysis", "reeb_dynamics")
_STAGE_RESULT = {"basic_geometry": "Basic_Geometry", "physics": "Physical_fields", "reeb": "Reeb_Graphs",
                 "ms_complex": "MSComplexAnalysis", "parametrization": "Parametrization", "trajectories": "Trajectories",
                 "graph_animation": "GraphAnimation", "dynamic_analysis": "DynamicAnalysis",
                 "motion_analysis": "MotionAnalysis", "reeb_dynamics": "ReebDynamics"}


def _has_meshes(folder):
    return any(f.is_file() and f.suffix in (".obj", ".mat") for f in Path(folder).iterdir())


def export_gifs(mesh_path, results_path=None, stages="all", gif_params=None, verbose=True):
    """
    Writes the GIF animations of results that are ALREADY COMPUTED (nothing is recomputed), exactly as the pipeline
    does with gif=True.

    mesh_path    : a scene folder with the meshes (e.g. Mesh_models/camel) or a root folder with one sub-folder per
                   scene (e.g. Mesh_models) - the same folders given to run_pipeline
    results_path : Results/<scene> of that scene; default = the pipeline layout <root>/../Results/<scene>
                   (for a root folder: <root>/../Results/<each scene>)
    stages       : 'all' (every stage whose results exist) or a list of
                   'basic_geometry', 'physics', 'reeb', 'ms_complex', 'parametrization', 'trajectories',
                   'graph_animation', 'dynamic_analysis'
    gif_params   : GIF_DEFAULTS overrides (fps, real_time, max_frames, panel_width, height, max_width, vectors, loop)
    The timing follows Results/<scene>/frame_times.csv when it exists (known frame times), else 1 / fps per frame.
    Returns {scene: {stage: [written files]}}.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.is_dir():
        raise FileNotFoundError(f"{mesh_path} is not a folder")
    if _has_meshes(mesh_path):
        scenes = [(mesh_path, Path(results_path) if results_path else mesh_path.parent.parent / "Results" / mesh_path.name)]
    else:                                              # root folder: every scene sub-folder
        roots = sorted(d for d in mesh_path.iterdir() if d.is_dir() and _has_meshes(d))
        base = Path(results_path) if results_path else mesh_path.parent / "Results"
        scenes = [(d, base / d.name) for d in roots]
    wanted = list(STAGES) if stages == "all" else ([stages] if isinstance(stages, str) else list(stages))
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; choose among {list(STAGES)}")
    report = {}
    for mesh_folder, res in scenes:
        if not res.is_dir():
            if verbose:
                print(f"[GIF] {mesh_folder.name}: no results in {res} - skipped")
            continue
        report[mesh_folder.name] = {}
        for st in wanted:
            sub = res / _STAGE_RESULT[st]
            if st != "basic_geometry" and not sub.is_dir():
                if verbose and stages != "all":
                    print(f"[GIF] {mesh_folder.name}: no {_STAGE_RESULT[st]} results - '{st}' skipped")
                continue
            log = export_stage_gifs(st, res, mesh_folder, gif_params)
            files = [str(f) for _, status, det in log if status == "ok" for f in (det or []) if str(f).endswith(".gif")]
            report[mesh_folder.name][st] = files
            if verbose:
                fails = [f"{lab}: {status}" for lab, status, _ in log if status != "ok"]
                print(f"[GIF] {mesh_folder.name} / {st}: {len(files)} GIF(s) in {sub / 'gif'}"
                      + (f"  (problems: {'; '.join(fails)})" if fails else ""))
    return report
