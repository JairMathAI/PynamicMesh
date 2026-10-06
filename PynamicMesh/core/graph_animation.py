"""
graph_animation.py  —  a graph of one frame (Reeb skeleton / Morse–Smale critical-point graph) moved by the trajectory
model of the sequence
=====================================================================================================================

The trajectories (trajectories.py) move ONE mesh, the reference mesh (connectivity F0), through the whole sequence:
observed frames X (T, n0, 3) and dense interpolations interpolation/<scheme>.npy (S, n0, 3). A graph computed on frame
k is anchored to that moving mesh, so it moves with the surface:

  * every node is a fixed linear combination of reference vertices (its ANCHOR):
      Reeb node with vertices     -> uniform weights over its vertices (the node is their mean: compute_approx_reeb_graph)
      Reeb node without vertices  -> barycentric weights of the triangle it lies in (iso-line points)
      Morse–Smale critical point  -> the vertex itself
      Morse–Smale 'center'        -> recomputed at every step as the area-weighted centre of the moving surface
    the vertices of frame k are mapped to reference vertices through X[k] (the reference mesh registered to frame k),
    so any frame can be chosen; for frame 0 the anchors are exact;
  * positions for all steps: one (n_nodes × n0) · (n0 × 3S) product (GPU through xp / CuPy when available);
  * the edges (topology) are those of the chosen graph: the skeleton keeps its structure and follows the motion.

Analysis (why it is useful): node speeds, edge strain (relative change of every edge length w.r.t. the source frame:
stretching / shortening of skeleton segments; for the Morse–Smale star graph the centre-to-maximum edges are the
extension of the protrusions), total skeleton length, and, at every observed frame, the distance between the animated
graph and the graph actually computed on that frame (large = the topology / structure of the frame differs from the
transported one: the skeleton changed, not just moved).

Several trajectory models can be used at once (model = a tuple of schemes, e.g. ('arap_polar', 'natural_cubic')): every
graph is animated with each of them, and Results/<scene>/GraphAnimation/<graph>_T<frame>_comparison/ compares them
(skeleton kinematics per model, deviation of the node positions of every model from the first one over time).

Results: Results/<scene>/GraphAnimation/<graph>_T<frame>_<model>/
    animation.npz        positions (S, n_nodes, 3), times, edges, node keys / types / f values, observed steps
    summary.csv          per step: time, observed, node speed (mean / max), skeleton length, |edge strain| (mean / max)
    nodes.csv            per step and node: position, speed, displacement from the source frame
    edge_strain.npy      (S, n_edges)
    comparison.csv       observed frames: animated vs actual graph (Chamfer / Hausdorff of the nodes, node counts)
    graphs/              networkx pickles of the animated graph (observed frames or every step, 'pos' updated)
    plots/
Viewer: PynamicMesh.utils.dynamics_visualizers.visualize_graph_animation.
"""
import json
import pickle
import re
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import cKDTree

from PynamicMesh.core.dyn_common import xp, GPU_AVAILABLE, to_gpu, to_cpu, as_faces, ensure_dir, load_frames
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots

GRAPH_SOURCES = {
    "reeb": ("Reeb_Graphs", "Reeb_T{:04d}.pkl"),
    "mscomplex": ("MSComplexAnalysis/MS_Graphs", "MSGraph_T{:04d}.pkl"),
    "reeb_controlled": ("Reeb_Graphs/Topology_Controlled", "Reeb_T{:04d}.pkl"),   # topology-controlled Reeb graphs
}


def _frame_id(path):
    m = re.search(r"T(\d+)", Path(path).stem)
    return int(m.group(1)) if m else -1


def graph_files(target_folder, graph):
    """{frame: path} of the graphs of one kind ('reeb' | 'mscomplex') computed by the pipeline."""
    sub, _ = GRAPH_SOURCES[graph]
    folder = Path(target_folder) / sub
    return {_frame_id(f): f for f in sorted(folder.glob("*.pkl"))} if folder.is_dir() else {}


def load_graph(target_folder, graph, frame):
    files = graph_files(target_folder, graph)
    if frame not in files:
        raise FileNotFoundError(f"no {graph} graph for frame {frame} in {Path(target_folder)} (available: {sorted(files)})")
    with open(files[frame], "rb") as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------------------------------------------------- #
#  Trajectory model
# ----------------------------------------------------------------------------------------------------------------- #
def trajectory_model(target_folder, model="best"):
    """
    Positions of the reference mesh along the sequence: model = 'observed' (the observed frames), a scheme name
    (interpolation/<scheme>.npy) or 'best' (the scheme with the lowest leave-one-out error that is on disk; the
    observed frames if none). Returns (P (S, n0, 3), times (S,), F0, observed step of every frame, model name, X).
    """
    tr = Path(target_folder) / "Trajectories"
    X = np.load(tr / "trajectories.npy"); F0 = np.load(tr / "reference_faces.npy")
    t_obs = np.load(tr / "time.npy") if (tr / "time.npy").exists() else np.arange(X.shape[0], dtype=float)
    interp = tr / "interpolation"
    available = sorted(f.stem for f in interp.glob("*.npy") if f.stem != "query_times") if interp.is_dir() else []
    name = model
    if model == "best":
        name = "observed"
        if available:
            name = available[0]
            try:
                v = pd.read_csv(interp / "validation.csv")
                col = "rel_error" if "rel_error" in v.columns else [c for c in v.columns if "error" in c.lower()][0]
                ranked = list(v.groupby(v.columns[0] if "scheme" not in v else "scheme")[col].median().sort_values().index)
                name = next((r for r in ranked if r in available), available[0])
            except Exception:  # noqa: BLE001
                pass
    if name == "observed":
        P, times = X, t_obs
    else:
        if name not in available:
            raise FileNotFoundError(f"interpolation '{name}' not on disk (available: {available + ['observed']})")
        P = np.load(interp / f"{name}.npy"); times = np.load(interp / "query_times.npy")
    obs_step = np.array([int(np.argmin(np.abs(times - t))) for t in t_obs])
    return P, times, F0, obs_step, name, X


# ----------------------------------------------------------------------------------------------------------------- #
#  Anchors
# ----------------------------------------------------------------------------------------------------------------- #
def reference_ids(X, frame, V_frame):
    """Reference vertex of every vertex of mesh `frame` (identity for the reference frame itself)."""
    n0 = X.shape[1]
    if V_frame.shape[0] == n0 and np.allclose(X[frame], V_frame, atol=1e-9 * (np.ptp(V_frame) + 1)):
        return np.arange(n0)
    return cKDTree(X[frame]).query(V_frame)[1]


def _barycentric_on_mesh(p, V, F, tree_c, centers, k=8):
    """(3 vertex ids, 3 weights) of the triangle of mesh (V, F) closest to p (projection clipped to the triangle)."""
    _, cand = tree_c.query(p, k=min(k, len(F)))
    best = (np.inf, None, None)
    for f in np.atleast_1d(cand):
        a, b, c = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
        v0, v1, v2 = b - a, c - a, p - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if den <= 1e-300:
            continue
        v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den; u = 1 - v - w
        bw = np.clip([u, v, w], 0, None); bw = bw / max(bw.sum(), 1e-300)
        q = bw[0] * a + bw[1] * b + bw[2] * c
        dist = np.linalg.norm(p - q)
        if dist < best[0]:
            best = (dist, F[f], bw)
    if best[1] is None:
        return np.array([0, 0, 0]), np.array([1.0, 0, 0])
    return best[1], best[2]


def node_anchors(G, V_frame, F_frame, ref_ids, n0):
    """
    Sparse (n_nodes × n0) anchor matrix A (rows sum to 1) and the list of nodes recomputed as the area-weighted centre
    of the moving surface. Rows follow list(G.nodes).
    """
    keys = list(G.nodes)
    rows, cols, vals, centre, kinds = [], [], [], [], []
    tree_c = centers = None
    for r, key in enumerate(keys):
        d = G.nodes[key]
        verts = np.asarray(d.get("vertices", []), dtype=np.int64).ravel()
        verts = verts[(verts >= 0) & (verts < V_frame.shape[0])]
        vtx = int(d.get("vertex", -1)) if d.get("vertex", None) is not None else -1
        if d.get("type") == "center" or key == "center":
            centre.append(r); kinds.append("centre"); continue
        if verts.size:
            ids, w = ref_ids[verts], np.full(verts.size, 1.0 / verts.size); kinds.append("vertex set")
        elif 0 <= vtx < V_frame.shape[0]:
            ids, w = np.array([ref_ids[vtx]]), np.array([1.0]); kinds.append("vertex")
        else:                                               # position only (e.g. iso-line nodes): barycentric anchor
            if tree_c is None:
                centers = V_frame[F_frame].mean(axis=1); tree_c = cKDTree(centers)
            tri, bw = _barycentric_on_mesh(np.asarray(d["pos"], float), V_frame, F_frame, tree_c, centers)
            ids, w = ref_ids[tri], bw; kinds.append("barycentric")
        rows += [r] * len(ids); cols += list(ids); vals += list(w)
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(len(keys), n0)).tocsr()
    return A, centre, keys, kinds


def _vertex_areas(V, F):
    fa = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    A = np.zeros(V.shape[0])
    for k in range(3):
        np.add.at(A, F[:, k], fa / 3.0)
    return A


def animate_positions(A, centre_rows, P, F0):
    """Node positions for every step: A · P_s (all steps in one product, GPU when available) + area-weighted centres."""
    S, n0, _ = P.shape
    Pr = np.ascontiguousarray(P.transpose(1, 0, 2).reshape(n0, 3 * S))
    if GPU_AVAILABLE:
        try:
            import cupyx.scipy.sparse as cps
            out = to_cpu(cps.csr_matrix(A) @ to_gpu(Pr))
        except Exception:  # noqa: BLE001 - dense product on the GPU
            out = to_cpu(to_gpu(A.toarray()) @ to_gpu(Pr))
    else:
        out = A @ Pr
    pos = np.asarray(out).reshape(A.shape[0], S, 3).transpose(1, 0, 2).copy()
    if centre_rows:
        F0 = as_faces(F0)
        for s in range(S):
            w = _vertex_areas(P[s], F0)
            c = (P[s] * w[:, None]).sum(axis=0) / max(w.sum(), 1e-300)
            pos[s, centre_rows] = c
    return pos


# ----------------------------------------------------------------------------------------------------------------- #
#  Analysis
# ----------------------------------------------------------------------------------------------------------------- #
def _chamfer(Pa, Pb):
    da, _ = cKDTree(Pb).query(Pa); db, _ = cKDTree(Pa).query(Pb)
    return 0.5 * (da.mean() + db.mean()), max(da.max(), db.max())


def animate_graph(target_folder, graph="reeb", frame=0, model="best", frames=None, mesh_folder=None, loader=None,
                  export_graphs="observed", plots=True, verbose=False):
    """
    Animates the `graph` ('reeb' | 'mscomplex') of `frame` with the trajectory `model` ('best' | 'observed' | a scheme)
    and writes the results (module docstring). `frames` / `mesh_folder` give the mesh of `frame` (needed when frame != 0
    or when the graph stores vertex indices); export_graphs: 'observed' | 'all' | 'none'. Returns the output folder.
    """
    target_folder = Path(target_folder)
    G = load_graph(target_folder, graph, frame)
    P, times, F0, obs_step, model_name, X = trajectory_model(target_folder, model)
    S, n0, _ = P.shape
    T = X.shape[0]
    if not 0 <= frame < T:
        raise ValueError(f"frame {frame} outside the {T} frames of the trajectories")
    if frames is None and mesh_folder is not None:
        frames = load_frames(mesh_folder, loader)
    if frames is not None and frame < len(frames):
        V_k, F_k = np.asarray(frames[frame]["V"], float), as_faces(frames[frame]["F"])
    elif frame == 0:
        V_k, F_k = X[0], as_faces(F0)
    else:
        raise ValueError("the mesh of the chosen frame is needed (frames= or mesh_folder=)")
    ref = reference_ids(X, frame, V_k)
    A, centre_rows, keys, kinds = node_anchors(G, V_k, F_k, ref, n0)
    pos = animate_positions(A, centre_rows, P, F0)
    key_index = {k: i for i, k in enumerate(keys)}
    edges = np.array([(key_index[u], key_index[v]) for u, v in G.edges()], dtype=np.int64).reshape(-1, 2)
    diag = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))

    # the source frame: animated positions vs the stored ones (anchor quality)
    s_k = int(obs_step[frame])
    stored = np.array([np.asarray(G.nodes[k]["pos"], float) for k in keys])
    anchor_err = np.linalg.norm(pos[s_k] - stored, axis=1) / diag

    # kinematics of the skeleton
    dt = np.gradient(times) if S > 1 else np.ones(S)
    vel = np.gradient(pos, times, axis=0) if S > 1 else np.zeros_like(pos)
    speed = np.linalg.norm(vel, axis=2)
    L = np.linalg.norm(pos[:, edges[:, 0]] - pos[:, edges[:, 1]], axis=2) if len(edges) else np.zeros((S, 0))
    L0 = L[s_k] if len(edges) else np.zeros(0)
    strain = L / np.maximum(L0[None, :], 1e-12) - 1.0 if len(edges) else np.zeros((S, 0))
    obs_of = {int(s): j for j, s in enumerate(obs_step)}
    summary = pd.DataFrame({
        "step": np.arange(S), "time": times, "observed_frame": [obs_of.get(s, -1) for s in range(S)],
        "mean_node_speed": speed.mean(axis=1), "max_node_speed": speed.max(axis=1),
        "skeleton_length": L.sum(axis=1) if len(edges) else 0.0,
        "skeleton_length_ratio": (L.sum(axis=1) / max(L0.sum(), 1e-12)) if len(edges) else 1.0,
        "mean_abs_edge_strain": np.abs(strain).mean(axis=1) if len(edges) else 0.0,
        "max_abs_edge_strain": np.abs(strain).max(axis=1) if len(edges) else 0.0,
        "mean_displacement_rel": np.linalg.norm(pos - pos[s_k][None], axis=2).mean(axis=1) / diag,
    })
    ntype = [str(G.nodes[k].get("type", "reeb")) for k in keys]
    nodes_df = pd.DataFrame({
        "step": np.repeat(np.arange(S), len(keys)), "time": np.repeat(times, len(keys)),
        "node": np.tile([str(k) for k in keys], S), "type": np.tile(ntype, S),
        "x": pos[:, :, 0].ravel(), "y": pos[:, :, 1].ravel(), "z": pos[:, :, 2].ravel(),
        "speed": speed.ravel(), "displacement": np.linalg.norm(pos - pos[s_k][None], axis=2).ravel()})

    # comparison with the graph actually computed on every observed frame
    comp = []
    files = graph_files(target_folder, graph)
    for j in tqdm(range(T), desc="Comparison with the frame graphs", leave=False):
        if j not in files:
            continue
        with open(files[j], "rb") as fh:
            Gj = pickle.load(fh)
        Pj = np.array([np.asarray(d["pos"], float) for _, d in Gj.nodes(data=True)])
        if not len(Pj):
            continue
        ch, hd = _chamfer(pos[int(obs_step[j])], Pj)
        comp.append({"frame": j, "time": float(times[int(obs_step[j])]), "step": int(obs_step[j]),
                     "chamfer_rel": ch / diag, "hausdorff_rel": hd / diag,
                     "n_nodes_animated": len(keys), "n_nodes_actual": Gj.number_of_nodes(),
                     "n_edges_animated": len(edges), "n_edges_actual": Gj.number_of_edges(),
                     "cycles_actual": Gj.number_of_edges() - Gj.number_of_nodes() + nx.number_connected_components(Gj)})
    comp = pd.DataFrame(comp)

    out = ensure_dir(target_folder / "GraphAnimation" / f"{graph}_T{frame:04d}_{model_name}")
    for f in out.glob("graphs/*.pkl"):
        f.unlink()
    f_val = np.array([float(G.nodes[k].get("f_value", G.nodes[k].get("bin", 0.0))) for k in keys])
    np.savez_compressed(out / "animation.npz", positions=pos, times=times, edges=edges,
                        node_keys=np.array([str(k) for k in keys]), node_type=np.array(ntype), node_f=f_val,
                        node_anchor=np.array(kinds), observed_step=obs_step, source_frame=frame, source_step=s_k,
                        graph=graph, model=model_name, diag=diag)
    np.save(out / "edge_strain.npy", strain.astype(np.float32))
    summary.to_csv(out / "summary.csv", index=False)
    nodes_df.to_csv(out / "nodes.csv", index=False)
    if len(comp):
        comp.to_csv(out / "comparison.csv", index=False)
    if export_graphs in ("observed", "all"):
        d_g = ensure_dir(out / "graphs")
        steps = range(S) if export_graphs == "all" else sorted(set(int(s) for s in obs_step))
        for s in steps:
            Gs = G.copy()
            for i, k in enumerate(keys):
                Gs.nodes[k]["pos"] = tuple(float(c) for c in pos[s, i])
            Gs.graph.update(animated_from_frame=frame, model=model_name, time=float(times[s]))
            tag = f"T{obs_of[s]:04d}" if s in obs_of else f"S{s:04d}"
            with open(d_g / f"{graph}_anim_{tag}.pkl", "wb") as fh:
                pickle.dump(Gs, fh)
    with open(out / "info.json", "w") as fh:
        json.dump({"graph": graph, "source_frame": frame, "model": model_name, "n_steps": S, "n_nodes": len(keys),
                   "n_edges": int(len(edges)), "anchor_kinds": {k: kinds.count(k) for k in set(kinds)},
                   "anchor_error_rel_mean": float(np.nanmean(anchor_err)), "anchor_error_rel_max": float(np.nanmax(anchor_err)),
                   "gpu": bool(GPU_AVAILABLE)}, fh, indent=2)
    if plots:
        plot_graph_animation(out)
    if verbose:
        print(f"Graph animation: {graph} graph of frame {frame} ({len(keys)} nodes, {len(edges)} edges) moved by "
              f"'{model_name}' over {S} steps; anchor error {np.nanmean(anchor_err):.2e} of the size -> {out}")
    return out


def resolve_models(target_folder, model):
    """
    model: 'best' | 'observed' | a scheme name | a tuple / list of them. Returns the list of concrete model names
    ('best' resolved to the scheme it designates), in the given order, without duplicates; models whose data are not
    on disk are dropped with a message.
    """
    requested = [model] if isinstance(model, str) else list(model)
    out = []
    for m in requested:
        try:
            name = trajectory_model(target_folder, str(m))[4]
        except FileNotFoundError as exc:
            print(f"[Graph animation] model '{m}' skipped: {exc}")
            continue
        if name not in out:
            out.append(name)
    return out


def compute_graph_animation(target_folder, frames=None, mesh_folder=None, loader=None, graphs=("reeb", "mscomplex"),
                            frame=0, model="best", export_graphs="observed", plots=True, verbose=True):
    """
    Pipeline entry: animates every requested graph kind that exists for `frame` (missing kinds are skipped) with the
    trajectory model(s): model = 'best' | 'observed' | a scheme | a tuple / list of them (e.g. ('arap_polar',
    'natural_cubic', 'best')). With several models the graphs are animated with each of them and compared
    (<graph>_T<frame>_comparison/). Returns {graph: path} for one model, {graph: {model: path}} for several.
    """
    models = resolve_models(target_folder, model)
    if not models:
        print("[Graph animation] no usable trajectory model: nothing animated.")
        return {}
    if frames is None and mesh_folder is not None:
        frames = load_frames(mesh_folder, loader)          # loaded once for all the models
    outs = {}
    for g in ([graphs] if isinstance(graphs, str) else graphs):
        if frame not in graph_files(target_folder, g):
            if verbose:
                print(f"[Graph animation] no {g} graph for frame {frame}: skipped")
            continue
        per_model = {}
        for m in tqdm(models, desc=f"Graph animation ({g})", leave=False):
            per_model[m] = str(animate_graph(target_folder, g, frame, m, frames=frames, export_graphs=export_graphs,
                                             plots=plots, verbose=verbose))
        if len(per_model) > 1:
            compare_models(target_folder, g, frame, list(per_model), plots=plots, verbose=verbose)
        outs[g] = per_model[models[0]] if isinstance(model, str) else per_model
    return outs


def compare_models(target_folder, graph, frame, models, plots=True, verbose=True):
    """
    Compares the animations of one graph made with several trajectory models (all already on disk):
      models_summary.csv    per model and step: time, node speed, skeleton length ratio, |edge strain|
      models_deviation.csv  at the times common to all models: mean / max distance between the nodes of every model and
                            those of the first one (relative to the size): where the schemes disagree on the motion
      plots/1_models_kinematics.png, plots/2_models_deviation.png
    """
    base = Path(target_folder) / "GraphAnimation"
    out = ensure_dir(base / f"{graph}_T{frame:04d}_comparison")
    anim = {m: np.load(base / f"{graph}_T{frame:04d}_{m}" / "animation.npz", allow_pickle=True) for m in models}
    summ = pd.concat([pd.read_csv(base / f"{graph}_T{frame:04d}_{m}" / "summary.csv").assign(model=m) for m in models],
                     ignore_index=True)
    summ.to_csv(out / "models_summary.csv", index=False)
    ref = models[0]; t_ref = np.round(anim[ref]["times"], 8); diag = float(anim[ref]["diag"])
    common = set(t_ref.tolist())
    for m in models[1:]:
        common &= set(np.round(anim[m]["times"], 8).tolist())
    common = np.array(sorted(common))
    rows = []
    for m in models:
        tm = np.round(anim[m]["times"], 8)
        for t in common:
            a = anim[ref]["positions"][int(np.argmin(np.abs(t_ref - t)))]
            b = anim[m]["positions"][int(np.argmin(np.abs(tm - t)))]
            d = np.linalg.norm(a - b, axis=1) / diag
            rows.append({"model": m, "reference": ref, "time": float(t), "mean_deviation_rel": float(d.mean()),
                         "max_deviation_rel": float(d.max())})
    dev = pd.DataFrame(rows); dev.to_csv(out / "models_deviation.csv", index=False)
    with open(out / "info.json", "w") as fh:
        json.dump({"graph": graph, "source_frame": frame, "models": models, "reference": ref,
                   "n_common_times": int(len(common))}, fh, indent=2)
    if plots:
        d = ensure_dir(out / "plots")
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for m in models:
            sm = summ[summ.model == m]
            axes[0].plot(sm.time, sm.mean_node_speed, label=m)
            axes[1].plot(sm.time, sm.skeleton_length_ratio, label=m)
            axes[2].plot(sm.time, sm.mean_abs_edge_strain, label=m)
        axes[0].set_ylabel("mean node speed"); axes[1].set_ylabel("skeleton length / source"); axes[2].set_ylabel("mean |edge strain|")
        axes[2].set_xlabel("time"); axes[0].legend(fontsize=8)
        axes[0].set_title(f"{graph} graph of frame {frame}: skeleton kinematics with every trajectory model", fontweight="bold")
        fig.tight_layout(); fig.savefig(d / "1_models_kinematics.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for m in models[1:]:
            dm = dev[dev.model == m]
            ax.plot(dm.time, dm.mean_deviation_rel, marker="o", ms=3, label=f"{m}: mean")
            ax.plot(dm.time, dm.max_deviation_rel, ls="--", label=f"{m}: max")
        ax.set_xlabel("time"); ax.set_ylabel(f"node distance to '{ref}' / size"); ax.legend(fontsize=8)
        ax.set_title(f"Where the trajectory models disagree (reference: {ref}; 0 at the observed frames)", fontweight="bold")
        fig.tight_layout(); fig.savefig(d / "2_models_deviation.png", dpi=200); plt.close(fig)
    if verbose:
        worst = dev[dev.model != ref].groupby("model").max_deviation_rel.max() if len(models) > 1 else pd.Series(dtype=float)
        print(f"Graph animation: {graph} graph of frame {frame} compared over {len(models)} models "
              f"(largest node deviation from '{ref}': {', '.join(f'{k} {v:.3g}' for k, v in worst.items())}) -> {out}")
    return out


# ----------------------------------------------------------------------------------------------------------------- #
#  Plots
# ----------------------------------------------------------------------------------------------------------------- #
def plot_graph_animation(out):
    out = Path(out); d = ensure_dir(out / "plots")
    S_ = pd.read_csv(out / "summary.csv")
    obs = S_[S_.observed_frame >= 0]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(S_.time, S_.mean_node_speed, label="mean node speed"); axes[0].plot(S_.time, S_.max_node_speed, label="max node speed")
    axes[0].scatter(obs.time, obs.mean_node_speed, color="k", zorder=3, s=18, label="observed frames")
    axes[0].set_ylabel("speed"); axes[0].legend(fontsize=8); axes[0].set_title("Skeleton kinematics", fontweight="bold")
    axes[1].plot(S_.time, S_.skeleton_length_ratio, color="tab:green"); axes[1].axhline(1, color="gray", lw=0.8)
    axes[1].set_ylabel("skeleton length / source"); axes[1].set_title("Total edge length (stretching of the skeleton)", fontweight="bold")
    axes[2].plot(S_.time, S_.mean_abs_edge_strain, label="mean |edge strain|"); axes[2].plot(S_.time, S_.max_abs_edge_strain, label="max |edge strain|")
    axes[2].set_ylabel("|L / L_source − 1|"); axes[2].set_xlabel("time"); axes[2].legend(fontsize=8)
    axes[2].set_title("Edge strain", fontweight="bold")
    fig.tight_layout(); fig.savefig(d / "1_skeleton_kinematics.png", dpi=200); plt.close(fig)
    strain = np.load(out / "edge_strain.npy")
    if strain.size:
        fig, ax = plt.subplots(figsize=(11, 5))
        v = float(np.percentile(np.abs(strain), 99)) or 1.0
        im = ax.imshow(strain.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                       extent=[S_.time.iloc[0], S_.time.iloc[-1], -0.5, strain.shape[1] - 0.5])
        fig.colorbar(im, ax=ax, label="edge strain L / L_source − 1")
        ax.set_xlabel("time"); ax.set_ylabel("edge"); ax.set_title("Strain of every skeleton edge (red stretched, blue shortened)", fontweight="bold")
        fig.tight_layout(); fig.savefig(d / "2_edge_strain.png", dpi=200); plt.close(fig)
    if (out / "comparison.csv").exists():
        C = pd.read_csv(out / "comparison.csv")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(C.frame, C.chamfer_rel, marker="o", label="Chamfer distance (nodes)")
        ax.plot(C.frame, C.hausdorff_rel, marker="s", label="Hausdorff distance (nodes)")
        ax.set_ylabel("distance / size"); ax.set_xlabel("observed frame"); ax.legend(fontsize=8, loc="upper left")
        ax2 = ax.twinx(); ax2.plot(C.frame, C.n_nodes_actual, color="gray", ls="--", marker="^", label="nodes of the actual graph")
        ax2.axhline(C.n_nodes_animated.iloc[0], color="k", ls=":", label="nodes of the animated graph"); ax2.set_ylabel("nodes"); ax2.grid(False)
        ax2.legend(fontsize=8, loc="upper right")
        ax.set_title("Animated graph vs the graph computed on every frame (high = the structure changed, not only moved)",
                     fontweight="bold", fontsize=10)
        fig.tight_layout(); fig.savefig(d / "3_animated_vs_actual.png", dpi=200); plt.close(fig)