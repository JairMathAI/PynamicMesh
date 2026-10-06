import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from tqdm.auto import tqdm
from pyFM.mesh import TriMesh
import seaborn as sns
import networkx as nx
import pickle
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse import linalg as splinalg
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import wasserstein_distance
import warnings
from PynamicMesh.utils.tools import mesh_mat2object
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots

try:
    import cupy as xp
    GPU_AVAILABLE = True
    if __import__("multiprocessing").current_process().name == "MainProcess":            # worker processes import this module again: say it once
        print("[INFO] CuPy detected. Utilizing GPU for Reeb Graph spectral computations and scalar fields.")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    if __import__("multiprocessing").current_process().name == "MainProcess":            # worker processes import this module again: say it once
        print("[INFO] CuPy not found. Defaulting to CPU (NumPy).")


def to_gpu(arr):
    """Moves a numpy array to the GPU if available."""
    if GPU_AVAILABLE and isinstance(arr, np.ndarray):
        return xp.asarray(arr)
    return arr


def to_cpu(arr):
    """Moves a CuPy array back to the CPU for NetworkX/PyVista/Saving."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr


# ----------------------------------------------------------------------------- #
#  Dynamic graph analysis (unchanged behaviour)
# ----------------------------------------------------------------------------- #

def _frame_number(path):
    """Extracts the integer frame id from names like 'Reeb_T12.pkl' (falls back to name order)."""
    m = re.search(r'T(\d+)', Path(path).stem)
    return int(m.group(1)) if m else float('inf')


def _genus_diagnostic(G):
    """Topological-control diagnostic stored with the graph (compute_reeb_graph_controlled), if any."""
    g = getattr(G, "graph", {}) or {}
    if "surface_genus" not in g:
        return {}
    lo, hi = g.get("beta1_expected", (np.nan, np.nan))
    return {"Surface_Genus": g["surface_genus"], "Boundary_Loops": g.get("boundary_loops", 0),
            "Beta1_Expected_Min": lo, "Beta1_Expected_Max": hi, "Genus_Consistent": g.get("genus_consistent"),
            "Refinements": g.get("refinements", 0), "N_Slabs": g.get("n_slabs"), "Hidden_Handles": g.get("hidden_handles", 0)}


def _controlled_diagnostic(reeb_folder, frame):
    """Diagnostic of the topology-controlled graph of a frame (Reeb_Graphs/Topology_Controlled), if computed."""
    f = Path(reeb_folder) / "Topology_Controlled" / f"Reeb_T{int(frame):04d}.pkl"
    if not f.exists():
        return {}
    try:
        with open(f, "rb") as fh:
            Gc = pickle.load(fh)
    except Exception:  # noqa: BLE001
        return {}
    g = Gc.graph
    return {"Nodes_Controlled": Gc.number_of_nodes(), "Beta1_Controlled": g.get("beta1_graph"),
            "Genus_Consistent_Controlled": g.get("genus_consistent"), "Hidden_Handles_Controlled": g.get("hidden_handles"),
            "Refinements_Controlled": g.get("refinements"), "N_Slabs_Controlled": g.get("n_slabs")}


def _graph_topology(G):
    """Betti numbers of a graph: β0 = connected components, β1 = independent cycles (E − V + β0); component sizes."""
    v, e = G.number_of_nodes(), G.number_of_edges()
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True) if v > 0 else []
    b0 = len(comps)
    return {"nodes": v, "edges": e, "betti_0": b0, "betti_1": e - v + b0, "component_sizes": comps}


def _component_event(d0):
    return "" if d0 == 0 else ("split / new component" if d0 > 0 else "merge / lost component")


def graph_time_analysis(reeb_folder_path, single_file=True, analysis_folder=None):
    """
    Time analysis of a sequence of graphs (Reeb graphs, Morse–Smale critical-point graphs, ...).
      time_analysis.csv       one row per transition T_{i-1} -> T_i with the size and topology of the CURRENT graph
                              (Frame = i; Time_Step keeps the previous convention = i-1) and the distances between the two:
                              Betti_0_Components (connected components), Betti_1_Cycles, their changes Delta_Betti_0 /
                              Delta_Betti_1 and Component_Event (split / new component, merge / lost component)
      topology_per_frame.csv  every frame INCLUDING the first one: nodes, edges, Betti-0, Betti-1, component sizes
    Betti-0 of a Reeb graph = number of connected components of the surface (each piece of the shape has its own
    part of the graph), so splits and merges of the shape appear as jumps of Betti-0.
    """
    if single_file:
        print("\nStarting Dynamic Graph Analysis...")
    reeb_folder = Path(reeb_folder_path)
    # Natural sort on the frame number: a plain string sort puts Reeb_T10 before Reeb_T2.
    reeb_files = sorted([f for f in reeb_folder.iterdir() if f.is_file() and f.suffix == '.pkl'],
                        key=_frame_number)
    frame_ids = [_frame_number(f) for f in reeb_files]

    if len(reeb_files) < 2:
        print("Not enough Reeb graphs found to perform time analysis.")
        return

    analysis_folder = Path(analysis_folder) if analysis_folder is not None else reeb_folder.parent / 'Graph_analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    features_list = []

    # graph edit distances of all the transitions computed in parallel (independent; each call keeps its own 2 s
    # timeout on its own core, so the values are those of the sequential loop) - accel.parallel_map
    geds = None
    if len(reeb_files) > 2:
        try:
            from PynamicMesh.core import accel
            geds = accel.parallel_map(accel.graph_edit_distance_task,
                                      [(str(reeb_files[i - 1]), str(reeb_files[i]), 2) for i in range(1, len(reeb_files))],
                                      desc="Graph edit distances")
        except Exception:  # noqa: BLE001 - sequential fallback
            geds = None

    with open(reeb_files[0], 'rb') as f:
        G_prev = pickle.load(f)
    topo_prev = _graph_topology(G_prev)
    per_frame = [{"Frame": frame_ids[0], "Nodes": topo_prev["nodes"], "Edges": topo_prev["edges"],
                  "Betti_0_Components": topo_prev["betti_0"], "Betti_1_Cycles": topo_prev["betti_1"],
                  "Delta_Betti_0": 0, "Delta_Betti_1": 0, "Component_Event": "",
                  "Largest_Component_Nodes": topo_prev["component_sizes"][0] if topo_prev["component_sizes"] else 0,
                  "Component_Sizes": ";".join(map(str, topo_prev["component_sizes"])), **_genus_diagnostic(G_prev),
                  **_controlled_diagnostic(reeb_folder, frame_ids[0])}]

    for i in tqdm(range(1, len(reeb_files)), desc="Computing Graph Metrics", leave=single_file):
        with open(reeb_files[i], 'rb') as f:
            G_curr = pickle.load(f)

        v_prev, e_prev = G_prev.number_of_nodes(), G_prev.number_of_edges()
        v_curr, e_curr = G_curr.number_of_nodes(), G_curr.number_of_edges()

        topo_curr = _graph_topology(G_curr)
        c_prev, c_curr = topo_prev["betti_0"], topo_curr["betti_0"]
        betti_prev, betti_curr = topo_prev["betti_1"], topo_curr["betti_1"]

        if v_curr > 0 and c_curr > 0:
            largest_cc = max(nx.connected_components(G_curr), key=len)
            sub_G = G_curr.subgraph(largest_cc)
            diameter = nx.diameter(sub_G)
            radius = nx.radius(sub_G)
        else:
            diameter, radius = 0, 0

        degrees_prev = [d for n, d in G_prev.degree()] if v_prev > 0 else [0]
        degrees_curr = [d for n, d in G_curr.degree()] if v_curr > 0 else [0]

        deg_wasserstein = wasserstein_distance(degrees_prev, degrees_curr)

        if v_prev > 0 and v_curr > 0:
            lap_prev_np = np.asarray(nx.normalized_laplacian_matrix(G_prev).todense())
            lap_curr_np = np.asarray(nx.normalized_laplacian_matrix(G_curr).todense())

            lap_prev_gpu = to_gpu(lap_prev_np)
            lap_curr_gpu = to_gpu(lap_curr_np)

            evals_prev = xp.linalg.eigvalsh(lap_prev_gpu)
            evals_curr = xp.linalg.eigvalsh(lap_curr_gpu)

            max_len = max(len(evals_prev), len(evals_curr))
            e_p_pad = xp.pad(evals_prev, (0, max_len - len(evals_prev)))
            e_c_pad = xp.pad(evals_curr, (0, max_len - len(evals_curr)))

            spectral_dist = float(to_cpu(xp.linalg.norm(e_p_pad - e_c_pad)))
        else:
            spectral_dist = 0.0

        # NOTE: graph_edit_distance is exponential; with a timeout NetworkX returns the
        # best *upper bound* found so far, so this value is approximate and can vary run to run.
        ged = geds[i - 1] if geds is not None else nx.graph_edit_distance(G_prev, G_curr, timeout=2)
        if ged is None:
            ged = abs(v_curr - v_prev) + abs(e_curr - e_prev)

        features_list.append({
            'Transition': f"T{frame_ids[i-1]} -> T{frame_ids[i]}",
            'Time_Step': frame_ids[i-1],
            'Frame': frame_ids[i],                      # frame the size / topology columns refer to (the current graph)
            'Nodes_T': v_curr,
            'Edges_T': e_curr,
            'Delta_Nodes': v_curr - v_prev,
            'Delta_Edges': e_curr - e_prev,
            'Betti_0_Components': c_curr,
            'Delta_Betti_0': c_curr - c_prev,
            'Component_Event': _component_event(c_curr - c_prev),
            'Betti_1_Cycles': betti_curr,
            'Delta_Betti_1': betti_curr - betti_prev,
            'Graph_Density': nx.density(G_curr) if v_curr > 1 else 0,
            'LCC_Diameter': diameter,
            'LCC_Radius': radius,
            'Deg_Wasserstein_Dist': deg_wasserstein,
            'Spectral_Laplacian_Dist': spectral_dist,
            'Graph_Edit_Dist': ged
        })

        per_frame.append({"Frame": frame_ids[i], "Nodes": v_curr, "Edges": e_curr,
                          "Betti_0_Components": c_curr, "Betti_1_Cycles": betti_curr,
                          "Delta_Betti_0": c_curr - c_prev, "Delta_Betti_1": betti_curr - betti_prev,
                          "Component_Event": _component_event(c_curr - c_prev),
                          "Largest_Component_Nodes": topo_curr["component_sizes"][0] if topo_curr["component_sizes"] else 0,
                          "Component_Sizes": ";".join(map(str, topo_curr["component_sizes"])), **_genus_diagnostic(G_curr),
                          **_controlled_diagnostic(reeb_folder, frame_ids[i])})
        G_prev, topo_prev = G_curr, topo_curr

    df = pd.DataFrame(features_list)
    csv_out_path = analysis_folder / 'time_analysis.csv'
    # known acquisition times (Results/<scene>/frame_times.csv): Time / Dt and the distances per time unit
    from PynamicMesh.core.dyn_common import add_time_columns
    dist_cols = [c for c in df.columns if c.endswith('_Dist') or c.startswith('Delta_')]
    df = add_time_columns(df, reeb_folder, frame_col='Frame', prev_col='Time_Step', rate_cols=dist_cols)
    df.to_csv(csv_out_path, index=False)
    add_time_columns(pd.DataFrame(per_frame), reeb_folder, frame_col='Frame').to_csv(
        analysis_folder / 'topology_per_frame.csv', index=False)

    if single_file:
        print(f"Graph analysis complete. Data saved to: {csv_out_path}")

    return str(csv_out_path)


def plot_dynamic_graph_analysis(csv_path, single_file=True):
    """
    Generates time-series reports visualizing the dynamic features of the Reeb graphs.
    Saves the plots as PNG files in a 'plots' subdirectory.
    """
    if single_file:
        print("\nGenerating visual reports for Graph Dynamics...")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Could not find CSV file at {csv_path}")
        return

    df = pd.read_csv(csv_file)
    plots_folder = csv_file.parent / 'plots'
    os.makedirs(plots_folder, exist_ok=True)

    from PynamicMesh.core.dyn_common import time_axis
    time_steps, t_label = time_axis(df, 'Time_Step', None)    # acquisition time when known (Time column)
    sns.set_theme(style="whitegrid")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(time_steps, df['Nodes_T'], label='Nodes', marker='o', color='#1f77b4', linewidth=2)
    ax1.plot(time_steps, df['Edges_T'], label='Edges', marker='s', color='#ff7f0e', linewidth=2)
    ax1.set_title('Reeb Graph Structural Size Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel(t_label or 'Time Step (T)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(plots_folder / '1_Structural_Size.png', dpi=200)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes2[0].plot(time_steps, df['Deg_Wasserstein_Dist'], color='purple', marker='o')
    axes2[0].set_title('Degree Distribution Shift (Wasserstein Distance)')
    axes2[0].set_ylabel('Distance')

    axes2[1].plot(time_steps, df['Spectral_Laplacian_Dist'], color='teal', marker='D')
    axes2[1].set_title('Global Shape Shift (Spectral Laplacian Distance)')
    axes2[1].set_ylabel('L2 Norm Diff')

    axes2[2].plot(time_steps, df['Graph_Edit_Dist'], color='crimson', marker='X')
    axes2[2].set_title('Transformation Cost (Graph Edit Distance)')
    axes2[2].set_ylabel('Edit Cost')
    axes2[2].set_xlabel(t_label or 'Time Step (Transition $T_{n-1} \u2192 T_n$)')

    fig2.suptitle('Dynamic Graph Similarity Metrics', fontsize=16, fontweight='bold', y=0.98)
    fig2.tight_layout()
    fig2.savefig(plots_folder / '2_Graph_Distances.png', dpi=200)
    plt.close(fig2)

    # topology per frame (Betti-0 = connected components, Betti-1 = cycles); the per-frame table includes the first frame
    topo_csv = csv_file.parent / 'topology_per_frame.csv'
    if topo_csv.exists():
        tp = pd.read_csv(topo_csv)
        fr, b0, b1 = (tp['Time'] if 'Time' in tp else tp['Frame']), tp['Betti_0_Components'], tp['Betti_1_Cycles']
        ev = tp[tp['Delta_Betti_0'] != 0]
        if 'Time' in tp:
            ev = ev.assign(Frame=ev['Time'])                # event markers on the time axis
    else:                                                # older time_analysis.csv only
        fr = df['Frame'] if 'Frame' in df else time_steps
        b0 = df['Betti_0_Components'] if 'Betti_0_Components' in df else None
        b1 = df['Betti_1_Cycles']
        ev = df[df['Delta_Betti_0'] != 0] if 'Delta_Betti_0' in df else df.iloc[0:0]
        tp = None
    diam_x = df['Time'] if 'Time' in df else (df['Frame'] if 'Frame' in df else time_steps)
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    ax = axes3[0]
    if b0 is not None:
        ax.step(fr, b0, where='mid', color='black', lw=2, label='Betti-0 (connected components)')
        ax.plot(fr, b0, 'o', color='black', ms=4)
        for _, r in ev.iterrows():
            up = r['Delta_Betti_0'] > 0
            ax.scatter(r['Frame'], r['Betti_0_Components'], marker='^' if up else 'v', s=170, zorder=4,
                       color='seagreen' if up else 'crimson', label='split / new component' if up else 'merge / lost component')
            ax.annotate(f"{'+' if up else ''}{int(r['Delta_Betti_0'])}", (r['Frame'], r['Betti_0_Components']),
                        textcoords='offset points', xytext=(7, 6 if up else -14), color='seagreen' if up else 'crimson')
        h_, l_ = ax.get_legend_handles_labels(); uniq = dict(zip(l_, h_)); ax.legend(uniq.values(), uniq.keys(), fontsize=9)
        ax.set_ylim(0, max(float(np.nanmax(b0)), 1) + 1)
        ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel('components (Betti-0)', fontsize=12)
    ax.set_title('Connected components of the graph (Betti-0)', fontsize=13, fontweight='bold')
    ax = axes3[1]
    ax.step(fr, b1, where='mid', color='darkgreen', lw=2); ax.plot(fr, b1, '^', color='darkgreen', label='Betti-1 (Cycles)')
    ax.set_ylabel('Number of Cycles (Betti-1)', color='darkgreen', fontsize=12); ax.legend(fontsize=9)
    ax.set_title('Independent cycles (Betti-1)', fontsize=13, fontweight='bold')
    ax = axes3[2]
    ax.plot(diam_x, df['LCC_Diameter'], color='navy', marker='v', linestyle='--', label='Diameter (LCC)')
    ax.set_ylabel('Graph Diameter (Hops)', color='navy', fontsize=12); ax.set_xlabel(t_label or 'Frame (T)', fontsize=12); ax.legend(fontsize=9)
    ax.set_title('Spatial span of the largest component', fontsize=13, fontweight='bold')
    for a_ in axes3:
        for _, r in ev.iterrows():
            a_.axvline(r['Frame'], color='seagreen' if r['Delta_Betti_0'] > 0 else 'crimson', ls=':', lw=0.8, alpha=0.7)
    fig3.suptitle('Topology & Spatial Span', fontsize=15, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(plots_folder / '3_Internal_Topology.png', dpi=200)
    plt.close(fig3)

    if single_file:
        print(f"Visual reports generated successfully in: {plots_folder}")


# ----------------------------------------------------------------------------- #
#  Shared helpers
# ----------------------------------------------------------------------------- #

def _as_faces(faces):
    """Return faces as an (F, 3) int64 numpy array, accepting pyvista's padded format too."""
    f = np.asarray(to_cpu(faces))
    if f.ndim == 1:
        if f.size % 4 != 0 or not np.all(f[::4] == 3):
            raise ValueError("Flat face array must be in pyvista [3, i, j, k, ...] format.")
        f = f.reshape(-1, 4)[:, 1:]
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3); got {f.shape}.")
    return f.astype(np.int64)


def _area_weighted_center(vertices, faces):
    """Center of mass of the surface (area weighted), independent of vertex density."""
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    tri = v[f]                                          # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)              # (F,)
    centroid = tri.mean(axis=1)                         # (F, 3)
    total = area.sum()
    if total <= 0:
        return v.mean(axis=0)
    return (centroid * area[:, None]).sum(axis=0) / total


# --------------------------------------------------------------------------- #
#  Automatic reference points of the scalar fields (sources / sinks)
# --------------------------------------------------------------------------- #

AUTO_SPECS = ("auto", "auto_center", "auto_extremity")
# reference parameters of the fields, and what 'auto' means for each one
REFERENCE_PARAMS = {"geodesic": {"vertex_ref_index": "center"}, "heat_diffusion": {"source_idx": "center"},
                    "matern_kernel": {"source_idx": "center"},
                    "harmonic": {"source_idx": "extremity", "sink_idx": "far_from_source"}}


def _is_auto(spec):
    if isinstance(spec, str):
        return spec in AUTO_SPECS
    if isinstance(spec, (list, tuple)) and len(spec) and isinstance(spec[0], str):
        return spec[0] in AUTO_SPECS
    return False


def _geo(vertices, faces, idx):
    return compute_geodesic_distance(vertices, faces, [int(idx)], solver="heat")


def intrinsic_center(vertices, faces, n_samples=8, near=None, tol=0.01):
    """
    Intrinsic centre of the surface: the vertex of smallest AVERAGE geodesic distance to the surface (estimated from
    n_samples farthest-point samples, Hilaga et al.). It always lies on the surface and does not depend on the pose,
    unlike the vertex closest to the Euclidean centroid (outside non-convex shapes). Among the vertices within `tol` of
    the minimum, the one closest to `near` is returned (temporal consistency).
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    start = int(np.argmin(np.linalg.norm(v - _area_weighted_center(v, faces), axis=1)))
    D = []
    d = _geo(v, faces, start)
    for _ in range(int(n_samples)):
        s_ = int(np.argmax(d)) if not D else int(np.argmax(np.min(np.stack(D), axis=0)))
        ds = _geo(v, faces, s_); D.append(ds)
        d = ds
    agd = np.mean(np.stack(D), axis=0)
    cand = np.flatnonzero(agd <= agd.min() * (1 + tol) + 1e-12)
    if near is not None and len(cand) > 1:
        return int(cand[np.argmin(np.linalg.norm(v[cand] - near, axis=1))])
    return int(cand[np.argmin(agd[cand])])


def farthest_point(vertices, faces, from_idx, near=None, frac=0.6):
    """
    The geodesically farthest vertex from `from_idx`. With `near` (the transported reference of the previous frame) the
    same EXTREMITY is kept: the vertices with distance ≥ frac · max form one connected region per far part of the shape
    (limbs, tips); the region containing — or closest to — `near` is chosen and its farthest vertex (the tip of that
    part) returned, so the reference neither jumps to another tip nor slides along the limb.
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    d = _geo(v, faces, from_idx)
    if near is None:
        return int(np.argmax(d))
    cand = np.flatnonzero(d >= frac * d.max())
    f = _as_faces(faces)
    keep = np.zeros(len(v), bool); keep[cand] = True
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = e[keep[e[:, 0]] & keep[e[:, 1]]]
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v)))
    _, lab = connected_components(adj, directed=False)
    nearest = cand[np.argmin(np.linalg.norm(v[cand] - near, axis=1))]
    region = cand[lab[cand] == lab[nearest]]
    return int(region[np.argmax(d[region])])


def _transport(prev, vertices, prev_vertices=None, p2p=None, mode="auto"):
    """Position (current frame) of a reference chosen on the previous frame: through the point-to-point map (current
    vertex -> previous vertex) when available, else the previous position itself (aligned frames)."""
    if prev is None:
        return None
    pos = np.asarray(prev["position"], dtype=np.float64)
    if mode == "position":                             # aligned frames, small motion: immune to map flips
        return pos
    if p2p is not None and prev_vertices is not None:
        pv = np.asarray(to_cpu(prev_vertices), dtype=np.float64)
        p2p = np.asarray(p2p, dtype=np.int64)
        if len(p2p) == np.asarray(vertices).shape[0] and p2p.max() < len(pv):
            v = np.asarray(to_cpu(vertices), dtype=np.float64)
            i = int(np.argmin(np.linalg.norm(pv[p2p] - pos, axis=1)))
            # guard: the frames are aligned, so a reference moving more than a quarter of the size between two frames
            # means an unreliable map here (e.g. a symmetric flip): the previous position is used instead
            size = float(np.sqrt(np.prod(np.sort(np.ptp(v, axis=0))[-2:])))
            if mode == "map" or np.linalg.norm(v[i] - pos) <= 0.25 * size:
                return v[i]
    return pos


def resolve_auto_references(method, kwargs, vertices, faces, prev_state=None, prev_vertices=None, p2p=None, n_samples=8,
                            transport=None):
    """
    Replaces 'auto' / 'auto_center' / 'auto_extremity' reference points of a scalar field by vertex indices:
      geodesic (vertex_ref_index), heat_diffusion / matern_kernel (source_idx): 'auto' = intrinsic centre (radial
          fields: protrusions become the far ends); 'auto_extremity' = the farthest point from the centre (fields that
          sweep the shape along its main axis);
      harmonic: source 'auto' = the farthest point from the intrinsic centre, sink 'auto' = the farthest point from the
          source (the two ends of the shape: a fair harmonic function with few critical points, Ni et al. 2004).
    Temporal consistency, as for the automatic landmarks of the functional maps: the reference of the previous frame
    (prev_state) is transported to this frame (point-to-point map, or position) and the criterion is re-applied near
    it, so the reference stays the same part of the shape. Returns (kwargs with indices, state of this frame).
    """
    method = str(method).lower()
    params = REFERENCE_PARAMS.get(method, {})
    out = dict(kwargs or {}); state = {}
    # how the reference of the previous frame is carried over: 'auto' (point-to-point map, ignored when it would move
    # the reference by more than a quarter of the size), 'map' (always the map: large motion, good maps) or 'position'
    # (aligned frames with small motion between them: immune to symmetric flips of the maps)
    transport = transport or out.pop("reference_transport", "auto")
    out.pop("reference_transport", None)
    if not any(_is_auto(out.get(pname)) for pname in params):
        return out, state
    state["_transport"] = transport
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    center = None
    for pname, default_mode in params.items():
        spec = out.get(pname)
        if not _is_auto(spec):
            continue
        tag = spec if isinstance(spec, str) else spec[0]
        mode = {"auto": default_mode, "auto_center": "center", "auto_extremity": "extremity"}[tag]
        near = _transport((prev_state or {}).get(pname), v, prev_vertices, p2p, transport)
        if mode == "center":
            idx = intrinsic_center(v, faces, n_samples, near)
        elif mode == "extremity":
            center = center if center is not None else intrinsic_center(v, faces, n_samples)
            idx = farthest_point(v, faces, center, near)
        else:                                          # far_from_source (harmonic sink)
            src = out.get("source_idx")
            src = int(np.atleast_1d(src)[0]) if src is not None and not _is_auto(src) else intrinsic_center(v, faces, n_samples)
            idx = farthest_point(v, faces, src, near)
        out[pname] = [int(idx)] if pname == "vertex_ref_index" else int(idx)
        state[pname] = {"index": int(idx), "position": v[idx].tolist(), "mode": mode}
    return out, state


def _resolve_vertex_indices(spec, vertices, faces, default):
    """
    Normalizes a user-provided vertex selection into a 1-D int array.

    Accepted: int, list/array of ints, 'mass_center' (or a list whose first element
    is 'mass_center'), or None (-> `default`).
    """
    if spec is None:
        spec = default
    if isinstance(spec, str):
        spec = [spec]
    spec = list(np.atleast_1d(spec)) if not isinstance(spec, list) else spec

    if len(spec) > 0 and isinstance(spec[0], str) and spec[0] in AUTO_SPECS:
        # direct call without temporal context: the criterion on this mesh alone (the pipeline resolves 'auto'
        # beforehand with resolve_auto_references, consistently through time)
        if spec[0] == "auto_extremity":
            return np.array([farthest_point(vertices, faces, intrinsic_center(vertices, faces))])
        return np.array([intrinsic_center(vertices, faces)])
    if len(spec) > 0 and isinstance(spec[0], str):
        if spec[0] != 'mass_center':
            raise ValueError(f"Unknown vertex selector '{spec[0]}'. Use an index, a list of indices, 'mass_center' or 'auto'.")
        v = np.asarray(to_cpu(vertices), dtype=np.float64)
        center = _area_weighted_center(v, faces)
        return np.array([int(np.argmin(np.linalg.norm(v - center, axis=1)))])

    idx = np.asarray(spec, dtype=np.int64).ravel()
    n = np.asarray(vertices).shape[0]
    if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
        raise IndexError(f"Vertex indices {idx} out of range for a mesh with {n} vertices.")
    return idx


def _rank_normalize(x):
    """
    Histogram equalization: maps values to their normalized rank in [0, 1].

    The Reeb graph only depends on the *ordering* of the scalar field, so this is a
    topology-preserving transform. It guarantees that the bins used by
    compute_approx_reeb_graph are equally populated, which is far more robust for
    heavy-tailed fields (curvature, kernels) than uniform bins in value space.
    Ties are broken deterministically by index (stable sort).
    """
    x_gpu = to_gpu(np.asarray(x, dtype=np.float64))
    order = xp.argsort(x_gpu, kind='stable')
    ranks = xp.empty(len(order), dtype=xp.float64)
    ranks[order] = xp.arange(len(order), dtype=xp.float64)
    denom = max(len(order) - 1.0, 1.0)
    return to_cpu(ranks / denom)


def _robust_clip(x, lower_pct=0.5, upper_pct=99.5):
    """Replaces non-finite values and clips extreme outliers (typical for discrete curvature)."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.percentile(x[finite], [lower_pct, upper_pct])
    x = np.where(finite, x, np.median(x[finite]))
    return np.clip(x, lo, hi)


def _sanitize_field(field, num_vertices, method):
    """Final guard: correct shape, float64, finite, non-degenerate."""
    field = np.asarray(to_cpu(field), dtype=np.float64).ravel()
    if field.shape[0] != num_vertices:
        raise ValueError(f"Scalar field '{method}' has {field.shape[0]} values but mesh has {num_vertices} vertices.")
    if not np.all(np.isfinite(field)):
        n_bad = int(np.count_nonzero(~np.isfinite(field)))
        warnings.warn(f"Scalar field '{method}': {n_bad} non-finite values replaced (median / clipped).")
        field = _robust_clip(field, 0.0, 100.0)
    if np.ptp(field) <= 1e-12 * max(1.0, np.abs(field).max()):
        warnings.warn(f"Scalar field '{method}' is (numerically) constant; the Reeb graph will collapse to a single node.")
    return field


# ----------------------------------------------------------------------------- #
#  Scalar fields
# ----------------------------------------------------------------------------- #

def _get_mesh_adjacency(vertices, faces):
    """Sparse symmetric adjacency matrix weighted by Euclidean edge length."""
    v_gpu = to_gpu(np.asarray(to_cpu(vertices), dtype=np.float64))
    f_gpu = to_gpu(_as_faces(faces))

    edges_gpu = xp.vstack([f_gpu[:, [0, 1]], f_gpu[:, [1, 2]], f_gpu[:, [2, 0]]])
    weights_gpu = xp.linalg.norm(v_gpu[edges_gpu[:, 0]] - v_gpu[edges_gpu[:, 1]], axis=1)

    edges = to_cpu(edges_gpu)
    weights = to_cpu(weights_gpu)
    keep = edges[:, 0] != edges[:, 1]                    # drop degenerate edges
    edges, weights = edges[keep], weights[keep]

    n = np.asarray(vertices).shape[0]
    adj = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    return adj.maximum(adj.T)


def compute_geodesic_distance(vertices, faces, vertex_ref_index, solver="heat"):
    """
    Geodesic distance from a *set* of source vertices (distance to the nearest source).

    solver='heat'     : pyFM heat method (smooth, fast). Falls back to Dijkstra on failure
                        or if the result is not finite.
    solver='dijkstra' : exact shortest path on the edge graph (always finite, slightly
                        over-estimates true geodesics, but perfectly stable across frames).
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    refs = np.atleast_1d(np.asarray(vertex_ref_index, dtype=np.int64))
    n = v.shape[0]

    if solver == "heat":
        try:
            d = np.asarray(TriMesh(v, f).geod_from(refs.tolist() if refs.size > 1 else int(refs[0])))
            # pyFM returns (n,) for one source or (n, k) / (k, n) for several -> distance to the set
            if d.ndim == 2:
                d = d.min(axis=1) if d.shape[0] == n else d.min(axis=0)
            d = d.ravel()
            if d.shape[0] == n and np.all(np.isfinite(d)):
                return d
            warnings.warn("Heat-method geodesics returned an invalid result; falling back to Dijkstra.")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Heat-method geodesics failed ({exc}); falling back to Dijkstra.")

    adj = _get_mesh_adjacency(v, f)
    d = dijkstra(adj, directed=False, indices=refs, min_only=True)
    if np.isinf(d).any():
        # Disconnected components: give unreachable vertices a distance beyond the max.
        finite_max = d[np.isfinite(d)].max() if np.isfinite(d).any() else 0.0
        warnings.warn("Mesh has vertices unreachable from the geodesic source(s); assigning max+1.")
        d = np.where(np.isinf(d), finite_max + 1.0, d)
    return d


def compute_harmonic_field(trimesh_obj, source_idx, sink_idx):
    """
    Solves Δf = 0 with Dirichlet conditions f(source)=1, f(sink)=0 using exact
    constraint elimination:   W_II f_I = -W_IB f_B.
    This replaces the penalty formulation (1e8 on the diagonal), which is badly
    conditioned for the cotangent matrix and silently inaccurate on fine meshes.
    """
    W = csr_matrix(trimesh_obj.W)
    n = W.shape[0]
    src = np.atleast_1d(np.asarray(source_idx, dtype=np.int64))
    snk = np.atleast_1d(np.asarray(sink_idx, dtype=np.int64))
    if np.intersect1d(src, snk).size:
        raise ValueError("Harmonic field: source and sink sets overlap.")

    f = np.zeros(n)
    f[src] = 1.0
    boundary = np.union1d(src, snk)
    interior = np.setdiff1d(np.arange(n), boundary)

    W_II = W[interior][:, interior]
    W_IB = W[interior][:, boundary]
    rhs = -W_IB @ f[boundary]

    # A component without any constrained vertex makes W_II singular -> tiny Tikhonov term.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f[interior] = splinalg.spsolve(W_II.tocsc(), rhs)
            if not np.all(np.isfinite(f)):
                raise np.linalg.LinAlgError
        except Exception:  # noqa: BLE001
            eps = 1e-8 * (abs(W_II.diagonal()).mean() + 1e-12)
            f[interior] = splinalg.spsolve((W_II + eps * identity(W_II.shape[0])).tocsc(), rhs)
    return np.clip(f, 0.0, 1.0)     # discrete maximum principle; clip only removes round-off


def _spectral_basis(trimesh_obj):
    if trimesh_obj is None or getattr(trimesh_obj, "eigenvalues", None) is None:
        raise ValueError("trimesh_obj with a computed LB eigendecomposition (mesh.process(k=...)) is required.")
    evals = np.asarray(trimesh_obj.eigenvalues, dtype=np.float64)
    evecs = np.asarray(trimesh_obj.eigenvectors, dtype=np.float64)
    return evals, evecs


def _auto_heat_time(evals):
    """Geometric mean of the two extreme resolvable diffusion times, 1/λ_1 and 1/λ_K."""
    pos = evals[evals > 1e-12]
    if pos.size < 2:
        return 1.0
    return float(1.0 / np.sqrt(pos[0] * pos[-1]))


def compute_heat_diffusion(trimesh_obj, source_idx, t="auto"):
    """
    Heat kernel k_t(source, x) = Σ_k e^{-t λ_k} φ_k(source) φ_k(x), summed over the source set.

    t='auto' picks a scale-aware time (see _auto_heat_time). A fixed t (the old default
    was 10.0) is *not* mesh-independent: for un-normalized meshes with small eigenvalues
    it yields a nearly constant field, for large eigenvalues it collapses to a delta.
    """
    evals, evecs = _spectral_basis(trimesh_obj)
    src = np.atleast_1d(np.asarray(source_idx, dtype=np.int64))
    if t == "auto" or t is None:
        t = _auto_heat_time(evals)

    evals_gpu = to_gpu(evals)
    evecs_gpu = to_gpu(evecs)
    phi_src = evecs_gpu[to_gpu(src), :].sum(axis=0)             # (K,)
    weights = xp.exp(-float(t) * evals_gpu) * phi_src           # (K,)
    return to_cpu(evecs_gpu @ weights)


def compute_matern_field(trimesh_obj, source_idx, nu=1.5, lengthscale=1.0):
    """
    Matérn kernel on the surface via its Karhunen–Loève expansion in the LB basis:
        k(source, x) = Σ_k (2ν/ℓ² + λ_k)^{-(ν + d/2)} φ_k(source) φ_k(x),   d = 2.
    This is exactly what geometric_kernels computes internally; using the pyFM basis
    directly avoids a second eigendecomposition and the fragile private-attribute patching.
    """
    evals, evecs = _spectral_basis(trimesh_obj)
    src = np.atleast_1d(np.asarray(source_idx, dtype=np.int64))
    if np.isinf(nu):
        spectrum = np.exp(-0.5 * lengthscale ** 2 * evals)          # squared-exponential limit
    else:
        spectrum = (2.0 * nu / lengthscale ** 2 + evals) ** (-(nu + 1.0))

    evecs_gpu = to_gpu(evecs)
    phi_src = evecs_gpu[to_gpu(src), :].sum(axis=0)
    return to_cpu(evecs_gpu @ (to_gpu(spectrum) * phi_src))


def compute_curvature_field(vertices, faces, method):
    """
    Discrete curvatures via VTK (pyvista). VTK's estimates are noisy and can be ±inf on
    degenerate triangles / boundaries, so the result is robustly clipped. Note that these
    are scale dependent (units of 1/length), so compare frames only at a common scale.
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    faces_pv = np.hstack([np.full((f.shape[0], 1), 3, dtype=np.int64), f]).ravel()
    mesh_pv = pv.PolyData(v, faces_pv)

    H = _robust_clip(mesh_pv.curvature(curv_type="mean"))
    if method == "mean_curvature":
        return H
    K = _robust_clip(mesh_pv.curvature(curv_type="Gaussian"))
    if method == "gaussian_curvature":
        return K

    # Principal curvatures: κ1,2 = H ± sqrt(H² - K)   (K <= H² for a real surface)
    disc = np.sqrt(np.maximum(H ** 2 - K, 0.0))
    if method == "shape_index":
        # Koenderink shape index in [-1, 1]; undefined (0) on umbilical points where κ1 == κ2
        with np.errstate(divide="ignore", invalid="ignore"):
            S = (2.0 / np.pi) * np.arctan2(H, disc)
        return np.nan_to_num(S, nan=0.0)
    if method == "curvedness":
        return np.sqrt(np.maximum(2.0 * H ** 2 - K, 0.0))
    raise ValueError(method)


def compute_vertex_normals(vertices, faces):
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])   # area-weighted face normals
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(norms, 1e-12)


def compute_normal_displacement(vertices, faces, prev_vertices, p2p, signed=True):
    """
    Normal component of the displacement between the previous frame (mapped through the
    point-to-point map p2p: current vertex j -> previous vertex p2p[j]) and the current frame.
    Uses the project's flow decomposition if importable, otherwise a direct projection
    on area-weighted vertex normals.
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    pv_prev = np.asarray(to_cpu(prev_vertices), dtype=np.float64)
    p2p = np.asarray(to_cpu(p2p), dtype=np.int64).ravel()
    if p2p.shape[0] != v.shape[0]:
        raise ValueError(f"p2p has {p2p.shape[0]} entries but the current mesh has {v.shape[0]} vertices "
                         "(expected the mesh2 -> mesh1 map, p2p_21).")
    displacements = v - pv_prev[p2p]

    try:
        from PynamicMesh.core.physic_model import compute_flow_decomposition
        norm_mag, _ = compute_flow_decomposition(v, _as_faces(faces), displacements)
        return np.asarray(norm_mag, dtype=np.float64).ravel()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"compute_flow_decomposition unavailable ({exc}); using direct normal projection.")
        normals = compute_vertex_normals(v, faces)
        comp = np.einsum("ij,ij->i", displacements, normals)
        return comp if signed else np.abs(comp)


def get_scalar_field(vertices, faces, method="z", prev_vertices=None, p2p=None, trimesh_obj=None, **kwargs):
    """
    Computes a per-vertex scalar field f: V -> R used to build the Reeb graph.

    Common kwargs
    -------------
    equalize_histogram : bool   rank-normalize the field to [0,1] (topology preserving).
                                Default True for 'heat_diffusion'/'matern_kernel', False otherwise.
    Method-specific kwargs are documented inline.
    """
    method = method.lower()
    v_cpu = np.asarray(to_cpu(vertices), dtype=np.float64)
    num_vertices = v_cpu.shape[0]
    v_gpu = to_gpu(v_cpu)
    equalize_default = method in ("heat_diffusion", "matern_kernel")

    if method in ("x", "y", "z"):
        field = v_gpu[:, "xyz".index(method)]

    elif method in ("signed_dist_x", "signed_dist_y", "signed_dist_z"):
        axis = "xyz".index(method[-1])
        field = v_gpu[:, axis] - v_gpu[:, axis].mean()

    elif method in ("dist_x_axis", "dist_y_axis", "dist_z_axis"):
        # Distance to the axis (parallel to x, y or z) passing through the area-weighted center.
        axis = "xyz".index(method[5])
        others = [a for a in range(3) if a != axis]
        center = to_gpu(_area_weighted_center(v_cpu, faces))
        field = xp.linalg.norm((v_gpu - center)[:, others], axis=1)

    elif method == "dist_centroid":
        # Area-weighted center: the plain vertex mean is biased by vertex density.
        center = to_gpu(_area_weighted_center(v_cpu, faces))
        field = xp.linalg.norm(v_gpu - center, axis=1)

    elif method in ("geodesic", "mass_center_geodesic"):
        # kwargs: vertex_ref_index (int | list | 'mass_center'), geodesic_solver ('heat' | 'dijkstra')
        spec = ['mass_center'] if method == "mass_center_geodesic" else kwargs.get("vertex_ref_index", [0])
        refs = _resolve_vertex_indices(spec, v_cpu, faces, default=[0])
        field = compute_geodesic_distance(v_cpu, faces, refs, solver=kwargs.get("geodesic_solver", "heat"))

    elif method in ("mean_curvature", "gaussian_curvature", "shape_index", "curvedness"):
        field = compute_curvature_field(v_cpu, faces, method)

    elif method == "normal_displacement":
        if prev_vertices is None or p2p is None:
            warnings.warn("normal_displacement requested without prev_vertices/p2p (first frame?); returning zeros.")
            field = np.zeros(num_vertices)
        else:
            field = compute_normal_displacement(v_cpu, faces, prev_vertices, p2p, signed=kwargs.get("signed", True))

    elif method.startswith("lb_eigen_"):
        evals, evecs = _spectral_basis(trimesh_obj)
        idx = int(method.split("_")[-1])
        if idx >= evecs.shape[1]:
            raise ValueError(f"{method}: only {evecs.shape[1]} eigenfunctions were computed (mesh.process(k=...)).")
        if idx == 0:
            warnings.warn("lb_eigen_0 is the constant eigenfunction; the Reeb graph will be trivial. Use idx >= 1.")
        # Eigenvector sign is arbitrary between frames; the Reeb graph is invariant under f -> -f,
        # but we fix the sign (positive at the vertex of largest |value|) so bins are comparable.
        field = evecs[:, idx].copy()
        field *= np.sign(field[np.argmax(np.abs(field))]) or 1.0

    elif method == "heat_diffusion":
        # kwargs: source_idx (int | list | 'mass_center'), t (float | 'auto')
        src = _resolve_vertex_indices(kwargs.get("source_idx", 0), v_cpu, faces, default=[0])
        field = compute_heat_diffusion(trimesh_obj, src, t=kwargs.get("t", "auto"))

    elif method == "matern_kernel":
        # kwargs: source_idx, nu, lengthscale
        src = _resolve_vertex_indices(kwargs.get("source_idx", 0), v_cpu, faces, default=[0])
        field = compute_matern_field(trimesh_obj, src, nu=kwargs.get("nu", 1.5), lengthscale=kwargs.get("lengthscale", 1.0))

    elif method == "harmonic":
        # kwargs: source_idx / sink_idx (int | list | 'mass_center'); defaults: lowest / highest z
        if trimesh_obj is None or getattr(trimesh_obj, "W", None) is None:
            raise ValueError("trimesh_obj with stiffness matrix W is required for harmonic fields.")
        src = _resolve_vertex_indices(kwargs.get("source_idx"), v_cpu, faces, default=[int(np.argmin(v_cpu[:, 2]))])
        snk = _resolve_vertex_indices(kwargs.get("sink_idx"), v_cpu, faces, default=[int(np.argmax(v_cpu[:, 2]))])
        field = compute_harmonic_field(trimesh_obj, src, snk)

    elif method == "multi_pca":
        # kwargs: fields (list of method names). Each field is robust-standardized before PCA.
        fields = kwargs.get("fields", ["z", "mean_curvature", "gaussian_curvature"])
        sub_kwargs = {k: val for k, val in kwargs.items() if k not in ("fields", "equalize_histogram")}
        stacked = []
        for name in fields:
            val = get_scalar_field(v_cpu, faces, method=name, prev_vertices=prev_vertices, p2p=p2p,
                                   trimesh_obj=trimesh_obj, equalize_histogram=False, **sub_kwargs)
            med = np.median(val)
            mad = np.median(np.abs(val - med)) * 1.4826
            stacked.append((val - med) / (mad if mad > 1e-12 else (np.std(val) + 1e-8)))
        pca = PCA(n_components=1)
        field = pca.fit_transform(np.vstack(stacked).T).ravel()
        field *= np.sign(field[np.argmax(np.abs(field))]) or 1.0      # deterministic sign

    else:
        raise ValueError(f"Unknown scalar field method: {method}")

    field = _sanitize_field(field, num_vertices, method)
    if kwargs.get("equalize_histogram", equalize_default):
        field = _rank_normalize(field)
    return field


# ----------------------------------------------------------------------------- #
#  Reeb graph
# ----------------------------------------------------------------------------- #

def _bin_scalar_field(sf, num_bins):
    """Uniform bins over [min, max]; robust to constant fields and floating point at the ends."""
    sf = np.asarray(sf, dtype=np.float64).ravel()
    f_min, f_max = sf.min(), sf.max()
    if not np.isfinite(f_min) or not np.isfinite(f_max):
        raise ValueError("Scalar field contains non-finite values; sanitize it before building the Reeb graph.")
    if f_max - f_min <= 0.0:
        return np.zeros(sf.shape[0], dtype=np.int64), np.array([f_min, f_min])
    bin_edges = np.linspace(f_min, f_max, num_bins + 1)
    bin_idx = np.floor((sf - f_min) / (f_max - f_min) * num_bins).astype(np.int64)
    return np.clip(bin_idx, 0, num_bins - 1), bin_edges


def _bin_with_edges(sf, edges):
    """Bin index of every value for explicit (possibly non-uniform) increasing bin edges."""
    idx = np.searchsorted(edges, sf, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2).astype(np.int64), np.asarray(edges, dtype=np.float64)


def compute_approx_reeb_graph(vertices, faces, scalar_field, num_bins=20, bin_edges=None):
    """
    Discrete Reeb graph of (mesh, scalar_field) via slab decomposition.

    The range of f is cut into `num_bins` slabs. For each triangle and each slab it
    crosses, the piece "triangle ∩ slab" is one cell (it is convex, hence connected).
    Cells of neighbouring triangles are merged when they share a segment of the common
    edge inside that slab; cells around a vertex are merged through the vertex. Connected
    groups of cells are the nodes (one per connected component of f^{-1}(slab)); two nodes
    are joined when a triangle spans both consecutive slabs.

    Compared with binning *vertices* only (previous implementation), this handles mesh
    edges that jump over several slabs correctly: they no longer create shortcut edges
    between non-adjacent slabs, which produced hundreds of spurious cycles and inflated
    Betti-1 as soon as num_bins exceeded the mesh resolution. Complexity is O(F·s) where
    s is the mean number of slabs a triangle spans (≈1 for a well-resolved mesh).

    Node attributes: pos (3,), bin (int), f_value (slab center), n_vertices, vertices (array).
    bin_edges: explicit, increasing slab edges (non-uniform slabs, used by the topological control); default
    `num_bins` uniform slabs over [min f, max f].
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    f = _as_faces(faces)
    sf = np.asarray(to_cpu(scalar_field), dtype=np.float64).ravel()
    if sf.shape[0] != v.shape[0]:
        raise ValueError("scalar_field must have one value per vertex.")
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")

    # Drop degenerate triangles (repeated vertex index).
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    F = f.shape[0]
    graph = nx.Graph()
    if F == 0:
        return graph

    if bin_edges is None:
        bin_idx, bin_edges = _bin_scalar_field(sf, num_bins)
    else:
        bin_idx, bin_edges = _bin_with_edges(sf, bin_edges)
    fb = bin_idx[f]                                     # (F, 3) slab of each corner
    fb_min, fb_max = fb.min(axis=1), fb.max(axis=1)
    span = fb_max - fb_min + 1                          # slabs crossed by each triangle
    cell_offset = np.concatenate([[0], np.cumsum(span)])
    n_cells = int(cell_offset[-1])

    def cell_id(face_ids, slabs):
        return cell_offset[face_ids] + (slabs - fb_min[face_ids])

    def expand(starts, counts):
        """For groups with given start values and counts, returns (group_index, start+offset)."""
        total = int(counts.sum())
        grp = np.repeat(np.arange(len(counts)), counts)
        first = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        return grp, np.repeat(starts, counts) + (np.arange(total) - first)

    rows, cols = [], []

    # 1) Merge cells of triangles that share an edge, in every slab the edge crosses.
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    e_face = np.tile(np.arange(F), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, e_face = e[order], e_face[order]
    same = np.all(e[1:] == e[:-1], axis=1)              # consecutive rows = same undirected edge
    p = np.flatnonzero(same)
    if p.size:
        fa, fb_ = e_face[p], e_face[p + 1]
        lo = np.minimum(bin_idx[e[p, 0]], bin_idx[e[p, 1]])
        hi = np.maximum(bin_idx[e[p, 0]], bin_idx[e[p, 1]])
        grp, slab = expand(lo, hi - lo + 1)
        rows.append(cell_id(fa[grp], slab))
        cols.append(cell_id(fb_[grp], slab))

    # 2) Merge all cells around a vertex in the vertex's own slab (covers pinched vertices).
    corner_vertex = f.ravel()
    corner_face = np.repeat(np.arange(F), 3)
    corner_cell = cell_id(corner_face, bin_idx[corner_vertex])
    order = np.argsort(corner_vertex, kind="stable")
    cv, cc = corner_vertex[order], corner_cell[order]
    same_v = cv[1:] == cv[:-1]
    rows.append(cc[:-1][same_v])
    cols.append(cc[1:][same_v])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    adj = coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n_cells, n_cells))
    n_nodes, cell_label = connected_components(adj, directed=False)

    # Slab of every cell, hence of every node.
    _, cell_slab = expand(fb_min, span)
    node_slab = np.empty(n_nodes, dtype=np.int64)
    node_slab[cell_label] = cell_slab

    # Relabel nodes so ids increase with slab (deterministic, nicer for downstream plots).
    relabel = np.empty(n_nodes, dtype=np.int64)
    relabel[np.lexsort((np.arange(n_nodes), node_slab))] = np.arange(n_nodes)
    cell_label = relabel[cell_label]
    node_slab = node_slab[np.argsort(relabel)]

    # 3) Reeb edges: consecutive slabs inside the same triangle.
    grp, slab = expand(fb_min, span - 1)
    if grp.size:
        a = cell_label[cell_id(grp, slab)]
        b = cell_label[cell_id(grp, slab + 1)]
        reeb_edges = np.unique(np.sort(np.stack([a, b], axis=1), axis=1), axis=0)
    else:
        reeb_edges = np.empty((0, 2), dtype=np.int64)

    # 4) Node geometry: vertex -> node (well defined thanks to step 2).
    vertex_node = np.full(v.shape[0], -1, dtype=np.int64)
    vertex_node[corner_vertex] = cell_label[corner_cell]
    has_node = vertex_node >= 0
    pos_sum = np.zeros((n_nodes, 3))
    counts = np.zeros(n_nodes)
    np.add.at(pos_sum, vertex_node[has_node], v[has_node])
    np.add.at(counts, vertex_node[has_node], 1.0)

    # Nodes without vertices (slabs cutting through a triangle strictly between its corners):
    # place them on the iso-line f = slab center, interpolated along the triangle edges.
    empty = counts == 0
    if empty.any():
        f_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:]) if len(bin_edges) > 2 else np.array([bin_edges[0]])
        grp_c, slab_c = expand(fb_min, span)             # (cell -> face, cell -> slab)
        lab_c = cell_label[np.arange(n_cells)]
        sel = empty[lab_c]
        fc, sc, lc = grp_c[sel], slab_c[sel], lab_c[sel]
        target = f_mid[sc]
        for k in range(3):
            i0, i1 = f[fc, k], f[fc, (k + 1) % 3]
            f0, f1 = sf[i0], sf[i1]
            cross = ((f0 <= target) & (target <= f1)) | ((f1 <= target) & (target <= f0))
            cross &= np.abs(f1 - f0) > 1e-15
            tpar = np.where(cross, (target - f0) / np.where(cross, f1 - f0, 1.0), 0.0)
            pts = v[i0] + tpar[:, None] * (v[i1] - v[i0])
            np.add.at(pos_sum, lc[cross], pts[cross])
            np.add.at(counts, lc[cross], 1.0)
        still_empty = counts == 0                        # numerical corner case: use face centroids
        if still_empty.any():
            cen = v[f[fc]].mean(axis=1)
            m = still_empty[lc]
            np.add.at(pos_sum, lc[m], cen[m])
            np.add.at(counts, lc[m], 1.0)

    pos = pos_sum / counts[:, None]
    n_vertices = np.bincount(vertex_node[has_node], minlength=n_nodes)
    f_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) if len(bin_edges) > 2 else np.array([bin_edges[0]])
    vert_order = np.argsort(vertex_node[has_node], kind="stable")
    vert_sorted = np.flatnonzero(has_node)[vert_order]
    vert_split = np.split(vert_sorted, np.cumsum(n_vertices)[:-1])

    for node in range(n_nodes):
        graph.add_node(node, pos=pos[node], bin=int(node_slab[node]), f_value=float(f_centers[node_slab[node]]),
                       n_vertices=int(n_vertices[node]), vertices=vert_split[node])
    graph.add_edges_from(map(tuple, reeb_edges.tolist()))
    return graph


# --------------------------------------------------------------------------- #
#  Topological control (Extended Reeb Graph, Biasotti et al.) and node types
# --------------------------------------------------------------------------- #

def surface_topology(faces, n_vertices=None):
    """(components c, Euler characteristic χ, boundary loops b, genus g = (2c − χ − b)/2) of a triangle mesh
    (vertices not referenced by a face are ignored; orientable surfaces assumed)."""
    f = _as_faces(faces)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    used = np.unique(f)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    eu, cnt = np.unique(e, axis=0, return_counts=True)
    n = int(f.max()) + 1 if f.size else 0
    adj = coo_matrix((np.ones(len(eu)), (eu[:, 0], eu[:, 1])), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    c = len(np.unique(lab[used]))
    chi = len(used) - len(eu) + len(f)
    be = eu[cnt == 1]
    b = 0
    if len(be):
        badj = coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(n, n))
        _, bl = connected_components(badj, directed=False)
        b = len(np.unique(bl[np.unique(be)]))
    return int(c), int(chi), int(b), int(max(0, (2 * c - chi - b) // 2))


def _region_genus(faces, vertex_node, n_nodes):
    """Genus of the vertex-induced sub-surface of every node region (Euler formula χ = V − E + F, boundary loops
    b, components c: g = (2c − χ − b)/2). A region with g > 0 contains a handle that a single slab cannot show."""
    f = faces
    vn = vertex_node
    V = np.bincount(vn[vn >= 0], minlength=n_nodes)
    ff = f[(vn[f[:, 0]] == vn[f[:, 1]]) & (vn[f[:, 1]] == vn[f[:, 2]]) & (vn[f[:, 0]] >= 0)]
    F = np.bincount(vn[ff[:, 0]], minlength=n_nodes) if len(ff) else np.zeros(n_nodes, int)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    e = e[(vn[e[:, 0]] == vn[e[:, 1]]) & (vn[e[:, 0]] >= 0)]
    E = np.bincount(vn[e[:, 0]], minlength=n_nodes)
    n = len(vn)
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    has = vn >= 0
    comp = np.zeros(n_nodes, int)
    pairs = np.unique(np.stack([vn[has], lab[has]], 1), axis=0)
    np.add.at(comp, pairs[:, 0], 1)
    b = np.zeros(n_nodes, int)
    if len(ff):
        fe = np.sort(np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]]), axis=1)
        feu, fc = np.unique(fe, axis=0, return_counts=True)
        be = feu[fc == 1]
        if len(be):
            # boundary LOOPS = cycle rank of the boundary-edge graph (edges − vertices + components): loops that touch
            # at a vertex (a pinched region, e.g. where a level set nearly closes on itself) form one connected
            # component but are several loops — counting components would invent a handle
            badj = coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(n, n))
            _, bl = connected_components(badj, directed=False)
            bv = np.unique(be)
            Eb = np.bincount(vn[be[:, 0]], minlength=n_nodes)
            Vb = np.bincount(vn[bv], minlength=n_nodes)
            bp = np.unique(np.stack([vn[bv], bl[bv]], 1), axis=0)
            Cb = np.bincount(bp[:, 0], minlength=n_nodes)
            b = Eb - Vb + Cb
    chi = V - E + F
    return np.maximum(0, (2 * comp - chi - b) // 2)


def _multi_interface_bins(faces, vertex_node, graph):
    """Slabs whose adjacent regions touch along SEVERAL separate interfaces: a simple graph keeps one edge between
    two nodes, so the loop formed by the parallel connections would be lost; both slabs are refined."""
    f = faces; vn = vertex_node
    a = vn[f]
    ok = (a >= 0).all(axis=1)
    f, a = f[ok], a[ok]
    lo, hi = a.min(axis=1), a.max(axis=1)
    cross = (lo != hi) & (((a == lo[:, None]) | (a == hi[:, None])).all(axis=1))   # faces between two regions
    if not cross.any():
        return set()
    fc, lc, hc = f[cross], lo[cross], hi[cross]
    bad = set()
    key = lc.astype(np.int64) * (int(vn.max()) + 2) + hc
    for k in np.unique(key):
        sel = np.flatnonzero(key == k)
        if len(sel) < 2:
            continue
        sub = fc[sel]
        # faces of the interface connected through shared vertices
        n = int(sub.max()) + 1
        fv = coo_matrix((np.ones(sub.size), (np.repeat(np.arange(len(sub)), 3), sub.ravel())), shape=(len(sub), n)).tocsr()
        adj = fv @ fv.T
        ncomp, _ = connected_components(adj, directed=False)
        if ncomp > 1:
            bad.update({int(graph.nodes[int(lc[sel[0]])]["bin"]), int(graph.nodes[int(hc[sel[0]])]["bin"])})
    return bad


def reeb_node_types(graph):
    """Type of every node from its neighbours above / below (Reeb's degree-index relations): 'minimum',
    'maximum', 'split saddle' (one arc below, several above), 'merge saddle' (several below, one above),
    'multiple saddle' (several on both sides, degenerate), 'regular' (one below, one above)."""
    for v in graph.nodes:
        b = graph.nodes[v].get("bin", 0)
        up = sum(1 for w in graph.neighbors(v) if graph.nodes[w].get("bin", 0) > b)
        down = sum(1 for w in graph.neighbors(v) if graph.nodes[w].get("bin", 0) < b)
        if up == 0 and down == 0:
            t = "isolated"
        elif up == 0:
            t = "maximum"
        elif down == 0:
            t = "minimum"
        elif up == 1 and down == 1:
            t = "regular"
        elif down == 1:
            t = "split saddle"
        elif up == 1:
            t = "merge saddle"
        else:
            t = "multiple saddle"
        graph.nodes[v]["node_type"] = t
    return graph


def annotate_reeb_graph(G, faces, refinements=0, bin_edges=None):
    """Adds (never changes the graph) the node types and the genus–loop diagnostic of Biasotti et al. (Table 1):
    β1(K) = g without boundary, g ≤ β1(K) ≤ 2g + b − c with b boundary loops."""
    f_all = _as_faces(faces)
    f_all = f_all[(f_all[:, 0] != f_all[:, 1]) & (f_all[:, 1] != f_all[:, 2]) & (f_all[:, 0] != f_all[:, 2])]
    c, chi, b, g = surface_topology(f_all)
    cK = nx.number_connected_components(G) if G.number_of_nodes() else 0
    beta1 = G.number_of_edges() - G.number_of_nodes() + cK
    lo, hi = (g, g) if b == 0 else (g, 2 * g + b - c)
    n_slabs = (len(bin_edges) - 1) if bin_edges is not None else (len({d.get("bin") for _, d in G.nodes(data=True)}) or 0)
    G.graph.update({"refinements": int(refinements), "n_slabs": int(n_slabs),
                    "surface_genus": int(g), "surface_components": int(c), "boundary_loops": int(b),
                    "beta1_graph": int(beta1), "beta1_expected": (int(lo), int(hi)),
                    "genus_consistent": bool(lo <= beta1 <= hi),
                    # handles of the surface the graph does not represent (usually tiny handles — self-contacts or
                    # defects of the mesh — rather than real tunnels of the object)
                    "hidden_handles": int(max(0, lo - beta1))})
    if bin_edges is not None:
        G.graph["bin_edges"] = np.asarray(bin_edges)
    return reeb_node_types(G)


def compute_reeb_graph_controlled(vertices, faces, scalar_field, num_bins=20, max_refine=6, min_rel_width=1e-4):
    """
    Reeb graph with TOPOLOGICAL CONTROL (Extended Reeb Graph, Biasotti et al. 2008): after the slab decomposition,
    the genus of every slab region is checked with the Euler formula; a region containing a handle (that no slab
    boundary cuts, so the graph would miss its loop) gets its slab split in two, and the graph is rebuilt — only
    there, up to `max_refine` times. The graph then also carries the genus–loop diagnostic of Biasotti et al.
    (Table 1): for an orientable surface without boundary β1(K) = g, with b boundary loops g ≤ β1(K) ≤ 2g + b − c.
    Graph attributes: bin_edges, refinements, surface_genus, surface_components, boundary_loops, beta1_graph,
    beta1_expected (lo, hi), genus_consistent; node attribute node_type.
    """
    sf = np.asarray(to_cpu(scalar_field), dtype=np.float64).ravel()
    f_all = _as_faces(faces)
    f_all = f_all[(f_all[:, 0] != f_all[:, 1]) & (f_all[:, 1] != f_all[:, 2]) & (f_all[:, 0] != f_all[:, 2])]
    _, edges = _bin_scalar_field(sf, num_bins)
    edges = np.asarray(edges, dtype=np.float64)
    width0 = max(edges[-1] - edges[0], 1e-300)
    refinements = 0
    G = compute_approx_reeb_graph(vertices, faces, sf, num_bins=num_bins)
    c_s, _, b_s, g_s = surface_topology(f_all)
    lo_s = g_s                                          # Biasotti et al., Table 1: at least g loops
    for _ in range(int(max_refine)):
        if G.number_of_nodes() == 0:
            break
        # the graph already has the loops of the surface (always the case for genus 0: no handle can hide), so there
        # is nothing to control — the local tests only LOCATE missing loops, they never trigger a refinement
        if G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G) >= lo_s:
            break
        vn = np.full(len(sf), -1, dtype=np.int64)
        for node, d in G.nodes(data=True):
            vn[np.asarray(d.get("vertices", []), dtype=np.int64)] = node
        g_reg = _region_genus(f_all, vn, G.number_of_nodes())
        bad = {int(G.nodes[nd]["bin"]) for nd in np.flatnonzero(g_reg > 0)}       # handle inside one slab
        bad |= _multi_interface_bins(f_all, vn, G)                                 # loop through parallel interfaces
        bad_bins = sorted(bad)                     # local refinement only: the slab count stays comparable
        bad_bins = [k for k in bad_bins if (edges[k + 1] - edges[k]) > 2 * min_rel_width * width0]
        if not bad_bins:
            break
        mids = [0.5 * (edges[k] + edges[k + 1]) for k in bad_bins]
        edges = np.sort(np.concatenate([edges, mids]))
        refinements += 1
        G = compute_approx_reeb_graph(vertices, faces, sf, bin_edges=edges)
    return annotate_reeb_graph(G, f_all, refinements=refinements, bin_edges=edges)


def create_reeb_polydata(graph):
    if len(graph.nodes) == 0:
        return pv.PolyData()

    nodes = list(graph.nodes(data=True))
    node_map = {n: i for i, (n, data) in enumerate(nodes)}
    pts = np.array([data['pos'] for n, data in nodes])

    lines = []
    for u, v in graph.edges():
        lines.extend([2, node_map[u], node_map[v]])

    mesh = pv.PolyData(pts)
    if lines:
        mesh.lines = np.array(lines)
    # Node attributes for colouring in the viewers (legacy / manually added nodes -> bin / 0).
    mesh.point_data['f_value'] = np.array([float(data.get('f_value', data.get('bin', np.nan))) for _, data in nodes])
    mesh.point_data['bin'] = np.array([int(data.get('bin', 0)) for _, data in nodes])
    mesh.point_data['n_vertices'] = np.array([int(data.get('n_vertices', 0)) for _, data in nodes])
    mesh.point_data['degree'] = np.array([graph.degree(n) for n, _ in nodes])
    return mesh