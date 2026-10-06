import os
import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm.auto import tqdm
from scipy.stats import wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots


try:
    import cupy as xp
    GPU_AVAILABLE = True
    if __import__("multiprocessing").current_process().name == "MainProcess":            # worker processes import this module again: say it once
        print("[INFO] CuPy detected. Utilizing GPU for Spectral computations.")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    if __import__("multiprocessing").current_process().name == "MainProcess":            # worker processes import this module again: say it once
        print("[INFO] CuPy not found. Defaulting to CPU (NumPy).")


def to_cpu(arr):
    """Safely moves a CuPy array/scalar back to the CPU."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr


def _frame_number(path):
    """Integer frame id from names like 'Reeb_T0012.pkl' / 'Reeb_T12.pkl' (name order as fallback)."""
    m = re.search(r'T(\d+)', Path(path).stem)
    return int(m.group(1)) if m else float('inf')


def node_function_value(data):
    """
    Reeb function value of a node.

    Nodes produced by compute_approx_reeb_graph store 'f_value' (center of the slab in the
    units of the scalar field). Older pickles only have 'bin'. The z coordinate of 'pos' is
    used as a last resort only: it equals the Reeb function solely when reeb_scalar == 'z'.
    """
    if 'f_value' in data:
        return float(data['f_value'])
    if 'bin' in data:
        return float(data['bin'])
    if 'pos' in data and len(data['pos']) >= 3:
        return float(data['pos'][2])
    return 0.0


def _sorted_linf(a, b):
    """L-infinity distance between the sorted values of two sets (the shorter padded with its last value)."""
    a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    a = np.pad(a, (0, max_len - len(a)), mode='edge')
    b = np.pad(b, (0, max_len - len(b)), mode='edge')
    return float(np.max(np.abs(a - b)))


def labeled_interleaving_proxy(g1, g2, label_key='label'):
    """
    Labelled interleaving proxy used by 'labeled_interleaving_distance': nodes grouped by `label_key`, sorted-value
    L-infinity per group, maximum over groups; a group missing in one graph is compared with [0.0].
    """
    def grouped(G):
        groups = {}
        for n, data in G.nodes(data=True):
            groups.setdefault(str(data.get(label_key, 'unlabeled')), []).append(node_function_value(data))
        return groups

    g1_groups, g2_groups = grouped(g1), grouped(g2)
    all_labels = set(g1_groups) | set(g2_groups)
    if not all_labels:
        return 0.0
    return max(_sorted_linf(g1_groups.get(label, [0.0]), g2_groups.get(label, [0.0])) for label in all_labels)


# Origin-separated comparison (Morse–Smale graphs with the convex-hull protrusion fusion only)
ORIGIN_GROUPS = ("maximum_ms", "maximum_hull", "saddle_ms", "saddle_hull", "minimum_ms", "center")
ORIGIN_MS_GROUPS = ("maximum_ms", "saddle_ms", "minimum_ms", "center")
ORIGIN_HULL_GROUPS = ("maximum_hull", "saddle_hull")
ORIGIN_COLORS = {"maximum_ms": "red", "maximum_hull": "deepskyblue", "saddle_ms": "limegreen",
                 "saddle_hull": "teal", "minimum_ms": "royalblue", "center": "goldenrod"}


def node_origin_group(data):
    """'<type>_<origin>' of a Morse–Smale graph node (SMComplex.ms_star_graph 'origin_label'); None if absent."""
    if data.get('type') == 'center':
        return 'center'
    lab = data.get('origin_label')
    if lab is None and 'origin' in data and 'type' in data:
        lab = f"{data['type']}_{'hull' if data['origin'] == 'hull' else 'ms'}"
    return lab


def graph_has_hull_origin(G):
    """True when the graph contains critical points added by the convex-hull protrusion detection."""
    if G.graph.get('has_hull_origin') is not None:
        return bool(G.graph['has_hull_origin'])
    return any(d.get('origin') == 'hull' for _, d in G.nodes(data=True))


def _origin_group_values(G):
    groups = {}
    for _, d in G.nodes(data=True):
        g = node_origin_group(d)
        if g is not None:
            groups.setdefault(g, []).append(node_function_value(d))
    return groups


def _hull_tips_below_col(G):
    """Hull tips whose scalar value is below the value of their own col (they are not maxima of the field)."""
    col = {}
    for _, d in G.nodes(data=True):
        if d.get('type') == 'saddle' and int(d.get('pair_extremum', -1)) >= 0:
            col[int(d['pair_extremum'])] = node_function_value(d)
    below = 0
    for _, d in G.nodes(data=True):
        if node_origin_group(d) == 'maximum_hull' and int(d.get('vertex', -1)) in col:
            below += int(node_function_value(d) < col[int(d['vertex'])])
    return below


def origin_group_similarity(graph_folder_path, single_file=True):
    """
    Like-with-like comparison of consecutive Morse–Smale graphs when the protrusions were completed by the convex
    hull (SMComplex.fuse_hull_protrusions). The current 'labeled_interleaving_distance' groups the nodes by type only,
    so the tips added by the hull (minima of the hull depth, NOT maxima of the scalar field: their value can even be
    below their own col) and their cols are compared together with the Morse–Smale maxima and saddles. Here every
    group is type x origin: maximum_ms, maximum_hull, saddle_ms, saddle_hull, minimum_ms, center ('ms+hull' maxima are
    field maxima -> 'ms').

    Per transition (<tag>_origin_group_similarity.csv):
      labeled_interleaving_distance        the current metric (same code), for reference
      origin_labeled_interleaving_distance maximum over the type x origin groups (groups present in both graphs)
      ms_only_interleaving_distance        maximum over the Morse–Smale groups: what the scalar field alone shows
      hull_only_interleaving_distance      maximum over the hull groups: the protrusions recovered by the hull
      dominant_group_current / dominant_group_origin
      d_<group>, n_prev_<group>, n_curr_<group>, dn_<group>: per-group distance and counts (a count change means the
      value was affected by the padding of the sorted comparison); a group missing in one graph gives NaN (no [0.0]
      comparison) and its count change is recorded instead.
    Also <tag>_origin_group_frames.csv (per frame and group: n, min / mean / max value, hull tips below their col) and
    <tag>_origin_group_nodes.csv (every node with its group and value).
    Returns the transitions csv path, or None when no graph carries hull-origin nodes (Reeb graphs, MS complexes
    computed without the fusion): only the usual analysis applies then.
    """
    folder = Path(graph_folder_path)
    files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix == '.pkl'], key=_frame_number)
    if len(files) < 2:
        return None
    graphs = []
    for f in files:
        with open(f, 'rb') as fh:
            graphs.append(pickle.load(fh))
    if not any(graph_has_hull_origin(G) for G in graphs):
        return None
    if not all(any(node_origin_group(d) is not None for _, d in G.nodes(data=True)) for G in graphs):
        print("[Warning] origin-separated comparison skipped: some Morse-Smale graphs carry no 'origin' attribute "
              "(computed with an older SMComplex); recompute the MS complex.")
        return None
    tag = folder.name
    frames = [_frame_number(f) for f in files]
    analysis_folder = folder.parent / 'Graph_analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    node_rows, frame_rows = [], []
    for t, G in zip(frames, graphs):
        below = _hull_tips_below_col(G)
        vals = _origin_group_values(G)
        for n, d in G.nodes(data=True):
            g = node_origin_group(d)
            if g is not None:
                node_rows.append({'Time_Step': t, 'node': n, 'group': g, 'type': d.get('type'), 'origin': d.get('origin', 'ms'),
                                  'source': d.get('source', ''), 'f_value': node_function_value(d)})
        for g in ORIGIN_GROUPS:
            v = np.asarray(vals.get(g, []), dtype=float)
            frame_rows.append({'Time_Step': t, 'group': g, 'n': int(v.size),
                               'f_min': float(v.min()) if v.size else np.nan, 'f_mean': float(v.mean()) if v.size else np.nan,
                               'f_max': float(v.max()) if v.size else np.nan,
                               'n_hull_tips_below_col': below if g == 'maximum_hull' else np.nan})

    rows = []
    for i in range(1, len(graphs)):
        g1, g2 = graphs[i - 1], graphs[i]
        v1, v2 = _origin_group_values(g1), _origin_group_values(g2)
        row = {'Transition': f"T{frames[i-1]} -> T{frames[i]}", 'Time_Step': frames[i],
               'labeled_interleaving_distance': labeled_interleaving_proxy(g1, g2, 'label')}
        per = {}
        for g in ORIGIN_GROUPS:
            a, b = v1.get(g, []), v2.get(g, [])
            d = _sorted_linf(a, b) if (len(a) and len(b)) else np.nan
            per[g] = d
            row[f'd_{g}'] = d
            row[f'n_prev_{g}'] = len(a); row[f'n_curr_{g}'] = len(b); row[f'dn_{g}'] = len(b) - len(a)

        def nanmax(keys):
            vals_ = [per[k] for k in keys if np.isfinite(per[k])]
            return float(max(vals_)) if vals_ else np.nan

        row['origin_labeled_interleaving_distance'] = nanmax(ORIGIN_GROUPS)
        row['ms_only_interleaving_distance'] = nanmax(ORIGIN_MS_GROUPS)
        row['hull_only_interleaving_distance'] = nanmax(ORIGIN_HULL_GROUPS)
        fin = {k: v for k, v in per.items() if np.isfinite(v)}
        row['dominant_group_origin'] = max(fin, key=fin.get) if fin else ''
        # which type group gives the current value, and whether hull nodes are inside it
        cur = {}
        for lab in {str(d.get('label')) for _, d in g1.nodes(data=True)} | {str(d.get('label')) for _, d in g2.nodes(data=True)}:
            a = [node_function_value(d) for _, d in g1.nodes(data=True) if str(d.get('label')) == lab] or [0.0]
            b = [node_function_value(d) for _, d in g2.nodes(data=True) if str(d.get('label')) == lab] or [0.0]
            cur[lab] = _sorted_linf(a, b)
        row['dominant_group_current'] = max(cur, key=cur.get) if cur else ''
        row['n_hull_nodes_prev'] = sum(len(v1.get(g, [])) for g in ORIGIN_HULL_GROUPS)
        row['n_hull_nodes_curr'] = sum(len(v2.get(g, [])) for g in ORIGIN_HULL_GROUPS)
        rows.append(row)

    out = analysis_folder / f'{tag}_origin_group_similarity.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(frame_rows).to_csv(analysis_folder / f'{tag}_origin_group_frames.csv', index=False)
    pd.DataFrame(node_rows).to_csv(analysis_folder / f'{tag}_origin_group_nodes.csv', index=False)
    if single_file:
        print(f"Origin-separated (MS / convex hull) comparison saved to: {out}")
    return str(out)


def graph_similarity(reeb_folder_path, metrics_list, single_file=True, analysis_folder=None, tag=None):
    """
    Computes pairwise metrics between consecutive Reeb graphs of a sequence.

    NOTE: 'interleaving_distance', 'labeled_interleaving_distance', 'function_distortion_distance'
    and 'branch_decomposition_distance' are cheap *proxies* (sorted-value / sorted-hop-distance
    L-infinity comparisons and feature-vector norms), not the exact TDA distances of the same name.
    'labeled_interleaving_distance' coincides with 'interleaving_distance' unless nodes carry a
    'label' attribute (Reeb graphs: never; Morse–Smale graphs: the critical-point type).
    'betti_0_distance' = |Δ connected components| and 'betti_1_distance' = |Δ independent cycles| between consecutive
    graphs (exact); 'branch_decomposition_distance' uses (Betti-0, Betti-1, branching nodes), so a split / merge of the
    shape changes it too.

    Morse–Smale graphs whose protrusions were completed by the convex hull additionally get the origin-separated
    (like-with-like) comparison of origin_group_similarity (own csv files, plotted by plot_graph_similarity); for Reeb
    graphs and MS graphs without hull points only the metrics above are computed.

    Args:
        reeb_folder_path (str/Path): folder containing the .pkl Reeb graphs.
        metrics_list (list of str | 'all'): requested metrics.
    Returns:
        str: path to the generated CSV file (None if fewer than two graphs are found).
    """
    reeb_folder = Path(reeb_folder_path)
    tag = tag or ('Reeb' if 'reeb' in str(reeb_folder).lower() else reeb_folder.name)

    if single_file:
        print("Starting metric similarity among graphs")

    if isinstance(metrics_list, str):
        metrics_list = [
            'degree_wasserstein',
            'spectral_laplacian',
            'interleaving_distance',
            'labeled_interleaving_distance',
            'function_distortion_distance',
            'branch_decomposition_distance',
            'betti_0_distance',
            'betti_1_distance'
        ]

    # Sort on the frame number: a plain string sort puts Reeb_T10 before Reeb_T2 (un-padded files).
    reeb_files = sorted([f for f in reeb_folder.iterdir() if f.is_file() and f.suffix == '.pkl'],
                        key=_frame_number)
    frame_ids = [_frame_number(f) for f in reeb_files]

    if len(reeb_files) < 2:
        print("Error: Not enough Reeb graphs found to perform pairwise similarity analysis.")
        return None

    analysis_folder = Path(analysis_folder) if analysis_folder is not None else reeb_folder.parent / 'Graph_analysis'
    os.makedirs(analysis_folder, exist_ok=True)

    with open(reeb_files[0], 'rb') as f:
        G_prev = pickle.load(f)

    results = []

    def calc_degree_wasserstein(g1, g2):
        v1, v2 = g1.number_of_nodes(), g2.number_of_nodes()
        deg1 = [d for n, d in g1.degree()] if v1 > 0 else [0]
        deg2 = [d for n, d in g2.degree()] if v2 > 0 else [0]
        return wasserstein_distance(deg1, deg2)

    def calc_spectral_distance(g1, g2):
        """Dense spectral distance between normalized Laplacians (spectra zero-padded to equal length)."""
        v1, v2 = g1.number_of_nodes(), g2.number_of_nodes()
        if v1 == 0 or v2 == 0:
            return 0.0

        lap_prev_np = np.asarray(nx.normalized_laplacian_matrix(g1).todense())
        lap_curr_np = np.asarray(nx.normalized_laplacian_matrix(g2).todense())

        evals_prev = xp.linalg.eigvalsh(xp.asarray(lap_prev_np))
        evals_curr = xp.linalg.eigvalsh(xp.asarray(lap_curr_np))

        max_len = max(len(evals_prev), len(evals_curr))
        e_p_pad = xp.pad(evals_prev, (0, max_len - len(evals_prev)))
        e_c_pad = xp.pad(evals_curr, (0, max_len - len(evals_curr)))

        return float(to_cpu(xp.linalg.norm(e_p_pad - e_c_pad)))

    def extract_scalars(G):
        if G.number_of_nodes() == 0:
            return np.array([0.0])
        return np.array([node_function_value(data) for _, data in G.nodes(data=True)])

    def calc_interleaving_proxy(g1, g2):
        return _sorted_linf(extract_scalars(g1), extract_scalars(g2))

    def calc_function_distortion_proxy(g1, g2):
        if g1.number_of_nodes() == 0 or g2.number_of_nodes() == 0:
            return np.nan
        try:
            d1 = [d for _, dists in nx.all_pairs_shortest_path_length(g1) for d in dists.values()]
            d2 = [d for _, dists in nx.all_pairs_shortest_path_length(g2) for d in dists.values()]
            return _sorted_linf(d1, d2)
        except Exception:
            return np.nan

    def _betti(g):
        v, e = g.number_of_nodes(), g.number_of_edges()
        c = nx.number_connected_components(g) if v > 0 else 0
        return c, max(0, e - v + c)

    def calc_branch_decomposition_proxy(g1, g2):
        def branch_features(g):
            b0, b1 = _betti(g)
            branch_nodes = sum(1 for n, d in g.degree() if d > 2)
            return np.array([b0, b1, branch_nodes])          # components, cycles, branching nodes

        return float(np.linalg.norm(branch_features(g1) - branch_features(g2)))

    def calc_betti_0_distance(g1, g2):
        """|Δ connected components| (Betti-0): > 0 when a piece of the shape split off or two pieces merged."""
        return float(abs(_betti(g2)[0] - _betti(g1)[0]))

    def calc_betti_1_distance(g1, g2):
        """|Δ independent cycles| (Betti-1)."""
        return float(abs(_betti(g2)[1] - _betti(g1)[1]))

    def calc_labeled_interleaving_proxy(g1, g2):
        return labeled_interleaving_proxy(g1, g2, 'label')

    metric_dispatch = {
        'degree_wasserstein': calc_degree_wasserstein,
        'spectral_laplacian': calc_spectral_distance,
        'interleaving_distance': calc_interleaving_proxy,
        'labeled_interleaving_distance': calc_labeled_interleaving_proxy,
        'function_distortion_distance': calc_function_distortion_proxy,
        'branch_decomposition_distance': calc_branch_decomposition_proxy,
        'betti_0_distance': calc_betti_0_distance,
        'betti_1_distance': calc_betti_1_distance,
    }

    for i in tqdm(range(1, len(reeb_files)), desc="Computing Fast Pairwise Similarities", leave=single_file):
        with open(reeb_files[i], 'rb') as f:
            G_curr = pickle.load(f)

        row_data = {
            'Transition': f"T{frame_ids[i-1]} -> T{frame_ids[i]}",
            'Time_Step': frame_ids[i]
        }

        for metric in metrics_list:
            metric_clean = metric.lower().strip()
            if metric_clean in metric_dispatch:
                val = metric_dispatch[metric_clean](G_prev, G_curr)
            else:
                print(f"[Warning] Unknown metric '{metric}'. Skipping.")
                val = np.nan
            row_data[metric] = val

        results.append(row_data)
        G_prev = G_curr

    df = pd.DataFrame(results)
    csv_out_path = analysis_folder / f'{tag}_pairwise_graph_similarity.csv'
    from PynamicMesh.core.dyn_common import add_time_columns     # known frame times: Time / Dt / <metric>_rate
    df = add_time_columns(df, reeb_folder_path, frame_col='Time_Step',
                          rate_cols=[c for c in df.columns if c not in ('Transition', 'Time_Step')])
    df.to_csv(csv_out_path, index=False)

    if single_file:
        print(f"Similarity analysis complete. Data saved to: {csv_out_path}")

    origin_group_similarity(reeb_folder, single_file=single_file)   # no-op for Reeb / MS graphs without hull points

    return str(csv_out_path)


def plot_graph_similarity(csv_path, single_file=True):
    """
    Reads the pairwise similarity CSV and generates a plot tracking the evolution
    of the chosen metrics over time. Each subplot has its own x-axis label.
    """
    if csv_path is None:
        return
    csv_file = Path(csv_path)

    tag = str(csv_file.name).split('_')[0]

    if single_file:
        print("\nGenerating visual reports for Graph Similarity...")

    if not csv_file.exists():
        print(f"Error: Could not find CSV file at {csv_path}")
        return

    df = pd.read_csv(csv_file)
    plots_folder = csv_file.parent / 'plots'
    os.makedirs(plots_folder, exist_ok=True)

    from PynamicMesh.core.dyn_common import time_axis
    time_steps, t_label = time_axis(df, 'Time_Step', None)

    exclude_cols = ['Transition', 'Time_Step', 'Time', 'Dt', 'Time_Unit'] + [c for c in df.columns if c.endswith('_rate')]
    valid_metrics = [col for col in df.columns if col not in exclude_cols and not df[col].isna().all()]

    if not valid_metrics:
        print("No valid metric data found in the CSV to plot.")
        return

    sns.set_theme(style="whitegrid")
    num_metrics = len(valid_metrics)

    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics), sharex=False)

    if num_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("husl", num_metrics)

    for ax, metric, color in zip(axes, valid_metrics, colors):
        ax.plot(time_steps, df[metric], color=color, marker='D', linewidth=2, markersize=6)

        formatted_title = metric.replace('_', ' ').title()
        ax.set_title(f'Evolution of {formatted_title}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance / Shift', fontsize=10)
        ax.set_xlabel(t_label or 'Time Step (Transition $T_{n-1} \\rightarrow T_n$)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()

    output_img_path = plots_folder / f'{tag}_Pairwise_Similarity_Evolution.png'
    fig.savefig(output_img_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    origin_csv = csv_file.parent / csv_file.name.replace('_pairwise_graph_similarity.csv', '_origin_group_similarity.csv')
    if origin_csv.exists() and origin_csv != csv_file:
        plot_origin_group_similarity(origin_csv, plots_folder, tag)

    if single_file:
        print(f"Visual report generated successfully in: {output_img_path}")


def plot_origin_group_similarity(origin_csv, plots_folder, tag):
    """
    Figures of the origin-separated comparison (Morse–Smale graphs with convex-hull protrusions):
      <tag>_Origin_Separated_Similarity.png  (a) current labelled interleaving vs like-with-like, MS-only and hull-only;
                                             (b) distance of every type x origin group (hollow marker = the count of the
                                             group changed, value affected by the padding); (c) node counts per group
                                             and hull tips below their own col; (d) group giving the maximum, current
                                             grouping vs separated grouping
      <tag>_Origin_Group_Values.png          scalar values of every group per frame (overlap of hull tips with saddles
                                             and Morse–Smale maxima)
    """
    origin_csv = Path(origin_csv)
    df = pd.read_csv(origin_csv)
    if df.empty:
        return
    frames_csv = origin_csv.parent / origin_csv.name.replace('_origin_group_similarity.csv', '_origin_group_frames.csv')
    nodes_csv = origin_csv.parent / origin_csv.name.replace('_origin_group_similarity.csv', '_origin_group_nodes.csv')
    fr = pd.read_csv(frames_csv) if frames_csv.exists() else None
    nd = pd.read_csv(nodes_csv) if nodes_csv.exists() else None
    sns.set_theme(style="whitegrid")
    t = df['Time_Step']
    xlabel = 'Time Step (Transition $T_{n-1} \\rightarrow T_n$)'

    fig, axes = plt.subplots(4, 1, figsize=(11, 17), gridspec_kw={'height_ratios': [1.1, 1.1, 1.0, 0.8]})
    ax = axes[0]
    ax.plot(t, df['labeled_interleaving_distance'], color='black', lw=2.6, marker='D', label='current: labelled by type (hull mixed in)')
    ax.plot(t, df['origin_labeled_interleaving_distance'], color='purple', lw=2, marker='o', label='like with like: type x origin')
    ax.plot(t, df['ms_only_interleaving_distance'], color='red', lw=1.6, marker='^', label='Morse-Smale points only')
    ax.plot(t, df['hull_only_interleaving_distance'], color='deepskyblue', lw=1.6, ls='--', marker='v', label='convex-hull points only')
    ax.set_title('Labelled interleaving: current grouping vs origin-separated grouping', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance / Shift'); ax.set_xlabel(xlabel); ax.legend(fontsize=9)

    ax = axes[1]
    for g in ORIGIN_GROUPS:
        col = f'd_{g}'
        if col not in df or df[col].isna().all():
            continue
        c = ORIGIN_COLORS[g]; ls = '--' if g.endswith('_hull') else '-'
        ax.plot(t, df[col], color=c, ls=ls, lw=1.8, label=g)
        changed = df[f'dn_{g}'] != 0
        ax.scatter(t[~changed], df[col][~changed], color=c, s=28, zorder=3)
        ax.scatter(t[changed], df[col][changed], facecolors='white', edgecolors=c, s=60, linewidths=1.8, zorder=4)
    ax.scatter([], [], facecolors='white', edgecolors='gray', s=60, linewidths=1.8, label='count of the group changed (padding)')
    ax.set_title('Distance of every type x origin group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance / Shift'); ax.set_xlabel(xlabel); ax.legend(fontsize=8, ncol=2)

    ax = axes[2]
    if fr is not None and len(fr):
        markers = {'maximum_ms': '^', 'maximum_hull': 'v', 'saddle_ms': 's', 'saddle_hull': 'D', 'minimum_ms': 'o', 'center': '*'}
        for k, g in enumerate(ORIGIN_GROUPS):
            sub = fr[fr.group == g]
            if sub.n.sum() == 0:
                continue
            dx = (k - (len(ORIGIN_GROUPS) - 1) / 2) * 0.06          # small dodge: equal counts stay visible
            ax.plot(sub.Time_Step + dx, sub.n, color=ORIGIN_COLORS[g], ls='--' if g.endswith('_hull') else '-',
                    marker=markers[g], ms=7, lw=1.6, label=g)
        below = fr[fr.group == 'maximum_hull']
        if len(below) and below.n_hull_tips_below_col.notna().any():
            ax.bar(below.Time_Step, below.n_hull_tips_below_col, color='deepskyblue', alpha=0.25, width=0.6,
                   label='hull tips below their own col')
    ax.set_title('Nodes per group (frames)', fontsize=12, fontweight='bold')
    ax.set_ylabel('count'); ax.set_xlabel('Time Step (T)')
    ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    ax = axes[3]
    cur = df['dominant_group_current'].value_counts(); org = df['dominant_group_origin'].value_counts()
    cats = [c for c in ('maximum', 'saddle', 'minimum', 'center') if c in cur.index] + [g for g in ORIGIN_GROUPS if g in org.index]
    x = np.arange(len(cats))
    ax.bar(x, [cur.get(c, 0) if c in cur.index else 0 for c in cats], color='black', alpha=0.75, label='current grouping (type)')
    ax.bar(x, [org.get(c, 0) if c in org.index else 0 for c in cats],
           color=[ORIGIN_COLORS.get(c, 'gray') for c in cats], alpha=0.85, label='separated grouping (type x origin)')
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20)
    ax.set_title('Which group gives the maximum (number of transitions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('transitions'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(plots_folder) / f'{tag}_Origin_Separated_Similarity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    if nd is not None and len(nd):
        order = [g for g in ORIGIN_GROUPS if g in set(nd.group)]
        fig, ax = plt.subplots(figsize=(max(9, 0.9 * nd.Time_Step.nunique() + 4), 6))
        sns.stripplot(data=nd, x='Time_Step', y='f_value', hue='group', hue_order=order, dodge=True, jitter=0.18, size=5,
                      palette={g: ORIGIN_COLORS[g] for g in order}, ax=ax, edgecolor='k', linewidth=0.3)
        ax.set_title('Scalar value of the graph nodes per group: convex-hull tips are not maxima of the field',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step (T)'); ax.set_ylabel('scalar field value (f_value)')
        ax.legend(fontsize=8, ncol=1, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
        fig.tight_layout()
        fig.savefig(Path(plots_folder) / f'{tag}_Origin_Group_Values.png', dpi=200, bbox_inches='tight')
        plt.close(fig)