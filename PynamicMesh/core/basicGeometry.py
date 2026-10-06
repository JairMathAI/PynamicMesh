import trimesh
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyFM.mesh import TriMesh
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from scipy.sparse import coo_matrix, diags
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    if __import__("multiprocessing").current_process().name == "MainProcess":            # worker processes import this module again: say it once
        print("[INFO] CuPy detected. Utilizing GPU for Basic Geometry.")
except ImportError:
    CUPY_AVAILABLE = False
    if __import__("multiprocessing").current_process().name == "MainProcess":            # worker processes import this module again: say it once
        print("[INFO] CuPy not found. Defaulting to CPU (NumPy).")


# Metric groups -> CSV columns they produce
METRIC_COLUMNS = {
    'n_vertices': ['n_vertices'],
    'n_faces': ['n_faces'],
    'area': ['area'],
    'volume': ['volume', 'is_watertight'],
    'sphericity': ['sphericity'],
    'convexity': ['convexity'],
    'center_mass': ['cm_x', 'cm_y', 'cm_z'],
    'gaussian_curvature': ['mean_gaussian_curvature', 'mean_abs_gaussian_curvature', 'total_abs_gaussian_curvature', 'gaussian_curvature_std'],
    'mean_curvature': ['mean_mean_curvature', 'mean_abs_mean_curvature', 'willmore_energy'],
    'topology': ['euler_number', 'genus', 'n_boundary_edges', 'n_components', 'betti_0', 'betti_1', 'betti_2'],
    # connected components of the surface (Betti-0): number, closed ones, share of the largest, per-component
    # areas / volumes (';'-separated, largest first) - splits and merges of the shape show up as changes of n_components
    'components': ['n_components', 'n_closed_components', 'largest_component_area_fraction', 'component_areas',
                   'component_volumes'],
    'bounding_box': ['bbox_dx', 'bbox_dy', 'bbox_dz', 'bbox_diagonal'],
    'shape': ['elongation', 'flatness', 'surface_to_volume', 'radius_of_gyration'],
}
AVAILABLE_METRICS = set(METRIC_COLUMNS)


# ----------------------------------------------------------------------------- #
#  Geometry helpers (numpy only)
# ----------------------------------------------------------------------------- #

def _faces(mesh):
    f = np.asarray(mesh.faces)
    if f.ndim == 1:                                     # pyvista padded format
        f = f.reshape(-1, 4)[:, 1:]
    return f.astype(np.int64)


def face_areas(v, f):
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def vertex_areas(v, f):
    fa = face_areas(v, f)
    va = np.zeros(v.shape[0])
    for k in range(3):
        np.add.at(va, f[:, k], fa / 3.0)
    return va


def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def signed_volume(v, f):
    """Enclosed volume by the divergence theorem (exact for closed, consistently oriented meshes)."""
    return float(np.einsum('ij,ij->i', v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0)


def gaussian_curvature(v, f):
    """Discrete Gaussian curvature K_i = (2π - Σ angles_i) / A_i (angle deficit per vertex area)."""
    angles_sum = np.zeros(v.shape[0])
    for k in range(3):
        p0, p1, p2 = v[f[:, k]], v[f[:, (k + 1) % 3]], v[f[:, (k + 2) % 3]]
        a, b = p1 - p0, p2 - p0
        cosang = np.einsum('ij,ij->i', a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-16)
        np.add.at(angles_sum, f[:, k], np.arccos(np.clip(cosang, -1.0, 1.0)))
    va = vertex_areas(v, f)
    used = va > 0
    K = np.zeros(v.shape[0])
    K[used] = (2.0 * np.pi - angles_sum[used]) / va[used]
    return K, va, used


def mean_curvature(v, f):
    """Signed discrete mean curvature H_i = (W x)_i · n_i / (2 A_i) with the cotangent stiffness W."""
    n = v.shape[0]
    I, J, Wv = [], [], []
    for k in range(3):
        i, j, l = f[:, k], f[:, (k + 1) % 3], f[:, (k + 2) % 3]
        a, b = v[i] - v[l], v[j] - v[l]
        cot = np.einsum('ij,ij->i', a, b) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-16)
        I += [i, j]; J += [j, i]; Wv += [0.5 * cot, 0.5 * cot]
    Wm = coo_matrix((np.concatenate(Wv), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr()
    W = diags(np.asarray(Wm.sum(axis=1)).ravel()) - Wm
    va = vertex_areas(v, f)
    used = va > 0
    H = np.zeros(n)
    H[used] = np.einsum('ij,ij->i', (W @ v)[used], vertex_normals(v, f)[used]) / (2.0 * va[used])
    return H, va, used


def connected_components_of_surface(v, f):
    """
    Connected components of the surface = of the vertices REFERENCED by faces (an isolated / unreferenced vertex is
    not a piece of surface; counting every vertex of the array made each of them a spurious component).
    Returns (n_components, face_label (n_faces,) with the component of every face).
    """
    from scipy.sparse.csgraph import connected_components
    used = np.unique(f)
    if used.size == 0:
        return 0, np.zeros(0, dtype=np.int64)
    remap = -np.ones(v.shape[0], dtype=np.int64); remap[used] = np.arange(used.size)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    adj = coo_matrix((np.ones(len(e)), (remap[e[:, 0]], remap[e[:, 1]])), shape=(used.size, used.size))
    n, lab = connected_components(adj, directed=False)
    return int(n), lab[remap[f[:, 0]]]


def component_statistics(v, f):
    """Per component (largest area first): area, enclosed volume (divergence theorem), closed (no boundary edge)."""
    n, face_lab = connected_components_of_surface(v, f)
    fa = face_areas(v, f)
    comps = []
    for c in range(n):
        fc = f[face_lab == c]
        n_b, _ = boundary_edge_count(fc)
        comps.append({"area": float(fa[face_lab == c].sum()), "volume": abs(signed_volume(v, fc)), "closed": n_b == 0})
    comps.sort(key=lambda d: -d["area"])
    return comps


def boundary_edge_count(f):
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def to_trimesh(v, f):
    """
    Trimesh WITHOUT vertex merging. The default `process=True` welds coincident vertices; when two
    parts of the shape touch (legs, belly, ...) this creates non-manifold edges, `is_watertight`
    becomes False and every volume-based metric was lost, although the original mesh was closed.
    Only the face orientation is made consistent (outward normals).
    """
    m = trimesh.Trimesh(vertices=v, faces=f, process=False, validate=False)
    if not m.is_winding_consistent:
        trimesh.repair.fix_winding(m)
    if m.is_watertight:
        trimesh.repair.fix_inversion(m)
    return m


# ----------------------------------------------------------------------------- #
#  Per-mesh metrics
# ----------------------------------------------------------------------------- #

def compute_mesh_geometry(pyfm_mesh, metrics='all'):
    """
    Computes geometric properties of a single pyFM TriMesh.

    :param pyfm_mesh: A pyFM.mesh.TriMesh object (or any object with .vertices / .faces).
    :param metrics: 'all', a single string, or a list of strings among
        'n_vertices', 'n_faces', 'area', 'volume', 'sphericity', 'convexity', 'center_mass',
        'gaussian_curvature', 'mean_curvature', 'topology', 'components', 'bounding_box', 'shape'.
    :return: Dictionary of metrics (see METRIC_COLUMNS for the columns each metric produces).

    Volume-based metrics are always computed with the divergence theorem on the original faces;
    the boolean `is_watertight` tells whether the mesh is closed and manifold (exact volume) or not
    (volume is then the approximation for the almost closed surface).
    """
    if isinstance(metrics, str):
        metrics_to_compute = set(AVAILABLE_METRICS) if metrics.lower() == 'all' else {metrics.lower()}
    else:
        metrics_to_compute = {str(m).lower() for m in metrics}
    unknown = metrics_to_compute - AVAILABLE_METRICS
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}. Available: {sorted(AVAILABLE_METRICS)}")

    v = np.asarray(pyfm_mesh.vertices, dtype=np.float64)
    f = _faces(pyfm_mesh)
    results = {}

    if 'n_vertices' in metrics_to_compute:
        results['n_vertices'] = int(v.shape[0])
    if 'n_faces' in metrics_to_compute:
        results['n_faces'] = int(f.shape[0])

    fa = face_areas(v, f)
    area = float(fa.sum())
    if 'area' in metrics_to_compute:
        results['area'] = area

    needs_volume = metrics_to_compute & {'volume', 'sphericity', 'convexity', 'center_mass', 'shape'}
    vol, watertight, t_mesh = np.nan, False, None
    if needs_volume:
        t_mesh = to_trimesh(v, f)
        watertight = bool(t_mesh.is_watertight)
        vol = abs(signed_volume(np.asarray(t_mesh.vertices), np.asarray(t_mesh.faces)))

    if 'volume' in metrics_to_compute:
        results['volume'] = vol
        results['is_watertight'] = watertight

    if 'sphericity' in metrics_to_compute:
        results['sphericity'] = (np.pi ** (1 / 3) * (6 * vol) ** (2 / 3)) / area if (vol > 0 and area > 0) else np.nan

    if 'convexity' in metrics_to_compute:
        try:
            hull_vol = abs(float(t_mesh.convex_hull.volume))
            results['convexity'] = vol / hull_vol if hull_vol > 0 else np.nan
        except Exception:  # noqa: BLE001 - degenerate hull
            results['convexity'] = np.nan

    if 'center_mass' in metrics_to_compute:
        # Volume-based center of mass for closed meshes, area-weighted surface centroid otherwise
        # (trimesh.center_mass is meaningless on open surfaces).
        if watertight:
            cm = np.asarray(t_mesh.center_mass, dtype=np.float64)
        else:
            centroids = v[f].mean(axis=1)
            cm = (centroids * fa[:, None]).sum(axis=0) / max(area, 1e-16)
        results['cm_x'], results['cm_y'], results['cm_z'] = float(cm[0]), float(cm[1]), float(cm[2])

    if 'gaussian_curvature' in metrics_to_compute:
        K, va, used = gaussian_curvature(v, f)
        w = va[used] / va[used].sum()
        results['mean_gaussian_curvature'] = float(np.sum(w * K[used]))          # = 2πχ / A for closed meshes (Gauss-Bonnet)
        results['mean_abs_gaussian_curvature'] = float(np.sum(w * np.abs(K[used])))
        results['total_abs_gaussian_curvature'] = float(np.sum(va[used] * np.abs(K[used])))   # bending complexity, scale invariant
        results['gaussian_curvature_std'] = float(np.sqrt(np.sum(w * (K[used] - results['mean_gaussian_curvature']) ** 2)))

    if 'mean_curvature' in metrics_to_compute:
        H, va, used = mean_curvature(v, f)
        w = va[used] / va[used].sum()
        results['mean_mean_curvature'] = float(np.sum(w * H[used]))
        results['mean_abs_mean_curvature'] = float(np.sum(w * np.abs(H[used])))
        results['willmore_energy'] = float(np.sum(va[used] * H[used] ** 2))      # ∫H² dA, scale invariant (4π for a sphere)

    comps = None
    if metrics_to_compute & {'topology', 'components'}:
        comps = component_statistics(v, f)

    if 'topology' in metrics_to_compute:
        n_boundary, n_nonmanifold = boundary_edge_count(f)
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        n_edges = np.unique(e, axis=0).shape[0]
        n_used = np.unique(f).size                            # unreferenced vertices are not part of the surface
        euler = int(n_used - n_edges + f.shape[0])
        n_comp = len(comps)
        n_closed = sum(c["closed"] for c in comps)
        results['euler_number'] = euler
        results['genus'] = int(round((2 * n_comp - euler) / 2)) if n_boundary == 0 else np.nan
        results['n_boundary_edges'] = n_boundary
        results['n_components'] = n_comp
        # Betti numbers of the surface (orientable 2-manifold): β0 components, β2 closed components, and
        # β1 from Euler's relation β0 − β1 + β2 = χ (β1 = 2·genus for closed surfaces, + boundary loops otherwise)
        results['betti_0'] = n_comp
        results['betti_2'] = int(n_closed)
        results['betti_1'] = int(n_comp + n_closed - euler)
        if n_used < v.shape[0]:
            results['n_unreferenced_vertices'] = int(v.shape[0] - n_used)
        if n_nonmanifold:
            results['n_nonmanifold_edges'] = n_nonmanifold

    if 'components' in metrics_to_compute:
        total = sum(c["area"] for c in comps) or 1.0
        results['n_components'] = len(comps)
        results['n_closed_components'] = int(sum(c["closed"] for c in comps))
        results['largest_component_area_fraction'] = float(comps[0]["area"] / total) if comps else np.nan
        results['component_areas'] = ";".join(f"{c['area']:.6g}" for c in comps)
        results['component_volumes'] = ";".join(f"{c['volume']:.6g}" for c in comps)

    if 'bounding_box' in metrics_to_compute:
        ext = np.ptp(v, axis=0)
        results['bbox_dx'], results['bbox_dy'], results['bbox_dz'] = map(float, ext)
        results['bbox_diagonal'] = float(np.linalg.norm(ext))

    if 'shape' in metrics_to_compute:
        # Area-weighted PCA of the surface: principal extents (std along principal axes).
        centroids = v[f].mean(axis=1)
        w = fa / max(area, 1e-16)
        c = (centroids * w[:, None]).sum(axis=0)
        cov = ((centroids - c) * w[:, None]).T @ (centroids - c)
        ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
        s = np.sqrt(ev)
        results['elongation'] = float(s[0] / s[1]) if s[1] > 0 else np.nan     # 1 = isotropic in the two largest axes
        results['flatness'] = float(s[1] / s[2]) if s[2] > 0 else np.nan
        results['surface_to_volume'] = float(area / vol) if vol > 0 else np.nan
        results['radius_of_gyration'] = float(np.sqrt(ev.sum()))

    return results


# ----------------------------------------------------------------------------- #
#  Plotting
# ----------------------------------------------------------------------------- #

def generate_plots_from_csv(csv_path, dpi=200):
    """
    Reads the CSV, computes consecutive pairwise distances for the Center of Mass,
    and plots all properties inside a single integrated dashboard figure with
    independent, optimized y-axis scaling. Frames whose mesh is not watertight are
    marked with a red cross on the volume-based curves.
    """
    csv_path = Path(csv_path)
    plot_path = csv_path.parent
    os.makedirs(plot_path, exist_ok=True)

    df = pd.read_csv(csv_path)
    x = np.arange(len(df))
    # known acquisition times (column 'time', written by the pipeline when frame times are given): time axis,
    # centre-of-mass SPEED instead of displacement, relative growth rates of area / volume
    has_time = 'time' in df.columns and df['time'].notna().all() and len(df) >= 2
    t_unit = str(df['time_unit'].iloc[0]) if has_time and 'time_unit' in df.columns else 's'
    xs = df['time'].to_numpy(dtype=float) if has_time else x
    x_label = f'time ({t_unit})' if has_time else 'Mesh Sequence Index'
    rate_panels = []
    if has_time:
        tt = df['time'].to_numpy(dtype=float)
        for col, lab in (('area', 'area'), ('volume', 'volume')):
            if col in df.columns and df[col].notna().sum() >= 2:
                v = df[col].to_numpy(dtype=float)
                rate_panels.append((f'relative {lab} growth rate (1/{lab[0].upper()}) d{lab[0].upper()}/dt (1/{t_unit})',
                                    np.gradient(v, tt) / np.where(np.abs(v) > 0, v, np.nan)))

    cm_cols = {'cm_x', 'cm_y', 'cm_z'}
    flag_cols = {'is_watertight', 'time', 'frame', 'size', 'area_rel', 'volume_rel'}   # size-normalised copies: csv only
    comp_cols = {'n_components', 'betti_0', 'n_closed_components'}     # drawn together in the components panel
    volume_based = {'volume', 'sphericity', 'convexity', 'surface_to_volume'}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scalar_metrics = [c for c in numeric_cols if c not in cm_cols | flag_cols | comp_cols and not df[c].isna().all()]

    # connected components (Betti-0): splits (+) and merges (-) between consecutive meshes
    ncomp_col = 'n_components' if 'n_components' in df.columns else ('betti_0' if 'betti_0' in df.columns else None)
    events = pd.DataFrame(columns=['frame', 'n_before', 'n_after', 'change', 'event'])
    if ncomp_col is not None and len(df) >= 2:
        nc = df[ncomp_col].to_numpy(dtype=float)
        rows_ = [{'frame': int(i), 'n_before': int(nc[i - 1]), 'n_after': int(nc[i]), 'change': int(nc[i] - nc[i - 1]),
                  'event': 'split / new component' if nc[i] > nc[i - 1] else 'merge / lost component'}
                 for i in range(1, len(nc)) if np.isfinite(nc[i]) and np.isfinite(nc[i - 1]) and nc[i] != nc[i - 1]]
        events = pd.DataFrame(rows_, columns=events.columns)
        if has_time:
            events['time'] = [float(df['time'].iloc[int(f)]) for f in events['frame']]
        events.to_csv(plot_path / 'component_events.csv', index=False)

    not_watertight = None
    if 'is_watertight' in df.columns:
        flags = df['is_watertight'].astype(str).str.lower().isin(['false', '0', '0.0'])
        not_watertight = np.flatnonzero(flags.values)

    has_cm_displacement = False
    distances = None
    if cm_cols.issubset(df.columns):
        coords = df[['cm_x', 'cm_y', 'cm_z']].values
        if len(coords) >= 2:
            has_cm_displacement = True
            if CUPY_AVAILABLE:
                gpu_coords = cp.asarray(coords)
                distances = cp.asnumpy(cp.linalg.norm(cp.diff(gpu_coords, axis=0), axis=1))
            else:
                distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)

    num_plots = len(scalar_metrics) + (1 if has_cm_displacement else 0) + (1 if ncomp_col is not None else 0) + len(rate_panels)
    if num_plots == 0:
        print("[INFO] No columns found to plot.")
        return

    cols = 2 if num_plots > 1 else 1
    rows = int(np.ceil(num_plots / cols))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes = axes.flatten()

    def pad_flat(ax, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        y_min, y_max = values.min(), values.max()
        if np.isclose(y_min, y_max):
            ax.set_ylim(-1, 1) if y_min == 0 else ax.set_ylim(min(y_min * 0.9, y_min * 1.1), max(y_min * 0.9, y_min * 1.1))

    def mark_events(ax):
        for _, ev in events.iterrows():
            ax.axvline(xs[int(ev['frame'])], color='seagreen' if ev['change'] > 0 else 'crimson', lw=0.8, ls=':', alpha=0.7)

    plot_idx = 0
    if ncomp_col is not None:                              # first panel: connected components (Betti-0)
        ax = axes[plot_idx]
        nc = df[ncomp_col].to_numpy(dtype=float)
        ax.step(xs, nc, where='mid', color='black', lw=2, label='connected components (Betti-0)')
        ax.plot(xs, nc, 'o', color='black', ms=4)
        if 'n_closed_components' in df.columns:
            ax.plot(xs, df['n_closed_components'], 's', mfc='none', color='tab:blue', ms=7, label='closed components')
        for _, ev in events.iterrows():
            up = ev['change'] > 0
            ax.scatter(xs[int(ev['frame'])], ev['n_after'], marker='^' if up else 'v', s=160, color='seagreen' if up else 'crimson',
                       zorder=4, label=('split / new component' if up else 'merge / lost component'))
            ax.annotate(f"{'+' if up else ''}{int(ev['change'])}", (xs[int(ev['frame'])], ev['n_after']), textcoords='offset points',
                        xytext=(6, 6 if up else -14), fontsize=9, color='seagreen' if up else 'crimson')
        h_, l_ = ax.get_legend_handles_labels(); uniq = dict(zip(l_, h_))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8)
        ax.set_title('Connected Components (Betti-0): splits and merges', fontweight='bold')
        ax.set_xlabel(x_label); ax.set_ylabel('components')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, max(np.nanmax(nc), 1) + 1); ax.grid(True, linestyle='--', alpha=0.5)
        plot_idx += 1

    for metric in scalar_metrics:
        ax = axes[plot_idx]
        mark_events(ax)
        ax.plot(xs, df[metric], marker='o', linestyle='-', color='b')
        if metric in volume_based and not_watertight is not None and not_watertight.size:
            ax.scatter(xs[not_watertight], df[metric].values[not_watertight], marker='x', s=90, color='red', zorder=3,
                       label='not watertight (approx.)')
            ax.legend(fontsize=8)
        ax.set_title(f'Evolution of {metric.replace("_", " ").title()}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=0.1)
        pad_flat(ax, df[metric])
        plot_idx += 1

    if has_cm_displacement:
        ax = axes[plot_idx]
        if has_time:                                   # known intervals: speed of the centre of mass
            speeds = distances / np.diff(xs)
            ax.plot(xs[1:], speeds, marker='s', linestyle='-', color='b')
            ax.set_title('Center of Mass Speed')
            ax.set_xlabel(f'{x_label} (end of the interval)')
            ax.set_ylabel(f'speed (length / {t_unit})')
            distances = speeds
        else:
            ax.plot(np.arange(len(distances)), distances, marker='s', linestyle='-', color='b')
            ax.set_title('Center of Mass Consecutive Displacement')
            ax.set_xlabel('Sequence Transition Interval (i to i+1)')
            ax.set_ylabel('Euclidean Distance')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=0.1)
        pad_flat(ax, distances)
        plot_idx += 1

    for title_, vals in rate_panels:                   # growth rates (only with known frame times)
        ax = axes[plot_idx]
        ax.plot(xs, vals, marker='o', linestyle='-', color='darkgreen'); ax.axhline(0, color='gray', lw=0.8)
        mark_events(ax)
        ax.set_title(title_, fontsize=10); ax.set_xlabel(x_label); ax.set_ylabel(f'1/{t_unit}')
        ax.grid(True, linestyle='--', alpha=0.5); ax.margins(y=0.1)
        plot_idx += 1

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    fig.savefig(plot_path / 'mesh_evolution_summary.png', dpi=dpi)
    plt.close(fig)