import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import seaborn as sns
from scipy.sparse import coo_matrix, diags
from PynamicMesh.utils.tools import mesh_mat2object
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots
try:
    from PynamicMesh.utils.tools import load_aligned_mesh
except ImportError:                       # older tools module
    load_aligned_mesh = mesh_mat2object

try:
    import cupy as xp
    import cupyx as xxp
    GPU_AVAILABLE = True
    if __import__("multiprocessing").current_process().name == "MainProcess":            # worker processes import this module again: say it once
        print("[INFO] CuPy detected. Utilizing GPU for Physics and metric computations.")
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
    """Moves a CuPy array back to the CPU for PyVista/Plotting/Saving."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr


def _scatter_add(target, idx, values):
    """np.add.at / cupyx.scatter_add on the right backend."""
    if GPU_AVAILABLE and hasattr(target, 'get'):
        xxp.scatter_add(target, idx, values)
    else:
        np.add.at(target, to_cpu(idx), to_cpu(values))


def _frame_number(path, default=None):
    m = re.search(r'(\d+)(?!.*\d)', Path(path).stem)
    return int(m.group(1)) if m else default


def natural_sort_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', Path(path).name)]


def _as_faces(faces):
    f = np.asarray(faces)
    if f.ndim == 1:
        f = f.reshape(-1, 4)[:, 1:]
    return f.astype(np.int64)


# ----------------------------------------------------------------------------- #
#  Functional-map heatmap analysis
# ----------------------------------------------------------------------------- #

def compute_heatmap_similarity(matrix1, matrix2):
    """
    Cross-heatmap similarity between two consecutive functional maps.
    Maps refined with ZoomOut may have different sizes: the common top-left block is compared.
    JSD uses log2 so that it lives in [0, 1], matching the plot label.
    """
    m1c, m2c = np.asarray(matrix1), np.asarray(matrix2)
    r, c = min(m1c.shape[0], m2c.shape[0]), min(m1c.shape[1], m2c.shape[1])
    m1 = to_gpu(np.ascontiguousarray(m1c[:r, :c], dtype=np.float64)).ravel()
    m2 = to_gpu(np.ascontiguousarray(m2c[:r, :c], dtype=np.float64)).ravel()

    euclidean = xp.linalg.norm(m1 - m2)
    manhattan = xp.sum(xp.abs(m1 - m2))

    pearson = 0.0 if (xp.std(m1) == 0 or xp.std(m2) == 0) else xp.corrcoef(m1, m2)[0, 1]

    r1 = xp.argsort(xp.argsort(m1)).astype(xp.float64)
    r2 = xp.argsort(xp.argsort(m2)).astype(xp.float64)
    spearman = 0.0 if (xp.std(r1) == 0 or xp.std(r2) == 0) else xp.corrcoef(r1, r2)[0, 1]

    p, q = m1 ** 2, m2 ** 2
    sum_p, sum_q = xp.sum(p), xp.sum(q)
    p = p / sum_p if sum_p > 0 else xp.ones_like(p) / len(p)
    q = q / sum_q if sum_q > 0 else xp.ones_like(q) / len(q)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_p = xp.sum(p * xp.log2((p + eps) / (m + eps)))
    kl_q = xp.sum(q * xp.log2((q + eps) / (m + eps)))
    jsd = 0.5 * kl_p + 0.5 * kl_q

    return float(to_cpu(jsd)), float(to_cpu(pearson)), float(to_cpu(spearman)), float(to_cpu(manhattan)), float(to_cpu(euclidean))


def plot_similarity_metrics(similarity_history, path, dpi=150):
    """Plots the evolution of cross-heatmap similarity metrics across timesteps."""
    if not similarity_history:
        return

    hist = np.array(similarity_history)
    timesteps = np.arange(2, len(hist) + 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(timesteps, hist[:, 1], marker='o', label='Pearson', color='royalblue')
    axes[0].plot(timesteps, hist[:, 2], marker='s', label='Spearman', color='darkorange')
    axes[0].set_title("Cross-Heatmap Alignment Profile")
    axes[0].set_xlabel(r"Timestep (Transition $T_{i-1} \to T_i$)")
    axes[0].set_ylabel("Correlation Value")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(timesteps, hist[:, 3], marker='o', label='Manhattan ($L_1$)', color='crimson')
    axes[1].plot(timesteps, hist[:, 4], marker='s', label='Euclidean ($L_2$)', color='purple')
    axes[1].set_title("Spectral Coordinate Absolute Distance")
    axes[1].set_xlabel(r"Timestep (Transition $T_{i-1} \to T_i$)")
    axes[1].set_ylabel("Distance Magnitude")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(timesteps, hist[:, 0], marker='o', label='Jensen-Shannon Div.', color='forestgreen')
    axes[2].set_title("Spectral Energy Field Divergence (JSD)")
    axes[2].set_xlabel(r"Timestep (Transition $T_{i-1} \to T_i$)")
    axes[2].set_ylabel("Divergence [0, 1] (log2)")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_tranformation_heatmap(matrix, i, path, dpi=150):
    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(matrix, annot=False, cmap='coolwarm', center=0, cbar=True)
    plt.title(f'Transformation Heatmap Mesh T{i-1} → Mesh T{i}')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"Mesh T{i-1} Eigenfunctions")
    plt.ylabel(f"Mesh T{i} Eigenfunctions")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def Diagonal_metrics(matrix, alpha=0.5):
    """Diagonality metrics of a functional map (energy concentration around the diagonal)."""
    mat_gpu = to_gpu(np.asarray(matrix, dtype=np.float64))
    n, m = mat_gpu.shape
    matrix_sq = mat_gpu ** 2
    total_energy = xp.sum(matrix_sq)

    if float(to_cpu(total_energy)) == 0:
        return 0.0, 0.0, np.zeros(max(n, m))

    I, J = xp.ogrid[:n, :m]
    dist = xp.abs(I - J)
    max_dist = max(max(n, m) - 1, 1)

    inertia = xp.sum((dist ** 2) * matrix_sq) / (total_energy * (max_dist ** 2))
    inertia_metric = 1.0 - inertia
    decay_metric = xp.sum(matrix_sq * xp.exp(-alpha * dist)) / total_energy

    # Cumulative energy within bandwidth k, vectorized (energy per |i-j| then cumsum).
    energy_by_dist = xp.bincount(dist.ravel(), weights=matrix_sq.ravel(), minlength=max(n, m))
    cdf_bandwidth = xp.cumsum(energy_by_dist)[:max(n, m)] / total_energy

    return float(to_cpu(inertia_metric)), float(to_cpu(decay_metric)), to_cpu(cdf_bandwidth)


def plot_diagonal(inertia_history, decay_history, cdf_history, path, dpi=150):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(inertia_history, marker='o', color='blue')
    axes[0].set_title("Moment of Inertia Metric")
    axes[0].set_xlabel("Timestep")

    axes[1].plot(decay_history, marker='o', color='green')
    axes[1].set_title("Exponential Decay Metric")
    axes[1].set_xlabel("Timestep")

    cmap = plt.get_cmap('viridis')
    for i, cdf in enumerate(cdf_history):
        axes[2].plot(cdf, color=cmap(i / max(len(cdf_history), 1)), alpha=0.6)
    axes[2].set_title("Evolution of Energy CDFs")
    axes[2].set_xlabel("Bandwidth (k)")
    axes[2].set_ylabel("Cumulative Energy Ratio")

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- #
#  Geometry helpers
# ----------------------------------------------------------------------------- #

def get_spatial_rgb_base(vertices):
    min_val = vertices.min(axis=0)
    max_val = vertices.max(axis=0)
    denom = max_val - min_val
    denom[denom == 0] = 1.0
    return (vertices - min_val) / denom


def face_areas_and_normals(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    area = 0.5 * np.linalg.norm(n, axis=1)
    return area, n / np.maximum(2.0 * area, 1e-16)[:, None]


def vertex_areas(vertices, faces):
    area, _ = face_areas_and_normals(vertices, faces)
    va = np.zeros(np.asarray(vertices).shape[0])
    for k in range(3):
        np.add.at(va, faces[:, k], area / 3.0)
    return va


def vertex_normals(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def cotangent_stiffness(vertices, faces):
    """Cotangent stiffness matrix W (positive semi-definite): (W x)_i = Σ_j w_ij (x_i - x_j)."""
    v = np.asarray(vertices, dtype=np.float64)
    n = v.shape[0]
    I, J, Wv = [], [], []
    for k in range(3):
        i, j, l = faces[:, k], faces[:, (k + 1) % 3], faces[:, (k + 2) % 3]
        a, b = v[i] - v[l], v[j] - v[l]
        cot = np.einsum('ij,ij->i', a, b) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-16)
        I += [i, j]; J += [j, i]; Wv += [0.5 * cot, 0.5 * cot]
    Wm = coo_matrix((np.concatenate(Wv), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr()
    return diags(np.asarray(Wm.sum(axis=1)).ravel()) - Wm


def mean_curvature(vertices, faces):
    """Signed discrete mean curvature H = (W x)_i · n_i / (2 A_i)  (cotangent formula)."""
    v = np.asarray(vertices, dtype=np.float64)
    Hvec = cotangent_stiffness(v, faces) @ v
    A = np.maximum(vertex_areas(v, faces), 1e-16)
    return np.einsum('ij,ij->i', Hvec, vertex_normals(v, faces)) / (2.0 * A)


def signed_volume(vertices, faces):
    """Enclosed volume by the divergence theorem (meaningful for closed, consistently oriented meshes)."""
    v = np.asarray(vertices, dtype=np.float64)
    return float(np.einsum('ij,ij->i', v[faces[:, 0]], np.cross(v[faces[:, 1]], v[faces[:, 2]])).sum() / 6.0)


def _average_faces_to_vertices(face_values, faces, weights, valid, num_vertices):
    """Weighted average of per-face quantities onto vertices, ignoring invalid faces."""
    w = np.where(valid, weights, 0.0)
    num = np.zeros(num_vertices)
    den = np.zeros(num_vertices)
    vals = np.where(valid, face_values, 0.0)
    for k in range(3):
        np.add.at(num, faces[:, k], vals * w)
        np.add.at(den, faces[:, k], w)
    out = np.zeros(num_vertices)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out, ok


# ----------------------------------------------------------------------------- #
#  Deformation fields (all use p2p = mesh2 -> mesh1 map: p2p[j] is the mesh1 vertex matched to mesh2 vertex j)
# ----------------------------------------------------------------------------- #

def compute_displacement_velocity(vertices1, vertices2, p2p, dt=1.0):
    """Displacement (and its magnitude / dt = velocity) of every mesh2 vertex from its matched mesh1 vertex."""
    v1_gpu, v2_gpu = to_gpu(np.asarray(vertices1, dtype=np.float64)), to_gpu(np.asarray(vertices2, dtype=np.float64))
    p2p_gpu = to_gpu(np.asarray(p2p, dtype=np.int64))

    displacements = v2_gpu - v1_gpu[p2p_gpu]
    magnitudes = xp.linalg.norm(displacements, axis=1) / dt

    return to_cpu(magnitudes), to_cpu(displacements)


def compute_finite_element_strain(vertices1, vertices2, faces2, p2p, rel_tol=1e-3):
    """
    Edge (engineering) strain averaged on vertices: (l2 - l1) / l1.
    Edges whose reference length collapses (two mesh2 vertices matched to the same or nearly the same
    mesh1 vertex) are *excluded* instead of being clamped to 1e-6, which produced strains of ~1e6.
    """
    faces2 = _as_faces(faces2)
    edges = np.vstack((faces2[:, [0, 1]], faces2[:, [1, 2]], faces2[:, [2, 0]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)

    v1, v2 = np.asarray(vertices1, dtype=np.float64), np.asarray(vertices2, dtype=np.float64)
    p2p = np.asarray(p2p, dtype=np.int64)

    lengths2 = np.linalg.norm(v2[edges[:, 0]] - v2[edges[:, 1]], axis=1)
    lengths1 = np.linalg.norm(v1[p2p[edges[:, 0]]] - v1[p2p[edges[:, 1]]], axis=1)

    valid = lengths1 > rel_tol * np.median(lengths2)
    edge_strain = np.where(valid, (lengths2 - lengths1) / np.where(valid, lengths1, 1.0), 0.0)

    n = v2.shape[0]
    vertex_strain = np.zeros(n)
    count = np.zeros(n)
    w = valid.astype(np.float64)
    for col in range(2):
        np.add.at(vertex_strain, edges[:, col], edge_strain * w)
        np.add.at(count, edges[:, col], w)
    ok = count > 0
    vertex_strain[ok] /= count[ok]
    return vertex_strain


def compute_area_strain(vertices1, vertices2, faces2, p2p, rel_tol=1e-3):
    """Face area strain (A2 - A1)/A1 averaged on vertices; collapsed reference triangles are excluded."""
    faces2 = _as_faces(faces2)
    v1, v2 = np.asarray(vertices1, dtype=np.float64), np.asarray(vertices2, dtype=np.float64)
    p2p = np.asarray(p2p, dtype=np.int64)

    areas2, _ = face_areas_and_normals(v2, faces2)
    areas1, _ = face_areas_and_normals(v1, p2p[faces2])

    valid = areas1 > rel_tol * np.median(areas2)
    face_area_strain = np.where(valid, (areas2 - areas1) / np.where(valid, areas1, 1.0), 0.0)
    strain, _ = _average_faces_to_vertices(face_area_strain, faces2, np.ones(len(faces2)), valid, v2.shape[0])
    return strain


def compute_deformation_gradient_metrics(vertices1, vertices2, faces2, p2p, rel_tol=1e-3):
    """
    Continuum-mechanics deformation measures from the per-triangle deformation gradient F
    (reference = mesh1 triangle pulled back through p2p, deformed = mesh2 triangle).

    Right Cauchy–Green C = FᵀF, principal stretches λ1 ≥ λ2 = sqrt(eig(C)).
    Per-vertex (area-weighted average over valid incident faces):
        stretch_max, stretch_min        principal stretches (1 = no deformation)
        principal_strain_max/min        Green–Lagrange principal strains ½(λ² − 1)
        shear_anisotropy                log(λ1/λ2): 0 for pure dilation, grows with shearing
        max_shear_strain                ½(ε1 − ε2)
        elastic_energy_density          ARAP / Saint-Venant style (λ1−1)² + (λ2−1)²
        area_ratio                      λ1·λ2 = A2/A1 (dilatation)
        dilatation_log                  log(λ1·λ2), symmetric for growth / shrinkage
    Also returns the per-face quantities and the reference face areas for global integration.
    """
    faces2 = _as_faces(faces2)
    v1, v2 = np.asarray(vertices1, dtype=np.float64), np.asarray(vertices2, dtype=np.float64)
    p2p = np.asarray(p2p, dtype=np.int64)

    X = v1[p2p[faces2]]                  # (F,3,3) reference triangles
    x = v2[faces2]                       # (F,3,3) deformed triangles
    e1r, e2r = X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]
    e1d, e2d = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]

    nr = np.cross(e1r, e2r)
    area_r = 0.5 * np.linalg.norm(nr, axis=1)
    area_d = 0.5 * np.linalg.norm(np.cross(e1d, e2d), axis=1)
    valid = (area_r > rel_tol * np.median(area_d)) & (area_d > 0)

    # Orthonormal in-plane frame of the reference triangle -> 2D material coordinates.
    u = e1r / np.maximum(np.linalg.norm(e1r, axis=1, keepdims=True), 1e-16)
    w = nr / np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-16)
    vv = np.cross(w, u)
    Dm = np.stack([np.stack([np.einsum('ij,ij->i', e1r, u), np.einsum('ij,ij->i', e2r, u)], -1),
                   np.stack([np.einsum('ij,ij->i', e1r, vv), np.einsum('ij,ij->i', e2r, vv)], -1)], 1)   # (F,2,2)
    Ds = np.stack([e1d, e2d], axis=-1)                                                                   # (F,3,2)

    det = Dm[:, 0, 0] * Dm[:, 1, 1] - Dm[:, 0, 1] * Dm[:, 1, 0]
    safe_det = np.where(np.abs(det) > 1e-16, det, 1.0)
    Dm_inv = np.empty_like(Dm)
    Dm_inv[:, 0, 0], Dm_inv[:, 1, 1] = Dm[:, 1, 1] / safe_det, Dm[:, 0, 0] / safe_det
    Dm_inv[:, 0, 1], Dm_inv[:, 1, 0] = -Dm[:, 0, 1] / safe_det, -Dm[:, 1, 0] / safe_det

    Fg = np.einsum('fij,fjk->fik', Ds, Dm_inv)               # (F,3,2)
    C = np.einsum('fji,fjk->fik', Fg, Fg)                    # (F,2,2)
    tr = C[:, 0, 0] + C[:, 1, 1]
    dt = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0]
    disc = np.sqrt(np.maximum(tr ** 2 - 4.0 * dt, 0.0))
    lam1 = np.sqrt(np.maximum(0.5 * (tr + disc), 0.0))
    lam2 = np.sqrt(np.maximum(0.5 * (tr - disc), 1e-16))

    with np.errstate(divide='ignore', invalid='ignore'):
        shear = np.where(valid, np.log(np.maximum(lam1, 1e-16) / lam2), 0.0)
        dil_log = np.where(valid, np.log(np.maximum(lam1 * lam2, 1e-16)), 0.0)
    face = {
        'stretch_max': lam1,
        'stretch_min': lam2,
        'principal_strain_max': 0.5 * (lam1 ** 2 - 1.0),
        'principal_strain_min': 0.5 * (lam2 ** 2 - 1.0),
        'shear_anisotropy': shear,
        'max_shear_strain': 0.25 * (lam1 ** 2 - lam2 ** 2),
        'elastic_energy_density': (lam1 - 1.0) ** 2 + (lam2 - 1.0) ** 2,
        'area_ratio': lam1 * lam2,
        'dilatation_log': dil_log,
    }
    n = v2.shape[0]
    vertex = {k: _average_faces_to_vertices(val, faces2, area_r, valid, n)[0] for k, val in face.items()}
    return vertex, face, area_r, valid


def compute_flow_decomposition(vertices2, faces2, displacements):
    """Normal (signed) and tangential (magnitude) components of the displacement on mesh2."""
    faces2 = _as_faces(faces2)
    v2 = np.asarray(vertices2, dtype=np.float64)
    try:
        faces_pv = np.hstack([np.full((faces2.shape[0], 1), 3, dtype=np.int64), faces2]).ravel()
        temp_mesh = pv.PolyData(v2, faces_pv).compute_normals(cell_normals=False, point_normals=True,
                                                             split_vertices=False, consistent_normals=True)
        normals = np.asarray(temp_mesh['Normals'])
        if normals.shape != v2.shape or not np.all(np.isfinite(normals)):
            raise ValueError("pyvista normals have unexpected shape")
    except Exception:
        normals = vertex_normals(v2, faces2)

    displacements = np.asarray(displacements, dtype=np.float64)
    normal_vel_mag = np.einsum('ij,ij->i', displacements, normals)
    tangential_vel_vec = displacements - normals * normal_vel_mag[:, np.newaxis]
    tangential_vel_mag = np.linalg.norm(tangential_vel_vec, axis=1)

    return normal_vel_mag, tangential_vel_mag


def compute_normal_rotation(vertices1, faces1, vertices2, faces2, p2p):
    """Angle (rad) between the surface normal at a mesh2 vertex and at its matched mesh1 vertex (bending/twist)."""
    n1 = vertex_normals(vertices1, _as_faces(faces1))[np.asarray(p2p, dtype=np.int64)]
    n2 = vertex_normals(vertices2, _as_faces(faces2))
    return np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2), -1.0, 1.0))


def compute_curvature_change(vertices1, faces1, vertices2, faces2, p2p):
    """Change of mean curvature H2(x) - H1(p2p(x)); bending deformation of the surface."""
    H1 = mean_curvature(vertices1, _as_faces(faces1))
    H2 = mean_curvature(vertices2, _as_faces(faces2))
    return H2 - H1[np.asarray(p2p, dtype=np.int64)]


def compute_global_metrics(vertices1, faces1, vertices2, faces2, p2p, v_mag, face_energy, area_r, valid, dt=1.0):
    """Integrated quantities for one transition (one CSV row)."""
    faces1, faces2 = _as_faces(faces1), _as_faces(faces2)
    p2p = np.asarray(p2p, dtype=np.int64)
    A1 = vertex_areas(vertices1, faces1).sum()
    A2 = vertex_areas(vertices2, faces2).sum()
    Vol1, Vol2 = signed_volume(vertices1, faces1), signed_volume(vertices2, faces2)
    va2 = vertex_areas(vertices2, faces2)
    return {
        'area_ratio': A2 / A1 if A1 > 0 else np.nan,
        'volume_ratio': Vol2 / Vol1 if abs(Vol1) > 1e-12 else np.nan,
        'mean_speed': float(np.average(v_mag, weights=va2)),
        'max_speed': float(v_mag.max()),
        'total_elastic_energy': float(np.sum(face_energy[valid] * area_r[valid])),
        'mean_elastic_energy_density': float(np.sum(face_energy[valid] * area_r[valid]) / max(area_r[valid].sum(), 1e-16)),
        'p2p_injectivity': float(np.unique(p2p).size / vertices1.shape[0]),
        'collapsed_faces_fraction': float(1.0 - valid.mean()),
    }


# ----------------------------------------------------------------------------- #
#  Output
# ----------------------------------------------------------------------------- #

FIELD_KEYS = ('velocity', 'strain', 'area_strain', 'normal_flow', 'tangential_flow',
              'acceleration', 'stretch_max', 'stretch_min', 'principal_strain_max', 'principal_strain_min',
              'shear_anisotropy', 'max_shear_strain', 'elastic_energy_density', 'area_ratio', 'dilatation_log',
              'normal_rotation', 'curvature_change')


def pv_field_name(key):
    """npz key -> pyvista point_data name ('area_strain' -> 'Area_Strain')."""
    return key.replace('_', ' ').title().replace(' ', '_')


def create_pv_polydata(d):
    """
    PolyData for one frame of computing_fields(). Frames saved by the current computing_fields
    carry aligned vertices (aligned=True) and are already in the display frame; legacy frames
    (raw mesh_mat2object vertices, no flag) receive the historical x90/z90 rotation.
    """
    faces = _as_faces(d['faces'])
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()

    mesh = pv.PolyData(np.asarray(d['vertices'], dtype=np.float64), faces_pv)
    mesh.point_data['RGB'] = (np.clip(d['colors'], 0, 1) * 255).astype(np.uint8)
    for key in FIELD_KEYS:
        if key in d:
            mesh.point_data[pv_field_name(key)] = np.asarray(d[key], dtype=np.float64)

    if not bool(np.asarray(d.get('aligned', False))):
        mesh.rotate_x(90, inplace=True)
        mesh.rotate_z(90, inplace=True)
    return mesh


def plot_global_physical_metrics(csv_path, dpi=150, single_file=True):
    """
    Time-series report of global_physical_metrics.csv (one row per transition T{i-1} -> T{i}).
    Four figures are saved in a 'plots' folder next to the CSV:
        1_Size_and_Kinematics   : area / volume ratios, mean and max speed
        2_Strain_and_Energy     : elastic energy, mean |strain|, mean |area strain|, shear anisotropy
        3_Flow_and_Bending      : normal / tangential flow, normal rotation, curvature change
        4_Map_Quality           : p2p injectivity and collapsed-face fraction (trust indicators)
    Columns missing from the CSV (or entirely NaN, e.g. volume ratio of open meshes) are skipped.
    Returns the plots folder path.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Could not find CSV file at {csv_path}")
        return None
    df = pd.read_csv(csv_file)
    if df.empty:
        print("No transitions found in the physical metrics CSV; nothing to plot.")
        return None
    plots_folder = csv_file.parent / 'plots'
    os.makedirs(plots_folder, exist_ok=True)

    if single_file:
        print("\nGenerating visual reports for the global physical metrics...")

    if 'Time' in df:                                       # acquisition times known (frame_times)
        t = df['Time']
        xlabel = f"time ({df['Time_Unit'].iloc[0] if 'Time_Unit' in df else 's'}) at the end of the transition"
    else:
        t = df['Time_Step'] if 'Time_Step' in df else np.arange(1, len(df) + 1)
        xlabel = r"Time Step (Transition $T_{i-1} \to T_i$)"

    def has(col):
        return col in df.columns and not df[col].isna().all()

    def line(ax, col, label, color, marker='o', linestyle='-'):
        if has(col):
            ax.plot(t, df[col], label=label, color=color, marker=marker, linestyle=linestyle, linewidth=2, markersize=5)
            return True
        return False

    def finish(ax, title, ylabel, reference=None):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)
        if reference is not None:
            ax.axhline(reference, color='gray', linestyle=':', linewidth=1)
        ax.grid(True, linestyle='--', alpha=0.6)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)

    sns.set_theme(style="whitegrid")

    # 1) size and kinematics --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    line(axes[0], 'area_ratio', 'Area ratio $A_{t_i}/A_{t_{i-1}}$', 'royalblue')
    line(axes[0], 'volume_ratio', 'Volume ratio $V_{t_i}/V_{t_{i-1}}$', 'darkorange', marker='s')
    finish(axes[0], "Global Size Change", "Ratio (1 = no change)", reference=1.0)
    line(axes[1], 'mean_speed', 'Mean speed (area weighted)', 'seagreen')
    line(axes[1], 'max_speed', 'Max speed', 'crimson', marker='^', linestyle='--')
    finish(axes[1], "Kinematics", r"Displacement / $\Delta t$")
    fig.suptitle("Size & Kinematics", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plots_folder / '1_Size_and_Kinematics.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # 2) strain and energy ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    line(axes[0], 'total_elastic_energy', 'Total elastic energy', 'purple')
    finish(axes[0], "Elastic Energy of the Transition", "Energy (ARAP)")
    ax_tw = axes[0].twinx()
    if line(ax_tw, 'mean_elastic_energy_density', 'Mean energy density', 'orchid', marker='d', linestyle='--'):
        ax_tw.set_ylabel("Energy density", fontsize=10)
        ax_tw.legend(loc='upper right', fontsize=9)
        axes[0].legend(loc='upper left', fontsize=9)
    line(axes[1], 'mean_abs_strain', 'Mean |edge strain|', 'teal')
    line(axes[1], 'mean_abs_area_strain', 'Mean |area strain|', 'darkgoldenrod', marker='s')
    finish(axes[1], "Stretch / Compression Intensity", "Strain")
    line(axes[2], 'mean_shear_anisotropy', r'Mean $\log(\lambda_1/\lambda_2)$', 'firebrick')
    finish(axes[2], "Shape Change vs. Size Change", "Shear anisotropy (0 = pure dilation)", reference=0.0)
    fig.suptitle("Strain & Energy", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plots_folder / '2_Strain_and_Energy.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # 3) flow and bending -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    line(axes[0], 'mean_normal_flow', 'Mean normal flow (+ out / - in)', 'navy')
    line(axes[0], 'mean_tangential_flow', 'Mean tangential flow', 'darkorange', marker='s')
    finish(axes[0], "Flow Decomposition", "Displacement", reference=0.0)
    line(axes[1], 'mean_normal_rotation_deg', 'Mean normal rotation [deg]', 'slateblue')
    finish(axes[1], "Bending", "Degrees")
    ax_tw = axes[1].twinx()
    if line(ax_tw, 'mean_abs_curvature_change', r'Mean $|\Delta H|$', 'darkgreen', marker='v', linestyle='--'):
        ax_tw.set_ylabel("Mean curvature change", fontsize=10)
        ax_tw.legend(loc='upper right', fontsize=9)
        axes[1].legend(loc='upper left', fontsize=9)
    fig.suptitle("Flow & Bending", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(plots_folder / '3_Flow_and_Bending.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # 4) map quality ----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    line(ax, 'p2p_injectivity', 'p2p injectivity (1 = bijection)', 'black')
    line(ax, 'collapsed_faces_fraction', 'Collapsed faces fraction', 'red', marker='x', linestyle='--')
    finish(ax, "Functional Map Quality (trust indicators of the physical fields)", "Fraction")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(plots_folder / '4_Map_Quality.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if single_file:
        print(f"Physical metrics reports generated in: {plots_folder}")
    return str(plots_folder)


def _find_p2p_file(matrix_folder, i):
    """New zero-padded naming first, legacy 'FMV_{i}{i-1}.npy' second."""
    for name in (f'FMV_T{i:04d}_T{i-1:04d}.npy', f'FMV_{i}{i-1}.npy'):
        p = matrix_folder / name
        if p.exists():
            return p
    return None


def computing_fields(mesh_folder_path, matrix_folder_path, output_folder_path, single_file=True, dt=1.0, loader=None, plot=True,
                     times=None):
    """
    Computes all physics-based deformation fields step by step and caches them as .npz files,
    plus a CSV with integrated (global) quantities per transition (global_physical_metrics.csv)
    and, if `plot` is True, its time-series report in the 'plots' subfolder.

    times: acquisition times of the frames (FrameTimes or array, see dyn_common.resolve_frame_times). When given,
    every transition uses its own interval (irregular time-lapses are handled exactly): velocity = displacement / dt_i,
    acceleration = (v_i - v_(i-1)) / ((dt_i + dt_(i-1)) / 2), and the csv gets Time / Dt and the rates of area and
    volume change. Without times the uniform `dt` is used as before (the csv is unchanged).

    Meshes are loaded with the same loader used for the functional maps (aligned meshes), so the
    displacement field does not include the rigid alignment transform.
    """
    loader = loader or load_aligned_mesh
    mesh_folder = Path(mesh_folder_path)
    matrix_folder = Path(matrix_folder_path)
    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    obj_files = sorted([f for f in mesh_folder.iterdir() if f.is_file() and f.suffix in ('.obj', '.mat')], key=natural_sort_key)
    if not obj_files:
        print(f"Error: No files found in target directory: {mesh_folder_path}")
        return False

    if single_file:
        print(f"\nComputing physics tracking fields for {len(obj_files)} timesteps...")

    meshn_1 = loader(obj_files[0])
    v_prev, f_prev = np.asarray(meshn_1.vertices, dtype=np.float64), _as_faces(meshn_1.faces)
    tracking_colors = get_spatial_rgb_base(v_prev)
    n0 = v_prev.shape[0]

    # Base frame: no deformation -> 0 for strains / flows, 1 for stretches and area ratio.
    base_fields = {k: (np.ones(n0) if k in ('stretch_max', 'stretch_min', 'area_ratio') else np.zeros(n0)) for k in FIELD_KEYS}
    np.savez(output_folder / 'frame_0000.npz', vertices=v_prev, faces=f_prev, colors=tracking_colors, aligned=True, **base_fields)

    prev_disp = None
    global_rows = []
    t_arr = None
    if times is not None:
        t_arr = np.asarray(getattr(times, "t", times), dtype=np.float64)
        if t_arr.size != len(obj_files):
            print(f"[Warning] {t_arr.size} frame times for {len(obj_files)} meshes: the uniform dt is used.")
            t_arr = None
    unit = getattr(times, "unit", "s") if t_arr is not None else None
    dt_prev = None

    for i in tqdm(range(1, len(obj_files)), desc='Processing structural metrics', leave=single_file):
        meshn = loader(obj_files[i])
        v_cur, f_cur = np.asarray(meshn.vertices, dtype=np.float64), _as_faces(meshn.faces)

        p2p_file = _find_p2p_file(matrix_folder, i)
        if p2p_file is None:
            print(f"Error: Required point-to-point map for transition T{i-1}->T{i} missing in {matrix_folder}. Aborting.")
            return False
        p2p_zo = np.asarray(np.load(p2p_file), dtype=np.int64)
        if p2p_zo.shape[0] != v_cur.shape[0]:
            print(f"Error: {p2p_file.name} has {p2p_zo.shape[0]} entries but frame {i} has {v_cur.shape[0]} vertices.")
            return False

        tracking_colors = tracking_colors[p2p_zo]
        dt_i = float(t_arr[i] - t_arr[i - 1]) if t_arr is not None else dt
        v_mag, displacements = compute_displacement_velocity(v_prev, v_cur, p2p_zo, dt=dt_i)
        strain = compute_finite_element_strain(v_prev, v_cur, f_cur, p2p_zo)
        area_strain = compute_area_strain(v_prev, v_cur, f_cur, p2p_zo)
        norm_mag, tang_mag = compute_flow_decomposition(v_cur, f_cur, displacements)
        defo, face_defo, area_r, valid = compute_deformation_gradient_metrics(v_prev, v_cur, f_cur, p2p_zo)
        normal_rot = compute_normal_rotation(v_prev, f_prev, v_cur, f_cur, p2p_zo)
        curv_change = compute_curvature_change(v_prev, f_prev, v_cur, f_cur, p2p_zo)

        # Acceleration: change of the displacement of the *same material point* between two transitions.
        if prev_disp is not None:
            if t_arr is None:
                acceleration = np.linalg.norm(displacements - prev_disp[p2p_zo], axis=1) / dt ** 2
            else:                                          # irregular intervals: change of velocity over the mean interval
                acceleration = (np.linalg.norm(displacements / dt_i - prev_disp[p2p_zo] / dt_prev, axis=1)
                                / (0.5 * (dt_i + dt_prev)))
        else:
            acceleration = np.zeros(v_cur.shape[0])
        prev_disp = displacements; dt_prev = dt_i

        np.savez(output_folder / f'frame_{i:04d}.npz', vertices=v_cur, faces=f_cur, colors=tracking_colors, aligned=True,
                 velocity=v_mag, strain=strain, area_strain=area_strain, normal_flow=norm_mag, tangential_flow=tang_mag,
                 acceleration=acceleration, normal_rotation=normal_rot, curvature_change=curv_change, **defo)

        row = {'Transition': f"T{i-1} -> T{i}", 'Time_Step': i}
        if t_arr is not None:
            row.update({'Time': float(t_arr[i]), 'Dt': dt_i, 'Time_Unit': unit})
        row.update(compute_global_metrics(v_prev, f_prev, v_cur, f_cur, p2p_zo, v_mag,
                                          face_defo['elastic_energy_density'], area_r, valid, dt=dt_i))
        if t_arr is not None:                               # relative rates of change per time unit
            row['area_change_rate'] = (row['area_ratio'] - 1.0) / dt_i
            row['volume_change_rate'] = (row['volume_ratio'] - 1.0) / dt_i if np.isfinite(row['volume_ratio']) else np.nan
        row.update({
            'mean_abs_strain': float(np.mean(np.abs(strain))),
            'mean_abs_area_strain': float(np.mean(np.abs(area_strain))),
            'mean_shear_anisotropy': float(np.mean(defo['shear_anisotropy'])),
            'mean_normal_flow': float(np.mean(norm_mag)),
            'mean_tangential_flow': float(np.mean(tang_mag)),
            'mean_normal_rotation_deg': float(np.degrees(np.mean(normal_rot))),
            'mean_abs_curvature_change': float(np.mean(np.abs(curv_change))),
        })
        global_rows.append(row)

        meshn_1, v_prev, f_prev = meshn, v_cur, f_cur

    csv_path = output_folder / 'global_physical_metrics.csv'
    pd.DataFrame(global_rows).to_csv(csv_path, index=False)
    if plot:
        plot_global_physical_metrics(csv_path, single_file=single_file)
    return True