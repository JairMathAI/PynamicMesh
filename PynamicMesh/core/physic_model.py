import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from tqdm.auto import tqdm
import seaborn as sns
from PynamicMesh.utils.tools import  mesh_mat2object

try:
    import cupy as xp
    import cupyx as xxp 
    GPU_AVAILABLE = True
    print("[INFO] CuPy detected. Utilizing GPU for Physics and metric computations.")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
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

def compute_heatmap_similarity(matrix1, matrix2):
    """Computes cross-heatmap similarity metrics on the GPU."""
    m1 = to_gpu(matrix1).flatten()
    m2 = to_gpu(matrix2).flatten()
    
    euclidean = xp.linalg.norm(m1 - m2)
    manhattan = xp.sum(xp.abs(m1 - m2))
    
    if xp.std(m1) == 0 or xp.std(m2) == 0:
        pearson = 0.0
    else:
        pearson = xp.corrcoef(m1, m2)[0, 1]
        
    r1 = xp.argsort(xp.argsort(m1))
    r2 = xp.argsort(xp.argsort(m2))
    if xp.std(r1) == 0 or xp.std(r2) == 0:
        spearman = 0.0
    else:
        spearman = xp.corrcoef(r1, r2)[0, 1]
        
    p = (m1 ** 2)
    q = (m2 ** 2)
    
    sum_p, sum_q = xp.sum(p), xp.sum(q)
    p = p / sum_p if sum_p > 0 else xp.ones_like(p) / len(p)
    q = q / sum_q if sum_q > 0 else xp.ones_like(q) / len(q)
    
    m = 0.5 * (p + q)
    eps = 1e-12 
    kl_p = xp.sum(p * xp.log((p + eps) / (m + eps)))
    kl_q = xp.sum(q * xp.log((q + eps) / (m + eps)))
    jsd = 0.5 * kl_p + 0.5 * kl_q
    
    # Return metrics to CPU as standard floats
    return float(to_cpu(jsd)), float(to_cpu(pearson)), float(to_cpu(spearman)), float(to_cpu(manhattan)), float(to_cpu(euclidean))


def plot_similarity_metrics(similarity_history, path, dpi=150):
    """
    Plots the evolution of cross-heatmap similarity metrics across timesteps.
    """
    if not similarity_history:
        return
        
    hist = np.array(similarity_history)  
    timesteps = np.arange(2, len(hist) + 2)  
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(timesteps, hist[:, 1], marker='o', label='Pearson', color='royalblue')
    axes[0].plot(timesteps, hist[:, 2], marker='s', label='Spearman', color='darkorange')
    axes[0].set_title("Cross-Heatmap Alignment Profile")
    axes[0].set_xlabel("Timestep (Transition $T_{i-1} \to T_i$)")
    axes[0].set_ylabel("Correlation Value")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(timesteps, hist[:, 3], marker='o', label='Manhattan ($L_1$)', color='crimson')
    axes[1].plot(timesteps, hist[:, 4], marker='s', label='Euclidean ($L_2$)', color='purple')
    axes[1].set_title("Spectral Coordinate Absolute Distance")
    axes[1].set_xlabel("Timestep (Transition $T_{i-1} \to T_i$)")
    axes[1].set_ylabel("Distance Magnitude")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    axes[2].plot(timesteps, hist[:, 0], marker='o', label='Jensen-Shannon Div.', color='forestgreen')
    axes[2].set_title("Spectral Energy Field Divergence (JSD)")
    axes[2].set_xlabel("Timestep (Transition $T_{i-1} \to T_i$)")
    axes[2].set_ylabel("Divergence Scale [0, 1]")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def get_spatial_rgb_base(vertices):
    min_val = vertices.min(axis=0)
    max_val = vertices.max(axis=0)
    denom = max_val - min_val
    denom[denom == 0] = 1.0
    return (vertices - min_val) / denom


def compute_displacement_velocity(vertices1, vertices2, p2p):
    v1_gpu, v2_gpu = to_gpu(vertices1), to_gpu(vertices2)
    p2p_gpu = to_gpu(p2p)
    
    matched_vertices1 = v1_gpu[p2p_gpu]
    displacements = v2_gpu - matched_vertices1
    magnitudes = xp.linalg.norm(displacements, axis=1)
    
    return to_cpu(magnitudes), to_cpu(displacements)


def compute_finite_element_strain(vertices1, vertices2, faces2, p2p):
    """Vectorized and GPU-accelerated finite element strain calculation."""
    # 1. Topology extraction is safer and highly optimized on CPU NumPy
    edges = np.vstack((faces2[:, [0, 1]], faces2[:, [1, 2]], faces2[:, [2, 0]]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0) # Removes duplicates globally
    
    # 2. Move to GPU for heavy math
    edges_gpu = to_gpu(edges)
    v1_gpu, v2_gpu = to_gpu(vertices1), to_gpu(vertices2)
    p2p_gpu = to_gpu(p2p)
    
    lengths2 = xp.linalg.norm(v2_gpu[edges_gpu[:, 0]] - v2_gpu[edges_gpu[:, 1]], axis=1)
    
    p1_u = v1_gpu[p2p_gpu[edges_gpu[:, 0]]]
    p1_v = v1_gpu[p2p_gpu[edges_gpu[:, 1]]]
    lengths1 = xp.linalg.norm(p1_u - p1_v, axis=1)
    
    lengths1_safe = xp.where(lengths1 == 0, 1e-6, lengths1)
    edge_strain = (lengths2 - lengths1_safe) / lengths1_safe
    
    num_vertices = vertices2.shape[0]
    vertex_strain = xp.zeros(num_vertices)
    vertex_edge_count = xp.zeros(num_vertices)
    
    # Scatter-add operations are needed for aggregating edge data to vertices
    # xxp.scatter_add is CuPy's equivalent to np.add.at
    if GPU_AVAILABLE:
        xxp.scatter_add(vertex_strain, edges_gpu[:, 0], edge_strain)
        xxp.scatter_add(vertex_strain, edges_gpu[:, 1], edge_strain)
        xxp.scatter_add(vertex_edge_count, edges_gpu[:, 0], 1)
        xxp.scatter_add(vertex_edge_count, edges_gpu[:, 1], 1)
    else:
        np.add.at(vertex_strain, edges[:, 0], edge_strain)
        np.add.at(vertex_strain, edges[:, 1], edge_strain)
        np.add.at(vertex_edge_count, edges[:, 0], 1)
        np.add.at(vertex_edge_count, edges[:, 1], 1)
        
    vertex_edge_count[vertex_edge_count == 0] = 1
    vertex_strain /= vertex_edge_count
    
    return to_cpu(vertex_strain)


def compute_area_strain(vertices1, vertices2, faces2, p2p):
    """Vectorized and GPU-accelerated area strain calculation."""
    v1_gpu, v2_gpu = to_gpu(vertices1), to_gpu(vertices2)
    f2_gpu, p2p_gpu = to_gpu(faces2), to_gpu(p2p)
    
    v0_2, v1_2, v2_2 = v2_gpu[f2_gpu[:, 0]], v2_gpu[f2_gpu[:, 1]], v2_gpu[f2_gpu[:, 2]]
    areas2 = 0.5 * xp.linalg.norm(xp.cross(v1_2 - v0_2, v2_2 - v0_2), axis=1)

    v0_1, v1_1, v2_1 = v1_gpu[p2p_gpu[f2_gpu[:, 0]]], v1_gpu[p2p_gpu[f2_gpu[:, 1]]], v1_gpu[p2p_gpu[f2_gpu[:, 2]]]
    areas1 = 0.5 * xp.linalg.norm(xp.cross(v1_1 - v0_1, v2_1 - v0_1), axis=1)

    areas1_safe = xp.where(areas1 == 0, 1e-8, areas1)
    face_area_strain = (areas2 - areas1_safe) / areas1_safe

    num_vertices = vertices2.shape[0]
    vertex_area_strain = xp.zeros(num_vertices)
    vertex_face_count = xp.zeros(num_vertices)

    for i in range(3):
        if GPU_AVAILABLE:
            xxp.scatter_add(vertex_area_strain, f2_gpu[:, i], face_area_strain)
            xxp.scatter_add(vertex_face_count, f2_gpu[:, i], 1)
        else:
            np.add.at(vertex_area_strain, faces2[:, i], to_cpu(face_area_strain))
            np.add.at(vertex_face_count, faces2[:, i], 1)

    vertex_face_count[vertex_face_count == 0] = 1
    strain_res = vertex_area_strain / vertex_face_count
    
    return to_cpu(strain_res)


def compute_flow_decomposition(vertices2, faces2, displacements):
    faces_pv = np.empty((faces2.shape[0], 4), dtype=int)
    faces_pv[:, 0] = 3
    faces_pv[:, 1:] = faces2
    
    temp_mesh = pv.PolyData(vertices2, faces_pv.flatten())
    temp_mesh = temp_mesh.compute_normals(cell_normals=False, point_normals=True)
    normals = temp_mesh['Normals']
    
    normal_vel_mag = np.einsum('ij,ij->i', displacements, normals)
    tangential_vel_vec = displacements - normals * normal_vel_mag[:, np.newaxis]
    tangential_vel_mag = np.linalg.norm(tangential_vel_vec, axis=1)
    
    return normal_vel_mag, tangential_vel_mag


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
    """GPU-accelerated metric calculation for Functional Maps."""
    mat_gpu = to_gpu(matrix)
    n, m = mat_gpu.shape
    matrix_sq = mat_gpu**2
    total_energy = xp.sum(matrix_sq)
    
    if total_energy == 0:
        return 0, 0, np.zeros(max(n, m))

    # Create coordinate grids natively on GPU
    I, J = xp.ogrid[:n, :m]
    dist = xp.abs(I - J)
    
    max_dist = max(n, m) - 1
    inertia = xp.sum((dist**2) * matrix_sq) / (total_energy * (max_dist**2))
    inertia_metric = 1.0 - inertia
    
    decay_metric = xp.sum(matrix_sq * xp.exp(-alpha * dist)) / total_energy
    
    cdf_bandwidth = xp.zeros(max(n, m))
    for k in range(max(n, m)):
        mask = (dist <= k)
        cdf_bandwidth[k] = xp.sum(matrix_sq[mask]) / total_energy
        
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
        color = cmap(i / len(cdf_history))
        axes[2].plot(cdf, color=color, alpha=0.6)
    axes[2].set_title("Evolution of Energy CDFs")
    axes[2].set_xlabel("Bandwidth (k)")
    axes[2].set_ylabel("Cumulative Energy Ratio")

    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def create_pv_polydata(d):
    faces_pv = np.empty((d['faces'].shape[0], 4), dtype=int)
    faces_pv[:, 0] = 3
    faces_pv[:, 1:] = d['faces']
    faces_pv = faces_pv.flatten()
    
    mesh = pv.PolyData(d['vertices'], faces_pv)
    mesh.point_data['RGB'] = (np.clip(d['colors'], 0, 1) * 255).astype(np.uint8)
    mesh.point_data['Velocity'] = d['velocity']
    mesh.point_data['Strain'] = d['strain']
    mesh.point_data['Area_Strain'] = d['area_strain']
    mesh.point_data['Normal_Flow'] = d['normal_flow']
    mesh.point_data['Tangential_Flow'] = d['tangential_flow']

    mesh.rotate_x(90, inplace=True)
    mesh.rotate_z(90, inplace=True)

    return mesh


def computing_fields(mesh_folder_path, matrix_folder_path, output_folder_path,single_file=True):
    """
    Computes all physics-based deformation fields step-by-step and caches them 
    in compressed format (.npz) within the designated directory.
    """
    mesh_folder = Path(mesh_folder_path)
    matrix_folder = Path(matrix_folder_path)
    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)
    
    obj_files = sorted([f for f in mesh_folder.iterdir() if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')])
    if not obj_files:
        print(f"Error: No files found in target directory: {mesh_folder_path}")
        return False
    
    if single_file:
        print(f"\nComputing physics tracking fields for {len(obj_files)} timesteps...")
    
    meshn_1 = mesh_mat2object(obj_files[0]) 
    tracking_colors = get_spatial_rgb_base(meshn_1.vertices)
    num_vertices_0 = meshn_1.vertices.shape[0]
    
    # Save base/initial frame
    np.savez(
        output_folder / 'frame_0.npz',
        vertices=meshn_1.vertices,
        faces=meshn_1.faces,
        colors=tracking_colors,
        velocity=np.zeros(num_vertices_0),
        strain=np.zeros(num_vertices_0),
        area_strain=np.zeros(num_vertices_0),
        normal_flow=np.zeros(num_vertices_0),
        tangential_flow=np.zeros(num_vertices_0)
    )
    
    for i in tqdm(range(1, len(obj_files)), desc='Processing structural metrics',leave=single_file):
        meshn = mesh_mat2object(obj_files[i]) 
        
        p2p_file = matrix_folder / f'FMV_{i}{i-1}.npy'
        if not p2p_file.exists():
            print(f"Error: Required file '{p2p_file.name}' missing. Aborting calculations.")
            return False
            
        p2p_zo = np.load(p2p_file)
        
        tracking_colors = tracking_colors[p2p_zo]
        v_mag, displacements = compute_displacement_velocity(meshn_1.vertices, meshn.vertices, p2p_zo)
        strain = compute_finite_element_strain(meshn_1.vertices, meshn.vertices, meshn.faces, p2p_zo)
        area_strain = compute_area_strain(meshn_1.vertices, meshn.vertices, meshn.faces, p2p_zo)
        norm_mag, tang_mag = compute_flow_decomposition(meshn.vertices, meshn.faces, displacements)
        
        np.savez(
            output_folder / f'frame_{i}.npz',
            vertices=meshn.vertices,
            faces=meshn.faces,
            colors=tracking_colors,
            velocity=v_mag,
            strain=strain,
            area_strain=area_strain,
            normal_flow=norm_mag,
            tangential_flow=tang_mag
        )
        
        meshn_1 = meshn
    return True