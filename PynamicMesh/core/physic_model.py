import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from tqdm.auto import tqdm
from pyFM.mesh import TriMesh
import seaborn as sns


def compute_heatmap_similarity(matrix1, matrix2):
    """
    Computes cross-heatmap similarity metrics between two functional map matrices.
    """
    m1_flat = matrix1.flatten()
    m2_flat = matrix2.flatten()
    
    euclidean = np.linalg.norm(m1_flat - m2_flat)
    manhattan = np.sum(np.abs(m1_flat - m2_flat))
    
    if np.std(m1_flat) == 0 or np.std(m2_flat) == 0:
        pearson = 0.0
    else:
        pearson = np.corrcoef(m1_flat, m2_flat)[0, 1]
        if np.isnan(pearson): pearson = 0.0
        
    r1 = np.argsort(np.argsort(m1_flat))
    r2 = np.argsort(np.argsort(m2_flat))
    if np.std(r1) == 0 or np.std(r2) == 0:
        spearman = 0.0
    else:
        spearman = np.corrcoef(r1, r2)[0, 1]
        if np.isnan(spearman): spearman = 0.0
        
    p = (matrix1 ** 2).flatten()
    q = (matrix2 ** 2).flatten()
    
    sum_p, sum_q = np.sum(p), np.sum(q)
    p = p / sum_p if sum_p > 0 else np.ones_like(p) / len(p)
    q = q / sum_q if sum_q > 0 else np.ones_like(q) / len(q)
    
    m = 0.5 * (p + q)
    eps = 1e-12  # Prevent log(0)
    kl_p = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_q = np.sum(q * np.log((q + eps) / (m + eps)))
    jsd = 0.5 * kl_p + 0.5 * kl_q
    
    return jsd, pearson, spearman, manhattan, euclidean


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
    matched_vertices1 = vertices1[p2p]
    displacements = vertices2 - matched_vertices1
    magnitudes = np.linalg.norm(displacements, axis=1)
    return magnitudes, displacements


def compute_finite_element_strain(vertices1, vertices2, faces2, p2p):
    num_vertices = vertices2.shape[0]
    edges = set()
    for face in faces2:
        for i in range(3):
            u, v = face[i], face[(i+1)%3]
            if u > v: u, v = v, u
            edges.add((u, v))
    edges = np.array(list(edges))
    
    lengths2 = np.linalg.norm(vertices2[edges[:, 0]] - vertices2[edges[:, 1]], axis=1)
    
    p1_u = vertices1[p2p[edges[:, 0]]]
    p1_v = vertices1[p2p[edges[:, 1]]]
    lengths1 = np.linalg.norm(p1_u - p1_v, axis=1)
    
    lengths1_safe = np.where(lengths1 == 0, 1e-6, lengths1)
    edge_strain = (lengths2 - lengths1_safe) / lengths1_safe
    
    vertex_strain = np.zeros(num_vertices)
    vertex_edge_count = np.zeros(num_vertices)
    
    for idx, (u, v) in enumerate(edges):
        vertex_strain[u] += edge_strain[idx]
        vertex_strain[v] += edge_strain[idx]
        vertex_edge_count[u] += 1
        vertex_edge_count[v] += 1
        
    vertex_edge_count[vertex_edge_count == 0] = 1
    vertex_strain /= vertex_edge_count
    
    return vertex_strain


def compute_area_strain(vertices1, vertices2, faces2, p2p):
    v0_2 = vertices2[faces2[:, 0]]
    v1_2 = vertices2[faces2[:, 1]]
    v2_2 = vertices2[faces2[:, 2]]
    areas2 = 0.5 * np.linalg.norm(np.cross(v1_2 - v0_2, v2_2 - v0_2), axis=1)

    v0_1 = vertices1[p2p[faces2[:, 0]]]
    v1_1 = vertices1[p2p[faces2[:, 1]]]
    v2_1 = vertices1[p2p[faces2[:, 2]]]
    areas1 = 0.5 * np.linalg.norm(np.cross(v1_1 - v0_1, v2_1 - v0_1), axis=1)

    areas1_safe = np.where(areas1 == 0, 1e-8, areas1)
    face_area_strain = (areas2 - areas1_safe) / areas1_safe

    num_vertices = vertices2.shape[0]
    vertex_area_strain = np.zeros(num_vertices)
    vertex_face_count = np.zeros(num_vertices)

    for i in range(3):
        np.add.at(vertex_area_strain, faces2[:, i], face_area_strain)
        np.add.at(vertex_face_count, faces2[:, i], 1)

    vertex_face_count[vertex_face_count == 0] = 1
    return vertex_area_strain / vertex_face_count


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
    n, m = matrix.shape
    matrix_sq = matrix**2
    total_energy = np.sum(matrix_sq)
    
    if total_energy == 0:
        return 0, 0, np.zeros(max(n, m))

    I, J = np.ogrid[:n, :m]
    dist = np.abs(I - J)
    
    max_dist = max(n, m) - 1
    inertia = np.sum((dist**2) * matrix_sq) / (total_energy * (max_dist**2))
    inertia_metric = 1.0 - inertia
    
    decay_metric = np.sum(matrix_sq * np.exp(-alpha * dist)) / total_energy
    
    cdf_bandwidth = np.zeros(max(n, m))
    for k in range(max(n, m)):
        mask = (dist <= k)
        cdf_bandwidth[k] = np.sum(matrix_sq[mask]) / total_energy
        
    return inertia_metric, decay_metric, cdf_bandwidth


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


def launch_physics_viewer(frames_data):
    print("\nStarting interactive 3D Multi-Physics Gallery...")
    meshes = [create_pv_polydata(d) for d in frames_data]
    
    def get_clim(key, symmetric=False):
        arr = np.concatenate([d[key] for d in frames_data])
        if symmetric:
            v_max = np.max(np.abs(arr))
            return [-v_max, v_max]
        return [np.min(arr), np.max(arr)]

    vel_clim = get_clim('velocity')
    strain_clim = get_clim('strain', symmetric=True)
    area_strain_clim = get_clim('area_strain', symmetric=True)
    norm_flow_clim = get_clim('normal_flow', symmetric=True)
    tang_flow_clim = get_clim('tangential_flow')

    pl = pv.Plotter(shape=(2, 3))
    pl.title = "Cell Dynamics Multi-Physics Gallery"
    state = {'frame': 0, 'total': len(meshes)}
    
    
    for i in range(2):
        for j in range(3):
            pl.subplot(i, j)
            pl.add_axes()
            pl.camera_position = 'iso' 
    
    def update_frame(frame_idx):
        pl.subplot(0, 0)
        pl.add_mesh(meshes[frame_idx], scalars='RGB', rgb=True, name='color_mesh', show_scalar_bar=False, render=False)
        pl.add_text(f"Color Transfer ({frame_idx + 1}/{state['total']})", name='t0', font_size=10, position='upper_left')
        
        pl.subplot(0, 1)
        pl.add_mesh(meshes[frame_idx], scalars='Velocity', cmap='viridis', clim=vel_clim, name='vel_mesh', render=False)
        pl.add_text(f"Velocity Magnitude", name='t1', font_size=10, position='upper_left')
        
        pl.subplot(0, 2)
        pl.add_mesh(meshes[frame_idx], scalars='Strain', cmap='coolwarm', clim=strain_clim, name='strain_mesh', render=False)
        pl.add_text(f"Linear Edge Strain", name='t2', font_size=10, position='upper_left')

        pl.subplot(1, 0)
        pl.add_mesh(meshes[frame_idx], scalars='Area_Strain', cmap='coolwarm', clim=area_strain_clim, name='area_mesh', render=False)
        pl.add_text(f"Areal Expansion/Strain", name='t3', font_size=10, position='upper_left')

        pl.subplot(1, 1)
        pl.add_mesh(meshes[frame_idx], scalars='Normal_Flow', cmap='Spectral', clim=norm_flow_clim, name='norm_mesh', render=False)
        pl.add_text(f"Normal Protrusion Flow", name='t4', font_size=10, position='upper_left')

        pl.subplot(1, 2)
        pl.add_mesh(meshes[frame_idx], scalars='Tangential_Flow', cmap='plasma', clim=tang_flow_clim, name='tang_mesh', render=False)
        pl.add_text(f"Tangential Lateral Flow", name='t5', font_size=10, position='upper_left')
        
    update_frame(0)
    for i in range(2):
        for j in range(3):
            pl.subplot(i, j)
            pl.reset_camera()
            
    pl.link_views()  
    
    def step_next():
        if state['frame'] < state['total'] - 1:
            state['frame'] += 1
            update_frame(state['frame'])
            pl.render()

    def step_prev():
        if state['frame'] > 0:
            state['frame'] -= 1
            update_frame(state['frame'])
            pl.render()
            
    pl.add_key_event('Right', step_next)   
    pl.add_key_event('Left', step_prev)    
     
    pl.subplot(0, 0)
    pl.add_text("Time Control:\n  right arrow key : Next Mesh\n   left arrow key : Prev Mesh", position='lower_left', font_size=6, color='black')
    pl.show(full_screen=True)


def computing_fields(mesh_folder_path, matrix_folder_path, output_folder_path,single_file=True):
    """
    Computes all physics-based deformation fields step-by-step and caches them 
    in compressed format (.npz) within the designated directory.
    """
    mesh_folder = Path(mesh_folder_path)
    matrix_folder = Path(matrix_folder_path)
    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)
    
    obj_files = sorted([f for f in mesh_folder.iterdir() if f.is_file() and f.suffix == '.obj'])
    if not obj_files:
        print(f"Error: No .obj files found in target directory: {mesh_folder_path}")
        return False
    
    if single_file:
        print(f"\nComputing physics tracking fields for {len(obj_files)} timesteps...")
    
    meshn_1 = TriMesh(str(obj_files[0]))
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
        meshn = TriMesh(str(obj_files[i]))
        
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


def visualize_physics(mesh_folder_path, matrix_folder_path, on_time=True):
    """
    Loads spatial frame computations and deploys the viewer. If on_time is True,
    it computes transformations live. If False, it uses precomputed arrays.
    """
    matrix_folder = Path(matrix_folder_path)
    output_folder = matrix_folder.parent / 'physical_fields'
    
    if on_time:
        success = computing_fields(mesh_folder_path, matrix_folder_path, output_folder)
        if not success:
            return

    print(f"\nGathering structural data streams from {output_folder}...")
    npz_files = sorted(
        [f for f in output_folder.iterdir() if f.is_file() and f.suffix == '.npz' and f.name.startswith('frame_')],
        key=lambda x: int(x.stem.split('_')[1])
    )
    
    if not npz_files:
        print(f"Error: Missing physical fields dependencies in target: {output_folder}")
        return

    frames_data = []
    for npz_file in npz_files:
        with np.load(npz_file) as loaded_data:
            frames_data.append({key: loaded_data[key] for key in loaded_data.files})
            
    launch_physics_viewer(frames_data)