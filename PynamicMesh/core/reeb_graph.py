import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from tqdm.auto import tqdm
from pyFM.mesh import TriMesh
import seaborn as sns
import networkx as nx
import pickle
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix 
from scipy.sparse import linalg as splinalg
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.linalg import eigvalsh
import warnings
import copy
from PynamicMesh.utils.tools import mesh_mat2object

try:
    import cupy as xp
    GPU_AVAILABLE = True
    print("[INFO] CuPy detected. Utilizing GPU for Reeb Graph spectral computations and scalar fields.")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    print("[INFO] CuPy not found. Defaulting to CPU (NumPy).")


try:
    from geometrickernels.spaces import Mesh as GKMesh
    from geometrickernels.kernels import MaternKarhunenLoeveKernel
    import lab as B
    GK_AVAILABLE = True
except ImportError:
    GK_AVAILABLE = False

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


def graph_time_analysis(reeb_folder_path, single_file=True):
    if single_file:
        print("\nStarting Dynamic Graph Analysis...")
    reeb_folder = Path(reeb_folder_path)
    reeb_files = sorted([f for f in reeb_folder.iterdir() if f.is_file() and f.suffix == '.pkl'])
    
    if len(reeb_files) < 2:
        print("Not enough Reeb graphs found to perform time analysis.")
        return
        
    analysis_folder = reeb_folder.parent / 'Graph_analysis'
    os.makedirs(analysis_folder, exist_ok=True)
    
    features_list = []
    
    with open(reeb_files[0], 'rb') as f:
        G_prev = pickle.load(f)

    for i in tqdm(range(1, len(reeb_files)), desc="Computing Graph Metrics", leave=single_file):
        with open(reeb_files[i], 'rb') as f:
            G_curr = pickle.load(f)
            
        v_prev, e_prev = G_prev.number_of_nodes(), G_prev.number_of_edges()
        v_curr, e_curr = G_curr.number_of_nodes(), G_curr.number_of_edges()
        
        c_prev = nx.number_connected_components(G_prev) if v_prev > 0 else 0
        c_curr = nx.number_connected_components(G_curr) if v_curr > 0 else 0
        betti_prev = e_prev - v_prev + c_prev
        betti_curr = e_curr - v_curr + c_curr
        
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

        ged = nx.graph_edit_distance(G_prev, G_curr, timeout=2)
        if ged is None:  
            ged = abs(v_curr - v_prev) + abs(e_curr - e_prev)

        features_list.append({
            'Transition': f"T{i-1} -> T{i}",
            'Time_Step': i-1,
            'Nodes_T': v_curr,
            'Edges_T': e_curr,
            'Delta_Nodes': v_curr - v_prev,
            'Delta_Edges': e_curr - e_prev,
            'Betti_1_Cycles': betti_curr,
            'Graph_Density': nx.density(G_curr) if v_curr > 1 else 0,
            'LCC_Diameter': diameter,
            'LCC_Radius': radius,
            'Deg_Wasserstein_Dist': deg_wasserstein,
            'Spectral_Laplacian_Dist': spectral_dist,
            'Graph_Edit_Dist': ged
        })
        
        G_prev = G_curr

    df = pd.DataFrame(features_list)
    csv_out_path = analysis_folder / 'time_analysis.csv'
    df.to_csv(csv_out_path, index=False)

    if single_file:
        print(f"Graph analysis complete. Data saved to: {csv_out_path}")
    
    return str(csv_out_path)


def plot_dynamic_graph_analysis(csv_path,single_file=True):
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
        
    df = pd.DataFrame(pd.read_csv(csv_file))
    plots_folder = csv_file.parent / 'plots'
    os.makedirs(plots_folder, exist_ok=True)
    
    time_steps = df['Time_Step']
    sns.set_theme(style="whitegrid")


    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(time_steps, df['Nodes_T'], label='Nodes', marker='o', color='#1f77b4', linewidth=2)
    ax1.plot(time_steps, df['Edges_T'], label='Edges', marker='s', color='#ff7f0e', linewidth=2)
    ax1.set_title('Reeb Graph Structural Size Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Step (T)', fontsize=12)
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
    axes2[2].set_xlabel('Time Step (Transition $T_{n-1} \u2192 T_n$)')
    
    fig2.suptitle('Dynamic Graph Similarity Metrics', fontsize=16, fontweight='bold', y=0.98)
    fig2.tight_layout()
    fig2.savefig(plots_folder / '2_Graph_Distances.png', dpi=200)
    plt.close(fig2)


    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3_tw = ax3.twinx()
    
    l1 = ax3.plot(time_steps, df['Betti_1_Cycles'], color='darkgreen', marker='^', label='Betti-1 (Cycles)')
    l2 = ax3_tw.plot(time_steps, df['LCC_Diameter'], color='navy', marker='v', linestyle='--', label='Diameter (LCC)')
    
    ax3.set_title('Topology & Spatial Span', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Step (T)', fontsize=12)
    ax3.set_ylabel('Number of Cycles (Betti-1)', color='darkgreen', fontsize=12)
    ax3_tw.set_ylabel('Graph Diameter (Hops)', color='navy', fontsize=12)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    fig3.tight_layout()
    fig3.savefig(plots_folder / '3_Internal_Topology.png', dpi=200)
    plt.close(fig3)
    
    if single_file:
        print(f"Visual reports generated successfully in: {plots_folder}")


def _get_mesh_adjacency(vertices, faces):
    """Constructs a sparse adjacency matrix optimized on the GPU."""
    v_gpu = to_gpu(vertices)
    f_gpu = to_gpu(faces)
    
    edges_gpu = xp.vstack([f_gpu[:, [0, 1]], f_gpu[:, [1, 2]], f_gpu[:, [2, 0]]])
    weights_gpu = xp.linalg.norm(v_gpu[edges_gpu[:, 0]] - v_gpu[edges_gpu[:, 1]], axis=1)
    
    edges = to_cpu(edges_gpu)
    weights = to_cpu(weights_gpu)
    
    n = len(vertices)
    adj = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))
    return adj.maximum(adj.T)

def compute_geodesic_distance(vertices, faces, vertex_ref_index):
    # TriMesh requires standard numpy arrays on CPU host memory
    dist_matrix = TriMesh(to_cpu(vertices), to_cpu(faces)).geod_from(vertex_ref_index)
    return np.asarray(dist_matrix).flatten()

def compute_harmonic_field(trimesh_obj, source_idx, sink_idx):
    """Solves the Laplace equation Δf = 0 with Dirichlet boundary conditions."""
    W = trimesh_obj.W.tocsr() # Cotangent stiffness matrix from pyFM
    n = W.shape[0]
    
    b = np.zeros(n)
    b[source_idx] = 1.0
    b[sink_idx] = 0.0
    
    W_mod = W.copy()
    penalty = 1e8
    W_mod[source_idx, source_idx] += penalty
    W_mod[sink_idx, sink_idx] += penalty
    b[source_idx] *= penalty
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = splinalg.spsolve(W_mod, b)
    return f

def compute_heat_diffusion(trimesh_obj, source_idx, t=10.0):
    """Fully vectorized and parallelized GPU calculation of the Heat Kernel Signature."""
    evals_gpu = to_gpu(trimesh_obj.eigenvalues)
    evecs_gpu = to_gpu(trimesh_obj.eigenvectors)
    
    weights = xp.exp(-t * evals_gpu) * evecs_gpu[source_idx, :]
    heat_gpu = evecs_gpu @ weights
    
    return to_cpu(heat_gpu)

def get_scalar_field(vertices, faces, method="z", prev_vertices=None, p2p=None, trimesh_obj=None, **kwargs):
    method = method.lower()
    num_vertices = vertices.shape[0]
    

    v_gpu = to_gpu(vertices)
    

    if method == "x": return to_cpu(v_gpu[:, 0])
    elif method == "y": return to_cpu(v_gpu[:, 1])
    elif method == "z": return to_cpu(v_gpu[:, 2])
    elif method == "dist_centroid": 
        centroid = v_gpu.mean(axis=0)
        return to_cpu(xp.linalg.norm(v_gpu - centroid, axis=1))
    elif method == "signed_dist_x": return to_cpu(v_gpu[:, 0] - v_gpu.mean(axis=0)[0])
    elif method == "signed_dist_y": return to_cpu(v_gpu[:, 1] - v_gpu.mean(axis=0)[1])
    elif method == "signed_dist_z": return to_cpu(v_gpu[:, 2] - v_gpu.mean(axis=0)[2])

    elif method == "mass_center_geodesic":
        vertices_cpu = to_cpu(vertices)
        faces_cpu = to_cpu(faces)
        center = TriMesh(vertices_cpu, faces_cpu).center_mass
        dist_to_center = np.linalg.norm(vertices_cpu - center, axis=1)
        central_vertex_idx = np.argmin(dist_to_center)
        return compute_geodesic_distance(vertices_cpu, faces_cpu, [central_vertex_idx])
    

    elif method in ["mean_curvature", "gaussian_curvature", "shape_index", "curvedness"]:
        vertices_cpu = to_cpu(vertices)
        faces_cpu = to_cpu(faces)
        faces_pv = np.empty((faces_cpu.shape[0], 4), dtype=int)
        faces_pv[:, 0] = 3
        faces_pv[:, 1:] = faces_cpu
        mesh_pv = pv.PolyData(vertices_cpu, faces_pv.flatten())
        
        if method == "mean_curvature":
            return np.nan_to_num(mesh_pv.curvature(curv_type="mean"))
        elif method == "gaussian_curvature":
            return np.nan_to_num(mesh_pv.curvature(curv_type="Gaussian"))
        else:
            H = np.nan_to_num(mesh_pv.curvature(curv_type="mean"))
            K = np.nan_to_num(mesh_pv.curvature(curv_type="Gaussian"))
            discriminant = np.maximum(H**2 - K, 1e-8) 
            
            if method == "shape_index":
                S = (2.0 / np.pi) * np.arctan(H / np.sqrt(discriminant))
                return np.nan_to_num(S)
            elif method == "curvedness":
                C = np.sqrt(np.maximum(2 * H**2 - K, 0))
                return np.nan_to_num(C)


    elif method == "normal_displacement":
        if prev_vertices is None or p2p is None: return np.zeros(num_vertices)
        matched_prev = to_cpu(prev_vertices)[to_cpu(p2p)]
        displacements = to_cpu(vertices) - matched_prev
        from physic_model import compute_flow_decomposition  # dynamic reference to flow solver
        norm_mag, _ = compute_flow_decomposition(to_cpu(vertices), to_cpu(faces), displacements)
        return norm_mag

    elif method == "geodesic":
        vertex_ref_index = kwargs.get("vertex_ref_index", [0])

        if vertex_ref_index == 'mass_center' or (isinstance(vertex_ref_index, list) and len(vertex_ref_index) > 0 and vertex_ref_index[0] == 'mass_center'):
            vertices_cpu = to_cpu(vertices)
            faces_cpu = to_cpu(faces)
        
            center = TriMesh(vertices_cpu, faces_cpu).center_mass
            
            dist_to_center = np.linalg.norm(vertices_cpu - center, axis=1)
            central_vertex_idx = np.argmin(dist_to_center)
            
            vertex_ref_index = [central_vertex_idx]

        return compute_ge


    elif method.startswith("lb_eigen_"):
        if trimesh_obj is None: raise ValueError("trimesh_obj is required for LB eigenfunctions.")
        idx = int(method.split("_")[-1])
        return trimesh_obj.eigenvectors[:, idx]
        
    elif method == "heat_diffusion":
        if trimesh_obj is None: raise ValueError("trimesh_obj is required for heat diffusion.")
        source_idx = kwargs.get("source_idx", 0)
        
        if source_idx == 'mass_center' or (isinstance(source_idx, list) and len(source_idx) > 0 and source_idx[0] == 'mass_center'):
            vertices_cpu = to_cpu(vertices)
            faces_cpu = to_cpu(faces)
            center = TriMesh(vertices_cpu, faces_cpu).center_mass
            dist_to_center = np.linalg.norm(vertices_cpu - center, axis=1)
            source_idx = int(np.argmin(dist_to_center))

        t = kwargs.get("t", 10.0)
        
        heat_raw = compute_heat_diffusion(trimesh_obj, source_idx, t)
        
        if kwargs.get("equalize_histogram", True):
            heat_gpu = to_gpu(heat_raw)
            temp = heat_gpu.argsort()
            ranks = xp.empty_like(temp)
            ranks[temp] = xp.arange(len(heat_gpu))
            heat_normalized = ranks / (len(heat_gpu) - 1.0)
            return to_cpu(heat_normalized)
            
        return heat_raw
        
    elif method == "matern_kernel":
        if trimesh_obj is None: raise ValueError("trimesh_obj is required for matern_kernel.")
        source_idx = kwargs.get("source_idx", 0)
        
        if source_idx == 'mass_center' or (isinstance(source_idx, list) and len(source_idx) > 0 and source_idx[0] == 'mass_center'):
            vertices_cpu = to_cpu(vertices)
            faces_cpu = to_cpu(faces)
            center = TriMesh(vertices_cpu, faces_cpu).center_mass
            dist_to_center = np.linalg.norm(vertices_cpu - center, axis=1)
            source_idx = int(np.argmin(dist_to_center))

        nu = kwargs.get("nu", 1.5)
        lengthscale = kwargs.get("lengthscale", 1.0)

        global GK_AVAILABLE
        if GK_AVAILABLE:
            try:
                gk_mesh = GKMesh(to_cpu(vertices), to_cpu(faces))
                kernel = MaternKarhunenLoeveKernel(gk_mesh, num_eigenfunctions=len(trimesh_obj.eigenvalues))
                
                if hasattr(gk_mesh, '_eigenvalues'):
                    gk_mesh._eigenvalues = trimesh_obj.eigenvalues
                    gk_mesh._eigenfunctions = trimesh_obj.eigenvectors

                X_source = np.array([[source_idx]])
                X_all = np.arange(len(vertices)).reshape(-1, 1)

                if hasattr(kernel, "init_params_and_state"):
                    params, state = kernel.init_params_and_state()
                    params["nu"] = np.array([nu])
                    params["lengthscale"] = np.array([lengthscale])
                    K_vals = kernel.K(params, state, X_all, X_source)
                else:
                    params = {"nu": np.array([nu]), "lengthscale": np.array([lengthscale])}
                    K_vals = kernel.K(params, X_all, X_source)
                matern_raw = B.to_numpy(K_vals).flatten()
            except Exception:
                GK_AVAILABLE = False  

        if not GK_AVAILABLE:
            evals = trimesh_obj.eigenvalues
            evecs = trimesh_obj.eigenvectors
            power = -(nu + 1.0)
            weights = ((2 * nu) / (lengthscale**2) + evals) ** power
            phi_source = evecs[source_idx, :]
            matern_raw = (evecs * weights) @ phi_source

        if kwargs.get("equalize_histogram", True):
            matern_gpu = to_gpu(matern_raw)
            temp = matern_gpu.argsort()
            ranks = xp.empty_like(temp)
            ranks[temp] = xp.arange(len(matern_gpu))
            matern_normalized = ranks / (len(matern_gpu) - 1.0)
            return to_cpu(matern_normalized)
            
        return to_cpu(matern_raw)

    elif method == "harmonic":
        if trimesh_obj is None: raise ValueError("trimesh_obj is required for harmonic fields.")
        vertices_cpu = to_cpu(vertices)
        source_idx = kwargs.get("source_idx", np.argmin(vertices_cpu[:, 2])) 
        sink_idx = kwargs.get("sink_idx", np.argmax(vertices_cpu[:, 2]))
        return compute_harmonic_field(trimesh_obj, source_idx, sink_idx)

    elif method == "multi_pca":
        fields = kwargs.get("fields", ["z", "mean_curvature", "gaussian_curvature"])
        stacked_features = []
        for f in fields:
            val = get_scalar_field(to_cpu(vertices), to_cpu(faces), method=f, trimesh_obj=trimesh_obj, **kwargs)
            val = (val - np.mean(val)) / (np.std(val) + 1e-8)
            stacked_features.append(val)
            
        feature_matrix = np.vstack(stacked_features).T
        pca = PCA(n_components=1)
        return pca.fit_transform(feature_matrix).flatten()

    else:
        raise ValueError(f"Unknown scalar field method: {method}")

def compute_approx_reeb_graph(vertices, faces, scalar_field, num_bins=20):
    
    v_cpu = to_cpu(vertices)
    f_cpu = to_cpu(faces)
    sf_cpu = to_cpu(scalar_field)
    
    f_min, f_max = sf_cpu.min(), sf_cpu.max()
    bins = np.linspace(f_min, f_max + 1e-8, num_bins + 1)
    bin_indices = np.digitize(sf_cpu, bins) - 1
    
    edges = set()
    for face in f_cpu:
        for i in range(3):
            u, v = face[i], face[(i+1)%3]
            edges.add((min(u, v), max(u, v)))
    edges = np.array(list(edges))
    
    graph = nx.Graph()
    node_id_counter = 0
    node_of_vertex = {}
    
    for b in range(num_bins):
        v_in_bin = np.where(bin_indices == b)[0]
        if len(v_in_bin) == 0:
            continue
            
        v_map = {v: i for i, v in enumerate(v_in_bin)}
        local_edges = [[v_map[u], v_map[v]] for u, v in edges if u in v_map and v in v_map]
        
        if local_edges:
            local_edges = np.array(local_edges)
            adj = coo_matrix((np.ones(len(local_edges)), (local_edges[:,0], local_edges[:,1])), shape=(len(v_in_bin), len(v_in_bin)))
            adj = adj.maximum(adj.T)
            n_components, labels = connected_components(adj, directed=False)
        else:
            n_components = len(v_in_bin)
            labels = np.arange(len(v_in_bin))
            
        for comp in range(n_components):
            comp_vertices = v_in_bin[labels == comp]
            center_pos = v_cpu[comp_vertices].mean(axis=0)
            
            graph.add_node(node_id_counter, pos=center_pos, bin=b)
            for v in comp_vertices:
                node_of_vertex[v] = node_id_counter
            node_id_counter += 1
            
    for u, v in edges:
        n_u = node_of_vertex.get(u)
        n_v = node_of_vertex.get(v)
        if n_u is not None and n_v is not None and n_u != n_v:
            graph.add_edge(n_u, n_v)
            
    return graph

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
    return mesh