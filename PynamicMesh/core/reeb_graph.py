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
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse import linalg as splinalg
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.linalg import eigvalsh
import warnings
import copy

def edit_graph(mesh_folder_path, reeb_folder_path):
    print("\nStarting Interactive Split-Screen Graph Editor with Undo...")
    
    mesh_path = Path(mesh_folder_path)
    reeb_path = Path(reeb_folder_path)
    
    if not mesh_path.exists() or not reeb_path.exists():
        print("Error: Invalid mesh or reeb graph directory paths.")
        return

    obj_files = sorted([f for f in mesh_path.iterdir() if f.is_file() and f.suffix == '.obj'])
    reeb_files = sorted([f for f in reeb_path.iterdir() if f.is_file() and f.suffix == '.pkl'])
    scalar_files = sorted([f for f in reeb_path.iterdir() if f.is_file() and f.name.startswith('Scalar') and f.suffix == '.npy'])
    
    if not obj_files or not reeb_files or not scalar_files:
        print("Error: Missing .obj, .pkl, or Scalar .npy files for visualization.")
        return

    num_frames = min(len(obj_files), len(reeb_files), len(scalar_files))
    scene_name = mesh_path.name
    
    out_root = mesh_path.parent.parent / 'Results'
    target_folder = out_root / scene_name / 'Reeb_graph_manual_trim'
    os.makedirs(target_folder, exist_ok=True)
    
    # Load all graphs into memory (coordinates are already aligned by the pipeline)
    graphs = []
    for i in range(num_frames):
        with open(reeb_files[i], 'rb') as f:
            G = pickle.load(f)
            graphs.append(G)
            
    state = {
        'frame': 0,
        'total': num_frames,
        'graphs': graphs,
        'modified': [False] * num_frames,
        'history': [],
        'current_nodes_pv': None,
        'node_ids': [], 
        'diag_size': 1.0,
        'selected_node': None,
        'mode': 'normal' # Modes: 'normal', 'link', 'inner', 'outer', 'edge_delete'
    }

    plotter = pv.Plotter(shape=(1, 2), title="Reeb Graph Split-Screen Editor")
    plotter.add_axes()

    def ensure_node_limits(node_id, G, mesh_pv):
        """Safely initializes and extracts localized geometry limits and thickness parameters for a node."""
        node_data = G.nodes[node_id]
        if 'orig_pos' not in node_data:
            node_data['orig_pos'] = node_data['pos'].copy()
        if 'current_depth' not in node_data:
            node_data['current_depth'] = 0.0
        if 'normal' not in node_data:
            v_idx = mesh_pv.find_closest_point(node_data['orig_pos'])
            node_data['normal'] = mesh_pv.point_data['Normals'][v_idx]
        if 'max_depth' not in node_data:
            start_ray = node_data['orig_pos'] - node_data['normal'] * (state['diag_size'] * 1e-4)
            end_ray = node_data['orig_pos'] - node_data['normal'] * (state['diag_size'] * 2.0)
            hits, _ = mesh_pv.ray_trace(start_ray, end_ray)
            if len(hits) > 0:
                node_data['max_depth'] = np.linalg.norm(hits[0] - node_data['orig_pos'])
            else:
                node_data['max_depth'] = state['diag_size'] * 0.5
        return node_data

    def update_frame(frame_idx):
        tm = TriMesh(str(obj_files[frame_idx]))
        pad = np.full((tm.faces.shape[0], 1), 3, dtype=np.int64)
        pv_faces = np.hstack((pad, tm.faces)).flatten()
        mesh_pv = pv.PolyData(tm.vertices, pv_faces)
        mesh_pv.rotate_x(90, inplace=True)
        mesh_pv.rotate_z(90, inplace=True)
        
        bounds = mesh_pv.bounds
        state['diag_size'] = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        
        mesh_pv = mesh_pv.compute_normals(point_normals=True, cell_normals=False)
        state['mesh_pv'] = mesh_pv 
        
        plotter.subplot(0, 0)
        scalar_array = np.load(scalar_files[frame_idx])
        mesh_pv.point_data['Dynamic_Scalar'] = scalar_array
        
        plotter.add_mesh(mesh_pv, scalars='Dynamic_Scalar', cmap='viridis', name='z_mesh', show_scalar_bar=True, pickable=True)
        
        plotter.add_points(
            mesh_pv.points, color='darkgray', point_size=4, 
            render_points_as_spheres=True, name='mesh_vertices_spheres', pickable=False
        )
        
        plotter.add_text(
            f"Scalar Field - Frame {frame_idx + 1}/{state['total']}\nMesh: {obj_files[frame_idx].name}", 
            name='t1', font_size=8, position='upper_left'
        )
        
        if state['modified'][frame_idx]:
            plotter.add_text("MODIFIED (Unsaved changes)", name='mod_text_L', font_size=10, position='lower_left', color='orange')
        else:
            plotter.remove_actor('mod_text_L')

        plotter.subplot(0, 1)
        plotter.add_mesh(mesh_pv, color='white', opacity=0.25, name='ghost_mesh', show_scalar_bar=False, pickable=True)
        
        G = state['graphs'][frame_idx]
        if G.number_of_nodes() > 0:
            nodes = list(G.nodes(data=True))
            state['node_ids'] = [int(n) for n, data in nodes] 
            pts = np.array([data['pos'] for n, data in nodes])
            
            nodes_pv = pv.PolyData(pts)
            state['current_nodes_pv'] = nodes_pv
            
            lines = []
            node_idx_map = {int(n): i for i, n in enumerate(state['node_ids'])}
            for u, v in G.edges():
                u, v = int(u), int(v) 
                if u in node_idx_map and v in node_idx_map:
                    lines.extend([2, node_idx_map[u], node_idx_map[v]])
            
            if lines:
                edges_pv = pv.PolyData(pts)
                edges_pv.lines = np.array(lines)
                tube_radius = state['diag_size'] * 0.002
                plotter.add_mesh(edges_pv.tube(radius=tube_radius), color="blue", name='graph_edges')
            else:
                plotter.remove_actor('graph_edges')
            
            point_size = state['diag_size'] * 0.005
            plotter.add_mesh(
                pv.PolyData(pts).glyph(geom=pv.Sphere(radius=point_size), scale=False, orient=False), 
                color="red", name='graph_nodes'
            )
            
            sel_node = state.get('selected_node')
            if sel_node is not None and G.has_node(sel_node):
                node_data = ensure_node_limits(sel_node, G, mesh_pv)
                sel_pos = node_data['pos']
                sel_pv = pv.PolyData(np.array([sel_pos]))
                
                plotter.add_mesh(
                    sel_pv.glyph(geom=pv.Sphere(radius=0.002), scale=False, orient=False),
                    color="yellow", name='selected_node_highlight'
                )
            else:
                plotter.remove_actor('selected_node_highlight')
                
        else:
            state['current_nodes_pv'] = None
            state['node_ids'] = []
            plotter.remove_actor('graph_nodes')
            plotter.remove_actor('graph_edges')
            plotter.remove_actor('selected_node_highlight')

        mode_str = state['mode'].upper().replace('_', ' ')
        instruction_text = (
            f"Frame {frame_idx + 1}/{state['total']} | CURRENT MODE: [{mode_str}]\n"
            "------------------------------------------------------------------\n"
            "LEFT PANE: Click mesh to ADD a new node (snaps to vertex).\n"
            "RIGHT PANE INTERACTIONS:\n"
            "  - [ESC] Normal: Click node to CONNECT/DELETE.\n"
            "  - [C] Link Mode: Click 2 existing nodes to connect them.\n"
            "  - [D] Edge Delete: Click an edge to remove it.\n"
            "  - [I] Inner Mode: Click node to push it inside mesh (adaptive step).\n"
            "  - [O] Outer Mode: Click node to pull it outside (adaptive step).\n"
            "SPACE BAR to UNDO action.\n"
        )
        plotter.add_text(instruction_text, name='t2', font_size=8, position='upper_left')

    def set_mode(new_mode):
        state['mode'] = 'normal' if state['mode'] == new_mode else new_mode
        state['selected_node'] = None 
        update_frame(state['frame'])
        plotter.render()

    def clear_selection():
        state['mode'] = 'normal'
        state['selected_node'] = None
        update_frame(state['frame'])
        plotter.render()

    def point_to_segment_dist(p, a, b):
        ab = b - a
        ap = p - a
        if np.dot(ab, ab) == 0:
            return np.linalg.norm(ap)
        t = max(0, min(1, np.dot(ap, ab) / np.dot(ab, ab)))
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def pick_callback(coord):
        if coord is None:
            return
            
        click_x, click_y = plotter.mouse_position
        is_left_pane = click_x < (plotter.window_size[0] / 2)
        coord = np.array(coord)
        G = state['graphs'][state['frame']]
        mesh_pv = state['mesh_pv']
        mode = state['mode']
        
        # Save history BEFORE modifications
        state['history'].append((
            state['frame'], copy.deepcopy(G), 
            state['modified'][state['frame']], state.get('selected_node')
        ))

        if is_left_pane:
            if state.get('selected_node') is not None:
                state['history'].pop() 
                return
            
            idx = mesh_pv.find_closest_point(coord)
            vertex_coord = mesh_pv.points[idx]
            normal = mesh_pv.point_data['Normals'][idx]
            
            new_id = 0 if len(G.nodes) == 0 else max(G.nodes) + 1
            while G.has_node(new_id): 
                new_id += 1
            
            G.add_node(new_id, pos=vertex_coord, bin=0, 
                       orig_pos=vertex_coord, normal=normal, current_depth=0.0) 
            state['selected_node'] = new_id 
            
        else:
            if mode == 'edge_delete':
                closest_edge = None
                min_dist = float('inf')
                
                for u, v in G.edges():
                    pos_u = G.nodes[u]['pos']
                    pos_v = G.nodes[v]['pos']
                    dist = point_to_segment_dist(coord, pos_u, pos_v)
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge = (u, v)

                pick_tolerance = state['diag_size'] * 0.04
                if closest_edge is not None and min_dist < pick_tolerance:
                    G.remove_edge(*closest_edge)
                else:
                    state['history'].pop() 
                    return
                
            else:
                clicked_node_id = None
                if state['current_nodes_pv'] is not None and state['current_nodes_pv'].n_points > 0:
                    idx = state['current_nodes_pv'].find_closest_point(coord)
                    node_pos = state['current_nodes_pv'].points[idx]
                    dist = np.linalg.norm(node_pos - coord)
                    
                    sel_node = state.get('selected_node')
                    
                    if mode == 'link' and sel_node is not None:
                        clicked_node_id = int(state['node_ids'][idx])
                    elif dist < (state['diag_size'] * 0.04):
                        clicked_node_id = int(state['node_ids'][idx])

                if clicked_node_id is not None:
                    sel_node = state.get('selected_node')

                    if mode == 'link':
                        if sel_node is None:
                            state['selected_node'] = clicked_node_id
                            state['history'].pop() 
                            update_frame(state['frame'])
                            plotter.render()
                            return
                        else:
                            if sel_node != clicked_node_id and G.has_node(sel_node) and G.has_node(clicked_node_id):
                                G.add_edge(sel_node, clicked_node_id)
                            state['selected_node'] = None
                    
                    elif mode in ['inner', 'outer']:
                        state['selected_node'] = clicked_node_id 
                        node_data = ensure_node_limits(clicked_node_id, G, mesh_pv)
                        
                        step = node_data['max_depth'] / 20.0
                        
                        if mode == 'inner':
                            node_data['current_depth'] += step
                            if node_data['current_depth'] > node_data['max_depth']:
                                node_data['current_depth'] = node_data['max_depth'] 
                        else: 
                            node_data['current_depth'] -= step
                            if node_data['current_depth'] < 0:
                                node_data['current_depth'] = 0 
                                
                        node_data['pos'] = node_data['orig_pos'] - (node_data['normal'] * node_data['current_depth'])
                        
                    else: 
                        if sel_node is not None:
                            if sel_node != clicked_node_id and G.has_node(sel_node) and G.has_node(clicked_node_id):
                                G.add_edge(sel_node, clicked_node_id)
                            state['selected_node'] = None 
                        else:
                            G.remove_node(clicked_node_id)
                else:
                    state['history'].pop()
                    return
            
        state['modified'][state['frame']] = True
        update_frame(state['frame'])
        plotter.render()

    def undo_action():
        if not state['history']: return 
        prev_frame, prev_G, prev_modified, prev_selected = state['history'].pop()
        state['graphs'][prev_frame] = prev_G
        state['modified'][prev_frame] = prev_modified
        state['selected_node'] = prev_selected
        if state['frame'] != prev_frame:
            state['frame'] = prev_frame
        update_frame(state['frame'])
        plotter.render()

    plotter.add_key_event('Right', lambda: set_mode('normal') or step_next())
    plotter.add_key_event('Left', lambda: set_mode('normal') or step_prev())
    plotter.add_key_event('space', undo_action)
    plotter.add_key_event('Escape', clear_selection)

    plotter.add_key_event('c', lambda: set_mode('link'))
    plotter.add_key_event('i', lambda: set_mode('inner'))
    plotter.add_key_event('o', lambda: set_mode('outer'))
    plotter.add_key_event('d', lambda: set_mode('edge_delete'))
    
    def step_next():
        if state['frame'] < state['total'] - 1:
            state['frame'] += 1
            update_frame(state['frame'])
            plotter.render()

    def step_prev():
        if state['frame'] > 0:
            state['frame'] -= 1
            update_frame(state['frame'])
            plotter.render()

    plotter.enable_surface_point_picking(callback=pick_callback, show_message=False, left_clicking=True)
    update_frame(0)
    
    plotter.subplot(0, 0)
    plotter.reset_camera()
    plotter.camera_position = 'iso'
    plotter.subplot(0, 1)
    plotter.reset_camera()
    plotter.camera_position = 'iso'
    
    plotter.link_views() 
    plotter.show(full_screen=True)
    
    print("\nClosing editor panel...")
    saved_count = 0
    for i in range(state['total']):
        if state['modified'][i]:
            save_path = target_folder / reeb_files[i].name
            
            with open(save_path, 'wb') as f:
                pickle.dump(state['graphs'][i], f)
            saved_count += 1
            print(f"Saved modified graph update: {save_path.name}")
            
    if saved_count == 0:
        print("No changes detected across frames. Save skipped.")
    else:
        print(f"[SUCCESS] Exported {saved_count} updated topological structures to: {target_folder}")

def graph_time_analysis(reeb_folder_path,single_file=True):
    """
    Analyzes the temporal evolution of Reeb graphs and extracts a robust set 
    of topological and distance features between consecutive timesteps.
    """
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
    
    # Load first graph
    with open(reeb_files[0], 'rb') as f:
        G_prev = pickle.load(f)

    for i in tqdm(range(1, len(reeb_files)), desc="Computing Graph Metrics", leave=single_file):
        with open(reeb_files[i], 'rb') as f:
            G_curr = pickle.load(f)
            
        # Basic Graph Sizes
        v_prev, e_prev = G_prev.number_of_nodes(), G_prev.number_of_edges()
        v_curr, e_curr = G_curr.number_of_nodes(), G_curr.number_of_edges()
        
        # Betti-1 Number (Cycles): E - V + C
        c_prev = nx.number_connected_components(G_prev) if v_prev > 0 else 0
        c_curr = nx.number_connected_components(G_curr) if v_curr > 0 else 0
        betti_prev = e_prev - v_prev + c_prev
        betti_curr = e_curr - v_curr + c_curr
        
        # Diameter & Radius of the Largest Connected Component
        if v_curr > 0 and c_curr > 0:
            largest_cc = max(nx.connected_components(G_curr), key=len)
            sub_G = G_curr.subgraph(largest_cc)
            diameter = nx.diameter(sub_G)
            radius = nx.radius(sub_G)
        else:
            diameter, radius = 0, 0
            
        # Degree Distribution
        degrees_prev = [d for n, d in G_prev.degree()] if v_prev > 0 else [0]
        degrees_curr = [d for n, d in G_curr.degree()] if v_curr > 0 else [0]
        
        # --- Similarity & Distance Metrics ---
        
        # 1. Degree Wasserstein Distance (Earth Mover's Distance)
        deg_wasserstein = wasserstein_distance(degrees_prev, degrees_curr)
        
        # 2. Spectral Distance (L2 norm of Normalized Laplacian Eigenvalues differences)
        if v_prev > 0 and v_curr > 0:
            lap_prev = nx.normalized_laplacian_matrix(G_prev).todense()
            lap_curr = nx.normalized_laplacian_matrix(G_curr).todense()
            evals_prev = eigvalsh(lap_prev)
            evals_curr = eigvalsh(lap_curr)
            
            # Pad the smaller eigenvalue array with zeros for direct comparison
            max_len = max(len(evals_prev), len(evals_curr))
            e_p_pad = np.pad(evals_prev, (0, max_len - len(evals_prev)))
            e_c_pad = np.pad(evals_curr, (0, max_len - len(evals_curr)))
            spectral_dist = np.linalg.norm(e_p_pad - e_c_pad)
        else:
            spectral_dist = 0.0

        # 3. Approximate Graph Edit Distance (GED)
        # Bounded to 2 seconds to prevent hanging on dense topological structures
        ged = nx.graph_edit_distance(G_prev, G_curr, timeout=2)
        if ged is None:  # If it timed out, default to a structural proxy
            ged = abs(v_curr - v_prev) + abs(e_curr - e_prev)

        # Append feature vector
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

    # Save to CSV
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
    """Constructs a sparse adjacency matrix weighted by Euclidean edge lengths."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    n = len(vertices)
    adj = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))
    return adj.maximum(adj.T)

def compute_geodesic_distance(vertices, faces, vertex_ref_index):
    """Computes the shortest path geodesic distance to a set of landmark vertices."""
    adj = _get_mesh_adjacency(vertices, faces)
    # Dijkstra from all landmarks, returning the minimum distance to the closest one
    dist_matrix = csgraph.dijkstra(adj, indices=vertex_ref_index, directed=False)
    if dist_matrix.ndim > 1:
        return np.min(dist_matrix, axis=0)
    return dist_matrix

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
    
    # Suppress sparse efficiency warnings for direct modification
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = splinalg.spsolve(W_mod, b)
    return f

def compute_heat_diffusion(trimesh_obj, source_idx, t=10.0):
    """Approximates the heat kernel signature using LBO eigenfunctions."""
    evals = trimesh_obj.eigenvalues
    evecs = trimesh_obj.eigenvectors
    
    heat = np.zeros(evecs.shape[0])
    for i in range(len(evals)):
        # H_t(x, y) ≈ sum exp(-t * lambda_i) * phi_i(x) * phi_i(y)
        heat += np.exp(-t * evals[i]) * evecs[:, i] * evecs[source_idx, i]
    return heat

def get_scalar_field(vertices, faces, method="z", prev_vertices=None, p2p=None, 
                     trimesh_obj=None, **kwargs):
    """
    Computes a specified scalar field dynamically for Reeb graph extraction.
    Supports geometric, spectral, and multi-scalar Mapper-type fields.
    """
    method = method.lower()
    centroid = vertices.mean(axis=0)
    num_vertices = vertices.shape[0]
    
    # --- Standard Geometric & Coordinate Fields ---
    if method == "x": return vertices[:, 0]
    elif method == "y": return vertices[:, 1]
    elif method == "z": return vertices[:, 2]
    elif method == "dist_centroid": return np.linalg.norm(vertices - centroid, axis=1)
    elif method == "signed_dist_x": return vertices[:, 0] - centroid[0]
    elif method == "signed_dist_y": return vertices[:, 1] - centroid[1]
    elif method == "signed_dist_z": return vertices[:, 2] - centroid[2]
    
    # --- Local Surface Metrics (Curvatures) ---
    elif method in ["mean_curvature", "gaussian_curvature", "shape_index", "curvedness"]:
        faces_pv = np.empty((faces.shape[0], 4), dtype=int)
        faces_pv[:, 0] = 3
        faces_pv[:, 1:] = faces
        mesh_pv = pv.PolyData(vertices, faces_pv.flatten())
        
        if method == "mean_curvature":
            return np.nan_to_num(mesh_pv.curvature(curv_type="mean"))
        elif method == "gaussian_curvature":
            return np.nan_to_num(mesh_pv.curvature(curv_type="Gaussian"))
        else:
            H = np.nan_to_num(mesh_pv.curvature(curv_type="mean"))
            K = np.nan_to_num(mesh_pv.curvature(curv_type="Gaussian"))
            
            # Discriminant for principal curvatures: H^2 - K
            discriminant = np.maximum(H**2 - K, 1e-8) 
            
            if method == "shape_index":
                S = (2.0 / np.pi) * np.arctan(H / np.sqrt(discriminant))
                return np.nan_to_num(S)
            elif method == "curvedness":
                # C = sqrt( (k1^2 + k2^2) / 2 ) = sqrt( 2H^2 - K )
                C = np.sqrt(np.maximum(2 * H**2 - K, 0))
                return np.nan_to_num(C)

    # --- Displacement Flow ---
    elif method == "normal_displacement":
        if prev_vertices is None or p2p is None: return np.zeros(num_vertices)
        matched_prev = prev_vertices[p2p]
        displacements = vertices - matched_prev
        norm_mag, _ = compute_flow_decomposition(vertices, faces, displacements)
        return norm_mag

    # --- Geodesic Distance ---
    elif method == "geodesic":
        vertex_ref_index = kwargs.get("vertex_ref_index", [0]) # Default to vertex 0 if none provided
        return compute_geodesic_distance(vertices, faces, vertex_ref_index)

    # --- Spectral Methods (Requires trimesh_obj) ---
    elif method.startswith("lb_eigen_"):
        if trimesh_obj is None: raise ValueError("trimesh_obj is required for LB eigenfunctions.")
        idx = int(method.split("_")[-1])
        return trimesh_obj.eigenvectors[:, idx]
        

    elif method == "heat_diffusion":
        if trimesh_obj is None: raise ValueError("trimesh_obj is required for heat diffusion.")
        source_idx = kwargs.get("source_idx", 0)
        t = kwargs.get("t", 10.0)
        heat_raw = compute_heat_diffusion(trimesh_obj, source_idx, t)
        
        if kwargs.get("equalize_histogram", True):
            temp = heat_raw.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(heat_raw))
            heat_normalized = ranks / (len(heat_raw) - 1.0)
            return heat_normalized
            
        return heat_raw

    elif method == "harmonic":
        if trimesh_obj is None: raise ValueError("trimesh_obj is required for harmonic fields.")
        # Default: harmonic field between furthest points
        source_idx = kwargs.get("source_idx", np.argmin(vertices[:, 2])) 
        sink_idx = kwargs.get("sink_idx", np.argmax(vertices[:, 2]))

        return compute_harmonic_field(trimesh_obj, source_idx, sink_idx)

    elif method == "multi_pca":
        # Extracts 1st Principal Component from multiple fields to create a 1D Mapper lens
        fields = kwargs.get("fields", ["z", "mean_curvature", "gaussian_curvature"])
        stacked_features = []
        for f in fields:
            # Recursively call get_scalar_field for each feature
            val = get_scalar_field(vertices, faces, method=f, trimesh_obj=trimesh_obj, **kwargs)
            # Normalize to avoid dominance of large scales
            val = (val - np.mean(val)) / (np.std(val) + 1e-8)
            stacked_features.append(val)
            
        feature_matrix = np.vstack(stacked_features).T
        pca = PCA(n_components=1)
        return pca.fit_transform(feature_matrix).flatten()

    else:
        raise ValueError(f"Unknown scalar field method: {method}")

def compute_approx_reeb_graph(vertices, faces, scalar_field, num_bins=20):
    f_min, f_max = scalar_field.min(), scalar_field.max()
    bins = np.linspace(f_min, f_max + 1e-8, num_bins + 1)
    bin_indices = np.digitize(scalar_field, bins) - 1
    
    edges = set()
    for face in faces:
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
            adj = coo_matrix((np.ones(len(local_edges)), (local_edges[:,0], local_edges[:,1])), 
                             shape=(len(v_in_bin), len(v_in_bin)))
            adj = adj.maximum(adj.T)
            n_components, labels = connected_components(adj, directed=False)
        else:
            n_components = len(v_in_bin)
            labels = np.arange(len(v_in_bin))
            
        for comp in range(n_components):
            comp_vertices = v_in_bin[labels == comp]
            center_pos = vertices[comp_vertices].mean(axis=0)
            
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

def launch_reeb_viewer(mesh_files, reeb_files, scalar_files):
    print("\nStarting interactive Reeb Graph orchestrator...")
    
    pl = pv.Plotter(shape=(1, 2))
    pl.title = "Cell Topology Evolution (Reeb Graphs)"
    state = {'frame': 0, 'total': len(mesh_files)}
    pl.add_axes()
    
    pl.subplot(0, 0)
    pl.camera_position = 'iso'
    pl.subplot(0, 1)
    pl.camera_position = 'iso'
    
    def update_frame(frame_idx):
        tm = TriMesh(mesh_files[frame_idx])

        faces_pv = np.empty((tm.faces.shape[0], 4), dtype=int)
        faces_pv[:, 0] = 3
        faces_pv[:, 1:] = tm.faces
        meshn = pv.PolyData(tm.vertices, faces_pv.flatten())
        meshn.rotate_x(90, inplace=True)
        meshn.rotate_z(90, inplace=True)
        
        bounds = meshn.bounds
        diag_size = np.linalg.norm([
            bounds[1] - bounds[0], 
            bounds[3] - bounds[2], 
            bounds[5] - bounds[4]
        ])
        node_radius = diag_size * 0.005
        edge_radius = diag_size * 0.002

        scalar_array = np.load(scalar_files[frame_idx])
        meshn.point_data['Dynamic_Scalar'] = scalar_array
        
        with open(reeb_files[frame_idx], 'rb') as f:
            graph = pickle.load(f)
        
        reeb_pv = create_reeb_polydata(graph)
        
        pl.subplot(0, 0)
        pl.add_mesh(meshn, scalars='Dynamic_Scalar', cmap='viridis', name='z_mesh', 
                    show_scalar_bar=True, render=False)
        pl.add_text(f"Scalar Field - Frame {frame_idx + 1}/{state['total']}", 
                    name='t1', font_size=10, position='upper_left')

        pl.subplot(0, 1)
        pl.add_mesh(meshn, color='white', opacity=0.25, name='ghost_mesh', render=False)
        
        if reeb_pv.n_points > 0:
            spheres = reeb_pv.glyph(geom=pv.Sphere(radius=node_radius), scale=False, orient=False)
            pl.add_mesh(spheres, color='red', name='reeb_nodes', render=False)
            
            if reeb_pv.n_lines > 0:
                tubes = reeb_pv.tube(radius=edge_radius) 
                pl.add_mesh(tubes, color='blue', name='reeb_edges', render=False)
            else:
                pl.remove_actor('reeb_edges')

        pl.add_text(f"Level-Set Reeb Graph - Frame {frame_idx + 1}/{state['total']}", 
                    name='t2', font_size=10, position='upper_left')
    
    update_frame(0)
    pl.subplot(0, 0)
    pl.reset_camera()
    pl.subplot(0, 1)
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
    pl.add_text("Time Control:\n  right arrow key : Next Mesh\n   left arrow key : Prev Mesh", 
                position='lower_left', font_size=6, color='black')
    pl.show(full_screen=True)

def visualize_reeb_graphs(mesh_folder_path, reeb_folder_path):
    mesh_folder = Path(mesh_folder_path)
    reeb_folder = Path(reeb_folder_path)
    
    obj_files = sorted([f for f in mesh_folder.iterdir() if f.is_file() and f.suffix == '.obj'])
    reeb_files = sorted([f for f in reeb_folder.iterdir() if f.is_file() and f.suffix == '.pkl'])
    scalar_files = sorted([f for f in reeb_folder.iterdir() if f.is_file() and f.name.startswith('Scalar') and f.suffix == '.npy'])
    
    if not obj_files or not reeb_files or not scalar_files:
        print("Error: Missing obj, pkl, or npy files for Reeb visualization.")
        return

    min_len = min(len(obj_files), len(reeb_files), len(scalar_files))
    
    launch_reeb_viewer(
        [str(f) for f in obj_files[:min_len]], 
        [str(f) for f in reeb_files[:min_len]],
        [str(f) for f in scalar_files[:min_len]]
    )