import numpy as np
from pyFM.functional import FunctionalMapping
import pyvista as pv
import pickle
import os
from pathlib import Path
from tqdm.auto import tqdm
import vtk
import os
import numpy as np
import pyvista as pv
from pathlib import Path
from PynamicMesh.utils.tools import  mesh_mat2object 
from PynamicMesh.core.physic_model import computing_fields, create_pv_polydata
from PynamicMesh.core.reeb_graph import create_reeb_polydata
import copy




##################################################################################################################### Landmarks #########################################################################################################################################

def visual_selection_edition(scene_folder_path, mood='FM'):
    """
    Dynamically loads and visualizes a sequence of meshes in a single interactive window.
    Supports modes:
      - 'FM': Enforces that ALL meshes contain the exact same number of landmarks. Saves to landmarks.npy.
      - 'geodesic': Allows variable size selection per mesh. Empty = None. Saves to vert_ref_geo.npy.
      - 'heat_diffusion': Allows max 1 vertex or None per mesh. Saves to sources.npy.
      - 'source_sink': Allows pairs (exactly 2) or None per mesh. Saves to source_sink.npy.
    """
    print(f"\nStarting Dynamic Landmark Editor [Mode: {mood}]...")
    
    path = Path(scene_folder_path)
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{scene_folder_path}' is not a valid directory.")
        return

    obj_files = sorted([f for f in path.iterdir() if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')])
    if not obj_files:
        print(f"No .obj files found in {path}")
        return

    scene_name = path.name
    out_root = path.parent.parent / 'Results'
    target_folder = out_root / scene_name
    os.makedirs(target_folder, exist_ok=True)
    
    # Configure filename based on selected mood
    if mood == 'FM':
        landmarks_file = target_folder / 'landmarks.npy'
    elif mood == 'geodesic':
        landmarks_file = target_folder / 'vert_ref_geo.npy'
    elif mood == 'heat_diffusion':
        landmarks_file = target_folder / 'sources.npy'
    elif mood == 'harmonic':
        landmarks_file = target_folder / 'source_sink.npy'
    else:
        print(f"Error: Unknown mood '{mood}'. Choose from 'FM', 'geodesic', 'heat_diffusion', 'source_sink'.")
        return

    num_meshes = len(obj_files)
    picks = [[] for _ in range(num_meshes)]
    
    if landmarks_file.exists():
        print(f"Loading existing data from {landmarks_file}")
        loaded_data = np.load(landmarks_file, allow_pickle=True)
        if mood == 'FM':
            for i, trans in enumerate(loaded_data):
                if trans is not None and len(trans) > 0:
                    if not picks[i]:  
                        picks[i] = list(trans[:, 0])
                    picks[i+1] = list(trans[:, 1])
        else:
            for i, entry in enumerate(loaded_data):
                if i < num_meshes:
                    picks[i] = list(entry) if entry is not None else []
    else:
        print(f"No existing data found. A new workspace will be created at {landmarks_file}")

    state = {
        'frame': 0,
        'total': num_meshes,
        'picks': picks,
        'current_mesh': None,
        'drawn_actors': []
    }
    
    def check_validity():
        lengths = [len(p) for p in state['picks']]
        if mood == 'FM':
            if len(set(lengths)) <= 1:
                return True, "Valid: All meshes have matching landmark counts. Safe to close."
            mode_len = max(set(lengths), key=lengths.count) 
            errors = [f"Frame {i+1} ({l} pts)" for i, l in enumerate(lengths) if l != mode_len]
            return False, f"INVALID: Missing/Extra points. Check: {', '.join(errors)}"
        
        elif mood == 'geodesic':
            return True, "Valid (Geodesic): Any number of vertex selections allowed."
            
        elif mood == 'heat_diffusion':
            errors = [f"Frame {i+1} ({l} pts)" for i, l in enumerate(lengths) if l > 1]
            if not errors:
                return True, "Valid (Heat Diffusion): All frames have <= 1 point."
            return False, f"INVALID: Max 1 vertex allowed per mesh. Check: {', '.join(errors)}"
            
        elif mood == 'harmonic':
            errors = [f"Frame {i+1} ({l} pts)" for i, l in enumerate(lengths) if l not in [0, 2]]
            if not errors:
                return True, "Valid (Source-Sink): All frames have either 0 or 2 vertices (pairs)."
            return False, f"INVALID: Must have exactly a pair (2 vertices) or none (0). Check: {', '.join(errors)}"
            
        return False, "Unknown mood constraint validation."

    while True:
        plotter = pv.Plotter(title=f"Dynamic Landmark Viewer & Editor [{mood}]")
        plotter.add_axes()

        def hover_callback(caller, event):
            if state['current_mesh'] is None:
                return
            click_pos = plotter.iren.get_event_position()
            picker = vtk.vtkPointPicker()
            picker.SetTolerance(0.005)
            picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
            idx = picker.GetPointId()
            
            if idx != -1:
                pick_pos = picker.GetPickPosition()
                mesh_idx = state['current_mesh'].find_closest_point(pick_pos)
                plotter.add_text(f"Hover Vertex: {mesh_idx}", name='hover_info', 
                                 position='upper_right', font_size=12, color='green')
            else:
                plotter.add_text("Hover Vertex: -", name='hover_info', 
                                 position='upper_right', font_size=12, color='white')

        plotter.iren.add_observer("MouseMoveEvent", hover_callback)

        def redraw_labels():
            for actor in state['drawn_actors']:
                plotter.remove_actor(actor)
            state['drawn_actors'].clear()

            curr_picks = state['picks'][state['frame']]
            if not curr_picks:
                return

            mesh_pv = state['current_mesh']
            points = [mesh_pv.points[idx] for idx in curr_picks]
            labels = [str(i + 1) for i in range(len(curr_picks))]

            bounds = mesh_pv.bounds
            sphere_radius = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 0.003

            for pt in points:
                actor = plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=pt), color="blue")
                state['drawn_actors'].append(actor)

            label_actor = plotter.add_point_labels(
                points, labels, point_size=0, font_size=15, 
                text_color='black', shape_color='white', shape_opacity=0.7, margin=3
            )
            state['drawn_actors'].append(label_actor)

        def update_frame(frame_idx):
            tm = mesh_mat2object(obj_files[frame_idx])
            pad = np.full((tm.faces.shape[0], 1), 3, dtype=np.int64)
            pv_faces = np.hstack((pad, tm.faces)).flatten()
            mesh_pv = pv.PolyData(tm.vertices, pv_faces)
            mesh_pv.rotate_x(90, inplace=True)
            mesh_pv.rotate_z(90, inplace=True)
            state['current_mesh'] = mesh_pv

            plotter.add_mesh(mesh_pv, name='main_mesh', color="white", show_edges=True, edge_color="green", opacity=1.0)
            plotter.add_points(mesh_pv.points, name='main_points', color="yellow", render_points_as_spheres=True, point_size=5)

            instruction_text = (
                f"Frame {frame_idx + 1} / {state['total']} : {obj_files[frame_idx].name}\n"
                f"Mode: {mood}\n"
                "--------------------------------------------------\n"
                "LEFT CLICK to add/remove a point.\n"
                "ARROWS (Left/Right) to switch meshes.\n"
                "Close window when finished to save."
            )
            plotter.add_text(instruction_text, name='ui_text', font_size=6, position='upper_left')
            plotter.add_text("Hover Vertex: -", name='hover_info', position='upper_right', font_size=12, color='white')

            valid, msg = check_validity()
            plotter.add_text(msg, name='status_text', font_size=10, position='lower_left', color="green" if valid else "red")

            redraw_labels()

        def step_next():
            if state['frame'] < state['total'] - 1:
                state['frame'] += 1
                update_frame(state['frame'])

        def step_prev():
            if state['frame'] > 0:
                state['frame'] -= 1
                update_frame(state['frame'])

        def pick_callback(coord):
            if state['current_mesh'] is None: return
            idx = state['current_mesh'].find_closest_point(coord)
            curr_picks = state['picks'][state['frame']]
            
            if idx in curr_picks:
                curr_picks.remove(idx)
            else:
                curr_picks.append(idx)
            
            redraw_labels()
            valid, msg = check_validity()
            plotter.add_text(msg, name='status_text', font_size=6, position='lower_left', color="green" if valid else "red")

        plotter.add_key_event('Right', step_next)
        plotter.add_key_event('Left', step_prev)
        plotter.enable_point_picking(callback=pick_callback, show_message=False, left_clicking=True)

        update_frame(state['frame'])
        plotter.show(full_screen=True)
        
        is_valid, error_msg = check_validity()
        if is_valid:
            break
        
        print(f"\n[ACTION REQUIRED] {error_msg}")
        print("Reopening the editor. Please correct the constraint violations before exiting.")

    # Save format processing based on mood rule sets
    if mood == 'FM':
        new_transitions = []
        for i in range(state['total'] - 1):
            src = state['picks'][i]
            tgt = state['picks'][i + 1]
            if len(src) == 0:
                new_transitions.append(None)
            else:
                pairs = np.array([[src[j], tgt[j]] for j in range(len(src))], dtype=int)
                new_transitions.append(pairs)
        np.save(landmarks_file, np.array(new_transitions, dtype=object), allow_pickle=True)
    else:
        saved_picks = []
        for i, p in enumerate(state['picks']):
            if len(p) > 0:
                saved_picks.append(p)
            else:
                if mood == 'harmonic':
                    # Load mesh to get total vertex count for max index
                    tm = mesh_mat2object(obj_files[i])
                    min_index = 0
                    max_index = tm.vertices.shape[0] - 1
                    saved_picks.append([min_index, max_index])
                else:
                    saved_picks.append(None)
                    
        np.save(landmarks_file, np.array(saved_picks, dtype=object), allow_pickle=True)
        
    print(f"\n[SUCCESS] Updated sequence selections saved to: {landmarks_file}\n")

def precompute_landmarks(path_str, mood='FM'):
    """
    Iterates through folders and allows manual landmark/vertex selection matching 
    specific conditions dictated by the mood parameter.
    """
    path = Path(path_str)
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{path_str}' is not a valid directory.")
        return

    subdirectories = [f for f in path.iterdir() if f.is_dir()]
    if not subdirectories:
        print(f"No folders found in {path}")
        return

    out_root = path.parent / 'Results'

    for folder in tqdm(subdirectories, desc='Precomputing Folders'):
        itemsfiles = list(folder.iterdir())
        obj_files = sorted([f for f in itemsfiles if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')])
        if not obj_files:
            continue
            
        scene_name = obj_files[0].parent.name
        target_folder = out_root / scene_name
        os.makedirs(target_folder, exist_ok=True)
        
        if mood == 'FM':
            landmarks_file = target_folder / 'landmarks.npy'
        elif mood == 'geodesic':
            landmarks_file = target_folder / 'vert_ref_geo.npy'
        elif mood == 'heat_diffusion':
            landmarks_file = target_folder / 'sources.npy'
        elif mood == 'harmonic':
            landmarks_file = target_folder / 'source_sink.npy'
        else:
            print(f"Error: Unknown mood '{mood}'.")
            return

        if mood == 'FM':
            if len(obj_files) < 2:
                continue
            all_transitions = []
            persisted_target_picks = None
            meshn_1 = mesh_mat2object(obj_files[0])
            
            for i in range(1, len(obj_files)):
                meshn = mesh_mat2object(obj_files[i])
                if i == 1 or persisted_target_picks is None:
                    source_picks = pick_single_mesh(meshn_1.vertices, meshn_1.faces, f"{scene_name} - Mesh {i-1} (Source)", marker_color="blue")
                else:
                    source_picks = persisted_target_picks

                if not source_picks:
                    all_transitions.append(None)
                    persisted_target_picks = None
                else:
                    expected = len(source_picks)
                    target_picks = []
                    
                    while True:
                        target_picks = pick_single_mesh(
                            meshn.vertices, meshn.faces, 
                            f"{scene_name} - Mesh {i} (Target)\nEXPECTED: {expected} points", 
                            marker_color="blue", 
                            expected_count=expected,
                            initial_picks=target_picks
                        )
                        if len(target_picks) == expected:
                            break
                    
                    current_landmarks = [[source_picks[j], target_picks[j]] for j in range(expected)]
                    current_landmarks = np.array(current_landmarks, dtype=int)
                    all_transitions.append(current_landmarks)
                    persisted_target_picks = target_picks

                meshn_1 = meshn
            
            np.save(landmarks_file, np.array(all_transitions, dtype=object), allow_pickle=True)
            print(f"\n[SUCCESS] Precomputed landmarks saved to: {landmarks_file}\n")
            
        else:
            # Multi-mode step-by-mesh pipeline logic
            all_selections = []
            if landmarks_file.exists():
                print(f"Loading existing workspace records from {landmarks_file}")
                loaded_data = np.load(landmarks_file, allow_pickle=True)
                all_selections = [list(x) if x is not None else [] for x in loaded_data]
            
            while len(all_selections) < len(obj_files):
                all_selections.append([])
                
            for i in range(len(obj_files)):
                meshn = mesh_mat2object(obj_files[i])
                initial_picks = all_selections[i]
                
                while True:
                    title = f"{scene_name} - Mesh {i+1} ({obj_files[i].name})\nMode: {mood}"
                    if mood == 'heat_diffusion':
                        title += "\nCONSTRAINT: Max 1 vertex or none allowed."
                    elif mood == 'harmonic':
                        title += "\nCONSTRAINT: Exactly 2 vertices (pair) or 0 vertices allowed."
                    
                    target_picks = pick_single_mesh(
                        meshn.vertices, meshn.faces, 
                        title, 
                        marker_color="blue", 
                        initial_picks=initial_picks
                    )
                    
                    # Run validations per mesh during consecutive configuration walkthrough
                    if mood == 'heat_diffusion' and len(target_picks) > 1:
                        print(f"[WARNING] heat_diffusion mode allows at most 1 point. Selected {len(target_picks)}.")
                        initial_picks = target_picks
                        continue
                    if mood == 'harmonic' and len(target_picks) not in [0, 2]:
                        print(f"[WARNING] source_sink mode requires exactly 0 or 2 points. Selected {len(target_picks)}.")
                        initial_picks = target_picks
                        continue
                        
                    break
                
                all_selections[i] = target_picks

            saved_selections = []
            for i, p in enumerate(all_selections):
                if len(p) > 0:
                    saved_selections.append(p)
                else:
                    if mood == 'harmonic':
                        # Load mesh to get total vertex count for max index
                        tm = mesh_mat2object(obj_files[i])
                        min_index = 0
                        max_index = tm.vertices.shape[0] - 1
                        saved_selections.append([min_index, max_index])
                    else:
                        saved_selections.append(None)
                        
            np.save(landmarks_file, np.array(saved_selections, dtype=object), allow_pickle=True)
            print(f"\n[SUCCESS] Precomputed selection lists saved to: {landmarks_file}\n")
                

def pick_single_mesh(vertices, faces, title, marker_color="blue", expected_count=None, initial_picks=None):
    """
    Opens a SINGLE PyVista window to pick points sequentially.
    Points are visibly numbered (1, 2, 3...).
    Edges are drawn green, vertices are drawn dark grey.
    """
    def create_pv_mesh(v, f):
        pad = np.full((f.shape[0], 1), 3, dtype=np.int64)
        pv_faces = np.hstack((pad, f)).flatten()
        return pv.PolyData(v, pv_faces)

    mesh_pv = create_pv_mesh(vertices, faces)
    mesh_pv.rotate_x(90, inplace=True)
    mesh_pv.rotate_z(90, inplace=True)
    bounds = mesh_pv.bounds
    sphere_radius = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 0.003

    plotter = pv.Plotter(title=title)
    plotter.add_axes()
    
    plotter.add_mesh(mesh_pv, color="white", show_edges=True, edge_color="green", opacity=1.0)
    plotter.add_points(mesh_pv.points, color="yellow", render_points_as_spheres=True, point_size=5)

    instruction_text = f"{title}\nLEFT CLICK to pick points.\nThey will be numbered (1, 2, 3...) to define the pairing order.\nCLICK an existing point to remove it.\nClose window when done."

    if expected_count is not None:
        plotter.add_text(f"\n\n-> EXPECTED NUMBER OF POINTS: {expected_count}", font_size=6, position='lower_left', color='red')
    plotter.add_text(instruction_text, font_size=6, position='upper_left')

    picked_list = initial_picks.copy() if initial_picks else []      
    drawn_actors = []     

    def redraw_labels():
        for actor in drawn_actors:
            plotter.remove_actor(actor)
        drawn_actors.clear()

        if not picked_list:
            return

        points = [mesh_pv.points[idx] for idx in picked_list]
        labels = [str(i + 1) for i in range(len(picked_list))]

        for pt in points:
            actor = plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=pt), color=marker_color)
            drawn_actors.append(actor)

        label_actor = plotter.add_point_labels(
            points, labels,
            point_size=0, font_size=15, text_color='black', shape_color='white', shape_opacity=0.7, margin=3
        )
        drawn_actors.append(label_actor)

    def callback(coord):
        idx = mesh_pv.find_closest_point(coord)
        if idx in picked_list:
            picked_list.remove(idx)
        else:
            picked_list.append(idx)
        redraw_labels()

    plotter.enable_point_picking(callback=callback, show_message=False, left_clicking=True)

    if picked_list:
        redraw_labels()
        
    plotter.show(full_screen=True)

    return picked_list

##################################################################################################################### Physic #########################################################################################################################################

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

##################################################################################################################### Reeb Graphs #########################################################################################################################################

def edit_graph(mesh_folder_path, reeb_folder_path):
    print("\nStarting Interactive Split-Screen Graph Editor with Undo...")
    
    mesh_path = Path(mesh_folder_path)
    reeb_path = Path(reeb_folder_path)
    
    if not mesh_path.exists() or not reeb_path.exists():
        print("Error: Invalid mesh or reeb graph directory paths.")
        return

    obj_files = sorted([f for f in mesh_path.iterdir() if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')])
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
        tm = mesh_mat2object(obj_files[frame_idx]) 
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
        tm = mesh_mat2object(mesh_files[frame_idx])

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
    
    obj_files = sorted([f for f in mesh_folder.iterdir() if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')])
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