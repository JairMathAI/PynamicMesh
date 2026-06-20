import numpy as np
from pyFM.functional import FunctionalMapping
import pyvista as pv
import os
from pathlib import Path
from tqdm.auto import tqdm
from pyFM.mesh import TriMesh
import vtk
import os
import numpy as np
import pyvista as pv
from pathlib import Path
from pyFM.mesh import TriMesh

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

    obj_files = sorted([f for f in path.iterdir() if f.is_file() and f.suffix == '.obj'])
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
            tm = TriMesh(str(obj_files[frame_idx]))
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
                    tm = TriMesh(str(obj_files[i]))
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
        obj_files = sorted([f for f in itemsfiles if f.is_file() and f.suffix == '.obj'])
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
            meshn_1 = TriMesh(str(obj_files[0]))
            
            for i in range(1, len(obj_files)):
                meshn = TriMesh(str(obj_files[i]))
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
                meshn = TriMesh(str(obj_files[i]))
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
                        tm = TriMesh(str(obj_files[i]))
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

class CustomFunctionalMapping(FunctionalMapping):
    """
    Functional map wrapper with:
      - Combined descriptor support (WKS + HKS)
    """

    def _set_descriptors(self, descr1: np.ndarray, descr2: np.ndarray) -> None:
        self.descr1 = descr1
        self.descr2 = descr2
        self.A = descr1
        self.B = descr2

    def preprocess(
        self,
        K=(10, 10),
        n_descr=100,
        descr_type="WKS",
        landmarks=None,
        subsample_step=1,
        k_process=None,
        verbose=False,
        **kwargs,
    ):
        """
        Preprocess the meshes for functional map fitting.

        Supported descr_type:
          - "WKS"
          - "HKS"
          - "WKS+HKS" or "HKS+WKS" (Combined)
        """

        required_k1 = max(K[0], k_process if k_process else 100)
        required_k2 = max(K[1], k_process if k_process else 100)

        if self.mesh1.eigenvalues is None or len(self.mesh1.eigenvalues) < required_k1:
            self.mesh1.process(k=required_k1)
        if self.mesh2.eigenvalues is None or len(self.mesh2.eigenvalues) < required_k2:
            self.mesh2.process(k=required_k2)

        combined = isinstance(descr_type, str) and descr_type.upper() in {"WKS+HKS", "HKS+WKS"}

        if combined:
            super().preprocess(
                K=K,
                n_descr=n_descr,
                descr_type="WKS",
                landmarks=landmarks,
                subsample_step=subsample_step,
                k_process=required_k1, 
                verbose=verbose,
                **kwargs,
            )
            descr1_wks = self.descr1.copy()
            descr2_wks = self.descr2.copy()

            super().preprocess(
                K=K,
                n_descr=n_descr,
                descr_type="HKS",
                landmarks=landmarks,
                subsample_step=subsample_step,
                k_process=required_k1,
                verbose=verbose,
                **kwargs,
            )
            descr1_hks = self.descr1.copy()
            descr2_hks = self.descr2.copy()

            descr1_wks = (descr1_wks - descr1_wks.mean(axis=0)) / (descr1_wks.std(axis=0) + 1e-8)
            descr2_wks = (descr2_wks - descr2_wks.mean(axis=0)) / (descr2_wks.std(axis=0) + 1e-8)
            
            descr1_hks = (descr1_hks - descr1_hks.mean(axis=0)) / (descr1_hks.std(axis=0) + 1e-8)
            descr2_hks = (descr2_hks - descr2_hks.mean(axis=0)) / (descr2_hks.std(axis=0) + 1e-8)

            self._set_descriptors(
                np.hstack([descr1_wks, descr1_hks]),
                np.hstack([descr2_wks, descr2_hks]),
            )
        else:
            super().preprocess(
                K=K,
                n_descr=n_descr,
                descr_type=descr_type,
                landmarks=landmarks,
                subsample_step=subsample_step,
                k_process=required_k1,
                verbose=verbose,
                **kwargs,
            )
            self._set_descriptors(self.descr1, self.descr2)

        return self