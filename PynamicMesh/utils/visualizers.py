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
from PynamicMesh.utils.tools import  mesh_mat2object , natural_sort_key
from PynamicMesh.core.physic_model import computing_fields, create_pv_polydata, pv_field_name, FIELD_KEYS
import pandas as pd
from PynamicMesh.core.reeb_graph import create_reeb_polydata
import copy
import re



##################################################################################################################### Landmarks #########################################################################################################################################
LANDMARK_FILE_RE = re.compile(r"landmarks_T(\d+)_T(\d+)\.npy$")
# Distinct colours for the landmark pairs of a transition (matplotlib tab20 without the greys).
PAIR_COLORS = np.array([[0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16], [0.58, 0.40, 0.74],
                        [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.68, 0.78, 0.91],
                        [1.00, 0.73, 0.47], [0.60, 0.87, 0.54], [1.00, 0.60, 0.59], [0.77, 0.69, 0.84], [0.77, 0.61, 0.58],
                        [0.97, 0.71, 0.82], [0.86, 0.86, 0.55], [0.62, 0.85, 0.90]])


def _display_mesh(obj_file):
    """Mesh in the display frame used by every viewer (raw mesh + x90/z90 rotation = aligned frame)."""
    tm = mesh_mat2object(obj_file)
    pad = np.full((tm.faces.shape[0], 1), 3, dtype=np.int64)
    mesh_pv = pv.PolyData(np.asarray(tm.vertices, dtype=np.float64), np.hstack((pad, tm.faces)).ravel())
    mesh_pv.rotate_x(90, inplace=True)
    mesh_pv.rotate_z(90, inplace=True)
    return mesh_pv


def load_fm_transitions(target_folder, num_meshes, source='auto'):
    """
    Loads functional-map landmarks as one list of [src, tgt] pairs per transition T{i} -> T{i+1}.

    Two storages exist:
      * per-transition files written by the pipeline for the landmarks actually used (explicit,
        'precomputed' or 'auto'): Results/<scene>/Landmarks/landmarks_T0000_T0001.npy, (m, 2) arrays
      * the legacy single file read by landmarks='precomputed': Results/<scene>/landmarks.npy,
        object array with one (m, 2) array (or None) per transition
    source: 'auto' (per-transition files if present, else legacy) | 'transitions' | 'legacy'
    Returns (transitions, origin) with origin in {'transitions', 'legacy', None}.
    """
    target_folder = Path(target_folder)
    transitions = [[] for _ in range(max(num_meshes - 1, 0))]
    folder = target_folder / 'Landmarks'
    files = sorted(folder.glob('landmarks_T*_T*.npy')) if folder.is_dir() else []

    if source in ('auto', 'transitions') and files:
        for f in files:
            m = LANDMARK_FILE_RE.search(f.name)
            if not m:
                continue
            i = int(m.group(1))
            if int(m.group(2)) != i + 1 or i >= len(transitions):
                continue
            arr = np.load(f, allow_pickle=True)
            if arr is not None and np.size(arr) > 0:
                transitions[i] = [[int(a), int(b)] for a, b in np.asarray(arr, dtype=int).reshape(-1, 2)]
        return transitions, 'transitions'

    legacy = target_folder / 'landmarks.npy'
    if source in ('auto', 'legacy') and legacy.exists():
        loaded = np.load(legacy, allow_pickle=True)
        for i, trans in enumerate(loaded):
            if i < len(transitions) and trans is not None and len(trans) > 0:
                transitions[i] = [[int(a), int(b)] for a, b in np.asarray(trans, dtype=int).reshape(-1, 2)]
        return transitions, 'legacy'

    return transitions, None


def save_fm_transitions(target_folder, transitions):
    """Writes both storages (see load_fm_transitions) and removes stale per-transition files."""
    target_folder = Path(target_folder)
    folder = target_folder / 'Landmarks'
    os.makedirs(folder, exist_ok=True)
    legacy = []
    for i, pairs in enumerate(transitions):
        f = folder / f'landmarks_T{i:04d}_T{i+1:04d}.npy'
        if pairs:
            arr = np.asarray(pairs, dtype=int).reshape(-1, 2)
            np.save(f, arr)
            legacy.append(arr)
        else:
            legacy.append(None)
            if f.exists():
                f.unlink()
    legacy_file = target_folder / 'landmarks.npy'
    np.save(legacy_file, np.array(legacy, dtype=object), allow_pickle=True)
    return legacy_file, folder


def _edit_fm_transitions(obj_files, target_folder, source='auto'):
    """
    Interactive editor of functional-map landmark pairs, one transition T{i} -> T{i+1} at a time.
    The two meshes are shown side by side (the target is shifted along x); a pair is made by
    clicking a vertex on the source mesh and then a vertex on the target mesh.
    """
    num_meshes = len(obj_files)
    if num_meshes < 2:
        print("At least two meshes are required to edit functional-map landmarks.")
        return
    transitions, origin = load_fm_transitions(target_folder, num_meshes, source)
    if origin == 'transitions':
        print(f"Loading landmark pairs used by the pipeline from {Path(target_folder) / 'Landmarks'}")
    elif origin == 'legacy':
        print(f"Loading landmark pairs from {Path(target_folder) / 'landmarks.npy'}")
    else:
        print(f"No existing landmarks found. A new workspace will be created in {target_folder}")

    state = {'trans': 0, 'total': num_meshes - 1, 'transitions': transitions, 'pending': None,
             'src_mesh': None, 'tgt_mesh': None, 'offset': 0.0, 'drawn': []}

    def status():
        counts = [len(p) for p in state['transitions']]
        n_empty = sum(1 for c in counts if c == 0)
        pend = "  |  PENDING source point: click a target vertex" if state['pending'] is not None else ""
        msg = f"Pairs per transition: {counts}  ({n_empty} transition(s) without landmarks){pend}"
        return state['pending'] is None, msg

    plotter = pv.Plotter(title="Functional Map Landmark Pair Editor")
    plotter.add_axes()

    def clear_drawn():
        for actor in state['drawn']:
            plotter.remove_actor(actor, render=False)
        state['drawn'].clear()

    def redraw_pairs():
        clear_drawn()
        src_mesh, tgt_mesh = state['src_mesh'], state['tgt_mesh']
        if src_mesh is None:
            return
        pairs = state['transitions'][state['trans']]
        pts, labels, colors = [], [], []
        for k, (s, t) in enumerate(pairs):
            ps, pt = src_mesh.points[s], tgt_mesh.points[t]
            rgb = PAIR_COLORS[k % len(PAIR_COLORS)]
            # connecting line (thick) + the two end points drawn below as screen-space markers
            state['drawn'].append(plotter.add_mesh(pv.Line(ps, pt), color=rgb, line_width=4, render=False))
            pts += [ps, pt]
            labels += [str(k + 1), str(k + 1)]
            colors += [rgb, rgb]
        if pts:
            cloud = pv.PolyData(np.asarray(pts))
            cloud['rgb'] = (np.asarray(colors) * 255).astype(np.uint8)
            # Markers with a fixed size in pixels: visible whatever the scale of the mesh.
            state['drawn'].append(plotter.add_mesh(cloud, scalars='rgb', rgb=True, point_size=18,
                                                   render_points_as_spheres=True, render=False))
            state['drawn'].append(plotter.add_point_labels(np.asarray(pts), labels, point_size=0, font_size=14,
                                                           text_color='black', shape_color='white', shape_opacity=0.8,
                                                           margin=3, always_visible=True, render=False))
        if state['pending'] is not None:
            pp = src_mesh.points[state['pending']]
            state['drawn'].append(plotter.add_mesh(pv.PolyData(pp.reshape(1, 3)), color='orange', point_size=24,
                                                   render_points_as_spheres=True, render=False))
            state['drawn'].append(plotter.add_point_labels(pp.reshape(1, 3), ["? (click target)"], point_size=0, font_size=14,
                                                           text_color='black', shape_color='orange', shape_opacity=0.9,
                                                           margin=3, always_visible=True, render=False))
        valid, msg = status()
        plotter.add_text(msg, name='status_text', font_size=8, position='lower_left', color="green" if valid else "red")

    def update_transition(i):
        state['pending'] = None
        src_mesh = _display_mesh(obj_files[i])
        tgt_mesh = _display_mesh(obj_files[i + 1])
        b = src_mesh.bounds
        state['offset'] = 1.25 * (b[1] - b[0])
        tgt_mesh.translate([state['offset'], 0.0, 0.0], inplace=True)
        state['src_mesh'], state['tgt_mesh'] = src_mesh, tgt_mesh

        plotter.add_mesh(src_mesh, name='src_mesh', color="white", show_edges=True, edge_color="green", opacity=1.0)
        plotter.add_mesh(tgt_mesh, name='tgt_mesh', color="white", show_edges=True, edge_color="purple", opacity=1.0)
        plotter.add_points(src_mesh.points, name='src_points', color="yellow", render_points_as_spheres=True, point_size=5)
        plotter.add_points(tgt_mesh.points, name='tgt_points', color="yellow", render_points_as_spheres=True, point_size=5)
        plotter.add_text(f"SOURCE  T{i}: {obj_files[i].name}", name='src_label', position=(0.05, 0.9), viewport=True, font_size=9, color='green')
        plotter.add_text(f"TARGET  T{i+1}: {obj_files[i+1].name}", name='tgt_label', position=(0.55, 0.9), viewport=True, font_size=9, color='purple')
        plotter.add_text(
            f"Transition {i + 1} / {state['total']} : T{i} -> T{i+1}\n"
            "--------------------------------------------------\n"
            "LEFT CLICK a vertex on the SOURCE mesh, then on the TARGET mesh to create a pair.\n"
            "LEFT CLICK an existing point to remove its pair.\n"
            "ARROWS (Left/Right) to switch transitions.   d : delete last pair    c : clear transition\n"
            "Close window when finished to save.",
            name='ui_text', font_size=6, position='upper_left')
        plotter.add_text("Hover Vertex: -", name='hover_info', position='upper_right', font_size=12, color='white')
        redraw_pairs()
        if not state.get('camera_set'):
            plotter.view_xz()               # x to the right: source on the left, target on the right
            state['camera_set'] = True
        plotter.reset_camera()

    def locate(coord):
        """Which mesh was clicked and the closest vertex index on it."""
        s_idx = state['src_mesh'].find_closest_point(coord)
        t_idx = state['tgt_mesh'].find_closest_point(coord)
        ds = np.linalg.norm(state['src_mesh'].points[s_idx] - coord)
        dt = np.linalg.norm(state['tgt_mesh'].points[t_idx] - coord)
        return ('src', int(s_idx)) if ds <= dt else ('tgt', int(t_idx))

    def hover_callback(caller, event):
        if state['src_mesh'] is None:
            return
        pos = plotter.iren.get_event_position()
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.005)
        picker.Pick(pos[0], pos[1], 0, plotter.renderer)
        if picker.GetPointId() != -1:
            side, idx = locate(np.asarray(picker.GetPickPosition()))
            plotter.add_text(f"Hover Vertex: {idx} ({'source' if side == 'src' else 'target'})", name='hover_info',
                             position='upper_right', font_size=12, color='green' if side == 'src' else 'orchid')
        else:
            plotter.add_text("Hover Vertex: -", name='hover_info', position='upper_right', font_size=12, color='white')

    plotter.iren.add_observer("MouseMoveEvent", hover_callback)

    def pick_callback(coord):
        if state['src_mesh'] is None:
            return
        side, idx = locate(np.asarray(coord))
        pairs = state['transitions'][state['trans']]
        if side == 'src':
            existing = [p for p in pairs if p[0] == idx]
            if existing:
                pairs.remove(existing[0])
            elif state['pending'] == idx:
                state['pending'] = None
            else:
                state['pending'] = idx
        else:
            existing = [p for p in pairs if p[1] == idx]
            if existing:
                pairs.remove(existing[0])
            elif state['pending'] is not None:
                pairs.append([state['pending'], idx])
                state['pending'] = None
            else:
                plotter.add_text("Select a SOURCE vertex first.", name='status_text', font_size=8, position='lower_left', color="red")
                return
        redraw_pairs()

    def delete_last():
        pairs = state['transitions'][state['trans']]
        if state['pending'] is not None:
            state['pending'] = None
        elif pairs:
            pairs.pop()
        redraw_pairs()

    def clear_transition():
        state['transitions'][state['trans']].clear()
        state['pending'] = None
        redraw_pairs()

    def step_next():
        if state['trans'] < state['total'] - 1:
            state['trans'] += 1
            update_transition(state['trans'])

    def step_prev():
        if state['trans'] > 0:
            state['trans'] -= 1
            update_transition(state['trans'])

    plotter.add_key_event('Right', step_next)
    plotter.add_key_event('Left', step_prev)
    plotter.add_key_event('d', delete_last)
    plotter.add_key_event('c', clear_transition)
    plotter.enable_point_picking(callback=pick_callback, show_message=False, left_clicking=True)

    update_transition(0)
    plotter.show(full_screen=True)

    if state['pending'] is not None:
        print("[Warning] A source point without target was discarded.")
    legacy_file, folder = save_fm_transitions(target_folder, state['transitions'])
    counts = [len(p) for p in state['transitions']]
    print(f"\n[SUCCESS] Landmark pairs saved: {legacy_file} (read by landmarks='precomputed') and one file per "
          f"transition in {folder}. Pairs per transition: {counts}\n")


def visual_selection_edition(scene_folder_path, mood='FM', source='auto'):
    """
    Dynamically loads and visualizes a sequence of meshes in a single interactive window.
    Supports modes:
      - 'FM': Landmark PAIRS per transition T{i} -> T{i+1} (independent count per transition). Loads the
              pairs used by the pipeline (Results/<scene>/Landmarks/landmarks_Txxxx_Tyyyy.npy, written for
              explicit, 'precomputed' and 'auto' landmarks) or the legacy landmarks.npy, and saves both.
              `source` selects the storage to load: 'auto' | 'transitions' | 'legacy'.
      - 'geodesic': Allows variable size selection per mesh. Empty = None. Saves to vert_ref_geo.npy.
      - 'heat_diffusion': Allows max 1 vertex or None per mesh. Saves to sources.npy.
      - 'harmonic': Allows pairs (exactly 2) or None per mesh. Saves to source_sink.npy.
    """
    print(f"\nStarting Dynamic Landmark Editor [Mode: {mood}]...")

    path = Path(scene_folder_path)
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{scene_folder_path}' is not a valid directory.")
        return

    obj_files = sorted([f for f in path.iterdir() if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')], key=natural_sort_key)
    if not obj_files:
        print(f"No .obj files found in {path}")
        return

    scene_name = path.name
    out_root = path.parent.parent / 'Results'
    target_folder = out_root / scene_name
    os.makedirs(target_folder, exist_ok=True)

    if mood == 'FM':
        _edit_fm_transitions(obj_files, target_folder, source=source)
        return

    # Configure filename based on selected mood
    if mood == 'geodesic':
        landmarks_file = target_folder / 'vert_ref_geo.npy'
    elif mood == 'heat_diffusion':
        landmarks_file = target_folder / 'sources.npy'
    elif mood == 'harmonic':
        landmarks_file = target_folder / 'source_sink.npy'
    else:
        print(f"Error: Unknown mood '{mood}'. Choose from 'FM', 'geodesic', 'heat_diffusion', 'harmonic'.")
        return

    num_meshes = len(obj_files)
    picks = [[] for _ in range(num_meshes)]

    if landmarks_file.exists():
        print(f"Loading existing data from {landmarks_file}")
        loaded_data = np.load(landmarks_file, allow_pickle=True)
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
        if mood == 'geodesic':
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

    # Save format processing based on mood rule sets (frame based moods)
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
        obj_files = sorted([f for f in itemsfiles if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')], key=natural_sort_key)

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
            
            # Both storages: legacy landmarks.npy (read by landmarks='precomputed') and one file per transition
            save_fm_transitions(target_folder, [[] if t is None else np.asarray(t).tolist() for t in all_transitions])
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

# Gallery pages: (npz key, panel title, colormap, center). center=None -> percentile range,
# center=0/1 -> colour range symmetric around that value (strains around 0, stretches around 1).
PHYSICS_PAGES = [
    ("Kinematics & classic strains", [
        ('RGB', 'Color Transfer (tracking)', None, None),
        ('velocity', 'Velocity magnitude', 'viridis', None),
        ('acceleration', 'Acceleration magnitude', 'magma', None),
        ('strain', 'Linear edge strain', 'coolwarm', 0.0),
        ('area_strain', 'Areal strain (A2-A1)/A1', 'coolwarm', 0.0),
        ('normal_flow', 'Normal (protrusion) flow', 'Spectral', 0.0),
        ('tangential_flow', 'Tangential (lateral) flow', 'plasma', None),
        ('normal_rotation', 'Normal rotation angle [rad]', 'cividis', None),
        ('curvature_change', 'Mean curvature change dH', 'coolwarm', 0.0),
    ]),
    ("Continuum mechanics (deformation gradient)", [
        ('RGB', 'Color Transfer (tracking)', None, None),
        ('stretch_max', 'Max principal stretch l1', 'coolwarm', 1.0),
        ('stretch_min', 'Min principal stretch l2', 'coolwarm', 1.0),
        ('principal_strain_max', 'Green-Lagrange strain e1', 'coolwarm', 0.0),
        ('principal_strain_min', 'Green-Lagrange strain e2', 'coolwarm', 0.0),
        ('shear_anisotropy', 'Shear anisotropy log(l1/l2)', 'inferno', None),
        ('max_shear_strain', 'Max shear strain (e1-e2)/2', 'plasma', None),
        ('elastic_energy_density', 'Elastic energy density (ARAP)', 'hot', None),
        ('dilatation_log', 'Dilatation log(A2/A1)', 'coolwarm', 0.0),
    ]),
]


def launch_physics_viewer(frames_data, global_df=None, pct=(1.0, 99.0)):
    """
    Multi-physics gallery. Pages are switched with the Up/Down arrow keys (or 'p'), frames with
    Left/Right. Only the fields present in the frames are shown, so legacy .npz files (6 fields)
    and new ones (17 fields) both work. Colour limits are robust percentiles over all frames,
    so a single collapsed triangle no longer saturates the colour map.
    """
    print("\nStarting interactive 3D Multi-Physics Gallery...")
    meshes = [create_pv_polydata(d) for d in frames_data]
    available = set(k for k in FIELD_KEYS if all(k in d for d in frames_data)) | {'RGB'}

    # The base frame carries no deformation by construction; use the transition frames for the colour range.
    clim_frames = frames_data[1:] if len(frames_data) > 1 else frames_data

    def get_clim(key, center=None):
        arr = np.concatenate([np.asarray(d[key]).ravel() for d in clim_frames])
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return [0.0, 1.0]
        if center is not None:
            v = np.percentile(np.abs(arr - center), pct[1])
            v = v if v > 0 else 1e-6
            return [center - v, center + v]
        lo, hi = np.percentile(arr, pct)
        return [lo, hi if hi > lo else lo + 1e-6]

    pages = []
    for title, panels in PHYSICS_PAGES:
        panels = [p for p in panels if p[0] in available]
        if len(panels) > 1:
            pages.append((title, panels, {p[0]: (get_clim(p[0], p[3]) if p[0] != 'RGB' else None) for p in panels}))
    if not pages:
        print("Error: no displayable fields in the frames.")
        return

    n_panels = max(len(p[1]) for p in pages)
    rows = 2 if n_panels <= 6 else 3
    cols = int(np.ceil(n_panels / rows))
    pl = pv.Plotter(shape=(rows, cols))
    pl.title = "Cell Dynamics Multi-Physics Gallery"
    state = {'frame': 0, 'total': len(meshes), 'page': 0}

    for i in range(rows):
        for j in range(cols):
            pl.subplot(i, j)
            pl.add_axes()
            pl.camera_position = 'iso'

    def global_text(frame_idx):
        if global_df is None or frame_idx == 0:
            return ""
        row = global_df[global_df['Time_Step'] == frame_idx]
        if row.empty:
            return ""
        r = row.iloc[0]
        return (f"\narea ratio {r.get('area_ratio', np.nan):.4f} | volume ratio {r.get('volume_ratio', np.nan):.4f}"
                f"\nmean speed {r.get('mean_speed', np.nan):.3g} | elastic energy {r.get('total_elastic_energy', np.nan):.3g}"
                f"\np2p injectivity {r.get('p2p_injectivity', np.nan):.2f} | collapsed faces {r.get('collapsed_faces_fraction', np.nan):.1%}")

    def clear_scalar_bars():
        """Scalar bars are separate actors: drop them all before drawing a new page."""
        try:
            for title in list(pl.scalar_bars.keys()):
                pl.remove_scalar_bar(title, render=False)
        except Exception:  # noqa: BLE001 - older pyvista without ScalarBars API
            pass

    def update_frame(frame_idx, page_changed=False):
        page_title, panels, clims = pages[state['page']]
        mesh = meshes[frame_idx]
        if page_changed:
            clear_scalar_bars()
        k = 0
        for i in range(rows):
            for j in range(cols):
                pl.subplot(i, j)
                # Actor names must be unique per subplot: pyvista's remove_actor / name replacement act on
                # ALL renderers, so a shared name deletes the mesh from every other panel.
                mesh_name, title_name = f'field_mesh_{i}_{j}', f'panel_title_{i}_{j}'
                if page_changed or k >= len(panels):
                    pl.remove_actor(mesh_name, render=False)
                    pl.remove_actor(title_name, render=False)
                if k >= len(panels):
                    k += 1
                    continue
                key, title, cmap, _ = panels[k]
                if key == 'RGB':
                    pl.add_mesh(mesh, scalars='RGB', rgb=True, name=mesh_name, show_scalar_bar=False, render=False)
                    pl.add_text(f"{title}  ({frame_idx + 1}/{state['total']})\nPage {state['page'] + 1}/{len(pages)}: {page_title}"
                                + global_text(frame_idx), name=title_name, font_size=8, position='upper_left')
                else:
                    pl.add_mesh(mesh, scalars=pv_field_name(key), cmap=cmap, clim=clims[key], name=mesh_name,
                                scalar_bar_args={'title': pv_field_name(key), 'n_labels': 3}, render=False)
                    pl.add_text(title, name=title_name, font_size=10, position='upper_left')
                k += 1

    update_frame(0)
    for i in range(rows):
        for j in range(cols):
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

    def next_page():
        state['page'] = (state['page'] + 1) % len(pages)
        update_frame(state['frame'], page_changed=True)
        pl.render()

    def prev_page():
        state['page'] = (state['page'] - 1) % len(pages)
        update_frame(state['frame'], page_changed=True)
        pl.render()

    pl.add_key_event('Right', step_next)
    pl.add_key_event('Left', step_prev)
    pl.add_key_event('Up', next_page)
    pl.add_key_event('Down', prev_page)
    pl.add_key_event('p', next_page)

    pl.subplot(rows - 1, 0)
    pl.add_text("Controls:\n  Right/Left : next / previous frame\n  Up/Down or p : switch quantity page",
                position='lower_left', font_size=6, color='black', name='controls')
    pl.show(full_screen=True)


def visualize_physics(mesh_folder_path, matrix_folder_path, on_time=True):
    """
    Loads spatial frame computations and deploys the viewer. If on_time is True,
    it computes the fields live (same aligned loader as the pipeline). If False, it uses the
    arrays precomputed by the pipeline in Results/<scene>/Physical_fields.
    """
    matrix_folder = Path(matrix_folder_path)
    output_folder = matrix_folder.parent / 'Physical_fields'
    if not output_folder.exists() and (matrix_folder.parent / 'physical_fields').exists():
        output_folder = matrix_folder.parent / 'physical_fields'          # legacy lowercase folder

    if on_time:
        success = computing_fields(mesh_folder_path, matrix_folder_path, output_folder)
        if not success:
            return

    print(f"\nGathering structural data streams from {output_folder}...")
    npz_files = sorted(
        [f for f in output_folder.iterdir() if f.is_file() and f.suffix == '.npz' and f.name.startswith('frame_')],
        key=natural_sort_key
    )

    if not npz_files:
        print(f"Error: Missing physical fields dependencies in target: {output_folder}")
        return

    frames_data = []
    for npz_file in npz_files:
        with np.load(npz_file) as loaded_data:
            frames_data.append({key: loaded_data[key] for key in loaded_data.files})

    global_df = None
    csv_path = output_folder / 'global_physical_metrics.csv'
    if csv_path.exists():
        global_df = pd.read_csv(csv_path)

    launch_physics_viewer(frames_data, global_df=global_df)

##################################################################################################################### Reeb Graphs #########################################################################################################################################

def edit_graph(mesh_folder_path, reeb_folder_path):
    print("\nStarting Interactive Split-Screen Graph Editor with Undo...")
    
    mesh_path = Path(mesh_folder_path)
    reeb_path = Path(reeb_folder_path)
    
    if not mesh_path.exists() or not reeb_path.exists():
        print("Error: Invalid mesh or reeb graph directory paths.")
        return

    obj_files = sorted([f for f in mesh_path.iterdir() if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')],key=natural_sort_key)
    reeb_files = sorted([f for f in reeb_path.iterdir() if f.is_file() and f.suffix == '.pkl'],key=natural_sort_key)
    scalar_files = sorted([f for f in reeb_path.iterdir() if f.is_file() and f.name.startswith('Scalar') and f.suffix == '.npy'],key=natural_sort_key)
    
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
            
            # Level attributes consistent with compute_approx_reeb_graph (used by graph_sim / analyses).
            f_val = float(mesh_pv.point_data['Dynamic_Scalar'][idx])
            bins_existing = [d.get('bin', 0) for _, d in G.nodes(data=True)]
            num_bins = (max(bins_existing) + 1) if bins_existing else 1
            f_all = mesh_pv.point_data['Dynamic_Scalar']
            f_min, f_max = float(f_all.min()), float(f_all.max())
            new_bin = int(np.clip(np.floor((f_val - f_min) / (f_max - f_min) * num_bins), 0, num_bins - 1)) if f_max > f_min else 0
            G.add_node(new_id, pos=vertex_coord, bin=new_bin, f_value=f_val, n_vertices=1,
                       vertices=np.array([idx], dtype=np.int64),
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

def launch_reeb_viewer(mesh_files, reeb_files, scalar_files, graph='reeb'):
    graph_label = "Reeb" if graph == 'reeb' else "Morse-Smale critical-point"
    print(f"\nStarting interactive {graph_label} graph orchestrator...")
    
    pl = pv.Plotter(shape=(1, 2))
    pl.title = f"Cell Topology Evolution ({graph_label} Graphs)"
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
            graph_obj = pickle.load(f)
        
        reeb_pv = create_reeb_polydata(graph_obj)
        if graph == 'mscomplex' and reeb_pv.n_points > 0:
            # nodes at the vertices of the displayed mesh (on the surface, same rotation); the centre node
            # (vertex -1) at the mean of the displayed points
            nodes = list(graph_obj.nodes(data=True))
            vidx = np.array([int(d.get('vertex', -1)) for _, d in nodes])
            pts = np.array(reeb_pv.points, dtype=float)
            ok = (vidx >= 0) & (vidx < meshn.n_points)
            pts[ok] = np.asarray(meshn.points)[vidx[ok]]
            pts[~ok] = np.asarray(meshn.points).mean(axis=0)
            reeb_pv.points = pts
            reeb_pv.point_data['type_code'] = np.array([{'minimum': 0, 'saddle': 1, 'maximum': 2}.get(str(d.get('type', '')), 3) for _, d in nodes])
        
        pl.subplot(0, 0)
        pl.add_mesh(meshn, scalars='Dynamic_Scalar', cmap='viridis', name='z_mesh', 
                    show_scalar_bar=True, render=False)
        pl.add_text(f"Scalar Field - Frame {frame_idx + 1}/{state['total']}", 
                    name='t1', font_size=10, position='upper_left')

        pl.subplot(0, 1)
        pl.add_mesh(meshn, color='white', opacity=0.25, name='ghost_mesh', render=False)
        
        if reeb_pv.n_points > 0:
            spheres = reeb_pv.glyph(geom=pv.Sphere(radius=node_radius * (1.6 if graph == 'mscomplex' else 1.0)), scale=False, orient=False)
            fvals = reeb_pv.point_data['f_value'] if 'f_value' in reeb_pv.point_data else None
            if graph == 'mscomplex':
                # colour by critical point type (red max, blue min, green saddle, gold centre)
                import matplotlib.colors as mc
                type_cols = np.array([mc.to_rgb(c) for c in ('royalblue', 'limegreen', 'red', 'gold')])
                rep = spheres.n_points // max(reeb_pv.n_points, 1)
                cols = type_cols[reeb_pv.point_data['type_code']]
                spheres.point_data['rgb'] = (np.repeat(cols, rep, axis=0)[:spheres.n_points] * 255).astype(np.uint8)
                pl.add_mesh(spheres, scalars='rgb', rgb=True, name='reeb_nodes', render=False)
            elif fvals is not None and np.all(np.isfinite(fvals)) and np.ptp(fvals) > 0:
                # Same colour map and range as the scalar field on the left, so a node's colour tells its level.
                pl.add_mesh(spheres, scalars='f_value', cmap='viridis', clim=[float(scalar_array.min()), float(scalar_array.max())],
                            name='reeb_nodes', scalar_bar_args={'title': 'node level f'}, render=False)
            else:
                pl.add_mesh(spheres, color='red', name='reeb_nodes', render=False)
            
            if reeb_pv.n_lines > 0:
                tubes = reeb_pv.tube(radius=edge_radius) 
                pl.add_mesh(tubes, color='blue', name='reeb_edges', render=False)
            else:
                pl.remove_actor('reeb_edges')

        pl.add_text(f"{'Level-Set Reeb Graph' if graph == 'reeb' else 'Morse-Smale critical-point graph'} - Frame {frame_idx + 1}/{state['total']}", 
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

def visualize_graphs(mesh_folder_path, graph_folder_path, graph='reeb'):
    """
    Graph-over-mesh viewer (scalar field on the left, graph on the right).
      graph='reeb'      : Reeb graphs; Reeb_T####.pkl and Scalar_T####.npy live in the same folder
                          (Results/<scene>/Reeb_Graphs) - unchanged behaviour of visualize_reeb_graphs.
      graph='mscomplex' : critical-point graphs of the Morse-Smale complex (MSGraph_T####.pkl in
                          Results/<scene>/MSComplexAnalysis/MS_Graphs); the scalar fields are read from
                          the sibling folder MSComplexAnalysis/MS_Complex (Scalar_T####.npy) and the nodes
                          are placed at their mesh vertices ('vertex' attribute).
    """
    mesh_folder = Path(mesh_folder_path)
    graph_folder = Path(graph_folder_path)
    graph = str(graph).lower()
    if graph not in ('reeb', 'mscomplex'):
        raise ValueError("graph must be 'reeb' or 'mscomplex'")

    obj_files = sorted([f for f in mesh_folder.iterdir() if f.is_file() and (f.suffix == '.obj' or f.suffix == '.mat')], key=natural_sort_key)
    graph_files = sorted([f for f in graph_folder.iterdir() if f.is_file() and f.suffix == '.pkl'], key=natural_sort_key)
    if graph == 'reeb':
        scalar_folder = graph_folder
    else:
        # MS_Graphs and MS_Complex are siblings under MSComplexAnalysis (accept the root folder too)
        root = graph_folder.parent if graph_folder.name == 'MS_Graphs' else graph_folder
        scalar_folder = root / 'MS_Complex'
        if graph_folder.name != 'MS_Graphs' and (root / 'MS_Graphs').is_dir():
            graph_folder = root / 'MS_Graphs'
            graph_files = sorted([f for f in graph_folder.iterdir() if f.is_file() and f.suffix == '.pkl'], key=natural_sort_key)
    scalar_files = sorted([f for f in scalar_folder.iterdir() if f.is_file() and f.name.startswith('Scalar') and f.suffix == '.npy'],
                          key=natural_sort_key) if scalar_folder.is_dir() else []

    if not obj_files or not graph_files or not scalar_files:
        print(f"Error: Missing obj, pkl, or npy files for the {graph} graph visualization "
              f"(graphs: {graph_folder}, scalars: {scalar_folder}).")
        return

    # pair graphs and scalars by frame number (the MS complex may start at frame 1 for map-dependent fields)
    def frame_of(f):
        m = re.search(r'T(\d+)', f.stem)
        return int(m.group(1)) if m else None
    scal_by_frame = {frame_of(f): f for f in scalar_files}
    mesh_sel, graph_sel, scalar_sel = [], [], []
    for gf in graph_files:
        k = frame_of(gf)
        if k is None:
            continue
        if k in scal_by_frame and k < len(obj_files):
            mesh_sel.append(obj_files[k]); graph_sel.append(gf); scalar_sel.append(scal_by_frame[k])
    if not graph_sel:   # no frame numbers in the names: fall back to positional pairing
        min_len = min(len(obj_files), len(graph_files), len(scalar_files))
        mesh_sel, graph_sel, scalar_sel = obj_files[:min_len], graph_files[:min_len], scalar_files[:min_len]

    launch_reeb_viewer([str(f) for f in mesh_sel], [str(f) for f in graph_sel], [str(f) for f in scalar_sel],
                       graph=graph)


def visualize_reeb_graphs(mesh_folder_path, reeb_folder_path):
    """Backward compatible alias of visualize_graphs(..., graph='reeb')."""
    return visualize_graphs(mesh_folder_path, reeb_folder_path, graph='reeb')


def visualize_obj_sequence(folder_path: str):
    """Dynamically loads and visualizes a sequence of .obj meshes."""
    path = Path(folder_path)
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return

    obj_files = sorted([f for f in path.iterdir() if f.is_file() and f.suffix == '.obj'], key=natural_sort_key)
    
    if not obj_files:
        print(f"No .obj files found in {path}")
        return

    pl = pv.Plotter(title="Sequential Mesh Visualizer")
    state = {'frame': 0, 'total': len(obj_files)}
    
    def update_frame(idx):
        pl.clear_actors()
        mesh = pv.read(obj_files[idx])
        pl.add_mesh(mesh, color='white', show_edges=True, edge_color="green")
        pl.add_text(f"Frame {idx + 1}/{state['total']}\nFile: {obj_files[idx].name}", 
                    position='upper_left', name='label', font_size=10)
        pl.add_text("Time Control:\n  Right Arrow : Next\n  Left Arrow : Prev", 
                    position='lower_left', font_size=8, color='black') 
        
    def step_next():
        if state['frame'] < state['total'] - 1:
            state['frame'] += 1
            update_frame(state['frame'])
            
    def step_prev():
        if state['frame'] > 0:
            state['frame'] -= 1
            update_frame(state['frame'])

    pl.add_key_event('Right', step_next) 
    pl.add_key_event('Left', step_prev) 
    
    update_frame(0)
    pl.show(full_screen=True)

##################################################################################################################### Morse–Smale Complex ###############################################################################################################################
MS_PALETTE = np.array([[0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16], [0.58, 0.40, 0.74],
                       [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.50, 0.50, 0.50], [0.74, 0.74, 0.13], [0.09, 0.75, 0.81],
                       [0.68, 0.78, 0.91], [1.00, 0.73, 0.47], [0.60, 0.87, 0.54], [1.00, 0.60, 0.59], [0.77, 0.69, 0.84],
                       [0.77, 0.61, 0.58], [0.97, 0.71, 0.82], [0.78, 0.78, 0.78], [0.86, 0.86, 0.55], [0.62, 0.85, 0.90],
                       [0.55, 0.83, 0.78], [1.00, 1.00, 0.70], [0.75, 0.73, 0.85], [0.98, 0.50, 0.45], [0.50, 0.69, 0.83],
                       [0.99, 0.71, 0.38], [0.70, 0.87, 0.41], [0.99, 0.80, 0.90], [0.85, 0.85, 0.85], [0.74, 0.50, 0.74],
                       [0.80, 0.92, 0.77], [1.00, 0.93, 0.44]])
MS_CP_COLORS = {"maximum": "red", "minimum": "royalblue", "saddle": "limegreen", "regular": "lightgrey", "lost": "black", "center": "gold"}


def _ms_rot(pd_):
    """Same cosmetic rotation as the Reeb viewer, applied to every polydata so they stay consistent."""
    pd_.rotate_x(90, inplace=True)
    pd_.rotate_z(90, inplace=True)
    return pd_


def _ms_palette(n):
    """n visually distinct colours: the first 32 from MS_PALETTE, then golden-angle hues with cycling
    saturation/lightness (no repetition, no warnings however many regions/tracks there are)."""
    import colorsys
    n = int(max(n, 1))
    if n <= len(MS_PALETTE):
        return MS_PALETTE[:n]
    extra = []
    for k in range(n - len(MS_PALETTE)):
        h = (0.61803398875 * k) % 1.0
        s_ = (0.55, 0.85, 0.7)[k % 3]
        l_ = (0.5, 0.65, 0.38)[(k // 3) % 3]
        extra.append(colorsys.hls_to_rgb(h, l_, s_))
    return np.vstack([MS_PALETTE, np.array(extra)])


def _ms_region_colors(labels, color_key, palette, boundary_edges=None):
    """RGB per vertex from the region ids. color_key: region id -> colour index (track id when a lineage
    is available, so a protrusion keeps its colour over time). If two *adjacent* regions received the
    same colour (indices wrapping around the palette) the one with the larger id is moved to the
    nearest free colour, so neighbouring regions are always distinguishable."""
    labels = np.asarray(labels)
    n = len(palette)
    idx_of = {int(r): int(color_key.get(int(r), int(r))) % n for r in np.unique(labels)}
    if boundary_edges is not None and len(boundary_edges):
        a, b = labels[boundary_edges[:, 0]], labels[boundary_edges[:, 1]]
        pairs = {(int(min(x, y)), int(max(x, y))) for x, y in zip(a, b) if x != y}
        neigh = {}
        for x, y in pairs:
            neigh.setdefault(x, set()).add(y); neigh.setdefault(y, set()).add(x)
        for r in sorted(idx_of):
            used = {idx_of[q] for q in neigh.get(r, ())}
            if idx_of[r] in used:
                for shift in range(1, n):
                    cand = (idx_of[r] + shift) % n
                    if cand not in used:
                        idx_of[r] = cand
                        break
    idx = np.array([idx_of[int(l)] for l in labels])
    return np.asarray(palette)[idx]


def launch_ms_viewer(mesh_files, ms_files, tracking_dir=None, lineage_csv=None):
    """
    Interactive viewer of the Morse–Smale segmentation.
      Right / Left   : next / previous frame
      m              : cycle the modality
                       'segmentation'  : regions of the maxima (protrusions), boundaries (yellow) and
                                         critical points (red max, blue min, green saddle)
                       'correspondence': left = previous frame segmentation; right = current mesh coloured
                                         with the previous regions transported by the functional map
                                         (hard p2p or soft spectral transport), current boundaries on top
                       'fates'         : current mesh with the images of the previous critical points
                                         coloured by their fate (kept type / became saddle / minimum / regular)
      s              : hard <-> soft transport (correspondence mode)
      b / c / e      : toggle boundaries / critical points / mesh edges
    Region colours are stable along a protrusion track when region_lineage.csv is available.
    """
    from PynamicMesh.core.SMComplex import MSComplex, create_ms_polydata, CP_TYPES
    print("\nStarting interactive Morse–Smale viewer...")
    n = len(mesh_files)
    tracking_dir = Path(tracking_dir) if tracking_dir else None
    # colour key: (frame, max vertex) -> track id  (stable colours along the lineage)
    color_key = {}
    if lineage_csv and Path(lineage_csv).exists():
        ln = pd.read_csv(lineage_csv)
        for _, r in ln.iterrows():
            if r.curr_max_vertex >= 0:
                color_key[(int(r.Time_Step), int(r.curr_max_vertex))] = int(r.curr_track_id)
            if r.prev_max_vertex >= 0:
                color_key[(int(r.Time_Step) - 1, int(r.prev_max_vertex))] = int(r.track_id)
    cache = {}
    ms_all = [MSComplex.from_pickle(f) for f in ms_files]
    n_colors = max([len(ms.maxima) for ms in ms_all] + [max(color_key.values(), default=0) + 1, 32])
    palette = _ms_palette(n_colors)

    def load(i):
        if i not in cache:
            tm = mesh_mat2object(mesh_files[i])
            ms = ms_all[i]
            mesh, bnd, _ = create_ms_polydata(ms, tm.vertices, tm.faces)
            _ms_rot(mesh); _ms_rot(bnd)
            # critical points at the vertices of the displayed (rotated) mesh: always on the surface
            idx = ms.critical.vertex.to_numpy(dtype=np.int64)
            cps = pv.PolyData(mesh.points[idx]) if len(idx) else pv.PolyData()
            if len(idx):
                cps.point_data["type_code"] = np.array([CP_TYPES.index(t) for t in ms.critical["type"]])
                cps.point_data["vertex"] = idx
            cache[i] = (ms, mesh, bnd, cps)
        return cache[i]

    def key_for(frame, ms):
        """region id -> colour index: the track id when known, else the rank of the maximum in the frame."""
        return {int(m): color_key.get((frame, int(m)), k) for k, m in enumerate(ms.maxima)}

    pl = pv.Plotter(shape=(1, 2))
    pl.title = "Morse–Smale complex: protrusion segmentation and correspondence"
    modes = ["segmentation", "correspondence", "fates"]
    state = {"frame": 0, "mode": 0, "soft": False, "bnd": True, "cps": True, "edges": False}
    pl.add_axes()

    def draw_mesh(sub, mesh, colors, name, opacity=1.0):
        pl.subplot(0, sub)
        m = mesh.copy()
        m.point_data["rgb"] = (np.asarray(colors) * 255).astype(np.uint8)
        pl.add_mesh(m, scalars="rgb", rgb=True, name=name, smooth_shading=True, opacity=opacity,
                    show_edges=state["edges"], edge_color="black", line_width=0.3, render=False)

    def draw_boundary(sub, bnd, name, color="yellow", radius=None):
        pl.subplot(0, sub)
        pl.remove_actor(name)
        if state["bnd"] and bnd.n_lines > 0:
            pl.add_mesh(bnd.tube(radius=radius), color=color, name=name, render=False)

    def draw_cps(sub, cps, name, radius, colors=None):
        pl.subplot(0, sub)
        pl.remove_actor(name)
        if state["cps"] and cps.n_points > 0:
            sph = cps.glyph(geom=pv.Sphere(radius=radius), scale=False, orient=False)
            if colors is None:
                cols = np.array([matplotlib_color(MS_CP_COLORS[CP_TYPES[int(t)]]) for t in cps.point_data["type_code"]])
            else:
                cols = np.asarray(colors)
            # glyph repeats the points: expand colours per glyph vertex
            rep = sph.n_points // max(cps.n_points, 1)
            sph.point_data["rgb"] = (np.repeat(cols, rep, axis=0)[:sph.n_points] * 255).astype(np.uint8)
            pl.add_mesh(sph, scalars="rgb", rgb=True, name=name, render=False)

    def matplotlib_color(c):
        import matplotlib.colors as mc
        return np.array(mc.to_rgb(c))

    def clear_all():
        for sub in (0, 1):
            pl.subplot(0, sub)
            for nm in ("mesh", "bnd", "cps", "bnd2", "cps2", "txt", "legend"):
                pl.remove_actor(nm)
            try:
                pl.remove_scalar_bar()
            except Exception:
                pass

    def update(i):
        clear_all()
        ms, mesh, bnd, cps = load(i)
        diag = np.linalg.norm(np.array(mesh.bounds[1::2]) - np.array(mesh.bounds[::2]))
        r_node, r_edge = diag * 0.010, diag * 0.002
        mode = modes[state["mode"]]
        key_c = key_for(i, ms)
        col_c = _ms_region_colors(ms.label_max, key_c, palette, ms.boundary_edges)
        c = ms.counts()
        if mode == "segmentation" or i == 0:
            draw_mesh(0, mesh, col_c, "mesh")
            draw_boundary(0, bnd, "bnd", radius=r_edge)
            draw_cps(0, cps, "cps", r_node)
            pl.subplot(0, 0)
            pl.add_text(f"Morse–Smale segmentation - Frame {i + 1}/{n}\nmax {c['n_max']}  min {c['n_min']}  saddle {c['n_saddle']}  "
                        f"regions {len(ms.maxima)}  (persistence {ms.persistence_threshold / ms.meta['field_range']:.2f})",
                        name="txt", font_size=9, position="upper_left")
            # right: scalar field with the critical points
            pl.subplot(0, 1)
            m2 = mesh.copy()
            pl.add_mesh(m2, scalars="scalar", cmap="viridis", name="mesh", smooth_shading=True, show_edges=state["edges"],
                        scalar_bar_args={"title": ms.meta.get("scalar_method", "f")}, render=False)
            draw_boundary(1, bnd, "bnd2", color="white", radius=r_edge)
            draw_cps(1, cps, "cps2", r_node)
            pl.add_text("Scalar field of the complex", name="txt", font_size=9, position="upper_left")
            return
        ms_p, mesh_p, bnd_p, cps_p = load(i - 1)
        key_p = key_for(i - 1, ms_p)
        col_p = _ms_region_colors(ms_p.label_max, key_p, palette, ms_p.boundary_edges)
        draw_mesh(0, mesh_p, col_p, "mesh")
        draw_boundary(0, bnd_p, "bnd", radius=r_edge)
        draw_cps(0, cps_p, "cps", r_node)
        pl.subplot(0, 0)
        pl.add_text(f"Frame {i}/{n} (previous)", name="txt", font_size=9, position="upper_left")
        npz = tracking_dir / f"mapped_T{i - 1:04d}_T{i:04d}.npz" if tracking_dir else None
        data = np.load(npz, allow_pickle=True) if (npz is not None and npz.exists()) else None
        if data is None:
            draw_mesh(1, mesh, col_c, "mesh")
            draw_boundary(1, bnd, "bnd2", radius=r_edge)
            draw_cps(1, cps, "cps2", r_node)
            pl.subplot(0, 1)
            pl.add_text(f"Frame {i + 1}/{n}: no tracking data (run with ms_track_regions=True)", name="txt", font_size=9, position="upper_left")
            return
        if mode == "correspondence":
            use_soft = state["soft"] and data["soft_label"].size > 0
            lab = data["soft_label"] if use_soft else data["hard_label"]
            col_m = _ms_region_colors(lab, key_p, palette)     # previous colours transported to the current mesh
            if use_soft and data["soft_confidence"].size:
                conf = np.clip(data["soft_confidence"], 0, 1)[:, None]
                col_m = conf * col_m + (1 - conf) * 0.85          # fade uncertain vertices
            draw_mesh(1, mesh, col_m, "mesh")
            draw_boundary(1, bnd, "bnd2", radius=r_edge)        # current segmentation boundaries on top
            draw_cps(1, cps, "cps2", r_node)
            iou = data["IoU"]
            pl.subplot(0, 1)
            pl.add_text(f"Frame {i + 1}/{n}: previous regions transported by the FM "
                        f"({'soft spectral transport' if use_soft else 'hard p2p transport'})\n"
                        f"matched {len(data['match_prev'])}/{len(data['prev_ids'])} regions, mean IoU "
                        f"{np.mean([iou[np.where(data['prev_ids'] == a)[0][0], np.where(data['curr_ids'] == b)[0][0]] for a, b in zip(data['match_prev'], data['match_curr'])]) if len(data['match_prev']) else 0:.2f}"
                        f"\nyellow = current boundaries; colours = previous regions (s: hard/soft)",
                        name="txt", font_size=9, position="upper_left")
        else:  # fates
            draw_mesh(1, mesh, 0.55 * col_c + 0.45, "mesh")
            draw_boundary(1, bnd, "bnd2", radius=r_edge)
            img = data["image_vertex"]; tn = data["cp_type_next"]; tp = data["cp_type_prev"]; gr = data["cp_growth"]
            ok = img >= 0
            pts = pv.PolyData(mesh.points[img[ok]]) if ok.any() else pv.PolyData()
            cols = np.array([matplotlib_color(MS_CP_COLORS.get(str(t), "black")) for t in tn[ok]]) if ok.any() else None
            if pts.n_points:
                pts.point_data["type_code"] = np.zeros(pts.n_points, dtype=int)
                draw_cps(1, pts, "cps2", r_node * 1.3, colors=cols)
            counts = {}
            for a, b in zip(tp, tn):
                counts[(str(a), str(b))] = counts.get((str(a), str(b)), 0) + 1
            summary = "  ".join(f"{a[:3]}->{b[:3]}:{k}" for (a, b), k in sorted(counts.items()))
            n_grow = int(np.sum((tp == "maximum") & (gr == "growing"))); n_ret = int(np.sum((tp == "maximum") & (gr == "retracting")))
            pl.subplot(0, 1)
            pl.add_text(f"Frame {i + 1}/{n}: fate of the previous critical points (colour = type reached)\n"
                        f"{summary}\nprotrusions growing {n_grow} / retracting {n_ret}",
                        name="txt", font_size=9, position="upper_left")

    def step(d):
        state["frame"] = int(np.clip(state["frame"] + d, 0, n - 1)); update(state["frame"]); pl.render()

    def toggle(k):
        def _f():
            if k == "mode":
                state["mode"] = (state["mode"] + 1) % len(modes)
            else:
                state[k] = not state[k]
            update(state["frame"]); pl.render()
        return _f

    for k_ in ("b", "c", "e", "m", "s"):          # drop pyvista's defaults on these keys
        try:
            pl.clear_events_for_key(k_)
        except Exception:
            pass
    pl.add_key_event("Right", lambda: step(1))
    pl.add_key_event("Left", lambda: step(-1))
    pl.add_key_event("m", toggle("mode"))
    pl.add_key_event("s", toggle("soft"))
    pl.add_key_event("b", toggle("bnd"))
    pl.add_key_event("c", toggle("cps"))
    pl.add_key_event("e", toggle("edges"))
    update(0)
    pl.subplot(0, 0); pl.reset_camera(); pl.subplot(0, 1); pl.reset_camera(); pl.link_views()
    pl.subplot(0, 0)
    pl.add_text("Controls: Right/Left frame | m modality (segmentation / correspondence / fates) | s hard-soft transport | b boundaries | c critical points | e edges",
                position="lower_left", font_size=6, color="black")
    pl.show(full_screen=True)


def visualize_ms_complex(mesh_folder_path, ms_root_path):
    """Launch the Morse–Smale viewer. ``ms_root_path`` is Results/<scene>/MSComplexAnalysis."""
    mesh_folder = Path(mesh_folder_path)
    ms_root = Path(ms_root_path)
    ms_dir = ms_root / "MS_Complex" if (ms_root / "MS_Complex").is_dir() else ms_root
    obj_files = sorted([f for f in mesh_folder.iterdir() if f.is_file() and f.suffix in (".obj", ".mat")], key=natural_sort_key)
    ms_files = sorted([f for f in ms_dir.iterdir() if f.is_file() and f.name.startswith("MS_T") and f.suffix == ".pkl"], key=natural_sort_key)
    if not obj_files or not ms_files:
        print("Error: missing meshes or MS_T####.pkl files for the Morse–Smale visualization.")
        return
    frames = [int(re.search(r"T(\d+)", f.stem).group(1)) for f in ms_files]
    mesh_sel = [obj_files[k] for k in frames if k < len(obj_files)]
    tracking = ms_root / "Region_tracking"
    launch_ms_viewer([str(f) for f in mesh_sel], [str(f) for f in ms_files[:len(mesh_sel)]],
                     tracking_dir=tracking if tracking.is_dir() else None,
                     lineage_csv=tracking / "region_lineage.csv" if (tracking / "region_lineage.csv").exists() else None)