import numpy as np
import pickle

def landmark_load(landmarks, target_folder, mood):
    if mood == 'FM':
        file = 'landmarks.npy'
    if mood == 'geodesic':
        file = 'vert_ref_geo.npy'
    if mood == 'heat_diffusion':
        file = 'sources.npy'
    if mood == 'harmonic':
        file = 'source_sink.npy'
    
    loaded_landmarks = None
    if isinstance(landmarks, str) and landmarks.lower() == 'precomputed':
        temp = []
        landmarks_file = target_folder / file
        if landmarks_file.exists():
            try:
                loaded_landmarks = np.load(landmarks_file, allow_pickle=True)
                if mood == 'FM':
                    loaded_landmarks = np.array([x for x in loaded_landmarks], dtype=int)
                if mood == 'geodesic' or mood == 'heat_diffusion':
                    for element in loaded_landmarks:
                        if isinstance(element, list):
                            temp.append(element)
                        else:
                            temp.append([0])
                    loaded_landmarks = temp
            except:
                if mood != 'FM':
                    raise ValueError(f'file not found: {landmarks_file}')
                loaded_landmarks = None
        else:
            loaded_landmarks = None 
    return loaded_landmarks


def landmark_parser(landmarks, loaded_landmarks, i, mood=None):
    current_landmarks = None
    if isinstance(landmarks, str):
        if landmarks.lower() == 'precomputed' and loaded_landmarks is not None:
            if mood == 'FM':
                current_landmarks = loaded_landmarks[i-1].astype(int)
            if mood == 'geodesic':
                current_landmarks = loaded_landmarks[i]
            if mood == 'heat_diffusion':
                current_landmarks = loaded_landmarks[i][0]
            if mood == 'harmonic':
                current_landmarks = loaded_landmarks[i]
    else:
        if landmarks is not None:
            if isinstance(landmarks, (list, tuple, np.ndarray)) and len(landmarks) > 0:
                first_elem = landmarks[0]
                is_multiple_mappings = False
                if isinstance(first_elem, (list, tuple, np.ndarray)):
                    if len(first_elem) > 0 and isinstance(first_elem[0], (list, tuple, np.ndarray)): 
                        is_multiple_mappings = True
                    elif any(len(x) != 2 for x in landmarks): 
                        is_multiple_mappings = True
                
                if is_multiple_mappings:
                    pair_idx = i - 1
                    if pair_idx < len(landmarks): 
                        current_landmarks = np.array(landmarks[pair_idx])
                    else: 
                        current_landmarks = None
                else: 
                    current_landmarks = np.array(landmarks)
            else: 
                current_landmarks = np.array(landmarks)
        else: 
            current_landmarks = None

    return current_landmarks


def resolve_scalar_args(reeb_scalar, scalar_kwargs, index, num_vertices, loaded_selections=None):
    kwargs_out = scalar_kwargs.copy()
    
    if reeb_scalar == 'geodesic':
        vertex_ref_index = scalar_kwargs.get("vertex_ref_index", None)
        if vertex_ref_index == 'precomputed':
            selection = landmark_parser(vertex_ref_index, loaded_selections, index, reeb_scalar)
            kwargs_out['vertex_ref_index'] = selection
        else:
            if vertex_ref_index is None:
                kwargs_out['vertex_ref_index'] = [0]
            else:
                is_list_of_lists = isinstance(vertex_ref_index, (list, tuple)) and any(isinstance(x, (list, tuple)) or x is None for x in vertex_ref_index)
                if is_list_of_lists:
                    if index < len(vertex_ref_index) and vertex_ref_index[index] is not None:
                        kwargs_out['vertex_ref_index'] = list(vertex_ref_index[index])
                    else:
                        kwargs_out['vertex_ref_index'] = [0]
                else:
                    if isinstance(vertex_ref_index, (int, float)):
                        kwargs_out['vertex_ref_index'] = [int(vertex_ref_index)]
                    else:
                        kwargs_out['vertex_ref_index'] = list(vertex_ref_index)

    elif reeb_scalar == 'heat_diffusion':
        source_idx = scalar_kwargs.get("source_idx", None)
        if source_idx == 'precomputed':
            selection = landmark_parser(source_idx, loaded_selections, index, reeb_scalar)
            kwargs_out['source_idx'] = selection
        else:
            if source_idx is None:
                kwargs_out['source_idx'] = 0
            elif isinstance(source_idx, (list, tuple)):
                if index < len(source_idx) and source_idx[index] is not None:
                    kwargs_out['source_idx'] = int(source_idx[index])
                else:
                    kwargs_out['source_idx'] = 0
            else:
                kwargs_out['source_idx'] = int(source_idx)

    elif reeb_scalar == 'harmonic':
        source_idx = scalar_kwargs.get("source_idx", None)
        sink_idx = scalar_kwargs.get("sink_idx", None)
        
        if source_idx == 'precomputed':
            selection = landmark_parser(source_idx, loaded_selections, index, reeb_scalar)
            if selection is not None and len(selection) >= 2:
                kwargs_out['source_idx'] = selection[0]
                kwargs_out['sink_idx'] = selection[1]
            else:
                kwargs_out['source_idx'] = 0
                kwargs_out['sink_idx'] = num_vertices - 1
        else:
            default_source = 0
            default_sink = num_vertices - 1
            
            is_list_of_pairs = isinstance(source_idx, (list, tuple)) and any(isinstance(x, (list, tuple)) or x is None for x in source_idx)
            
            if is_list_of_pairs:
                if index < len(source_idx) and source_idx[index] is not None:
                    val = source_idx[index]
                    if len(val) >= 2:
                        kwargs_out['source_idx'] = int(val[0])
                        kwargs_out['sink_idx'] = int(val[1])
                    else:
                        kwargs_out['source_idx'] = default_source
                        kwargs_out['sink_idx'] = default_sink
                else:
                    kwargs_out['source_idx'] = default_source
                    kwargs_out['sink_idx'] = default_sink
            else:
                if isinstance(source_idx, (list, tuple)):
                    kwargs_out['source_idx'] = int(source_idx[0]) if len(source_idx) > 0 else default_source
                    if len(source_idx) >= 2:
                        kwargs_out['sink_idx'] = int(source_idx[1])
                    else:
                        kwargs_out['sink_idx'] = int(sink_idx) if sink_idx is not None else default_sink
                else:
                    kwargs_out['source_idx'] = int(source_idx) if source_idx is not None else default_source
                    kwargs_out['sink_idx'] = int(sink_idx) if sink_idx is not None else default_sink

    return kwargs_out