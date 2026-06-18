import os
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PynamicMesh.core.custom_fm import CustomFunctionalMapping, pick_single_mesh
from pyFM.mesh import TriMesh
from PIL import Image
import pickle
from PynamicMesh.core.physic_model import (
    plot_diagonal, 
    generate_tranformation_heatmap, 
    Diagonal_metrics,
    compute_heatmap_similarity, 
    plot_similarity_metrics,
    computing_fields  
)
from PynamicMesh.core.reeb_graph import (
    get_scalar_field, 
    compute_approx_reeb_graph, 
    graph_time_analysis, 
    plot_dynamic_graph_analysis
)
from PynamicMesh.utils.tools import (
    landmark_load, 
    landmark_parser, 
    resolve_scalar_args
)  


def compute_FM(meshn_1, meshn, i, target_folder, descriptor, current_landmarks, k_eigenfunc, prev_FM_zo, inertia_history, decay_history, cdf_history, similarity_history, heatmap_paths, diagonal_analysis, isometric_analysis):
    os.makedirs(target_folder / 'Transform_Matrices', exist_ok=True)
    os.makedirs(target_folder / 'Diagonal_analysis', exist_ok=True)
    
    model = CustomFunctionalMapping(meshn_1, meshn)
    model.preprocess(K=(10, 10), descr_type=descriptor, landmarks=current_landmarks, k_process=k_eigenfunc)
    model.fit() 

    FM_zo = model.zoomout_refine(nit=30, step=2) 
    p2p_zo = model.get_p2p(FM_zo)  
    
    if diagonal_analysis:
        heatmap_filename = target_folder / 'Diagonal_analysis' / f'FM_{i-1}{i}.png'
        generate_tranformation_heatmap(FM_zo, i, heatmap_filename)
        heatmap_paths.append(heatmap_filename)
        
        inrt, decy, cdf = Diagonal_metrics(FM_zo)
        inertia_history.append(inrt)
        decay_history.append(decy)
        cdf_history.append(cdf)
        plot_diagonal(inertia_history, decay_history, cdf_history, target_folder / 'Diagonal_analysis' / 'Diagonal_metrics.png')

        diag_csv_path = target_folder / 'Diagonal_analysis' / 'Diagonal_Metrics.csv'
        with open(diag_csv_path, mode='w', encoding='utf-8') as diag_csv:
            diag_csv.write("Timestep,Map,Inertia_Metric,Decay_Metric\n")
            for idx_d, (inr_val, dec_val) in enumerate(zip(inertia_history, decay_history)):
                diag_csv.write(f"{idx_d+1},FM_{idx_d}{idx_d+1},{inr_val:.6f},{dec_val:.6f}\n")
    
    if prev_FM_zo is not None and isometric_analysis:
        jsd, pearson, spearman, manhattan, euclidean = compute_heatmap_similarity(prev_FM_zo, FM_zo)
        similarity_history.append([jsd, pearson, spearman, manhattan, euclidean])
        
        plot_similarity_metrics(similarity_history, target_folder / 'Diagonal_analysis' / 'Cross_Heatmap_Similarity.png')
        
        csv_path = target_folder / 'Diagonal_analysis' / 'Cross_Heatmap_Similarity.csv'
        with open(csv_path, mode='w', encoding='utf-8') as csv_file:
            csv_file.write("Comparison,JSD,Pearson_Corr,Spearman_Corr,Manhattan_Dist,Euclidean_Dist\n")
            for idx, values in enumerate(similarity_history):
                lbl = f"FM_{idx}{idx+1}_vs_FM_{idx+1}{idx+2}"
                csv_file.write(f"{lbl},{values[0]:.6f},{values[1]:.6f},{values[2]:.6f},{values[3]:.6f},{values[4]:.6f}\n")

    np.save(target_folder / 'Transform_Matrices' / f'FMC_{i-1}{i}.npy', FM_zo)
    np.save(target_folder / 'Transform_Matrices' / f'FMV_{i}{i-1}.npy', p2p_zo)
    
    return FM_zo, p2p_zo, str(target_folder / 'Transform_Matrices')


def compute_RG(meshn, i, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections, needs_spectral_processing, prev_vertices=None, p2p_zo=None):
    reeb_folder = target_folder / 'Reeb_Graphs'
    os.makedirs(reeb_folder, exist_ok=True)
    
    tn_kwargs = resolve_scalar_args(reeb_scalar, scalar_kwargs, i, meshn.vertices.shape[0], loaded_selections)
    scalar_tn = get_scalar_field(meshn.vertices, meshn.faces, method=reeb_scalar, prev_vertices=prev_vertices, p2p=p2p_zo, trimesh_obj=meshn if needs_spectral_processing else None, **tn_kwargs)

    reeb_tn = compute_approx_reeb_graph(meshn.vertices, meshn.faces, scalar_tn, num_bins=bins)
    with open(reeb_folder / f'Reeb_T{i}.pkl', 'wb') as f: 
        pickle.dump(reeb_tn, f)
    np.save(reeb_folder / f'Scalar_T{i}.npy', scalar_tn)

    return str(reeb_folder)


def headmap_gif(heatmap_paths, target_folder):
    frames = [Image.open(img_path) for img_path in heatmap_paths]
    gif_path = target_folder / 'Diagonal_analysis' / 'FM_Heatmap_Animation.gif'
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=700, loop=0)


def process_sequence(folder, path, matrix_tranformation, diagonal_analysis, isometric_analysis, k_eigenfunc, descriptor, landmarks, compute_reeb, reeb_scalar, bins, needs_spectral_processing, compute_physic_fields, scalar_kwargs):
    itemsfiles = list(folder.iterdir())
    obj_files = sorted([f for f in itemsfiles if f.is_file() and f.suffix == '.obj'])
    
    if len(obj_files) < 2:
        return None, None

    out_root = path.parent / 'Results'
    scene_name = obj_files[0].parent.name
    target_folder = out_root / scene_name
    os.makedirs(target_folder, exist_ok=True)
    
    loaded_landmarks_fm = None
    if matrix_tranformation:
        loaded_landmarks_fm = landmark_load(landmarks, target_folder, 'FM')
        
    loaded_selections_rg = None
    if compute_reeb:
        vertex_ref_index = scalar_kwargs.get("vertex_ref_index", None)
        source_idx = scalar_kwargs.get("source_idx", None)
        if reeb_scalar == 'geodesic' and vertex_ref_index == 'precomputed':
            loaded_selections_rg = landmark_load(vertex_ref_index, target_folder, reeb_scalar)
        elif reeb_scalar == 'heat_diffusion' and source_idx == 'precomputed':
            loaded_selections_rg = landmark_load(source_idx, target_folder, reeb_scalar)
        elif reeb_scalar == 'harmonic' and source_idx == 'precomputed':
            loaded_selections_rg = landmark_load(source_idx, target_folder, reeb_scalar)

    meshn_1 = TriMesh(str(obj_files[0]))
    if needs_spectral_processing:
        meshn_1.process(k=k_eigenfunc)

    if compute_reeb:
        compute_RG(meshn_1, 0, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections_rg, needs_spectral_processing)

    inertia_history, decay_history, cdf_history = [], [], []
    similarity_history = []  
    heatmap_paths = []
    RG_out_path = ' ' 
    FM_out_path = ' '
    prev_FM_zo = None  
    
    for i in tqdm(range(1, len(obj_files)), desc=f'processing {scene_name}', leave=False):
        meshn = TriMesh(str(obj_files[i]))
        p2p_zo = None
        
        if needs_spectral_processing: 
            meshn.process(k=k_eigenfunc)

        if matrix_tranformation:
            current_landmarks = landmark_parser(landmarks, loaded_landmarks_fm, i, 'FM')
            FM_zo, p2p_zo, FM_out_path = compute_FM(
                meshn_1, meshn, i, target_folder, descriptor, current_landmarks, 
                k_eigenfunc, prev_FM_zo, inertia_history, decay_history, 
                cdf_history, similarity_history, heatmap_paths, diagonal_analysis, isometric_analysis
            )
            prev_FM_zo = FM_zo.copy()
        
        if compute_reeb:
            prev_verts = meshn_1.vertices if matrix_tranformation else None
            RG_out_path = compute_RG(meshn, i, reeb_scalar, bins, scalar_kwargs, target_folder, loaded_selections_rg, needs_spectral_processing, prev_verts, p2p_zo)
            
        meshn_1 = meshn

    if matrix_tranformation and heatmap_paths:
        headmap_gif(heatmap_paths, target_folder)


    if compute_physic_fields:
        if not matrix_tranformation:
            print(f"\n[Warning] Cannot calculate fields for {scene_name} without track mappings.")
        else:
            matrix_folder_path = target_folder / 'Transform_Matrices'
            physical_fields_path = target_folder / 'Physical_fields'
            computing_fields(folder, matrix_folder_path, physical_fields_path)
    
    return FM_out_path, RG_out_path


def run_pipeline(path_str,
    matrix_tranformation=True,
    diagonal_analysis=True,
    isometric_analysis=True,
    k_eigenfunc=100,
    descriptor='WKS+HKS',
    landmarks=None,
    compute_reeb=True,
    time_graph_analysis=True,
    reeb_scalar="geodesic",
    bins=30,
    compute_physic_fields=True,  
    **scalar_kwargs
    ):

    path = Path(path_str)
    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{path_str}' is not a valid directory.")
        return

    spectral_methods = ["heat_diffusion", "harmonic", "multi_pca"]
    is_spectral = reeb_scalar in spectral_methods or reeb_scalar.startswith("lb_eigen_")
    needs_spectral_processing = matrix_tranformation or is_spectral

    subdirectories = [f for f in path.iterdir() if f.is_dir()]        
    if not subdirectories:
        print(f'No folders found on {path}')
        return
        
    for folder in tqdm(subdirectories, desc='processing folder'):
        FM_out_path, RG_out_path = process_sequence(
            folder=folder,
            path=path,
            matrix_tranformation=matrix_tranformation,
            diagonal_analysis=diagonal_analysis,
            isometric_analysis=isometric_analysis,
            k_eigenfunc=k_eigenfunc,
            descriptor=descriptor,
            landmarks=landmarks,
            compute_reeb=compute_reeb,
            reeb_scalar=reeb_scalar,
            bins=bins,
            needs_spectral_processing=needs_spectral_processing,
            compute_physic_fields=compute_physic_fields,  
            scalar_kwargs=scalar_kwargs            
        )

        if time_graph_analysis and RG_out_path and RG_out_path.strip():
            csv_path = graph_time_analysis(RG_out_path)
            plot_dynamic_graph_analysis(csv_path)