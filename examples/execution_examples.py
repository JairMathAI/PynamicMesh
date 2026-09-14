from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.core.reeb_graph import graph_time_analysis, plot_dynamic_graph_analysis  
from pathlib import Path
from PynamicMesh.core.graph_sim import graph_similarity, plot_graph_similarity
from PynamicMesh.utils.visualizers import (
    visualize_reeb_graphs,
    edit_graph,
    visualize_physics,
    visual_selection_edition, 
    precompute_landmarks,
    visualize_obj_sequence,
    visualize_ms_complex,
    visualize_graphs
)

####################################################################################################### Paths reference list ##########################################################################################################################################

####################################################################################################### Linux #########################################################################################################################################################
base_mesh_path = '/PynamicMesh/Mesh_models'
mesh_path = '/PynamicMesh/Mesh_models/scene1'
matrix_path = '/PynamicMesh/Results/scene1/Transform_Matrices'
reeb_path = '/PynamicMesh/Results/scene1/Reeb_Graphs'
csv_file_path = '/PynamicMesh/Results/scene1/Graph_analysis/time_analysis.csv'
mesh_objs_folder = '/Results/scene1/active_surface_lab/01_Passive_Relaxation'

###################################################################################################### Windows #########################################################################################################################################################
base_mesh_path = r'\PynamicMesh\Mesh_models'
mesh_path = r'\PynamicMesh\Mesh_models\scene1'
matrix_path = r'\PynamicMesh\Results\scene1\Transform_Matrices'
reeb_path = r'\PynamicMesh\Results\scene1\Reeb_Graphs'
csv_file_path = r'\Results\scene1\Graph_analysis\time_analysis.csv'
mesh_objs_folder = r'\Results\scene1\active_surface_lab\01_Passive_Relaxation'

###########################################################################################################################################################################################################################################################################

####################################################################################################### Precomputing Visual helpers ####################################################################################################################################

###################################################################################################### Landmarks Precompute Visual Launcher ############################################################################################################################
print('Visualizing or editing landmarks for FM...')
visual_selection_edition(mesh_path,'FM')

print('Precomputing landmarks for FM...')
precompute_landmarks(base_mesh_path,'FM') 

###################################################################################################### Vertex index reference Precompute Visual Launcer for 'geodesic' in Reebs ########################################################################################
print('Visualizing or editing Vertex index for RG...')
visual_selection_edition(mesh_path,'geodesic')

print('Precomputing Vertex index for RG...')
precompute_landmarks(base_mesh_path,'geodesic') 

###################################################################################################### Sources Vertex index  Precompute Visual Launcher for 'heat_diffusion' in Reebs ###################################################################################
print('Visualizing or editing Sources Vertex index for RG...')
visual_selection_edition(mesh_path,'heat_diffusion')

print('Precomputing Sources Vertex index for RG...')
precompute_landmarks(base_mesh_path,'heat_diffusion')

######################################################################################################  Source-sink Vertex index  Precompute Visual Launcher for 'harmonic' in Reebs ###################################################################################
print('Visualizing or editing Source-sink Vertex index index for RG...')
visual_selection_edition(mesh_path,'harmonic')

print('Precomputing Source-sink Vertex index for RG...')
precompute_landmarks(base_mesh_path,'harmonic')

#########################################################################################################################################################################################################################################################################


####################################################################################################### Runing Full pipeline over folder system ##########################################################################################################################

####################################################################################################### Functional Map Settings ##########################################################################################################################################
compute_FM = False
compute_FMdiagonal_analysis = True
compute_isometric_analysis = True
FM_k_eigenfunctions = (10,10)
FM_k_eigenvalues = 100
FM_descriptors = 'WKS+HKS+MKS'
FM_landmarks = 'precomputed'
compute_physic_fields = True
FM_symmetri = {'symmetry_mode': 'landmarks+orientation+extrinsic'}

####################################################################################################### Reeb Graph Settings ###############################################################################################################################################
compute_RG = False
compute_graph_time_analysis = True
reeb_scalar_field = 'geodesic' #'mass_center_geodesic' 
bins = 30
vertex_ref_index = [4896]

####################################################################################################### Basic Geometry Settings ############################################################################################################################################
compute_BasicGeo = True
plot_basicGeo = True
metrics = 'all' # ['n_vertices', 'n_faces', 'area', 'volume', 'sphericity', 'gaussian_curvature', 'convexity', 'center_mass']

####################################################################################################### Graph Similairty Meetrics ##########################################################################################################################################
compute_Graphsimilarity = True
graph_metrics = 'all' # ['degree_wasserstein','spectral_laplacian','interleaving_distance','labeled_interleaving_distance','function_distortion_distance','branch_decomposition_distance']

####################################################################################################### Morse-Smale Complex Settings ###################################################################################################################################
# Morse-Smale complex of a scalar field on every mesh: regions of the maxima = protrusion segmentation,
# critical points (max / min / saddle) after persistence simplification, csv + plots under
# Results/<scene>/MSComplexAnalysis. Any Reeb scalar field can be used; for protrusions the distance to the
# centre of mass ('dist_centroid' / 'mass_center_geodesic') is the natural choice.
compute_MSComplex = True
ms_scalar_field = 'dist_centroid'   # None -> same field as the Reeb graph (reused, not recomputed)
ms_persistence = 0.08               # extrema with persistence < 8% of the field range are merged (noise)
ms_min_region_area = 0.0            # optionally merge regions smaller than this fraction of the surface
mscomplex_graph = True              # star graph: critical points connected to the centre of mass ...
ms_graph_type = 'star+adjacency'    # 'star' | 'star+adjacency' (... plus edges between adjacent protrusion regions);
                                    # the Reeb-graph analyses (graph_time_analysis, graph_similarity) run on these graphs
ms_track_regions = True             # regions / critical points followed through the functional maps (needs compute_FM):
                                    # region lineage (continue / split / merge / birth / death), fate of every critical
                                    # point (max -> max / saddle / min / regular), inversions over several frames, growth
ms_tracker_params = {
    'iou_threshold': 0.25,          # minimum IoU (area weighted) for a matched region pair
    'dominance': 0.5,               # dominant-overlap fraction used for split / merge relations
    'fate_radius': 0.04,            # search radius of the nearest critical point (fraction of sqrt(area))
    'growth_tolerance': 0.002,      # |normal displacement| below this (fraction of sqrt(area)) is 'stable'
    'ghost_horizon': 6,             # frames a vanished critical point keeps being followed (to catch inversions)
}



####################################################################################################### Pipeline Runing ###################################################################################################################################################
print('Executing pipeline ...')
run_pipeline(
    path_str=base_mesh_path,
    compute_basicGeo = compute_BasicGeo,
    metrics = metrics,
    plot_basicGeo = plot_basicGeo,
    matrix_tranformation=compute_FM,
    diagonal_analysis=compute_FMdiagonal_analysis,
    isometric_analysis=compute_isometric_analysis,
    k_eigenfunctions=FM_k_eigenfunctions,
    k_eigenvalues=FM_k_eigenvalues,
    descriptor=FM_descriptors,
    landmarks=FM_landmarks,
    fm_params=FM_symmetri,
    compute_physic_fields=compute_physic_fields, 
    compute_reeb=compute_RG,
    time_graph_analysis=compute_graph_time_analysis,
    reeb_scalar=reeb_scalar_field,
    bins=bins,
    vertex_ref_index=vertex_ref_index,
    graph_sim = compute_Graphsimilarity,
    graph_metrics = graph_metrics,
    compute_mscomplex=compute_MSComplex,
    ms_scalar=ms_scalar_field,
    ms_persistence=ms_persistence,
    ms_min_region_area=ms_min_region_area,
    mscomplex_graph=mscomplex_graph,
    ms_graph_type=ms_graph_type,
    ms_track_regions=ms_track_regions,
    ms_graph_metrics=graph_metrics,
    ms_tracker_params=ms_tracker_params
    )

###########################################################################################################################################################################################################################################################################


#######################################################################################################  Executions Outside of the loop ###################################################################################################################################


####################################################################################################### Edit Created Reeb Graph  ##########################################################################################################################################
print('Editing reeb graph...') 
edit_graph(mesh_path, reeb_path)

####################################################################################################### Running Reeb Graph Time Analysis  ##################################################################################################################################
print('Graph path analysis...')
graph_time_analysis(reeb_path)

####################################################################################################### Generating Reeb Graph Time Analysis Plots  #########################################################################################################################
print('Plotting dynamic graph analysis...')
plot_dynamic_graph_analysis(csv_file_path)

####################################################################################################### Reeb Graph Visualizer Launcher  #####################################################################################################################################
print('Reeb visualizations...') 
visualize_graphs(mesh_path, reeb_path, graph='reeb')

####################################################################################################### Functional Map Visualizer Launcher  #################################################################################################################################
print('Multi-Physics Mapping visualizations...') 
visualize_physics(mesh_path, matrix_path, on_time=False)


####################################################################################################### Mesh sequence Visualizer Launcher  #################################################################################################################################
print('Mesh sequence Visualization...') 
visualize_obj_sequence(mesh_objs_folder)

####################################################################################################### Morse-Smale Visualizer Launcher  ####################################################################################################################################
print('Morse-Smale segmentation visualizations...')
visualize_ms_complex(mesh_path, ms_path)

####################################################################################################### Morse-Smale Graph Visualizer Launcher  ####################################################################################################################################
print('Morse-Smale graph visualizations...')
visualize_graphs(mesh_path, ms_graph_path, graph='mscomplex')

###################################################################################################### Similarity Metrics Among Graphs and ploting  ##########################################################################################################################
csv_sim_path = graph_similarity(reeb_folder_path=reeb_path,metrics_list=graph_metrics)
plot_graph_similarity(csv_sim_path)


################################################################################################################################################################################################################################################################################

####################################################################################################### .mat image/mesh visualizer #############################################################################################################################################
from PynamicMesh.utils.mat_files import MatViewer 
from pathlib import Path

folder_path = Path('path/to/.mat/folder')
viewer = MatViewer(folder_path)
viewer.show()

####################################################################################################### .mat image/mesh converter .obj/tiff #####################################################################################################################################
from PynamicMesh.utils.mat_files import mat_file_converter 
from pathlib import Path

folder_path = Path('path/to/.mat/folder')
mat_file_converter(folder_path)

#################################################################################################################################################################################################################################################################################

