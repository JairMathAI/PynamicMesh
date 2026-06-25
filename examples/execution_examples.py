from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.core.reeb_graph import graph_time_analysis, plot_dynamic_graph_analysis  
from pathlib import Path
from PynamicMesh.core.graph_sim import graph_similarity, plot_graph_similarity
from PynamicMesh.utils.visualizers import (
    visualize_reeb_graphs, 
    edit_graph,
    visualize_physics,
    visual_selection_edition, 
    precompute_landmarks
)

####################################################################################################### Paths reference list ##########################################################################################################################################

####################################################################################################### Linux #########################################################################################################################################################
base_mesh_path = '/PynamicMesh/Mesh_models'
mesh_path = '/PynamicMesh/Mesh_models/scene1'
matrix_path = '/PynamicMesh/Results/scene1/Transform_Matrices'
reeb_path = '/PynamicMesh/Results/scene1/Reeb_Graphs'
csv_file_path = '/PynamicMesh/Results/scene1/Graph_analysis/time_analysis.csv'

###################################################################################################### Windos #########################################################################################################################################################
base_mesh_path = r'\PynamicMesh\Mesh_models'
mesh_path = r'\PynamicMesh\Mesh_models\scene1'
matrix_path = r'\PynamicMesh\Results\scene1\Transform_Matrices'
reeb_path = r'\PynamicMesh\Results\scene1\Reeb_Graphs'
csv_file_path = r'\Results\scene1\Graph_analysis\time_analysis.csv'

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
FM_descriptors = 'WKS+HKS'
FM_landmarks = 'precomputed'
compute_physic_fields = True

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
    compute_physic_fields=compute_physic_fields, 
    compute_reeb=compute_RG,
    time_graph_analysis=compute_graph_time_analysis,
    reeb_scalar=reeb_scalar_field,
    bins=bins,
    vertex_ref_index=vertex_ref_index,
    graph_sim = compute_Graphsimilarity,
    graph_metrics = graph_metrics
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
visualize_reeb_graphs(mesh_path, reeb_path)

####################################################################################################### Functional Map Visualizer Launcher  #################################################################################################################################
print('Multi-Physics Mapping visualizations...') 
visualize_physics(mesh_path, matrix_path, on_time=False)

###################################################################################################### Similarity Metrics Among Graphs and ploting  ##################################################################################################################################
csv_sim_path = graph_similarity(reeb_folder_path=reeb_path,metrics_list=graph_metrics)
plot_graph_similarity(csv_sim_path)

###########################################################################################################################################################################################################################################################################

####################################################################################################### Paths reference list ##########################################################################################################################################

from PynamicMesh.utils.batch import run_batch
from PynamicMesh.utils.tools import extract_yaml

config_path = '/PynamicMesh/examples/config_batch.yaml' # vlinux
config_path = r'\PynamicMesh\examples\config_batch.yaml' # windows

config = extract_yaml(config_path)
data_cfg = config.get("Data", {})
path_str = data_cfg.get("path_str")
run_batch(config,path_str)

###########################################################################################################################################################################################################################################################################