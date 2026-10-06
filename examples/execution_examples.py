from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.core.reeb_graph import graph_time_analysis, plot_dynamic_graph_analysis
from pathlib import Path
from PynamicMesh.core.graph_sim import graph_similarity, plot_graph_similarity
from PynamicMesh.utils.dynamics_visualizers import (
    visualize_parametrization, visualize_trajectories, visualize_dynamic_trajectories, visualize_interpolation,
    visualize_hodge, visualize_ftle, visualize_shape_modes, visualize_graph_animation, visualize_motion_analysis, visualize_reeb_dynamics
)
from PynamicMesh.utils.visualizers import (
    visualize_reeb_graphs,
    edit_graph,
    visualize_physics,
    visual_selection_edition,
    precompute_landmarks,
    visualize_ms_complex,
    visualize_graphs,
    visualize_obj_sequence
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

####################################################################################################### GPU usage options ####################################################################################################################################
compute_devices = 'all'         # 'all' (every visible GPU) | 'cpu' | [0, 1] / '0,1'
n_workers = None                # None = CPU cores - 1 (at most 8); 1 = sequential
parallel_scenes = 'auto'        # 'auto' | True | False
compare_scenes = 'auto'         # comparison of the scenes processed together (Results/_comparison): 'auto' (>= 2 scenes) | False

####################################################################################################### Frame times definition ####################################################################################################################################
time_params = {
    'frame_interval': None,     # e.g. 30.0 -> one frame every 30 s (uniform time-lapse)
    'frame_times': None,        # e.g. [0, 30, 65, 90] or 'frame_times.csv' (relative to each scene folder) - irregular allowed
    'time_unit': 's',
    'auto_detect': False,        # use a times file found in the scene folder
}

####################################################################################################### Gif generation options ####################################################################################################################################
gif = True
gif_params = {
    'fps': 6,                   # frames per second (mean rate when real_time)
    'real_time': True,          # frame durations proportional to the real intervals (when the times are known)
    'max_frames': 240,          # longer sequences are subsampled evenly
    'panel_width': 900, 'height': 760,   # pixels per panel of the off-screen render
    'max_width': 1800,          # the written GIF is downscaled to this width
    'vectors': True,            # also the vector-field view ('_vectors' GIFs) of the modalities that have one
}

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

####################################################################################################### Functional Map Settings  #################################################################################################################################
compute_FM = True
compute_FMdiagonal_analysis = True
compute_isometric_analysis = True
FM_k_eigenfunctions = (10, 10)
FM_k_eigenvalues = 100
compute_physic_fields = True

common_fm_params = {
    'n_descr': 100,
    'subsample_step': 4,
    'descr_params': {'nu': 1.5, 'k_smooth': 30, 'xyz_weight': 0.25},
    'fit_params': {'w_descr': 1e-1, 'w_lap': 1e-3, 'w_dcomm': 1.0},
    'refine': 'auto',               # 'auto' (ZoomOut 30x2 if <= 10000 vertices and size ratio <= 1.5, else ICP) | 'icp' |
                                    # (nit, step) | 'zoomout_auto': ZoomOut up to zoomout_k_final for EVERY pair
                                    # (subsampled for large / mismatched pairs): better maps, slower
    'zoomout_k_final': 100,         # 'zoomout_auto': final size of the refined map (100 = Ovsjanikov et al.)
    'zoomout_step': 2,              # 'zoomout_auto': basis growth per ZoomOut iteration
    'zoomout_subsample': 'auto',    # 'zoomout_auto': points of the subsampled refinement ('auto': 3000 for large pairs)
    'dt': 1.0,                   # legacy uniform time step, used only when the frame times are unknown (time_params)
    'verbose': False,        # prints the descriptor plan and the landmarks kept for each pair
}

FM_EXAMPLES = {
    # ---------------------------------------------------------------------------------------------------------------
    # a) No landmarks. Only intrinsic point signatures: the map is determined up to the intrinsic symmetries
    #    of the shape (left/right legs can be swapped between frames).
    'a': dict(
        descriptor='WKS+HKS+MKS',           # equal energy shares; or e.g. '0.5*WKS + 0.3*HKS + 0.2*MKS'
        landmarks=None,
        fm_params={**common_fm_params, 'symmetry_mode': 'none'},
    ),
    # ---------------------------------------------------------------------------------------------------------------
    # b) Precomputed landmarks only. Uses the selections stored by precompute_landmarks() /
    #    visual_selection_edition() (mood='FM'); landmark-localized WKS descriptors break the symmetry.
    'b': dict(
        descriptor='WKS+HKS+MKS',
        landmarks='precomputed',
        fm_params={**common_fm_params, 'symmetry_mode': 'landmarks',
                   'landmark_params': {'weight': 1.0, 'descriptor': 'WKS'}},
    ),
    # ---------------------------------------------------------------------------------------------------------------
    # c) Automatic landmarks only. Farthest-point samples on frame t (extremities first), matched on frame t+1
    #    (descriptor match within a spatial radius), outliers rejected by displacement / duplicates / geodesic
    #    consistency. Landmarks used are saved in Results\<scene>\Landmarks\.
    'c': dict(
        descriptor='WKS+HKS+MKS',
        landmarks='auto',
        fm_params={**common_fm_params, 'symmetry_mode': 'landmarks',
                   'landmark_params': {'n_landmarks': 12, 'match': 'hybrid', 'max_rel_dist': 0.15,
                                       'max_distortion': 0.25, 'min_landmarks': 4, 'search_rel_radius': 0.10,
                                       'weight': 1.0}},
    ),
    # ---------------------------------------------------------------------------------------------------------------
    # d) Symmetry-aware descriptors only, no landmarks. 'extrinsic' adds an aligned-coordinates (XYZ) descriptor
    #    block (valid between aligned consecutive frames), 'orientation' adds the orientation-preserving operator
    #    term (w_orient) that penalizes the mirrored map. The XYZ block can also be requested explicitly in the
    #    descriptor string, e.g. '0.5*WKS + 0.3*MKS + 0.2*XYZ' (then 'extrinsic' is redundant).
    'd': dict(
        descriptor='WKS+HKS+MKS',
        landmarks=None,
        fm_params={**common_fm_params, 'symmetry_mode': 'orientation+extrinsic',
                   'fit_params': {**common_fm_params['fit_params'], 'w_orient': 1.0}},
    ),
}
fm_settings = FM_EXAMPLES[EXAMPLE]



####################################################################################################### Reeb Graph Settings ###############################################################################################################################################
compute_RG = True
compute_graph_time_analysis = True
reeb_scalar_field = 'geodesic'      # 'mass_center_geodesic' | 'heat_diffusion' | 'harmonic' | 'matern_kernel' | 'lb_eigen_1' | 'multi_pca' | 'normal_displacement' | ...
bins = 30
# Topological control of the Reeb graphs (Extended Reeb Graph, Biasotti et al. 2008), as a COMPLEMENT: the graphs of the
# pipeline stay the uniform-slab graphs (unchanged); the controlled graphs (slabs hiding a handle refined locally) are
# written to Reeb_Graphs/Topology_Controlled/ and their genus-loop diagnostic is reported next to the one of the main
# graphs (Graph_analysis/topology_per_frame.csv). False = no controlled graphs.
reeb_topology_control = True
vertex_ref_index = [4896]           # index, list of indices, 'mass_center', 'precomputed' or 'auto' | 'auto_center' | 'auto_extremity'

reeb_kwargs = {
    'vertex_ref_index': vertex_ref_index,
    'geodesic_solver': 'heat',      # 'heat' (pyFM, Dijkstra fallback) | 'dijkstra'
    'reference_transport': 'auto',  # automatic reference points from frame to frame: 'auto' (point-to-point map, ignored if it
                                    # moves the point > 1/4 of the size) | 'map' (large motion, good maps) | 'position' (aligned
                                    # frames with small motion: immune to symmetric flips of the maps)
    'equalize_histogram': False,    # True -> rank-normalized field, equally populated bins
}


# Motion analysis (Results/<scene>/MotionAnalysis): rigid vs non-rigid motion, physical fields from the trajectories
# (complement of Physical_fields), growth atlas, growth anisotropy + sliding-window FTLE, polarity, regional kinematics,
# protrusion kinetics, skeleton branches, signals + event timeline + synchronisation + periods + DMD. Needs the
# trajectories; uses every other result available (Morse-Smale tracking, shape space, graph animation, frame times).
# Time-varying Reeb graph (Edelsbrunner et al. 2008) on the tracked mesh (Results/<scene>/ReebDynamics): the Reeb field
# with the same material source in every frame, interpolated between the frames; branches (maximum/minimum-saddle
# persistence pairs) followed along their critical-point paths; birth / death / interchange / dominance events.
# Needs the trajectories; runs before the motion analysis (its events join the motion event timeline).
compute_ReebDynamics = True
reeb_dynamics_params = {
    'persistence': 0.08,            # relative persistence of a branch (fraction of the field range); kept down to half of it
    'substeps': 4,                  # time samples inside every observed interval (events between the frames)
    'match_radius': 0.12,           # largest move of a branch between samples (fraction of sqrt(area))
    'k_spectral': 60,               # eigenpairs for spectral fields (heat diffusion, ...) on the tracked mesh
    'plots': True,
}


####################################################################################################### Motion Analysis Settings ###############################################################################################################################################

compute_MotionAnalysis = True
motion_params = {
    'rigid': True, 'fields': True, 'growth_atlas': True, 'anisotropy': True,
    'ftle_window': 3,               # frames of the sliding-window FTLE
    'polarity': True, 'regions': True, 'protrusions': True, 'branches': True,
    'signals': True, 'timeline': True, 'synchronization': True, 'frequency': True,
    'max_lag': None,                # largest lag of the cross-correlations (None = a quarter of the sequence)
    'dmd': True, 'dmd_energy': 0.999, 'dmd_delays': 'auto', 'dmd_forecast': 3,
    'low_confidence': 0.5,          # frames whose correspondence confidence is below this x the median are flagged
    'flip_angle': 120.0,            # orientation jumps above this (deg) between frames = probable symmetric flip of the maps
    'graphs': None,                 # skeleton branches of these animated graph kinds (None = every animation computed)
    'plots': True,
}

####################################################################################################### Basic Geometry Settings ############################################################################################################################################
compute_BasicGeo = True
plot_basicGeo = True
metrics = 'all'

####################################################################################################### Graph Similarity Metrics ###########################################################################################################################################
compute_Graphsimilarity = True
graph_metrics = 'all'
# Graphs analysed by the time analysis and the similarity: 'reeb', 'reeb_controlled' (topology-controlled Reeb graphs,
# Graph_analysis/Topology_Controlled), 'mscomplex' (MSComplexAnalysis/Graph_analysis); only the computed ones are used
analysis_graphs = ('reeb', 'mscomplex')

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
    # size-adaptive matching: the IoU a perfectly tracked region can reach falls with its size (map error and
    # protrusion motion eps blur its transport), so the threshold of every pair is iou_threshold x expected IoU of a
    # region of that radius (iou_threshold keeps its meaning for large regions; small blebs get a lower bar)
    'adaptive': True,               # False -> the fixed iou_threshold for every region (previous behaviour)
    'min_iou': 0.05,                # floor of the adaptive IoU threshold
    'label_smoothing': 1,           # majority-filter iterations removing the speckle of the hard p2p transport
    'tip_rescue': True,             # an unmatched region whose tip lands in an unmatched overlapping region is matched
    'map_error': None,              # None = estimated per transition (map scatter + tip displacement); or fixed, / sqrt(area)
    'max_map_error': 0.15,          # cap of the estimated map error (/ sqrt(area))
}

####################################################################################################### Protrusion Detection (MS + convex hull) ##############################################################################################################
# The maxima of the MS scalar field miss the protrusions that are not local maxima of that field (flank blebs of an
# elongated cell for 'dist_centroid', small or nested protrusions). They are recovered from the minima of the depth
# below the convex hull of the cell (Huang, Wu & Yan 2024, Comput. Biol. Med. 173:108350), matched with the MS
# maxima and carved into the segmentation, so the region tracking (functional maps), the MS graphs and the reports
# use the fused protrusions. Results: MSComplexAnalysis\Protrusions\ (protrusions_detail / summary, hull_candidates)
# + plots\protrusion_detection_evolution.png; per frame HullDepth_T####.npy and Core_T####.npy (protrusion cores).
# All lengths are relative to the equivalent radius of the cell sqrt(area / 4 pi). Passed to run_pipeline below
# (ms_protrusion_params); for standalone use of SMComplex, configure_protrusion_detection(**params) sets the same values.
# Viewer: visualize_ms_complex -> 'm' until 'protrusions' (cores vs body, tips by source, hull depth + convex hull; 'h' hull).
ms_protrusion_params = {
    'enabled': True,                # False -> plain Morse-Smale segmentation (previous behaviour)
    'beta': 0.012,                  # Huang's beta: minimum trough height (depth persistence) of a protrusion
    'noise_factor': 1.0,            # beta is raised to noise_factor x the measured surface roughness (noisy meshes)
    'smooth_iters': 5,              # Taubin smoothing of the geometry used for the hull (0 = raw vertices)
    'alpha': 0.08,                  # Huang's alpha: tips closer than this geodesic distance are one protrusion
    'min_aspect': 0.12,             # height / cap radius: rejects broad convex patches of the cell body
    'max_cap_area': 0.30,           # a protrusion covers at most this fraction of the surface
    'max_tip_depth': 0.25,          # deeper minima of the depth are pits, not protrusions
    'core_fraction': 0.25,          # core of the MS-only protrusions: upper fraction of the relief of the field
    'temporal': True,               # hysteresis through the p2p map: weak candidates confirmed by the previous frame ...
    'temporal_beta_factor': 0.5,    # ... pass with beta and min_aspect scaled by this factor
    'temporal_radius': 2.0,         # ... when a previous protrusion lands within temporal_radius x alpha
    'prune_ms': False,              # drop MS maxima deep inside the hull without hull support (off: keep all MS maxima)
    'prune_depth': 0.35,
}

####################################################################################################### Explicit Parametrization Settings ##########################################################################################################################
# Closed-form parametrization of every frame (Results/<scene>/Parametrization):
#   genus 0  -> discrete harmonic map to the sphere (Gu et al. 2004) + spherical-harmonic series
#               x(θ,φ) = Σ c_lm Y_lm(θ,φ); coefficients per frame, rotationally aligned through the p2p maps
#               (coeffs_consistent.npy is a low-dimensional consistent encoding of the whole sequence);
#               SymPy closed form exported as .tex/.txt for a low degree.
#   any genus -> manifold-harmonics (Laplace–Beltrami eigenbasis) reconstruction;  disk -> ARAP planar map.
compute_Parametrization = True
param_params = {
    'sphere_method': 'conformal',   # 'conformal': stereographic harmonic map + alternating Schwarz sweeps (discrete conformal, K~1.02-1.07)
                                    #              followed by a density-equalising area correction; 'harmonic': the original gradient descent
    'area_weight': 0.7,             # angle/area trade-off of the sphere map: 0 conformal, 1 area-preserving (uniform sampling -> SPHARM
                                    # converges 3-5x faster on elongated cells), 0.5-0.8 balanced. quality.csv reports K and log-area std
    'temporal_weight': 0.0,         # >0: pull each map towards the previous frame's (via p2p) for smoother c_lm(t); costs distortion
    'L_max': 25,                    # spherical-harmonic degree: a number, or 'auto' = the smallest degree reaching
                                    # auto_target_error on the first frame (bisection with real fits), within the memory;
                                    # a number larger than the memory allows is reduced (message + auto_parameters.json)
    'reg': 1e-6,                    # bi-Laplacian Tikhonov weight of the SPHARM least squares
    'mh_modes': 80,                 # manifold-harmonics modes: a number, or 'auto' = the fewest modes reaching
                                    # auto_target_error on the first frame (40 -> 600), within the memory
    'auto_target_error': 0.02,      # relative reconstruction error aimed at by 'auto' (2 % of the size of the shape)
    'memory_fraction': 0.6,         # fraction of the free GPU / CPU memory the parametrization may use
    'export_degrees': (),   # reconstructed meshes written for these degrees (viewer key 'm')
    'symbolic_degree': 3,           # degree of the SymPy closed form (higher = long expressions)
    'align_with_p2p': True,         # rotate the sphere maps consistently through the functional maps
    'high_degree': True,            # + high-degree SPHARM with a fast spherical harmonic transform (Parametrization/SPHARM_HD)
    'hd_lmax': 'auto',              #   its degree ('auto' = enough for the mesh resolution, 32-160); complex shapes need 64-128
    'sht_backend': 'auto',          #   'auto' (torch-harmonics on the GPU if installed, else NumPy) | 'torch' | 'numpy'
    'plots': True,
}

####################################################################################################### Trajectory Interpolation Settings ##########################################################################################################################
# Trajectories of the vertices of the reference frame (needs the functional maps: compute_FM now, or Transform_Matrices
# already on disk), dense interpolation (Hermite / splines / ARAP, Sorkine–Alexa 2007), leave-one-frame-out validation,
# kinematics (velocity, acceleration, Frenet curvature, strain, ARAP rigidity).
# How the reference mesh follows the frames ('trajectory_method'):
#   'arap_tracking' (default): the reference mesh is registered to every frame in turn - one-step prediction through the
#                   FM displacement field, robust as-rigid-as-possible fit (wrong FM matches are down-weighted), refinement
#                   on the true surface of the frame - so the triangulation stays intact on thin, strongly moving parts
#                   (legs, tails, protrusions). Diagnostics: Trajectories/tracking.csv and plots/5_coverage.png.
#   'chain'       : previous behaviour (p2p maps composed back to frame 0 + harmonic inpainting): the correspondence
#                   errors accumulate and the triangles collapse / flip where the motion is strongest.
compute_Trajectories = True
traj_params = {
    'trajectory_method': 'arap_tracking',   # 'arap_tracking' | 'chain'
    'tracking_params': {                    # only used by 'arap_tracking' (defaults shown)
        'reference_rigidity': 0.5,          # memory of the reference shape: 0 = purely incremental (distortion can accumulate),
                                            # 1 = rigid w.r.t. the reference; lower it (0.2 - 0.3) for strongly growing cells
        'data_weight': 0.5,                 # pull towards the FM prediction (higher = follow the maps more, less rigid)
        'icp_rounds': 3,                    # refinement on the true surface of each frame (0 = follow the maps only)
        'icp_weight': 1.0,                  # pull towards the surface in the refinement
        'robust_rounds': 3,                 # re-weighting rounds that reject wrong FM matches
        'robust_scale': 2.5,                # outlier threshold (x robust spread of the residuals); lower = stricter
        'arap_iters': 6,                    # as-rigid-as-possible iterations per solve
        'normal_threshold': 0.3,            # closest points whose normal disagrees more than this (cosine) are ignored
        'max_icp_distance': 4.0,            # ... and those farther than this x the mean edge length
        'inpainted_confidence': 0.25,       # confidence of predictions from vertices the FM did not reach
    },
    'rigidity_params': {
        'enabled': True,          # False = the previous behaviour
        'mode': 'arap',           # 'arap' for the camel (near-isometric motion) | 'asap' (default) for growing cells
        'data_weight': 0.005,     # lower = more rigid; higher = follows the input more closely
        'iterations': 15,
        'keyframes': True,        # correct the observed frames before interpolating
        'interpolations': True,   # correct every interpolated sequence
    },
    'schemes': ('linear','hermite_finite_difference', 'kochanek_bartels', 'bspline','arap_rigid','hermite_catmull_rom', 'natural_cubic', 'arap_polar'),   # see trajectories.INTERPOLATORS
    'substeps': 4,                  # interpolated frames between two observed frames
    'validate': True,               # leave-one-frame-out ranking of all schemes (interpolation/validation.csv)
    'kinematics': True,
    'export_obj': True,             # dense OBJ sequence per scheme (interpolation/<scheme>/frame_####.obj)
    'plots': True,
}

####################################################################################################### Graph Animation Settings ######################################################################################################################################
# A Reeb graph (skeleton) or Morse-Smale critical-point graph of ONE frame, anchored to the moving reference mesh and moved
# by the trajectory model (needs compute_Trajectories now or Trajectories/ on disk, and the graphs of that frame:
# compute_RG / compute_MSComplex now or on disk). Results in Results/<scene>/GraphAnimation/<graph>_T<frame>_<model>/:
# animation.npz, summary.csv (node speed, skeleton length, edge strain), nodes.csv, comparison.csv (animated vs the graph
# computed on every frame: high = the structure changed, not only moved), graphs/ (animated graph pickles), plots/.
compute_GraphAnimation = True
graph_animation_params = {
    'graphs': ('reeb', 'mscomplex'),   # graph kinds to animate: 'reeb', 'mscomplex', 'reeb_controlled' (missing kinds are skipped)
    'frame': 0,                        # frame whose graph is animated (any frame; 0 = the reference mesh itself)
    'model': ('linear','hermite_finite_difference', 'kochanek_bartels', 'bspline','arap_rigid','hermite_catmull_rom', 'natural_cubic', 'arap_polar'),                   # 'best' (lowest leave-one-out error on disk) | 'observed' | a scheme, e.g. 'arap_polar'
                                       # | a tuple of them, e.g. ('arap_polar', 'natural_cubic', 'best'): the graphs are animated
                                       #   with every model and compared (<graph>_T<frame>_comparison/: kinematics per model,
                                       #   node deviation of every model from the first one)
    'export_graphs': 'observed',       # animated graph pickles: 'observed' frames | 'all' steps | 'none'
    'plots': True,
}

####################################################################################################### Dynamic Analysis Settings ###################################################################################################################################
# Descriptive analyses (Results/<scene>/DynamicAnalysis, report.md) - no dynamical model is fitted:
# Helmholtz–Hodge decomposition of the surface velocity (sources/sinks vs vortical cortical flow), finite-time
# Lyapunov exponents, shape space (PCA modes as meshes, MDS, recurrence, Shape-DNA), POD reduced coordinates,
# change points of the deformation regime, persistent homology of the shape trajectory.
compute_DynamicAnalysis = True
dyn_analysis_params = {
    'hodge': True, 'ftle': True, 'shape_space': True, 'change_points': True,
    'homology': True,               # needs ripser or gudhi (skipped otherwise)
    'n_shape_modes': 4,             # principal deformation modes exported as meshes
    'pod_energy': 0.99,             # reduced coordinates: POD rank by cumulative energy ...
    'pod_max_rank': 8,              # ... capped at this rank
    'shape_source': 'auto',         # shape coordinates of shape space / change points / homology: 'auto' (by reconstruction
                                    # quality) | 'spharm' | 'spharm_hd' | 'manifold' (harmonics of the tracked mesh) | 'trajectories'
    'shape_max_error': 0.05,        # 'auto': SPHARM only when its median relative reconstruction error is at most this
    'mh_modes': 'auto',             # manifold-harmonic modes of the tracked mesh ('auto': 60-300 until the errors are met)
    'hodge_nonrigid': True,         # also the Hodge decomposition of the deformation-only motion (Hodge_nonrigid/)
    'cp_permutations': 500,         # permutation p-values of the change points (0 = off); + joint change points
    'shape_max_worst_error': 0.15,  # 'auto' shape source: also the worst-region (95th percentile) error must be below this
    'spherical_spectra': True,      # power per degree of the growth and curvature fields on the reference sphere
    'spectral_lmax': 32,            #   its degree
    'sht_backend': 'auto',          #   torch-harmonics (GPU) / NumPy
    'plots': True,
}



####################################################################################################### Pipeline Runing ###################################################################################################################################################
print('Executing pipeline ...')
run_pipeline(
    path_str=base_mesh_path,
    compute_basicGeo=compute_BasicGeo,
    metrics=metrics,
    plot_basicGeo=plot_basicGeo,
    matrix_tranformation=compute_FM,
    diagonal_analysis=compute_FMdiagonal_analysis,
    isometric_analysis=compute_isometric_analysis,
    k_eigenfunctions=FM_k_eigenfunctions,
    k_eigenvalues=FM_k_eigenvalues,
    descriptor=fm_settings['descriptor'],
    landmarks=fm_settings['landmarks'],
    fm_params=fm_settings['fm_params'],
    compute_physic_fields=compute_physic_fields,
    compute_reeb=compute_RG,
    time_graph_analysis=compute_graph_time_analysis,
    reeb_scalar=reeb_scalar_field,
    bins=bins,
    graph_sim=compute_Graphsimilarity,
    graph_metrics=graph_metrics,
    compute_mscomplex=compute_MSComplex,
    ms_scalar=ms_scalar_field,
    ms_persistence=ms_persistence,
    ms_min_region_area=ms_min_region_area,
    mscomplex_graph=mscomplex_graph,
    ms_graph_type=ms_graph_type,
    ms_track_regions=ms_track_regions,
    ms_graph_metrics=graph_metrics,
    ms_tracker_params=ms_tracker_params,
    ms_protrusion_params=ms_protrusion_params,
    compute_parametrization=compute_Parametrization,
    param_params=param_params,
    compute_trajectories=compute_Trajectories,
    traj_params=traj_params,
    compute_dynamic_analysis=compute_DynamicAnalysis,
    dyn_analysis_params=dyn_analysis_params,
    compute_graph_animation=compute_GraphAnimation,
    graph_animation_params=graph_animation_params,
    time_params=time_params,
    gif=gif,
    gif_params=gif_params,
    compute_devices=compute_devices,
    n_workers=n_workers,
    parallel_scenes=parallel_scenes,
    reeb_topology_control=reeb_topology_control,
    analysis_graphs=analysis_graphs,
    compute_reeb_dynamics=compute_ReebDynamics,
    reeb_dynamics_params=reeb_dynamics_params,
    compute_motion_analysis=compute_MotionAnalysis,
    motion_params=motion_params,
    compare_scenes_flag=compare_scenes,
    **reeb_kwargs,
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

####################################################################################################### Parametrization Visualizer Launcher  ####################################################################################################################################
print('Parametrization visualizations...')
visualize_parametrization(mesh_path, scene_results_path)                     # (a) (θ,φ) iso-lines, sphere map, SPHARM reconstruction by degree (m)

####################################################################################################### Trajectories Visualizer Launcher  ####################################################################################################################################
print('Trajectories visualizations...')
visualize_trajectories(scene_results_path, mesh_path=mesh_path, stride=1,scheme=None)    # (c) m: displacement field mesh k->k+1 (all vertices, FM assignment) / scheme= (optional same as interpolations)
                                                                             #     all vertex trajectories / smooth interpolated trajectories / ghost stack

####################################################################################################### Dynamic Trajectories Visualizer Launcher  ####################################################################################################################################
print('Dynamic Trajectories visualizations...')
visualize_dynamic_trajectories(scene_results_path, mesh_path=mesh_path, trail=12, scheme=None)   # (c) the motion itself: particles moving along the
                                                                             #     interpolated trajectories, comet trails, history, streamers (space runs)

####################################################################################################### Interpolations Visualizer Launcher  ####################################################################################################################################
print('Interpolations Trajectories visualizations...')
visualize_interpolation(scene_results_path, 'arap_polar', compare_with='hermite_catmull_rom', mesh_path=mesh_path)   #  arap_polar (c) dense surface trajectory


####################################################################################################### Reeb Graph Animation Visualizer Launcher  ####################################################################################################################################
print('Reeb Graph Animation visualizations...')
visualize_graph_animation(scene_results_path, mesh_path=mesh_path, graph='reeb', frame=0, scheme='arap_polar')        # one model
visualize_graph_animation(scene_results_path, mesh_path=mesh_path, graph='reeb', frame=0,
                          scheme=('arap_polar', 'natural_cubic'))   # comparative mode: one panel per scheme, side by side

####################################################################################################### MSComplex Graph Animation Visualizer Launcher  ####################################################################################################################################
print('MSComplex Graph Animation visualizations...')
visualize_graph_animation(scene_results_path, mesh_path=mesh_path, graph='mscomplex', frame=0)   # (c2) graph of one frame moved by the
#   trajectory model | m: skeleton on the surface / node trails / edge strain / animated vs actual graph | g: next animation
#   h surface | p/l opacity | t nodes | k/j edge width | u/n node size | Up/Down trail | v velocities | s screenshot

####################################################################################################### Hodge Visualizer Launcher  ####################################################################################################################################
print('Hodge visualizations...')
visualize_hodge(scene_results_path, mesh_path=mesh_path, stride=2)           # (d) potentials + thin arrows; v: vectors only, x/y/z: axis projections

####################################################################################################### FTLE Visualizer Launcher  ####################################################################################################################################
print('FTLE visualizations...')
visualize_ftle(scene_results_path, mesh_path=mesh_path)                      # (d) Lagrangian coherent structures

####################################################################################################### Shape Modes Visualizer Launcher  ####################################################################################################################################
print('Shape Modes visualizations...')
visualize_shape_modes(scene_results_path)                                    # (d) mean shape and ±2σ of the principal deformation modes

####################################################################################################### Motion Analysis Visualizer Launcher  ####################################################################################################################################
print('Motion Analysis visualizations...')                       
visualize_motion_analysis(results_path, mesh_path=mesh_path)

####################################################################################################### Field Comparison  Visualizer Launcher  ####################################################################################################################################
print('Field Comparison visualizations...')                       
visualize_field_comparison(results_path, mesh_path=mesh_path)

####################################################################################################### Reeb Dynamic Visualizer Launcher  ####################################################################################################################################
print('Reeb Dynamic visualizations...')  
visualize_reeb_dynamics(results_path, mesh_path=mesh_path)

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

