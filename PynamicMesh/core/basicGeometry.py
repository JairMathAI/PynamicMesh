import trimesh
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyFM.mesh import TriMesh 
from pathlib import Path
from matplotlib.ticker import MaxNLocator

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[INFO] CuPy detected. Utilizing GPU for Basic Geometry.")
except ImportError:
    CUPY_AVAILABLE = False
    print("[INFO] CuPy not found. Defaulting to CPU (NumPy).")

def compute_mesh_geometry(pyfm_mesh, metrics='all'):

    """
    Computes geometric properties of a single pyFM TriMesh.
    
    :param pyfm_mesh: A pyFM.mesh.TriMesh object.
    :param metrics: 'all', a single string, or a list of strings of metrics to compute.
    :return: Dictionary containing the clean metrics.
    """
    AVAILABLE_METRICS = {
        'n_vertices', 'n_faces', 'area', 'volume', 'sphericity',
        'gaussian_curvature', 'convexity', 'center_mass'
    }

    if isinstance(metrics, str):
        metrics_to_compute = AVAILABLE_METRICS if metrics.lower() == 'all' else {metrics}
    else:
        metrics_to_compute = set(metrics)

    results = {}

    if 'n_vertices' in metrics_to_compute:
        results['n_vertices'] = pyfm_mesh.n_vertices
    if 'n_faces' in metrics_to_compute:
        results['n_faces'] = pyfm_mesh.n_faces
    if 'area' in metrics_to_compute:
        results['area'] = pyfm_mesh.area

    needs_trimesh = {'volume', 'sphericity', 'gaussian_curvature', 'convexity', 'center_mass'}
                     
    if metrics_to_compute.intersection(needs_trimesh):
        t_mesh = trimesh.Trimesh(vertices=pyfm_mesh.vertices, faces=pyfm_mesh.faces)

        if 'volume' in metrics_to_compute:
            results['volume'] = abs(t_mesh.volume) if t_mesh.is_watertight else np.nan
            
        if 'sphericity' in metrics_to_compute:
            vol = abs(t_mesh.volume) if t_mesh.is_watertight else np.nan
            area = pyfm_mesh.area
            if not np.isnan(vol) and area > 0:
                results['sphericity'] = (np.pi**(1/3) * (6 * vol)**(2/3)) / area
            else:
                results['sphericity'] = np.nan

        if 'gaussian_curvature' in metrics_to_compute:
            g_curv = trimesh.curvature.discrete_gaussian_curvature_measure(t_mesh, t_mesh.vertices, 1.0)
            results['mean_gaussian_curvature'] = np.mean(g_curv)


        if 'convexity' in metrics_to_compute:
            if t_mesh.is_watertight:
                convex_hull = t_mesh.convex_hull
                results['convexity'] = abs(t_mesh.volume) / abs(convex_hull.volume)
            else:
                results['convexity'] = np.nan

        if 'center_mass' in metrics_to_compute:
            cm = t_mesh.center_mass
            results['cm_x'], results['cm_y'], results['cm_z'] = cm[0], cm[1], cm[2]

    return results


def generate_plots_from_csv(csv_path, dpi=200):
    """
    Reads the CSV, computes consecutive pairwise distances for the Center of Mass,
    and plots all properties inside a single integrated dashboard figure with 
    independent, optimized y-axis scaling.
    """
    csv_path = Path(csv_path)
    plot_path = csv_path.parent

    os.makedirs(plot_path, exist_ok=True)

    df = pd.read_csv(csv_path)
    
    x = np.arange(len(df))
    
    cm_cols = {'cm_x', 'cm_y', 'cm_z'}
    
    # FIX 1: Filter out non-numeric columns (like mesh names or paths) 
    # to avoid plotting broken text-based categorical scales.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scalar_metrics = [col for col in numeric_cols if col not in cm_cols]

    has_cm_displacement = False
    distances = None
    
    if cm_cols.issubset(df.columns):
        coords = df[['cm_x', 'cm_y', 'cm_z']].dropna().values
        if len(coords) >= 2:
            has_cm_displacement = True
            if CUPY_AVAILABLE:
                gpu_coords = cp.asarray(coords)
                diffs = cp.diff(gpu_coords, axis=0)
                distances = cp.asnumpy(cp.sqrt(cp.sum(diffs**2, axis=1)))
            else:
                diffs = np.diff(coords, axis=0)
                distances = np.sqrt(np.sum(diffs**2, axis=1))

    num_plots = len(scalar_metrics)
    if has_cm_displacement:
        num_plots += 1

    if num_plots == 0:
        print("[INFO] No columns found to plot.")
        return

    cols = 2 if num_plots > 1 else 1
    rows = int(np.ceil(num_plots / cols))
    
    # Note: sharey=False is default, ensuring independent scaling per subplot
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes = axes.flatten()

    plot_idx = 0

    for metric in scalar_metrics:
        ax = axes[plot_idx]
        ax.plot(x, df[metric], marker='o', linestyle='-', color='b')
        ax.set_title(f'Evolution of {metric.replace("_", " ").title()}')
        ax.set_xlabel('Mesh Sequence Index')
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # FIX 2: Add 10% vertical padding to give data breathing room at edges
        ax.margins(y=0.1)
        
        # FIX 3: Safety check for flat lines (unchanging values like static vertex counts)
        y_min, y_max = df[metric].min(), df[metric].max()
        if np.isclose(y_min, y_max) or y_min == y_max:
            if y_min == 0:
                ax.set_ylim(-1, 1)
            else:
                ax.set_ylim(y_min * 0.9, y_min * 1.1)  # Pad by ±10% around the value

        plot_idx += 1

    if has_cm_displacement:
        ax = axes[plot_idx]
        x_dist = np.arange(len(distances))
        ax.plot(x_dist, distances, marker='s', linestyle='-', color='b')
        ax.set_title('Center of Mass Consecutive Displacement')
        ax.set_xlabel('Sequence Transition Interval (i to i+1)')
        ax.set_ylabel('Euclidean Distance')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Apply padding and flatline safety to displacement plot too
        ax.margins(y=0.1)
        if len(distances) > 0:
            d_min, d_max = np.min(distances), np.max(distances)
            if np.isclose(d_min, d_max):
                if d_min == 0:
                    ax.set_ylim(-1, 1)
                else:
                    ax.set_ylim(d_min * 0.9, d_min * 1.1)
                    
        plot_idx += 1

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    fig.savefig(plot_path / 'mesh_evolution_summary.png', dpi=dpi)
    plt.close(fig)