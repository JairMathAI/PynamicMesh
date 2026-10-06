"""
Discrete Morse–Smale complex of a scalar field on a triangle mesh, protrusion segmentation,
critical-point statistics, star graphs on the critical points and tracking of the segmented
regions through the functional maps of the pipeline.

The scalar fields are the ones of ``PynamicMesh.core.reeb_graph.get_scalar_field`` (geodesic,
dist_centroid, mean_curvature, heat_diffusion, ...), so every Reeb-graph field can also be
analysed through its Morse–Smale complex.

Outline
-------
* ``classify_critical_points``   PL (Banchoff) classification of every vertex from the sign
                                 changes of f - f(v) around its link: minimum, regular,
                                 saddle (with multiplicity) or maximum.
* ``steepest_manifolds``         descending manifolds of the maxima (region of every
                                 protrusion) and ascending manifolds of the minima, by
                                 steepest ascent / descent with pointer jumping.
* ``persistence_pairs``          merge-tree persistence of the maxima (super-level filtration)
                                 and minima (sub-level filtration); low-persistence extrema
                                 are merged into their neighbours (topological simplification).
* ``compute_ms_complex``         everything above -> ``MSComplex`` (labels, cells, critical
                                 points with type and persistence, boundaries, areas).
* ``ms_star_graph``              networkx graph: critical points connected to the centre of
                                 mass (optionally + adjacency of neighbouring regions).
* ``RegionTracker``              correspondence of the regions between consecutive frames
                                 through the functional map (hard p2p transport and soft
                                 spectral transport of region indicators), region lineage
                                 (continue / split / merge / birth / death), fate of every
                                 critical point (max -> max / saddle / min / regular, ...),
                                 protrusion growth measured along the surface normal.
* ``fuse_hull_protrusions``      protrusions missed by the scalar field are recovered from the minima
                                 of the depth below the convex hull (Huang, Wu & Yan 2024, Comput. Biol.
                                 Med. 173:108350): trough height β, merge distance α, caps = protrusion
                                 shapes, nested protrusions kept separate, noise-adaptive β, temporal
                                 hysteresis through the p2p map; protrusion cores (protrusion vs body).
* ``critical_points_report``, ``tracking_report``, ``protrusion_report``  csv + plots under MSComplexAnalysis/.
"""
import os
import re
import json
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
import pyvista as pv
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots

from PynamicMesh.core.reeb_graph import (
    get_scalar_field, _as_faces, _area_weighted_center, _get_mesh_adjacency, to_cpu,
)

CP_TYPES = ("minimum", "saddle", "maximum")
CP_COLORS = {"maximum": (0.90, 0.10, 0.10), "minimum": (0.10, 0.35, 0.95), "saddle": (0.10, 0.75, 0.20),
             "regular": (0.6, 0.6, 0.6), "center": (1.0, 0.85, 0.0)}
FATE_LABELS = ("maximum", "saddle", "minimum", "regular", "lost")



#  Mesh helpers

def _vertex_areas(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = _as_faces(faces)
    tri = v[f]
    area_f = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    A = np.zeros(v.shape[0])
    for k in range(3):
        np.add.at(A, f[:, k], area_f / 3.0)
    return A, area_f


def _vertex_normals(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = _as_faces(faces)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    N = np.zeros_like(v)
    for k in range(3):
        np.add.at(N, f[:, k], fn)
    nrm = np.linalg.norm(N, axis=1)
    return N / np.maximum(nrm, 1e-300)[:, None]


def _edges(faces):
    f = _as_faces(faces)
    E = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    return np.unique(E, axis=0)


def _total_order(f):
    """Strict total order of the vertices by (f, index): simulation of simplicity for ties."""
    f = np.asarray(f, dtype=np.float64)
    order = np.lexsort((np.arange(f.shape[0]), f))     # ascending f, then index
    rank = np.empty(f.shape[0], dtype=np.int64)
    rank[order] = np.arange(f.shape[0])
    return rank



#  Critical points (piecewise-linear classification)

def classify_critical_points(faces, f, n_vertices=None):
    """
    Banchoff / PL classification. For every vertex v the link (ring of neighbours) is scanned
    through the link edges (b, c) of the incident faces (a, b, c): every link edge whose ends
    are on opposite sides of f(v) is a crossing between the lower and the upper link.

        crossings = 0 & all neighbours lower  -> maximum
        crossings = 0 & all neighbours higher -> minimum
        crossings = 2                         -> regular
        crossings = 2k, k >= 2                -> saddle of multiplicity k - 1  (monkey saddles)

    Returns type codes (-1 min, 0 regular, 1 saddle, 2 max) and the saddle multiplicities.
    Vectorised (no per-vertex loop).
    """
    F = _as_faces(faces)
    f = np.asarray(f, dtype=np.float64)
    n = f.shape[0] if n_vertices is None else n_vertices
    rank = _total_order(f)
    crossings = np.zeros(n, dtype=np.int64)
    n_lower = np.zeros(n, dtype=np.int64)
    n_link = np.zeros(n, dtype=np.int64)
    for k in range(3):
        a, b, c = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        lb, lc = rank[b] < rank[a], rank[c] < rank[a]
        np.add.at(crossings, a, (lb != lc).astype(np.int64))
        # every link edge contributes its two ends; each neighbour is counted twice on a closed ring
        np.add.at(n_lower, a, lb.astype(np.int64) + lc.astype(np.int64))
        np.add.at(n_link, a, 2)
    types = np.zeros(n, dtype=np.int64)
    ncomp = crossings // 2
    types[(crossings == 0) & (n_lower == n_link) & (n_link > 0)] = 2      # maximum
    types[(crossings == 0) & (n_lower == 0) & (n_link > 0)] = -1          # minimum
    types[ncomp >= 2] = 1                                                 # saddle
    multiplicity = np.where(types == 1, ncomp - 1, 0)
    return types, multiplicity



#  Steepest ascent / descent manifolds

def _neighbour_structure(vertices, faces):
    """CSR adjacency (edge lengths) + arrays for vectorised neighbour ops."""
    adj = _get_mesh_adjacency(vertices, faces).tocsr()
    return adj


def steepest_manifolds(vertices, faces, f, adj=None):
    """
    Every vertex follows its steepest ascent neighbour (largest (f_j - f_i)/|e_ij|) up to a
    maximum -> label_max (descending manifold of that maximum = protrusion region), and its
    steepest descent neighbour down to a minimum -> label_min (ascending manifold).
    Pointer jumping makes the path compression O(log n) vector operations.
    Returns (label_max, label_min, next_up, next_down) as vertex indices.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = v.shape[0]
    adj = _neighbour_structure(v, faces) if adj is None else adj
    rank = _total_order(f)
    coo = adj.tocoo()
    i, j, w = coo.row, coo.col, np.maximum(coo.data, 1e-300)
    slope = (f[j] - f[i]) / w
    # tie-break with the strict order so that flat plateaus still have a direction
    slope = slope + 1e-12 * np.sign(rank[j] - rank[i])
    up = np.arange(n)
    best = np.full(n, -np.inf)
    # steepest ascent: for each i pick j maximising slope among neighbours with rank[j] > rank[i]
    asc = rank[j] > rank[i]
    order = np.lexsort((slope[asc], i[asc]))           # sort by i then slope -> last per i is the max
    ii, jj = i[asc][order], j[asc][order]
    last = np.r_[ii[1:] != ii[:-1], True]
    up[ii[last]] = jj[last]
    down = np.arange(n)
    desc = rank[j] < rank[i]
    order = np.lexsort((-slope[desc], i[desc]))        # most negative slope last
    ii, jj = i[desc][order], j[desc][order]
    last = np.r_[ii[1:] != ii[:-1], True]
    down[ii[last]] = jj[last]

    def compress(ptr):
        lab = ptr.copy()
        for _ in range(64):
            nxt = lab[lab]
            if np.array_equal(nxt, lab):
                break
            lab = nxt
        return lab
    return compress(up), compress(down), up, down



#  Persistence (merge trees) and simplification

class _UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a, b):
        self.p[b] = a


def persistence_pairs(adj, f, extremum="max"):
    """
    Merge-tree persistence of the maxima (extremum='max', super-level filtration: vertices are
    added by decreasing f) or of the minima (extremum='min', sub-level filtration).
    When a vertex v joins two components, the component whose extremum is less extreme dies:
        pair = (dead extremum, v, persistence = |f(extremum) - f(v)|).
    The global extremum never dies (persistence = inf, saddle = -1).
    Returns dict: extremum vertex -> (saddle vertex, persistence, killer extremum vertex).
    """
    f = np.asarray(f, dtype=np.float64)
    n = f.shape[0]
    rank = _total_order(f)
    order = np.argsort(-rank, kind="stable") if extremum == "max" else np.argsort(rank, kind="stable")
    indptr, indices = adj.indptr, adj.indices
    uf = _UnionFind(n)
    comp_ext = np.full(n, -1, dtype=np.int64)     # root -> extremum vertex of the component
    added = np.zeros(n, dtype=bool)
    pairs = {}
    better = (lambda a, b: rank[a] > rank[b]) if extremum == "max" else (lambda a, b: rank[a] < rank[b])
    for v in order:
        nb = indices[indptr[v]:indptr[v + 1]]
        nb = nb[added[nb]]
        added[v] = True
        if nb.size == 0:
            comp_ext[v] = v                      # a new local extremum
            continue
        roots = np.unique([uf.find(x) for x in nb])
        # attach v to the component with the most extreme extremum
        exts = comp_ext[roots]
        keep = roots[np.argmax(rank[exts])] if extremum == "max" else roots[np.argmin(rank[exts])]
        uf.union(keep, v)
        for r in roots:
            if r == keep:
                continue
            dead = comp_ext[r]
            pairs[int(dead)] = (int(v), float(abs(f[dead] - f[v])), int(comp_ext[keep]))
            uf.union(keep, r)
    survivors = {int(comp_ext[r]) for r in np.unique([uf.find(x) for x in range(n)])}
    for e in survivors:
        pairs.setdefault(e, (-1, np.inf, e))
    return pairs


def _simplify_labels(label, pairs, threshold):
    """Relabel vertices whose extremum has persistence < threshold to the extremum that killed it
    (following the chain until a surviving extremum)."""
    target = {}
    for e, (s, p, killer) in pairs.items():
        target[e] = e if p >= threshold else killer

    def final(e):
        seen = set()
        while target.get(e, e) != e and e not in seen:
            seen.add(e)
            e = target[e]
        return e
    remap = {e: final(e) for e in pairs}
    lut = np.arange(max(int(label.max()) + 1, 1))
    for e, t in remap.items():
        if e < lut.shape[0]:
            lut[e] = t
    return lut[label], remap



#  Morse–Smale complex container

@dataclass
class MSComplex:
    """Result of :func:`compute_ms_complex` for one frame (all arrays per vertex unless stated)."""
    frame: int
    scalar: np.ndarray                     # f
    label_max: np.ndarray                  # simplified descending manifold (id = maximum vertex)
    label_min: np.ndarray                  # simplified ascending manifold (id = minimum vertex)
    cell: np.ndarray                       # Morse–Smale cell id (pair max/min), 0..n_cells-1
    label_max_raw: np.ndarray
    label_min_raw: np.ndarray
    cp_type_raw: np.ndarray                # -1 min, 0 regular, 1 saddle, 2 max (PL classification)
    cp_multiplicity: np.ndarray
    critical: pd.DataFrame                 # simplified critical points: vertex, type, f, persistence, x,y,z, region_area
    maxima: np.ndarray                     # vertex ids of the surviving maxima
    minima: np.ndarray
    saddles: np.ndarray
    boundary_edges: np.ndarray             # (E, 2) edges between different maximum regions
    vertex_area: np.ndarray
    center: np.ndarray                     # area weighted centre of mass
    persistence_threshold: float
    n_vertices: int
    meta: Dict = field(default_factory=dict)
    # convex-hull protrusion fusion (None for complexes computed without it / older pickles)
    hull_depth: Optional[np.ndarray] = None          # depth below the convex hull boundary (Huang et al. 2024)
    core_label: Optional[np.ndarray] = None          # protrusion core (protrusion id = maximum vertex, -1 = cell body)
    protrusions: Optional[pd.DataFrame] = None       # one row per protrusion: source ms / hull / ms+hull, heights, areas
    hull_candidates: Optional[pd.DataFrame] = None   # every hull-depth minimum examined, with the rejection reason

    # -- convenience --------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        c = self.critical["type"].value_counts()
        return {"n_max": int(c.get("maximum", 0)), "n_min": int(c.get("minimum", 0)),
                "n_saddle": int(c.get("saddle", 0))}

    def region_areas(self) -> Dict[int, float]:
        out = {}
        for m in self.maxima:
            out[int(m)] = float(self.vertex_area[self.label_max == m].sum())
        return out

    def type_of_vertex(self, v: int) -> str:
        row = self.critical[self.critical["vertex"] == int(v)]
        return str(row["type"].iloc[0]) if len(row) else "regular"

    def to_pickle(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def from_pickle(path) -> "MSComplex":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def compute_ms_complex(vertices, faces, scalar_field, persistence=0.05, frame=0,
                       min_region_area=0.0, protrusion_params=None, prev_tips=None) -> MSComplex:
    """
    Morse–Smale complex of ``scalar_field`` on the mesh.

    persistence : float   simplification threshold as a fraction of the field range
                          (extrema with smaller persistence are merged into their killer);
                          0 keeps every PL critical point.
    min_region_area : float  additionally merge maxima whose region is smaller than this
                          fraction of the total area into the neighbouring region with the
                          longest shared boundary.
    Saddles of the simplified complex are the merge vertices of the surviving persistence
    pairs (one per surviving max/min except the global ones), which satisfies Euler's relation
    #min - #saddle + #max = 2 on closed genus-0 surfaces.
    protrusion_params : dict  convex-hull protrusion fusion (see PROTRUSION_DEFAULTS / configure_protrusion_detection);
                          enabled by default: the maxima of the field are completed with the minima of the
                          convex-hull depth (fuse_hull_protrusions). {'enabled': False} gives the plain complex.
    prev_tips : array     vertices of this mesh where the previous frame's protrusions land (temporal hysteresis).
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    F = _as_faces(faces)
    f = np.asarray(to_cpu(scalar_field), dtype=np.float64).ravel()
    n = v.shape[0]
    if f.shape[0] != n:
        raise ValueError(f"scalar field has {f.shape[0]} values for {n} vertices")
    adj = _neighbour_structure(v, F)
    A, _ = _vertex_areas(v, F)
    frange = float(np.ptp(f)) if np.ptp(f) > 0 else 1.0
    thr = float(persistence) * frange

    cp_raw, mult = classify_critical_points(F, f, n)
    lab_max_raw, lab_min_raw, up, down = steepest_manifolds(v, F, f, adj)
    pairs_max = persistence_pairs(adj, f, "max")
    pairs_min = persistence_pairs(adj, f, "min")
    # the steepest-ascent maxima and the filtration maxima coincide (both are the PL maxima);
    # make sure every label has a pair entry
    for m in np.unique(lab_max_raw):
        pairs_max.setdefault(int(m), (-1, np.inf, int(m)))
    for m in np.unique(lab_min_raw):
        pairs_min.setdefault(int(m), (-1, np.inf, int(m)))

    lab_max, remap_max = _simplify_labels(lab_max_raw, pairs_max, thr)
    lab_min, remap_min = _simplify_labels(lab_min_raw, pairs_min, thr)

    # optional area-based merging of tiny regions
    if min_region_area > 0:
        total = A.sum()
        E = _edges(F)
        for _ in range(20):
            areas = {int(m): float(A[lab_max == m].sum()) for m in np.unique(lab_max)}
            small = [m for m, a in areas.items() if a < min_region_area * total]
            if not small or len(areas) <= 1:
                break
            changed = False
            for m in small:
                lm = lab_max[E[:, 0]], lab_max[E[:, 1]]
                sel = (lm[0] == m) ^ (lm[1] == m)
                if not sel.any():
                    continue
                other = np.where(lm[0][sel] == m, lm[1][sel], lm[0][sel])
                vals, cnts = np.unique(other, return_counts=True)
                lab_max[lab_max == m] = vals[np.argmax(cnts)]
                changed = True
            if not changed:
                break

    maxima = np.unique(lab_max)
    minima = np.unique(lab_min)
    # saddles: merge vertices of the surviving pairs (excluding the global extremum)
    sad_max = {s: (e, p) for e, (s, p, k) in pairs_max.items() if s >= 0 and p >= thr and e in set(maxima.tolist())}
    sad_min = {s: (e, p) for e, (s, p, k) in pairs_min.items() if s >= 0 and p >= thr and e in set(minima.tolist())}
    saddles = np.array(sorted(set(sad_max) | set(sad_min)), dtype=np.int64)

    # Morse–Smale cells: unique (max, min) pairs
    pair_key = lab_max.astype(np.int64) * (n + 1) + lab_min.astype(np.int64)
    _, cell = np.unique(pair_key, return_inverse=True)

    # critical point table
    rows = []
    areas_max = {int(m): float(A[lab_max == m].sum()) for m in maxima}
    areas_min = {int(m): float(A[lab_min == m].sum()) for m in minima}
    for m in maxima:
        s, p, k = pairs_max[int(m)]
        rows.append(dict(vertex=int(m), type="maximum", f=float(f[m]), persistence=float(p),
                         x=v[m, 0], y=v[m, 1], z=v[m, 2], region_area=areas_max[int(m)], paired_saddle=int(s)))
    for m in minima:
        s, p, k = pairs_min[int(m)]
        rows.append(dict(vertex=int(m), type="minimum", f=float(f[m]), persistence=float(p),
                         x=v[m, 0], y=v[m, 1], z=v[m, 2], region_area=areas_min[int(m)], paired_saddle=int(s)))
    for s in saddles:
        if int(s) in sad_max:
            e, p = sad_max[int(s)]; killer = pairs_max[e][2]; kind = "max"
        else:
            e, p = sad_min[int(s)]; killer = pairs_min[e][2]; kind = "min"
        rows.append(dict(vertex=int(s), type="saddle", f=float(f[s]), persistence=float(p),
                         x=v[s, 0], y=v[s, 1], z=v[s, 2], region_area=np.nan, paired_saddle=-1,
                         pair_extremum=int(e), pair_killer=int(killer), pair_kind=kind))
    critical = pd.DataFrame(rows, columns=["vertex", "type", "f", "persistence", "x", "y", "z", "region_area",
                                           "paired_saddle", "pair_extremum", "pair_killer", "pair_kind"])
    critical["pair_extremum"] = critical["pair_extremum"].fillna(-1).astype(int)
    critical["pair_killer"] = critical["pair_killer"].fillna(-1).astype(int)
    critical["pair_kind"] = critical["pair_kind"].fillna("")
    critical["persistence_rel"] = critical["persistence"] / frange

    E = _edges(F)
    boundary = E[lab_max[E[:, 0]] != lab_max[E[:, 1]]]

    ms = MSComplex(frame=frame, scalar=f, label_max=lab_max, label_min=lab_min, cell=cell,
                     label_max_raw=lab_max_raw, label_min_raw=lab_min_raw, cp_type_raw=cp_raw,
                     cp_multiplicity=mult, critical=critical, maxima=maxima, minima=minima, saddles=saddles,
                     boundary_edges=boundary, vertex_area=A, center=_area_weighted_center(v, F),
                     persistence_threshold=thr, n_vertices=n,
                     meta={"field_range": frange, "n_max_raw": int((cp_raw == 2).sum()),
                           "n_min_raw": int((cp_raw == -1).sum()), "n_saddle_raw": int(mult.sum()),
                           "euler_raw": int((cp_raw == -1).sum() - mult.sum() + (cp_raw == 2).sum()),
                           "n_cells": int(cell.max() + 1)})
    ms.meta["n_max_ms"] = int(len(maxima))
    try:                                                   # topology of the surface: the expected Euler number
        from PynamicMesh.core.reeb_graph import surface_topology
        c_, chi_, b_, g_ = surface_topology(F)
        ms.meta.update({"surface_euler": int(chi_), "surface_genus": int(g_), "boundary_loops": int(b_)})
    except Exception:  # noqa: BLE001
        pass
    cfg = protrusion_config(protrusion_params)
    if cfg["enabled"]:
        try:
            ms = fuse_hull_protrusions(ms, v, F, params=cfg, prev_tips=prev_tips)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"convex-hull protrusion fusion failed on frame {frame} ({exc}); plain Morse–Smale result kept")
    return ms



#  Convex-hull depth protrusion detection (Huang, Wu & Yan 2024) and fusion with the MS complex

PROTRUSION_DEFAULTS = {
    "enabled": True,              # False -> plain Morse–Smale segmentation (previous behaviour)
    "beta": 0.012,                # Huang's β: minimum trough height (hull-depth persistence) / equivalent radius
    "noise_factor": 1.0,          # adaptive floor: β_eff = max(β, noise_factor · surface roughness / eq. radius)
    "smooth_iters": 5,            # Taubin λ|μ iterations of the geometry used for the hull (noise spikes would span it)
    "alpha": 0.08,                # Huang's α: minima closer than this geodesic distance (/ equivalent radius) are merged
    "max_tip_depth": 0.25,        # a tip deeper than this below the hull (/ eq. radius) is a pit, not a protrusion
    "min_aspect": 0.12,           # prominence / cap radius; rejects broad convex patches of the cell body
    "max_cap_area": 0.30,         # a protrusion cap covers at most this fraction of the surface
    "core_fraction": 0.25,        # protrusions without hull support: core = upper fraction of the relief of the scalar field
    "temporal": True,             # hysteresis through the p2p map: weak candidates confirmed by the previous frame
    "temporal_beta_factor": 0.5,  # ... accepted down to beta * factor and min_aspect * factor
    "temporal_radius": 2.0,       # ... when a previous protrusion maps within radius * alpha of the tip
    "prune_ms": False,            # drop MS maxima deep inside the hull without any hull support (off by default)
    "prune_depth": 0.35,          # ... deeper than this (/ eq. radius)
}
_PROTRUSION_ENV = "PYNAMICMESH_PROTRUSION_PARAMS"
_PROTRUSION_CONFIG: Dict = {}


def configure_protrusion_detection(**params) -> Dict:
    """
    Sets the parameters of the convex-hull / Morse–Smale protrusion fusion used by compute_MS (call it before
    run_pipeline). They are also exported to the environment so that worker processes see them.
    Unknown keys raise. Returns the effective configuration.
    """
    unknown = set(params) - set(PROTRUSION_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown protrusion parameters {sorted(unknown)}; valid: {sorted(PROTRUSION_DEFAULTS)}")
    _PROTRUSION_CONFIG.update(params)
    os.environ[_PROTRUSION_ENV] = json.dumps(_PROTRUSION_CONFIG)
    return protrusion_config()


def protrusion_config(overrides=None) -> Dict:
    """Defaults <- configure_protrusion_detection / environment <- explicit overrides."""
    cfg = dict(PROTRUSION_DEFAULTS)
    env = os.environ.get(_PROTRUSION_ENV)
    if env:
        try:
            cfg.update({k: v for k, v in json.loads(env).items() if k in cfg})
        except (ValueError, TypeError):
            pass
    cfg.update(_PROTRUSION_CONFIG)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if k in cfg})
    return cfg


def convex_hull_depth(vertices, chunk=1024):
    """
    Depth of every vertex below the boundary of the convex hull of the surface (Huang et al. 2024, §3.3):
    for a point inside a convex polytope the distance to the boundary is the minimum distance to the facet
    planes, depth_i = min_j -(n_j · p_i + d_j) (Qhull equations, n_j outward unit normals).
    Returns depth (n,), the index of the nearest facet (n,), the projection of every vertex onto the hull
    boundary p_i + depth_i n_j (Huang's projection ψ : M -> N) and the scipy ConvexHull.
    """
    from scipy.spatial import ConvexHull
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    try:
        hull = ConvexHull(v)
    except Exception:  # noqa: BLE001  (flat / degenerate input)
        hull = ConvexHull(v, qhull_options="QJ")
    eq = hull.equations
    n = v.shape[0]
    depth = np.empty(n); facet = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        d = -(v[s:s + chunk] @ eq[:, :3].T + eq[:, 3][None, :])
        k = np.argmin(d, axis=1)
        facet[s:s + chunk] = k
        depth[s:s + chunk] = d[np.arange(d.shape[0]), k]
    depth = np.maximum(depth, 0.0)
    proj = v + depth[:, None] * eq[facet, :3]
    return depth, facet, proj, hull


def surface_roughness(vertices, faces, adj=None):
    """Noise level of the surface: robust spread (1.4826 MAD) of the normal component of the umbrella Laplacian
    after removing its smooth (curvature) part by neighbour averaging. Same unit as the coordinates."""
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    adj = _neighbour_structure(v, faces) if adj is None else adj
    Wu = adj.copy(); Wu.data = np.ones_like(Wu.data)
    deg = np.maximum(np.diff(Wu.indptr), 1)
    ln = np.einsum("ij,ij->i", v - (Wu @ v) / deg[:, None], _vertex_normals(v, faces))
    r = ln - (Wu @ ln) / deg
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def taubin_smooth(vertices, adj, iters=5, lam=0.5, mu=-0.53):
    """Taubin λ|μ smoothing (uniform weights): removes vertex noise without shrinking the surface."""
    X = np.asarray(vertices, dtype=np.float64).copy()
    if iters <= 0:
        return X
    Wu = adj.copy(); Wu.data = np.ones_like(Wu.data)
    deg = np.maximum(np.diff(Wu.indptr), 1)[:, None]
    for _ in range(int(iters)):
        X = X + lam * ((Wu @ X) / deg - X)
        X = X + mu * ((Wu @ X) / deg - X)
    return X


def _superlevel_component(adj, g, seed, level):
    """Connected component of {g > level} containing seed (the 'mountain' of seed above its col)."""
    from collections import deque
    ip, ix = adj.indptr, adj.indices
    seen = np.zeros(g.shape[0], dtype=bool); seen[seed] = True
    q = deque([int(seed)]); out = [int(seed)]
    while q:
        u = q.popleft()
        for w in ix[ip[u]:ip[u + 1]]:
            if not seen[w] and g[w] > level:
                seen[w] = True; q.append(int(w)); out.append(int(w))
    return np.asarray(out, dtype=np.int64)


def hull_protrusion_candidates(vertices, faces, depth, adj=None, A=None, params=None, prev_tips=None):
    """
    Protrusion candidates of the convex-hull depth (local minima of the depth, Huang et al. 2024 §3.4–3.5),
    computed directly on the mesh: by Huang's Theorem 1 the critical points are invariant under the
    diffeomorphism to the hull boundary, so the spherical projection (a chart for finite differences) is not
    needed and the PL critical points + merge-tree persistence give the minima exactly.

      g = -depth / s   (s = sqrt(area / 4π), equivalent radius)  -> tips are maxima of g
      trough height (Huang's β) = persistence of the maximum of g; cap = mountain above its col
      (connected component of {g > g(col)}), i.e. the protrusion shape of 'ExtractCellProtrusions'.

    Accepted: height >= beta, tip depth <= max_tip_depth, aspect = height / cap radius >= min_aspect, cap area
    <= max_cap_area; tips closer than alpha (geodesic) are merged into the higher one (Huang's α).
    With prev_tips (vertices of this mesh where the protrusions of the previous frame land), candidates down to
    beta * temporal_beta_factor are accepted if a previous tip lies within temporal_radius * alpha.
    Returns (accepted list of dicts, DataFrame of every examined candidate with the reason of rejection).
    """
    p = protrusion_config(params)
    v = np.asarray(to_cpu(vertices), dtype=np.float64); F = _as_faces(faces)
    adj = _neighbour_structure(v, F) if adj is None else adj
    A = _vertex_areas(v, F)[0] if A is None else A
    s = float(np.sqrt(A.sum() / (4.0 * np.pi)))
    g = -np.asarray(depth, dtype=np.float64) / s
    pairs = persistence_pairs(adj, g, "max")
    beta_low = p["beta"] * (p["temporal_beta_factor"] if (p["temporal"] and prev_tips is not None and len(prev_tips)) else 1.0)
    killed_by: Dict[int, List[int]] = {}
    for e, (sd, pers, k) in pairs.items():
        if sd >= 0 and pers >= beta_low:               # significant merges only (noise peaks on a plateau ignored)
            killed_by.setdefault(int(k), []).append(int(sd))
    d_prev = None
    if prev_tips is not None and len(prev_tips):
        d_prev = dijkstra(adj, indices=np.asarray(prev_tips, dtype=np.int64), min_only=True,
                          limit=p["temporal_radius"] * p["alpha"] * s)
    rows, cand = [], []
    for e, (sd, pers, k) in pairs.items():
        if sd >= 0:
            level = g[sd]
        else:                                    # global maximum: its highest significant col
            cols = killed_by.get(int(e), [])
            level = max((g[c] for c in cols), default=float(g.min()))
        height = float(g[e] - level)
        if height < beta_low:
            continue
        cap = _superlevel_component(adj, g, e, level)
        a_cap = float(A[cap].sum())
        aspect = height / max(np.sqrt(a_cap / (np.pi * s * s)), 1e-12)
        tip_depth = float(depth[e] / s)
        # temporal hysteresis: a candidate on which a protrusion of the previous frame lands only has to pass
        # the relaxed thresholds (both the trough height and the aspect, by temporal_beta_factor)
        supported = d_prev is not None and bool(np.isfinite(d_prev[e]))
        relax = p["temporal_beta_factor"] if (p["temporal"] and supported) else 1.0
        temporal = bool(supported and (height < p["beta"] or aspect < p["min_aspect"]))
        reason = ""
        if tip_depth > p["max_tip_depth"]:
            reason = "tip too deep (pit)"
        elif height < p["beta"] * relax:
            reason = "below beta" + ("" if supported else " (no temporal support)")
        elif aspect < p["min_aspect"] * relax:
            reason = "flat convex patch (aspect)"
        elif a_cap > p["max_cap_area"] * A.sum():
            reason = "cap too large"
        row = {"vertex": int(e), "height_rel": height, "tip_depth_rel": tip_depth, "cap_area": a_cap,
               "cap_area_rel": a_cap / A.sum(), "aspect": float(aspect), "col_vertex": int(sd),
               "temporal": temporal, "accepted": reason == "", "reason": reason}
        rows.append(row)
        if not reason:
            cand.append({**row, "cap": cap})
    # Huang's α: merge tips closer than alpha (keep the higher trough); nested caps are distinct protrusions
    cand.sort(key=lambda c: -c["height_rel"])
    kept = []
    for c in cand:
        if kept:
            dist = dijkstra(adj, indices=[k_["vertex"] for k_ in kept], min_only=True, limit=p["alpha"] * s)
            if np.isfinite(dist[c["vertex"]]):
                for r in rows:
                    if r["vertex"] == c["vertex"]:
                        r["accepted"], r["reason"] = False, "merged (alpha)"
                continue
        kept.append(c)
    table = pd.DataFrame(rows, columns=["vertex", "height_rel", "tip_depth_rel", "cap_area", "cap_area_rel", "aspect",
                                        "col_vertex", "temporal", "accepted", "reason"])
    return kept, table


def _fix_region_connectivity(label, keep_vertex, E, max_iter=10):
    """Every label keeps only the component that contains its protrusion vertex; stray fragments are given to the
    neighbouring label with which they share most edges (repeated until stable)."""
    from scipy.sparse.csgraph import connected_components
    n = label.shape[0]
    for _ in range(max_iter):
        same = label[E[:, 0]] == label[E[:, 1]]
        Es = E[same]
        G = coo_matrix((np.ones(len(Es)), (Es[:, 0], Es[:, 1])), shape=(n, n))
        _, comp = connected_components(G, directed=False)
        good = np.zeros(comp.max() + 1, dtype=bool)
        for lab in np.unique(label):
            kv = keep_vertex.get(int(lab), int(lab))
            if 0 <= kv < n and label[kv] == lab:
                good[comp[kv]] = True
        stray = ~good[comp]
        if not stray.any():
            break
        ce = E[comp[E[:, 0]] != comp[E[:, 1]]]
        changed = False
        for c in np.unique(comp[stray]):
            sel = (comp[ce[:, 0]] == c) ^ (comp[ce[:, 1]] == c)
            if not sel.any():
                continue
            other = np.where(comp[ce[sel, 0]] == c, ce[sel, 1], ce[sel, 0])
            other = other[good[comp[other]]] if good[comp[other]].any() else other
            vals, cnts = np.unique(label[other], return_counts=True)
            label[comp == c] = vals[np.argmax(cnts)]
            changed = True
        if not changed:
            break
    return label


def fuse_hull_protrusions(ms: "MSComplex", vertices, faces, params=None, prev_tips=None) -> "MSComplex":
    """
    Completes the Morse–Smale protrusion segmentation with the convex-hull depth detection.

    1. hull candidates (see hull_protrusion_candidates) with their caps;
    2. matching: every MS maximum confirms the innermost cap that contains it (else the nearest tip within α);
       confirmed candidates are the same protrusion ('ms+hull');
    3. the unconfirmed candidates are protrusions the scalar field missed ('hull'): their caps are carved out of
       the MS regions (larger caps first, so nested protrusions are painted last and stay separate), stray
       fragments are re-attached, and every new maximum gets the saddle of its col (pair_kind 'max', killer =
       the region it was carved from), so #min - #saddle + #max is unchanged;
    4. protrusion cores (protrusion vs cell body): the hull cap for hull-supported protrusions, the upper part
       of the relief of the scalar field (core_fraction) for MS-only ones.
    The regions (label_max / maxima / critical / boundary_edges / cell) keep their meaning, so the tracker,
    graphs and viewers work unchanged; extra results: ms.hull_depth, ms.core_label, ms.protrusions,
    ms.hull_candidates.
    """
    p = protrusion_config(params)
    v = np.asarray(to_cpu(vertices), dtype=np.float64); F = _as_faces(faces)
    n = v.shape[0]
    adj = _neighbour_structure(v, F)
    A = ms.vertex_area
    s = float(np.sqrt(A.sum() / (4.0 * np.pi)))
    # noise-robust hull: noise spikes of a marching-cubes surface would span the hull and create spurious
    # minima, so the depth is measured on a Taubin-smoothed copy (same connectivity / vertex ids) and the
    # trough height threshold is raised to the noise level of the original surface
    noise_rel = surface_roughness(v, F, adj) / s
    beta_eff = max(float(p["beta"]), float(p["noise_factor"]) * noise_rel)
    p_eff = {**p, "beta": beta_eff}
    v_h = taubin_smooth(v, adj, p["smooth_iters"])
    depth, facet, proj, hull = convex_hull_depth(v_h)
    cands, table = hull_protrusion_candidates(v_h, F, depth, adj=adj, A=A, params=p_eff, prev_tips=prev_tips)
    f = ms.scalar
    frange = ms.meta.get("field_range", 1.0)
    label = ms.label_max.copy()
    ms_max = [int(m) for m in ms.maxima]
    E = _edges(F)

    # optional pruning of deep MS maxima without any hull support
    pruned = []
    if p["prune_ms"] and cands:
        in_cap = np.zeros(n, dtype=bool)
        for c in cands:
            in_cap[c["cap"]] = True
        for m in ms_max:
            if depth[m] / s > p["prune_depth"] and not in_cap[m] and len(ms_max) - len(pruned) > 1:
                pruned.append(m)
        if pruned:
            label[np.isin(label, pruned)] = -1
            ms_max = [m for m in ms_max if m not in pruned]
            for _ in range(n):                                    # flood the pruned regions from their neighbours
                bad = label < 0
                if not bad.any():
                    break
                e_ = E[(label[E[:, 0]] < 0) ^ (label[E[:, 1]] < 0)]
                src = np.where(label[e_[:, 0]] < 0, e_[:, 1], e_[:, 0]); dst = np.where(label[e_[:, 0]] < 0, e_[:, 0], e_[:, 1])
                label[dst] = label[src]

    # 2. matching MS maxima <-> hull candidates
    cap_sets = [set(c["cap"].tolist()) for c in cands]
    owner = {}                                   # candidate index -> list of MS maxima
    ms_to_cand = {}
    tips = np.array([c["vertex"] for c in cands], dtype=np.int64)
    for m in ms_max:
        inside = [k for k, cs in enumerate(cap_sets) if m in cs]
        if inside:
            k = min(inside, key=lambda k_: cands[k_]["cap_area"])
        elif tips.size:
            dist = dijkstra(adj, indices=m, limit=p["alpha"] * s)
            k = int(np.argmin(dist[tips])) if np.isfinite(dist[tips]).any() else -1
        else:
            k = -1
        if k >= 0:
            owner.setdefault(k, []).append(m); ms_to_cand[m] = k

    # 3. carve the caps (confirmed caps are painted with their MS maximum, new caps with their own tip)
    order = sorted(range(len(cands)), key=lambda k_: -cands[k_]["cap_area"])
    host_of = {}
    for k in order:
        c = cands[k]
        if k in owner:
            if len(owner[k]) == 1:                                   # one MS maximum: the cap is its protrusion
                label[c["cap"]] = owner[k][0]
            # several MS maxima on one hull cap: keep their Morse–Smale partition inside the cap
        else:
            vals, cnts = np.unique(label[c["cap"]], return_counts=True)
            host_of[c["vertex"]] = int(vals[np.argmax(cnts)])
            label[c["cap"]] = c["vertex"]
    keep_vertex = {int(l): int(l) for l in np.unique(label)}
    label = _fix_region_connectivity(label, keep_vertex, E)
    added = [cands[k]["vertex"] for k in range(len(cands)) if k not in owner and label[cands[k]["vertex"]] == cands[k]["vertex"]]

    maxima = np.unique(label)
    boundary = E[label[E[:, 0]] != label[E[:, 1]]]

    # critical point table: MS rows (pruned maxima and their saddles removed) + new maxima and their cols
    crit = ms.critical.copy()
    gone = set(pruned) | (set(ms_max) - set(maxima.tolist()))          # safety: maxima whose region vanished
    if gone:
        crit = crit[~(((crit.type == "maximum") & crit.vertex.isin(gone)) |
                      ((crit.type == "saddle") & (crit.pair_kind == "max") & crit.pair_extremum.isin(gone)))]
    g = -depth / s
    new_rows = []
    cand_of = {c["vertex"]: c for c in cands}
    for c_v in added:
        c = cand_of[c_v]
        reg = label == c_v
        pers_rel = float(c["height_rel"]); pers = pers_rel * frange
        new_rows.append(dict(vertex=int(c_v), type="maximum", f=float(f[c_v]), persistence=pers, x=v[c_v, 0], y=v[c_v, 1],
                             z=v[c_v, 2], region_area=float(A[reg].sum()), paired_saddle=-1, pair_extremum=-1,
                             pair_killer=-1, pair_kind="", persistence_rel=pers_rel))
        bnd = boundary[(label[boundary[:, 0]] == c_v) | (label[boundary[:, 1]] == c_v)]
        if bnd.size:
            # the saddle is the col of the protrusion: the pass where its mountain joins the neighbouring one
            # (merge vertex of the depth persistence pair). It lies outside the region by construction; if the
            # carving moved it, take the highest vertex just OUTSIDE the region (the inside boundary vertices are
            # always higher and sit next to the tip: that gave saddles glued to their maximum)
            outside = np.where(label[bnd[:, 0]] == c_v, bnd[:, 1], bnd[:, 0])
            sv = int(c["col_vertex"])
            if sv < 0 or label[sv] == c_v or sv not in set(outside.tolist()):
                ring = set(outside.tolist())
                sv = int(c["col_vertex"]) if (c["col_vertex"] >= 0 and label[c["col_vertex"]] != c_v
                                               and any(w in ring for w in adj.indices[adj.indptr[c["col_vertex"]]:adj.indptr[c["col_vertex"] + 1]])) \
                    else int(outside[np.argmax(g[outside])])
            other = np.where(label[bnd[:, 0]] == c_v, label[bnd[:, 1]], label[bnd[:, 0]])
            vals, cnts = np.unique(other, return_counts=True)
            killer = host_of.get(c_v, int(vals[np.argmax(cnts)]))
            killer = killer if killer in set(maxima.tolist()) else int(vals[np.argmax(cnts)])
            new_rows[-1]["paired_saddle"] = sv
            new_rows.append(dict(vertex=sv, type="saddle", f=float(f[sv]), persistence=pers, x=v[sv, 0], y=v[sv, 1],
                                 z=v[sv, 2], region_area=np.nan, paired_saddle=-1, pair_extremum=int(c_v),
                                 pair_killer=int(killer), pair_kind="max", persistence_rel=pers_rel))
    if new_rows:
        crit = pd.concat([crit, pd.DataFrame(new_rows)], ignore_index=True)
    is_max = crit.type == "maximum"
    crit.loc[is_max, "region_area"] = [float(A[label == vv].sum()) for vv in crit.loc[is_max, "vertex"]]
    crit["source"] = "ms"
    crit.loc[is_max & crit.vertex.isin(added), "source"] = "hull"
    crit.loc[is_max & crit.vertex.isin([m for m in ms_max if m in ms_to_cand]), "source"] = "ms+hull"
    crit.loc[(crit.type == "saddle") & crit.pair_extremum.isin(added), "source"] = "hull"
    crit["hull_depth_rel"] = depth[crit.vertex.to_numpy(dtype=np.int64)] / s
    crit = crit.reset_index(drop=True)

    # 4. protrusion cores and table
    core = np.full(n, -1, dtype=np.int64)
    for k in order:                                             # caps (nested ones last)
        c = cands[k]
        if k in owner and len(owner[k]) > 1:                    # shared cap: each MS region keeps its part
            core[c["cap"]] = label[c["cap"]]
        else:
            core[c["cap"]] = owner[k][0] if k in owner else c["vertex"]
    core[core >= 0] = np.where(label[core >= 0] == core[core >= 0], core[core >= 0], -1)
    prot_rows = []
    for m in maxima:
        m = int(m)
        reg = label == m
        k = ms_to_cand.get(m, None)
        if k is None and m in cand_of:
            k = next(i for i, c in enumerate(cands) if c["vertex"] == m)
        source = "hull" if m in added else ("ms+hull" if m in ms_to_cand else "ms")
        if not (core == m).any():                              # MS-only: upper part of the relief of the field
            idx = np.where(reg)[0]
            lvl = f[m] - p["core_fraction"] * (f[m] - f[idx].min())
            comp = _superlevel_component(adj, np.where(reg, f, -np.inf), m, lvl)
            core[comp] = m
        creg = core == m
        tip_h = int(np.where(creg)[0][np.argmin(depth[creg])]) if creg.any() else m
        cand = cands[k] if k is not None else None
        ms_row = ms.critical[(ms.critical.type == "maximum") & (ms.critical.vertex == m)]
        ms_pers = float(ms_row.persistence_rel.iloc[0]) if len(ms_row) else np.nan
        hull_h = float(cand["height_rel"]) if cand is not None else np.nan
        sc_ms = 0.0 if not (ms_pers == ms_pers) else (1.0 if np.isinf(ms_pers) else
                                                       min(1.0, ms_pers / max(2.0 * ms.persistence_threshold / frange, 1e-12)))
        sc_h = min(1.0, hull_h / (2.0 * beta_eff)) if np.isfinite(hull_h) else 0.0
        nrm = hull.equations[facet[tip_h], :3]
        prot_rows.append({"protrusion_id": m, "source": source, "tip_vertex": m, "hull_tip_vertex": tip_h,
                          "x": v[m, 0], "y": v[m, 1], "z": v[m, 2],
                          "tip_hull_depth_rel": float(depth[tip_h] / s), "hull_height_rel": hull_h,
                          "ms_persistence_rel": ms_pers, "temporal_support": bool(cand["temporal"]) if cand is not None else False,
                          "region_area": float(A[reg].sum()), "region_area_rel": float(A[reg].sum() / A.sum()),
                          "core_area": float(A[creg].sum()), "core_area_rel": float(A[creg].sum() / A.sum()),
                          "core_height_rel": float((depth[creg].max() - depth[tip_h]) / s) if creg.any() else 0.0,
                          "aspect": float(cand["aspect"]) if cand is not None else np.nan,
                          "direction_x": nrm[0], "direction_y": nrm[1], "direction_z": nrm[2],
                          "confidence": float(1.0 - (1.0 - sc_ms) * (1.0 - sc_h))})
    protrusions = pd.DataFrame(prot_rows)

    pair_key = label.astype(np.int64) * (n + 1) + ms.label_min.astype(np.int64)
    _, cell = np.unique(pair_key, return_inverse=True)
    saddles = np.array(sorted(set(crit.loc[crit.type == "saddle", "vertex"].astype(int).tolist())), dtype=np.int64)
    ms.label_max, ms.maxima, ms.saddles, ms.critical, ms.boundary_edges = label, maxima, saddles, crit, boundary
    ms.cell = cell
    ms.hull_depth, ms.core_label, ms.protrusions = depth, core, protrusions
    ms.hull_candidates = table
    ms.meta.update({"protrusion_method": "ms+convex_hull", "protrusion_params": {k: p[k] for k in p},
                    "n_cells": int(cell.max() + 1), "equivalent_radius": s, "n_hull_facets": int(len(hull.equations)),
                    "n_protrusions_ms_only": int((protrusions.source == "ms").sum()),
                    "n_protrusions_hull_only": int((protrusions.source == "hull").sum()),
                    "n_protrusions_both": int((protrusions.source == "ms+hull").sum()),
                    "n_ms_pruned": len(pruned), "noise_level_rel": noise_rel, "beta_effective": beta_eff})
    return ms


def _map_previous_tips(prev_ms, p2p, n_curr, prev_vertices=None, prev_faces=None):
    """Vertices of the current mesh where the protrusion tips of the previous frame land through the p2p map
    (p2p with n_curr entries = current -> previous, as the tracker's p2p_21; n_prev entries = previous -> current)."""
    if prev_ms is None or p2p is None:
        return None
    p2p = np.asarray(to_cpu(p2p), dtype=np.int64).ravel()
    tips = np.asarray(prev_ms.maxima, dtype=np.int64)
    if p2p.shape[0] == n_curr:
        vp = np.asarray(to_cpu(prev_vertices), float) if prev_vertices is not None else None
        adj_prev = _neighbour_structure(vp, prev_faces) if vp is not None and prev_faces is not None else None
        core_prev = getattr(prev_ms, "core_label", None)
        region_prev = core_prev if core_prev is not None else prev_ms.label_max
        out = []
        for u in tips:
            pre = np.where(p2p == u)[0]
            if pre.size:
                out.append(int(pre[0])); continue
            # no exact pre-image (the p2p map is not injective): pre-images of the previous protrusion core,
            # the one whose previous position is closest to the old tip
            pre = np.where(region_prev[p2p] == u)[0]
            if pre.size:
                if vp is not None:
                    pre = pre[[int(np.argmin(np.linalg.norm(vp[p2p[pre]] - vp[u], axis=1)))]]
                out.append(int(pre[0])); continue
            if adj_prev is not None:
                j, _ = RegionTracker._image_vertex(int(u), p2p, None, adj_prev)
                if j >= 0:
                    out.append(j)
        return np.asarray(out, dtype=np.int64)
    if p2p.shape[0] == prev_ms.n_vertices:
        return p2p[tips]
    return None


def protrusion_report(target_folder, single_file=True):
    """Protrusions/protrusions_detail.csv (every protrusion of every frame: source, hull depth, heights, areas,
    confidence), protrusions_summary.csv (per frame counts by source), hull_candidates.csv and the plot."""
    d = ms_folders(target_folder)
    out = d["root"] / "Protrusions"; os.makedirs(out, exist_ok=True)
    seq = load_ms_sequence(target_folder)
    det, cands, summ = [], [], []
    for ms in tqdm(seq, desc="Protrusions report", leave=False):
        pr = getattr(ms, "protrusions", None)
        if pr is None or not len(pr):
            continue
        det.append(pr.assign(Time_Step=ms.frame))
        hc = getattr(ms, "hull_candidates", None)
        if hc is not None and len(hc):
            cands.append(hc.assign(Time_Step=ms.frame))
        summ.append({"Time_Step": ms.frame, "n_protrusions": len(pr), "n_ms_only": int((pr.source == "ms").sum()),
                     "n_hull_only": int((pr.source == "hull").sum()), "n_both": int((pr.source == "ms+hull").sum()),
                     "n_temporal": int(pr.temporal_support.sum()), "mean_confidence": float(pr.confidence.mean()),
                     "mean_core_area_rel": float(pr.core_area_rel.mean()), "total_core_area_rel": float(pr.core_area_rel.sum()),
                     "mean_core_height_rel": float(pr.core_height_rel.mean()), "n_ms_pruned": ms.meta.get("n_ms_pruned", 0)})
    if not summ:
        if single_file:
            print("No convex-hull protrusion results (fusion disabled?).")
        return None
    pd.concat(det, ignore_index=True).to_csv(out / "protrusions_detail.csv", index=False)
    if cands:
        pd.concat(cands, ignore_index=True).to_csv(out / "hull_candidates.csv", index=False)
    df = pd.DataFrame(summ)
    from PynamicMesh.core.dyn_common import add_time_columns
    df = add_time_columns(df, target_folder, frame_col="Time_Step")
    df.to_csv(out / "protrusions_summary.csv", index=False)
    X_ = df.Time if "Time" in df else df.Time_Step
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.stackplot(X_, df.n_both, df.n_ms_only, df.n_hull_only,
                 labels=["MS + hull (both)", "MS only (scalar field)", "hull only (recovered)"],
                 colors=["tab:purple", CP_COLORS["maximum"], "tab:cyan"], alpha=0.8)
    ax.plot(X_, df.n_temporal, color="k", linestyle=":", marker=".", label="accepted by temporal support")
    ax.set_ylabel("protrusions"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Protrusion detection: Morse–Smale maxima fused with convex-hull depth minima", fontweight="bold")
    ax = axes[1]
    ax.plot(X_, df.total_core_area_rel, marker="o", color="tab:green", label="protrusion cores / surface area")
    ax2 = ax.twinx(); ax2.plot(X_, df.mean_core_height_rel, marker="^", color="tab:brown", label="mean core height / eq. radius")
    ax.set_xlabel(f"time ({df.Time_Unit.iloc[0]})" if "Time" in df else "Time step"); ax.set_ylabel("area fraction"); ax2.set_ylabel("height")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.set_title("Protrusion cores (protrusion vs cell body)", fontweight="bold")
    fig.tight_layout(); fig.savefig(d["plots"] / "protrusion_detection_evolution.png", dpi=200); plt.close(fig)
    if single_file:
        print(f"Protrusion report saved to {out}")
    return str(out / "protrusions_summary.csv")


#  Star graph on the critical points

def ms_star_graph(ms: MSComplex, vertices, faces, graph_type="star") -> nx.Graph:
    """
    Graph whose nodes are the simplified critical points (attributes: pos, f_value, type,
    persistence, vertex, region_area, label) and a 'center' node at the area weighted centre
    of mass, connected to every critical point (star graph).  graph_type='star+adjacency'
    also connects the maxima whose regions share a boundary (weight = shared boundary length),
    which makes the graph metrics sensitive to the arrangement of the protrusions.
    Node keys are 'cp_<vertex>' / 'center' so that graph_time_analysis / graph_similarity
    (node_function_value reads 'f_value') work unchanged.
    Detection origin (convex-hull protrusion fusion): every critical-point node carries
      source        'ms' | 'hull' | 'ms+hull' (maxima; saddles 'ms' | 'hull'; minima 'ms')
      origin        'hull' for the points added by the convex hull (tips and their cols), else 'ms'
                    ('ms+hull' maxima are maxima of the scalar field confirmed by the hull -> 'ms')
      origin_label  '<type>_<origin>' (e.g. 'maximum_hull', 'saddle_ms'), used by graph_sim's origin-separated
                    comparison; 'label' stays the plain type, so the existing labelled metric is unchanged
      pair_extremum saddles: the maximum whose persistence pair they close (-1 otherwise)
    G.graph['has_hull_origin'] is True when at least one node comes from the hull.
    """
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    G = nx.Graph()
    f = ms.scalar
    center = ms.center
    G.add_node("center", pos=tuple(float(c) for c in center), f_value=float(np.average(f, weights=ms.vertex_area)),
               type="center", persistence=np.inf, vertex=-1, label="center", region_area=float(ms.vertex_area.sum()),
               bin=0, n_vertices=int(ms.n_vertices))
    has_src = "source" in ms.critical.columns
    n_hull = 0
    for _, r in ms.critical.iterrows():
        key = f"cp_{int(r.vertex)}"
        src = str(r.source) if has_src and isinstance(r.source, str) else "ms"
        origin = "hull" if src == "hull" else "ms"
        n_hull += origin == "hull"
        pe = int(r.pair_extremum) if (r.type == "saddle" and str(r.pair_kind) == "max") else -1
        G.add_node(key, pos=(float(r.x), float(r.y), float(r.z)), f_value=float(r.f), type=str(r.type),
                   persistence=float(r.persistence), vertex=int(r.vertex), label=str(r.type),
                   region_area=float(r.region_area) if np.isfinite(r.region_area) else 0.0,
                   bin=int(CP_TYPES.index(r.type)), n_vertices=int((ms.label_max == r.vertex).sum()) if r.type == "maximum" else 1,
                   source=src, origin=origin, origin_label=f"{r.type}_{origin}", pair_extremum=pe)
        G.add_edge("center", key, weight=float(np.linalg.norm(v[int(r.vertex)] - center)), kind="star")
    prot = getattr(ms, "protrusions", None)
    if prot is not None and len(prot):
        for _, pr in prot.iterrows():
            key = f"cp_{int(pr.protrusion_id)}"
            if key in G:
                G.nodes[key].update(source=str(pr.source), hull_depth=float(pr.tip_hull_depth_rel),
                                    hull_height=float(pr.hull_height_rel) if np.isfinite(pr.hull_height_rel) else 0.0,
                                    core_area=float(pr.core_area), core_height=float(pr.core_height_rel),
                                    confidence=float(pr.confidence))
        G.graph["n_protrusions"] = int(len(prot))
        G.graph["protrusion_sources"] = prot.source.value_counts().to_dict()
    G.nodes["center"].update(source="ms", origin="ms", origin_label="center", pair_extremum=-1)
    G.graph["has_hull_origin"] = bool(n_hull > 0)
    if graph_type == "star+adjacency" and ms.boundary_edges.size:
        E = ms.boundary_edges
        l = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1)
        a, b = ms.label_max[E[:, 0]], ms.label_max[E[:, 1]]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        keys = lo.astype(np.int64) * (ms.n_vertices + 1) + hi
        for k in np.unique(keys):
            sel = keys == k
            m1, m2 = int(lo[sel][0]), int(hi[sel][0])
            G.add_edge(f"cp_{m1}", f"cp_{m2}", weight=float(l[sel].sum()), kind="adjacency")
    return G


def create_ms_polydata(ms: MSComplex, vertices, faces) -> Tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    """(mesh with region/critical scalars, boundary polyline, critical point cloud) for viewers."""
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    F = _as_faces(faces)
    faces_pv = np.c_[np.full(len(F), 3), F].ravel()
    mesh = pv.PolyData(v, faces_pv)
    _, inv = np.unique(ms.label_max, return_inverse=True)
    mesh.point_data["region"] = inv
    mesh.point_data["region_max_vertex"] = ms.label_max
    mesh.point_data["cell"] = ms.cell
    mesh.point_data["scalar"] = ms.scalar
    mesh.point_data["cp_type_raw"] = ms.cp_type_raw
    if getattr(ms, "hull_depth", None) is not None:
        mesh.point_data["hull_depth"] = ms.hull_depth
        mesh.point_data["protrusion_core"] = ms.core_label
        code = {"ms": 1, "hull": 2, "ms+hull": 3}
        src = dict(zip(ms.protrusions.protrusion_id, ms.protrusions.source)) if ms.protrusions is not None else {}
        lut = {int(m): code.get(src.get(int(m), "ms"), 1) for m in ms.maxima}
        mesh.point_data["protrusion_source"] = np.array([lut.get(int(l), 0) for l in ms.label_max])
    bnd = pv.PolyData(v)
    if ms.boundary_edges.size:
        bnd.lines = np.c_[np.full(len(ms.boundary_edges), 2), ms.boundary_edges].ravel()
    # positions from the vertex indices of the *given* mesh (never from stored coordinates), so the
    # points lie on the surface whatever alignment/transform the loader applies
    cps = pv.PolyData(v[ms.critical.vertex.to_numpy(dtype=np.int64)]) if len(ms.critical) else pv.PolyData()
    if len(ms.critical):
        cps.point_data["vertex"] = ms.critical.vertex.to_numpy(dtype=np.int64)
        cps.point_data["type_code"] = np.array([CP_TYPES.index(t) for t in ms.critical["type"]])
        cps.point_data["persistence"] = ms.critical["persistence_rel"].to_numpy()
        cps.point_data["f_value"] = ms.critical["f"].to_numpy()
    return mesh, bnd, cps



def create_hull_polydata(ms: MSComplex, vertices, faces) -> pv.PolyData:
    """Convex hull used by the protrusion detection of ``ms`` (built on the same Taubin-smoothed geometry),
    as a triangulated pyvista surface, for viewers. Empty PolyData if the complex has no hull results."""
    if getattr(ms, "hull_depth", None) is None:
        return pv.PolyData()
    from scipy.spatial import ConvexHull
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    F = _as_faces(faces)
    it = int(ms.meta.get("protrusion_params", {}).get("smooth_iters", PROTRUSION_DEFAULTS["smooth_iters"]))
    vh = taubin_smooth(v, _neighbour_structure(v, F), it)
    try:
        hull = ConvexHull(vh)
    except Exception:  # noqa: BLE001
        hull = ConvexHull(vh, qhull_options="QJ")
    return pv.PolyData(vh, np.c_[np.full(len(hull.simplices), 3), hull.simplices].ravel()).clean()


#  Per-frame step used by the pipeline

def _frame_number(path):
    m = re.search(r'T(\d+)', Path(path).stem)
    return int(m.group(1)) if m else float('inf')


def ms_folders(target_folder):
    root = Path(target_folder) / 'MSComplexAnalysis'
    d = {"root": root, "complex": root / 'MS_Complex', "graphs": root / 'MS_Graphs',
         "tracking": root / 'Region_tracking', "plots": root / 'plots'}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def compute_MS(vertices, faces, i, target_folder, scalar_field=None, scalar_method="dist_centroid",
               scalar_kwargs=None, persistence=0.05, min_region_area=0.0, build_graph=False,
               graph_type="star", trimesh_obj=None, prev_vertices=None, p2p=None, protrusion_params=None,
               prev_faces=None, prev_ms=None, use_disk_prev=True):
    """
    Morse–Smale complex of frame ``i`` saved under <target>/MSComplexAnalysis/MS_Complex/MS_T####.pkl
    (+ Scalar_T####.npy, Labels_T####.npy) and, if build_graph, the critical-point graph under
    MS_Graphs/MSGraph_T####.pkl (readable by graph_time_analysis / graph_similarity).
    ``scalar_field`` may be given (e.g. the Reeb field of the same frame); otherwise it is computed
    with get_scalar_field(method=scalar_method, **scalar_kwargs).
    Protrusions: the maxima of the field are fused with the convex-hull depth minima (Huang et al. 2024;
    parameters: protrusion_params, else configure_protrusion_detection(), else PROTRUSION_DEFAULTS). With the
    previous frame's complex (``prev_ms``, or MS_T{i-1}.pkl on disk when use_disk_prev) and ``p2p`` (current ->
    previous, one entry per current vertex, as pipelines.compute_FM's p2p_zo; previous -> current also accepted),
    weak hull candidates are confirmed by the protrusions of the previous frame (temporal hysteresis).
    Extra files: HullDepth_T####.npy, Core_T####.npy (protrusion cores, -1 = body).
    """
    d = ms_folders(target_folder)
    v = np.asarray(to_cpu(vertices), dtype=np.float64)
    F = _as_faces(faces)
    if scalar_field is None:
        scalar_field = get_scalar_field(v, F, method=scalar_method, prev_vertices=prev_vertices, p2p=p2p,
                                        trimesh_obj=trimesh_obj, **(scalar_kwargs or {}))
    cfg = protrusion_config(protrusion_params)
    prev_tips = None
    if cfg["enabled"] and cfg["temporal"] and p2p is not None and i > 0:
        prev_file = d["complex"] / f'MS_T{i - 1:04d}.pkl'
        if prev_ms is None and use_disk_prev and prev_file.exists():
            try:
                prev_ms = MSComplex.from_pickle(prev_file)
            except Exception:  # noqa: BLE001
                prev_ms = None
        if prev_ms is not None:
            try:
                prev_tips = _map_previous_tips(prev_ms, p2p, v.shape[0], prev_vertices, prev_faces)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"temporal protrusion support unavailable for frame {i} ({exc})")
    ms = compute_ms_complex(v, F, scalar_field, persistence=persistence, frame=i, min_region_area=min_region_area,
                            protrusion_params=cfg, prev_tips=prev_tips)
    ms.meta["scalar_method"] = scalar_method
    ms.meta["temporal_tips"] = int(len(prev_tips)) if prev_tips is not None else 0
    ms.to_pickle(d["complex"] / f'MS_T{i:04d}.pkl')
    np.save(d["complex"] / f'Scalar_T{i:04d}.npy', np.asarray(scalar_field))
    np.save(d["complex"] / f'Labels_T{i:04d}.npy', ms.label_max)
    if ms.hull_depth is not None:
        np.save(d["complex"] / f'HullDepth_T{i:04d}.npy', ms.hull_depth)
        np.save(d["complex"] / f'Core_T{i:04d}.npy', ms.core_label)
    if build_graph:
        G = ms_star_graph(ms, v, F, graph_type=graph_type)
        with open(d["graphs"] / f'MSGraph_T{i:04d}.pkl', 'wb') as fh:
            pickle.dump(G, fh)
    return ms


def load_ms_sequence(target_folder) -> List[MSComplex]:
    d = ms_folders(target_folder)
    files = sorted(d["complex"].glob('MS_T*.pkl'), key=_frame_number)
    return [MSComplex.from_pickle(f) for f in tqdm(files, desc="Loading MS complexes", leave=False)]



#  Critical point statistics over the sequence

def critical_points_report(target_folder, single_file=True):
    """critical_points.csv (per frame counts, raw counts, Euler check, persistence statistics),
    critical_points_detail.csv (every critical point of every frame) and the evolution plot."""
    d = ms_folders(target_folder)
    seq = load_ms_sequence(target_folder)
    if not seq:
        print("No Morse–Smale complexes found.")
        return None
    rows, detail = [], []
    for ms in tqdm(seq, desc="Critical points report", leave=False):
        c = ms.counts()
        crit = ms.critical
        rows.append({"Time_Step": ms.frame, **c, "n_critical": sum(c.values()),
                     "euler_simplified": c["n_min"] - c["n_saddle"] + c["n_max"],
                     "n_max_raw": ms.meta["n_max_raw"], "n_min_raw": ms.meta["n_min_raw"],
                     "n_saddle_raw": ms.meta["n_saddle_raw"], "euler_raw": ms.meta["euler_raw"],
                     # Morse–Euler check against the surface (#min − #saddle + #max = χ = V − E + F)
                     "surface_euler": ms.meta.get("surface_euler"), "surface_genus": ms.meta.get("surface_genus"),
                     "euler_consistent": (None if ms.meta.get("surface_euler") is None
                                          else bool(c["n_min"] - c["n_saddle"] + c["n_max"] == ms.meta["surface_euler"])),
                     "n_regions": len(ms.maxima), "n_ms_cells": ms.meta["n_cells"],
                     "mean_region_area": float(np.mean(list(ms.region_areas().values()))) if len(ms.maxima) else 0.0,
                     "max_persistence_rel_mean": float(crit.loc[(crit.type == "maximum") & np.isfinite(crit.persistence_rel), "persistence_rel"].mean()),
                     "min_persistence_rel_mean": float(crit.loc[(crit.type == "minimum") & np.isfinite(crit.persistence_rel), "persistence_rel"].mean()),
                     "persistence_threshold_rel": ms.persistence_threshold / ms.meta["field_range"],
                     "n_boundary_edges": int(len(ms.boundary_edges)),
                     "n_max_ms_field": int(ms.meta.get("n_max_ms", c["n_max"])),
                     "n_protrusions_hull_only": int(ms.meta.get("n_protrusions_hull_only", 0)),
                     "n_protrusions_both": int(ms.meta.get("n_protrusions_both", 0))})
        dd = crit.copy()
        dd.insert(0, "Time_Step", ms.frame)
        detail.append(dd)
    df = pd.DataFrame(rows)
    from PynamicMesh.core.dyn_common import add_time_columns       # known frame times: Time column + time axis
    df = add_time_columns(df, target_folder, frame_col="Time_Step")
    df.to_csv(d["root"] / 'critical_points.csv', index=False)
    X_ = df.Time if "Time" in df else df.Time_Step
    pd.concat(detail, ignore_index=True).to_csv(d["root"] / 'critical_points_detail.csv', index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    ax = axes[0]
    for col, color, mk in (("n_max", CP_COLORS["maximum"], "^"), ("n_min", CP_COLORS["minimum"], "v"),
                           ("n_saddle", CP_COLORS["saddle"], "x"), ("n_critical", "k", "o")):
        ax.plot(X_, df[col], marker=mk, color=color, linewidth=2, label=col)
    ax.set_title("Simplified critical points (Morse–Smale complex)", fontweight="bold")
    ax.set_ylabel("count"); ax.legend(ncol=4)
    ax = axes[1]
    for col, color, mk in (("n_max_raw", CP_COLORS["maximum"], "^"), ("n_min_raw", CP_COLORS["minimum"], "v"),
                           ("n_saddle_raw", CP_COLORS["saddle"], "x")):
        ax.plot(X_, df[col], marker=mk, color=color, linewidth=1.5, alpha=0.8, label=col)
    ax.plot(X_, df.euler_raw, color="grey", linestyle="--", label="Euler #min-#sad+#max")
    ax.set_title("Raw PL critical points (before persistence simplification)", fontweight="bold")
    ax.set_ylabel("count"); ax.legend(ncol=4)
    ax = axes[2]
    ax.plot(X_, df.n_regions, marker="s", color="tab:purple", label="protrusion regions")
    ax.plot(X_, df.n_ms_cells, marker="d", color="tab:brown", label="Morse–Smale cells")
    ax2 = ax.twinx()
    ax2.plot(X_, df.max_persistence_rel_mean, color=CP_COLORS["maximum"], linestyle=":", label="mean persistence of maxima")
    ax2.set_ylabel("relative persistence")
    ax.set_title("Segmentation size", fontweight="bold"); ax.set_ylabel("count"); ax.set_xlabel(f"time ({df.Time_Unit.iloc[0]})" if "Time" in df else "Time step")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    out = d["plots"] / 'critical_points_evolution.png'
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    if single_file:
        print(f"Critical point report saved to {d['root'] / 'critical_points.csv'} and {out}")
    protrusion_report(target_folder, single_file=single_file)
    return str(d["root"] / 'critical_points.csv')


#  Region tracking through the functional map

FATE_MEANING = {
    ("maximum", "maximum"): "protrusion persists",
    ("maximum", "saddle"): "protrusion absorbed: peak became the pass to a neighbouring protrusion",
    ("maximum", "minimum"): "protrusion inverted: local contraction / invagination",
    ("maximum", "regular"): "protrusion flattened out",
    ("minimum", "minimum"): "indentation persists",
    ("minimum", "saddle"): "indentation opened laterally: pit became a pass",
    ("minimum", "maximum"): "indentation inverted: local bulge / new protrusion",
    ("minimum", "regular"): "indentation filled",
    ("saddle", "saddle"): "ridge / pass persists",
    ("saddle", "maximum"): "ridge rose into a new protrusion (region split)",
    ("saddle", "minimum"): "pass sank into an indentation",
    ("saddle", "regular"): "pass smoothed out (regions merged)",
}


def forward_p2p_from_FM(FM_12, evects1, evects2):
    """Point-to-point map mesh1 -> mesh2 from C (k2 x k1): nearest neighbour of Φ1 Cᵀ in Φ2."""
    from PynamicMesh.core.accel import knn               # exact; GPU(s) when available, cKDTree otherwise
    k2, k1 = FM_12.shape
    emb1 = np.asarray(evects1)[:, :k1] @ np.asarray(FM_12).T
    emb2 = np.asarray(evects2)[:, :k2]
    return knn(emb2, emb1, k=1)


def soft_region_transport(FM_12, mesh1, mesh2, label1, ids1, area1=None):
    """
    Spectral (functional map) transport of the region indicator functions:
        a_R = Φ1ᵀ A1 1_R ,  b_R = C a_R ,  g_R = Φ2 b_R
    g_R (n2 x |ids1|) is a soft membership of every vertex of mesh2 to every region of mesh1.
    Returns (soft_label with the region ids of mesh1, confidence in [0,1], memberships).
    """
    k2, k1 = FM_12.shape
    Phi1 = np.asarray(mesh1.eigenvectors)[:, :k1]
    Phi2 = np.asarray(mesh2.eigenvectors)[:, :k2]
    n1 = Phi1.shape[0]
    if area1 is None:
        A = getattr(mesh1, "A", None)
        try:
            area1 = np.asarray(A.diagonal()).ravel() if A is not None else None
        except Exception:
            area1 = None
        if area1 is None or area1.shape[0] != n1:
            area1, _ = _vertex_areas(mesh1.vertices, mesh1.faces)
    ind = np.zeros((n1, len(ids1)))
    for c, r in enumerate(ids1):
        ind[label1 == r, c] = 1.0
    a = Phi1.T @ (area1[:, None] * ind)
    g = Phi2 @ (np.asarray(FM_12) @ a)
    g = np.maximum(g, 0.0)
    tot = g.sum(axis=1)
    conf = np.where(tot > 1e-12, g.max(axis=1) / np.maximum(tot, 1e-12), 0.0)
    soft = np.asarray(ids1)[np.argmax(g, axis=1)]
    return soft, conf, g


def _majority_filter(labels, adj, iters=1):
    """Each vertex takes the most frequent label of its closed 1-ring (ties keep the own label), `iters` times.
    Removes the isolated speckle of a point-to-point transport without moving real region boundaries much."""
    lab = np.asarray(labels, dtype=np.int64).copy()
    n = lab.shape[0]
    if iters <= 0 or n == 0:
        return lab
    coo = adj.tocoo()
    i = np.r_[coo.row, np.arange(n)].astype(np.int64); j = np.r_[coo.col, np.arange(n)].astype(np.int64)
    base = int(lab.max()) + 2
    for _ in range(int(iters)):
        key = i * base + lab[j]
        uk, cnt = np.unique(key, return_counts=True)
        ui = uk // base; ul = uk % base
        order = np.lexsort((cnt, ui))
        last = np.r_[ui[order][1:] != ui[order][:-1], True]
        bi, bl, bc = ui[order][last], ul[order][last], cnt[order][last]
        own_c = cnt[np.searchsorted(uk, np.arange(n) * base + lab)]
        best_c = np.zeros(n, dtype=np.int64); best_c[bi] = bc
        best_l = lab.copy(); best_l[bi] = bl
        new = np.where(own_c >= best_c, lab, best_l)
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def map_scatter(verts_prev, p2p_21, adj_curr):
    """Per-vertex inconsistency of a point-to-point map (current -> previous): distance between the image of a
    vertex and the mean image of its neighbours, on the previous mesh. ~0 for a smooth map; the size of the
    jumps for a noisy one."""
    vp = np.asarray(verts_prev, dtype=np.float64)
    X = vp[np.asarray(p2p_21, dtype=np.int64)]
    W = adj_curr.copy(); W.data = np.ones_like(W.data)
    deg = np.maximum(np.asarray(W.sum(axis=1)).ravel(), 1.0)
    return np.linalg.norm(X - (W @ X) / deg[:, None], axis=1)


def expected_overlap(radius, eps):
    """Overlap of a disk of radius r with itself shifted by eps (model of a perfectly tracked region whose
    transport is displaced / blurred by eps): returns (IoU, fraction of the region kept) in [0, 1]."""
    r = np.maximum(np.asarray(radius, dtype=np.float64), 1e-300)
    u = np.clip(eps / (2.0 * r), 0.0, 1.0)
    inter = (2.0 / np.pi) * (np.arccos(u) - u * np.sqrt(1.0 - u * u))        # intersection / disk area
    return inter / np.maximum(2.0 - inter, 1e-300), inter


class RegionTracker:
    """
    Tracks the Morse–Smale regions and critical points between consecutive frames.

    Size-adaptive matching (adaptive=True, default).  The IoU a perfectly tracked region can reach depends on its
    size: its transport is displaced / blurred by the map error and by the motion of the protrusion, ε, and a
    region of radius r then overlaps itself at most IoU_exp(r) = IoU of a disk shifted by ε (≈1 for large regions,
    0.1-0.4 for small blebs), so a fixed threshold rejects exactly the small protrusions. Per transition:
      * the hard transport is cleaned by a majority filter on the 1-ring (label_smoothing iterations);
      * ε = sqrt(scatter² + shift² + (h/2)²): scatter = median inconsistency of the p2p map (map_scatter), shift =
        median displacement between the transported previous tip and the current tip over the pairs matched
        confidently in a first pass (protrusion motion + map misalignment), h = mean edge length
        (or map_error, fixed, as a fraction of sqrt(area));
      * pair (a, b) is accepted if IoU >= max(min_iou, iou_threshold · IoU_exp(min(r_a, r_b))); the Hungarian
        assignment maximises IoU / IoU_exp, so small and large regions compete on the same scale;
      * split / merge dominance: dominance · (expected kept fraction of the region concerned);
      * tip rescue: an unmatched previous region whose tip lands in an unmatched current region that it overlaps
        is matched to it ('tip' match).
    adaptive=False gives the previous fixed-threshold behaviour.

    Correspondence.  The previous segmentation is transported to the current mesh with the
    point-to-point map of the functional map (hard transport ``label_prev[p2p_21]``) and, when
    the eigenbases are available, with the spectral transport of the region indicator functions
    (soft transport, Φ2 C Φ1ᵀ A1 1_R).  Regions are matched by area-weighted IoU with the
    Hungarian algorithm; the remaining relations use *dominant overlaps*: a current region whose
    dominant parent is p (and that is not matched elsewhere) is a split child of p, a previous
    region whose dominant child is c (and that is not matched elsewhere) is merged/absorbed into
    c; regions without any dominant relation are births / deaths.  Track ids give every
    protrusion a persistent identity (lineage).

    Fate of a critical point (per transition).  The image of the critical vertex on the current
    mesh is the forward map of the functional map (else the pre-image of p2p_21).
      * maximum : 'maximum' if the image lies inside the current region matched to its own
                  region (the peak may wander on a flat top without any topological change);
      * minimum : 'minimum' if the minimum of the ascending manifold containing the image is
                  within ``fate_radius``;
      * saddle  : 'saddle' if the two regions it separates are both matched and still adjacent;
      otherwise the nearest simplified critical point within ``fate_radius`` along the surface
      (fraction of sqrt(total area)), else 'regular'.
    Trajectories (multi-frame).  Every critical point is followed frame after frame through the
    chain of maps, also after it has become regular (ghost, up to ``ghost_horizon`` frames), so
    that slow processes such as protrusion -> flat -> indentation are reported as inversions.
    Growth of a protrusion is the displacement of the image of its maximum along the previous
    surface normal (> 0 outward), complemented by the change of persistence and region area.
    """

    def __init__(self, target_folder, iou_threshold=0.25, dominance=0.5, fate_radius=0.04,
                 growth_tolerance=0.002, ghost_horizon=6, adaptive=True, min_iou=0.05, label_smoothing=1,
                 tip_rescue=True, map_error=None, max_map_error=0.15):
        self.d = ms_folders(target_folder)
        self.iou_threshold = iou_threshold
        self.dominance = dominance
        self.adaptive = bool(adaptive)
        self.min_iou = float(min_iou)                 # absolute floor of the adaptive IoU threshold
        self.label_smoothing = int(label_smoothing)   # majority-filter iterations of the hard transport
        self.tip_rescue = bool(tip_rescue)
        self.map_error = map_error                    # fixed ε / sqrt(area) (None = estimated per transition)
        self.max_map_error = float(max_map_error)     # cap of the estimated ε / sqrt(area)
        self.fate_radius = fate_radius
        self.growth_tolerance = growth_tolerance
        self.ghost_horizon = ghost_horizon
        self.track_of: Dict[Tuple[int, int], int] = {}      # (frame, max vertex) -> track id
        self.next_track = 0
        self.lineage: List[Dict] = []
        self.fates: List[Dict] = []
        self.summary: List[Dict] = []
        self.traj: List[Dict] = []                           # critical point trajectories
        self.traj_by_cp: Dict[Tuple[int, int], int] = {}     # (frame, cp vertex) -> trajectory index

    # -- helpers -----------------------------------------------------------
    def _tid(self, frame, m):
        key = (int(frame), int(m))
        if key not in self.track_of:
            self.track_of[key] = self.next_track
            self.next_track += 1
        return self.track_of[key]

    @staticmethod
    def _image_vertex(v, p2p_21, p2p_12, adj_prev):
        """Image on the current mesh of previous vertex v: pre-image of the refined p2p map when it
        exists (exact inverse), else the forward map of the functional map, else ring expansion."""
        pre = np.where(p2p_21 == v)[0]
        if pre.size:
            if p2p_12 is not None and p2p_12[v] in pre:
                return int(p2p_12[v]), 0
            return int(pre[0]), 0
        if p2p_12 is not None:
            return int(p2p_12[v]), 0
        ring = {int(v)}
        frontier = {int(v)}
        for r in range(1, 4):
            nxt = set()
            for u in frontier:
                nxt.update(adj_prev.indices[adj_prev.indptr[u]:adj_prev.indptr[u + 1]].tolist())
            frontier = nxt - ring
            ring |= frontier
            pre = np.where(np.isin(p2p_21, list(frontier)))[0]
            if pre.size:
                return int(pre[0]), r
        return -1, -1

    def _classify_image(self, j, type_prev, v, ms_curr, adj_curr, radius, matches, cp_idx, cp_types, cp_pers,
                        adjacent, eps):
        """Type of the current frame at the image vertex j of a previous critical point (see class doc).
        eps: field tolerance (half the persistence threshold) for the 'at the top / at the bottom' tests."""
        f = ms_curr.scalar
        if type_prev == "maximum" and v in matches and int(ms_curr.label_max[j]) == matches[v]:
            m = matches[v]
            return "maximum", m, 0.0, float(cp_pers.get(m, np.nan))
        if type_prev == "saddle" and adjacent is not None:
            return "saddle", int(adjacent), 0.0, float(cp_pers.get(int(adjacent), np.nan))
        # at the top of a protrusion / at the bottom of a pit (value criterion, robust to wandering extrema)
        m_up, m_dn = int(ms_curr.label_max[j]), int(ms_curr.label_min[j])
        top, bottom = f[m_up] - f[j] <= eps, f[j] - f[m_dn] <= eps
        if top and not bottom:
            return "maximum", m_up, 0.0, float(cp_pers.get(m_up, np.nan))
        if bottom and not top:
            return "minimum", m_dn, 0.0, float(cp_pers.get(m_dn, np.nan))
        dist = dijkstra(adj_curr, indices=j, limit=radius, min_only=True)
        cand = dist[cp_idx]
        if np.isfinite(cand).any():
            k = int(np.argmin(cand))
            return str(cp_types[k]), int(cp_idx[k]), float(cand[k]), float(cp_pers.get(int(cp_idx[k]), np.nan))
        return "regular", -1, np.nan, 0.0

    # -- main --------------------------------------------------------------
    def track_pair(self, ms_prev: MSComplex, ms_curr: MSComplex, verts_prev, faces_prev, verts_curr, faces_curr,
                   p2p_21, FM_12=None, mesh_prev=None, mesh_curr=None, save=True) -> Dict:
        vp = np.asarray(to_cpu(verts_prev), float); vc = np.asarray(to_cpu(verts_curr), float)
        Fp = _as_faces(faces_prev); Fc = _as_faces(faces_curr)
        p2p_21 = np.asarray(to_cpu(p2p_21), dtype=np.int64).ravel()
        if p2p_21.shape[0] != vc.shape[0]:
            raise ValueError("p2p_21 must have one entry per vertex of the current mesh (mesh2 -> mesh1 map)")
        fp, fc = ms_prev.frame, ms_curr.frame
        Ac, Ap = ms_curr.vertex_area, ms_prev.vertex_area

        # ---- transport of the previous segmentation -----------------------------------
        adj_curr = _neighbour_structure(vc, Fc)
        hard_raw = ms_prev.label_max[p2p_21]
        hard = _majority_filter(hard_raw, adj_curr, self.label_smoothing) if self.adaptive else hard_raw
        soft, conf, p2p_12 = None, None, None
        if FM_12 is not None and mesh_prev is not None and mesh_curr is not None \
                and getattr(mesh_prev, "eigenvectors", None) is not None and getattr(mesh_curr, "eigenvectors", None) is not None:
            try:
                soft, conf, _ = soft_region_transport(FM_12, mesh_prev, mesh_curr, ms_prev.label_max, ms_prev.maxima, Ap)
                p2p_12 = forward_p2p_from_FM(FM_12, mesh_prev.eigenvectors, mesh_curr.eigenvectors)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"soft transport unavailable ({exc}); using the hard p2p transport only")
        mapped = hard

        # ---- overlaps between mapped previous regions and current regions ------------------
        ids_p, ids_c = [int(m) for m in ms_prev.maxima], [int(m) for m in ms_curr.maxima]
        ip = {m: k for k, m in enumerate(ids_p)}; ic = {m: k for k, m in enumerate(ids_c)}
        O = np.zeros((len(ids_p), len(ids_c)))
        rp = np.array([ip.get(int(m), -1) for m in mapped]); rc = np.array([ic[int(m)] for m in ms_curr.label_max])
        ok = rp >= 0
        np.add.at(O, (rp[ok], rc[ok]), Ac[ok])
        area_mapped = O.sum(axis=1); area_curr = O.sum(axis=0)
        IoU = O / np.maximum(area_mapped[:, None] + area_curr[None, :] - O, 1e-300)
        frac_prev = O / np.maximum(area_mapped[:, None], 1e-300)      # share of the previous region going to each current region
        frac_curr = O / np.maximum(area_curr[None, :], 1e-300)        # share of the current region coming from each previous region
        scale = float(np.sqrt(Ac.sum()))
        matches: Dict[int, int] = {}
        match_kind: Dict[int, str] = {}
        thr = np.full(IoU.shape, float(self.iou_threshold))          # IoU threshold of every pair
        e_iou = np.ones(IoU.shape); keep_p = np.ones(len(ids_p)); keep_c = np.ones(len(ids_c))
        map_eps = np.nan; scatter = np.nan; shift = np.nan
        if O.size and self.adaptive:
            r_p = np.sqrt(np.maximum(area_mapped, 1e-300) / np.pi); r_c = np.sqrt(np.maximum(area_curr, 1e-300) / np.pi)
            h = float(np.mean(np.linalg.norm(vc[Fc[:, 0]] - vc[Fc[:, 1]], axis=1)))
            if self.map_error is not None:
                map_eps = float(self.map_error) * scale
            else:
                scatter = float(np.median(map_scatter(vp, p2p_21, adj_curr)))
                # motion of the protrusions + map misalignment: displacement between the image of the previous
                # tip (through the p2p map) and the current tip, over the pairs confidently matched in a first pass
                # (fixed threshold); the median is robust to basins whose maximum jumped to another bleb
                first = IoU >= self.iou_threshold
                shifts = []
                if first.any():
                    adj_p0 = _neighbour_structure(vp, Fp)
                    ra, cb = linear_sum_assignment(-np.where(first, IoU, 0.0))
                    for a, b in zip(ra, cb):
                        if first[a, b]:
                            j, _ = self._image_vertex(ids_p[a], p2p_21, None, adj_p0)
                            if j >= 0:
                                shifts.append(float(np.linalg.norm(vc[j] - vc[ids_c[b]])))
                shift = float(np.median(shifts)) if shifts else 0.0
                map_eps = float(np.sqrt(scatter ** 2 + shift ** 2 + (0.5 * h) ** 2))
            map_eps = min(map_eps, self.max_map_error * scale)
            r_min = np.minimum(r_p[:, None], r_c[None, :])
            e_iou, _ = expected_overlap(r_min, map_eps)
            _, keep_p = expected_overlap(r_p, map_eps); _, keep_c = expected_overlap(r_c, map_eps)
            thr = np.maximum(self.min_iou, self.iou_threshold * e_iou)
            score = np.where((IoU >= thr) & (O > 0), IoU / np.maximum(e_iou, 1e-3), 0.0)
            ra, cb = linear_sum_assignment(-score)
            for a, b in zip(ra, cb):
                if score[a, b] > 0:
                    matches[ids_p[a]] = ids_c[b]; match_kind[ids_p[a]] = "iou"
        elif O.size:
            r, c = linear_sum_assignment(-IoU)
            for a, b in zip(r, c):
                if IoU[a, b] >= self.iou_threshold:
                    matches[ids_p[a]] = ids_c[b]; match_kind[ids_p[a]] = "iou"
        if self.tip_rescue and self.adaptive and O.size:
            adj_prev_ = _neighbour_structure(vp, Fp)
            taken = set(matches.values())
            for a, pm in enumerate(ids_p):
                if pm in matches:
                    continue
                j, _ = self._image_vertex(pm, p2p_21, None, adj_prev_)
                if j < 0:
                    continue
                cm = int(ms_curr.label_max[j]); b = ic.get(cm, -1)
                if b >= 0 and cm not in taken and O[a, b] > 0 and IoU[a, b] >= 0.5 * self.min_iou:
                    matches[pm] = cm; match_kind[pm] = "tip"; taken.add(cm)
        dom_thr_p = self.dominance * (keep_p if self.adaptive else np.ones(len(ids_p)))
        dom_thr_c = self.dominance * (keep_c if self.adaptive else np.ones(len(ids_c)))
        inv_matches = {c_: p_ for p_, c_ in matches.items()}
        dom_parent = {ids_c[b]: ids_p[int(np.argmax(frac_curr[:, b]))] for b in range(len(ids_c))} if len(ids_p) else {}
        dom_child = {ids_p[a]: ids_c[int(np.argmax(frac_prev[a]))] for a in range(len(ids_p))} if len(ids_c) else {}

        # ---- events and lineage (dominant overlap tracking) ------------------------------
        areas_p = ms_prev.region_areas(); areas_c = ms_curr.region_areas()
        pers_p = dict(zip(ms_prev.critical.vertex, ms_prev.critical.persistence_rel))
        pers_c = dict(zip(ms_curr.critical.vertex, ms_curr.critical.persistence_rel))
        n_ev = {"continue": 0, "split": 0, "merge": 0, "birth": 0, "death": 0}
        src_p = dict(zip(ms_prev.protrusions.protrusion_id, ms_prev.protrusions.source)) if getattr(ms_prev, "protrusions", None) is not None else {}
        src_c = dict(zip(ms_curr.protrusions.protrusion_id, ms_curr.protrusions.source)) if getattr(ms_curr, "protrusions", None) is not None else {}

        def link(p, c_, ev, tid_prev):
            self.lineage.append({"Transition": f"T{fp} -> T{fc}", "Time_Step": fc, "track_id": tid_prev,
                                 "prev_max_vertex": p, "curr_max_vertex": c_,
                                 "curr_track_id": self.track_of.get((fc, c_), -1) if c_ >= 0 else -1,
                                 "event": ev, "matched": int(matches.get(p, -1) == c_ and c_ >= 0),
                                 "IoU": float(IoU[ip[p], ic[c_]]) if (c_ >= 0 and p >= 0) else 0.0,
                                 "overlap_fraction_prev": float(frac_prev[ip[p], ic[c_]]) if (c_ >= 0 and p >= 0) else 0.0,
                                 "overlap_fraction_curr": float(frac_curr[ip[p], ic[c_]]) if (c_ >= 0 and p >= 0) else 0.0,
                                 "area_prev": areas_p.get(p, 0.0), "area_curr": areas_c.get(c_, 0.0),
                                 "area_change_rel": (areas_c.get(c_, 0.0) - areas_p.get(p, 0.0)) / max(areas_p.get(p, 0.0), 1e-300) if p >= 0 else np.nan,
                                 "persistence_prev": pers_p.get(p, np.nan), "persistence_curr": pers_c.get(c_, np.nan),
                                 "source_prev": src_p.get(p, "ms" if p >= 0 else ""), "source_curr": src_c.get(c_, "ms" if c_ >= 0 else ""),
                                 "match_kind": match_kind.get(p, "") if (p >= 0 and matches.get(p, -1) == c_ and c_ >= 0) else "",
                                 "iou_threshold_used": float(thr[ip[p], ic[c_]]) if (c_ >= 0 and p >= 0) else np.nan,
                                 "expected_IoU": float(e_iou[ip[p], ic[c_]]) if (c_ >= 0 and p >= 0) else np.nan})

        # matched pairs continue their track; split children of a matched parent
        for p in ids_p:
            tid = self._tid(fp, p)
            if p in matches:
                c_ = matches[p]
                self.track_of[(fc, c_)] = tid
        for p in ids_p:
            tid = self._tid(fp, p)
            children = [c_ for c_ in ids_c if c_ not in inv_matches and dom_parent.get(c_) == p
                        and frac_curr[ip[p], ic[c_]] >= dom_thr_c[ic[c_]]]
            if p in matches:
                c_ = matches[p]
                parents_of_c = [q for q in ids_p if q != p and q not in matches and dom_child.get(q) == c_
                                and frac_prev[ip[q], ic[c_]] >= dom_thr_p[ip[q]]]
                ev = "split" if children else ("merge" if parents_of_c else "continue")
                n_ev[ev] += 1
                link(p, c_, ev, tid)
                for c2 in children:
                    self._tid(fc, c2)                       # new track for the split child
                    link(p, c2, "split", tid)
            else:
                c_ = dom_child.get(p, -1)
                if c_ >= 0 and frac_prev[ip[p], ic[c_]] >= dom_thr_p[ip[p]]:
                    if c_ not in inv_matches and dom_parent.get(c_) == p:
                        # unmatched pair that dominate each other (IoU below threshold): continue
                        self.track_of[(fc, c_)] = tid
                        n_ev["continue"] += 1; link(p, c_, "continue", tid)
                    else:
                        n_ev["merge"] += 1; link(p, c_, "merge", tid)      # absorbed into c_
                else:
                    n_ev["death"] += 1; link(p, -1, "death", tid)
        for c_ in ids_c:
            if (fc, c_) in self.track_of:
                continue
            p = dom_parent.get(c_, -1)
            if p >= 0 and frac_curr[ip[p], ic[c_]] >= dom_thr_c[ic[c_]] and p in matches:
                self._tid(fc, c_)                           # split child (already linked above if dominant)
                if not any(r_["curr_max_vertex"] == c_ and r_["Time_Step"] == fc for r_ in self.lineage):
                    n_ev["split"] += 1; link(p, c_, "split", self._tid(fp, p))
            else:
                tid = self._tid(fc, c_)
                n_ev["birth"] += 1
                link(-1, c_, "birth", tid)
                self.lineage[-1]["curr_track_id"] = tid

        # ---- fate of every critical point + trajectories ---------------------------------
        adj_prev = _neighbour_structure(vp, Fp)
        Np = _vertex_normals(vp, Fp)
        radius = self.fate_radius * scale
        frange_p = ms_prev.meta["field_range"]
        rank_p = _total_order(ms_prev.scalar) / max(len(ms_prev.scalar) - 1, 1)
        rank_c = _total_order(ms_curr.scalar) / max(len(ms_curr.scalar) - 1, 1)
        cp_curr = ms_curr.critical
        cp_idx = cp_curr.vertex.to_numpy(dtype=np.int64)
        cp_types = cp_curr.type.to_numpy()
        # passes between adjacent current regions: the MS saddle separating the pair if it exists,
        # otherwise the highest vertex of their common boundary
        sad_by_pair = {}
        if ms_curr.boundary_edges.size:
            E = ms_curr.boundary_edges
            la, lb = ms_curr.label_max[E[:, 0]], ms_curr.label_max[E[:, 1]]
            fe = np.maximum(ms_curr.scalar[E[:, 0]], ms_curr.scalar[E[:, 1]])
            top_v = np.where(ms_curr.scalar[E[:, 0]] >= ms_curr.scalar[E[:, 1]], E[:, 0], E[:, 1])
            keys = np.minimum(la, lb).astype(np.int64) * (ms_curr.n_vertices + 1) + np.maximum(la, lb)
            for k in np.unique(keys):
                sel = keys == k
                i_best = np.argmax(fe[sel])
                sad_by_pair[frozenset((int(la[sel][0]), int(lb[sel][0])))] = int(top_v[sel][i_best])
        for _, r in cp_curr[cp_curr.type == "saddle"].iterrows():
            if r.pair_kind == "max":
                sad_by_pair[frozenset((int(r.pair_extremum), int(r.pair_killer)))] = int(r.vertex)
        eps = 0.5 * ms_curr.persistence_threshold
        growth = []
        fate_counts: Dict[Tuple[str, str], int] = {}

        def classify(v, j, type_prev, adjacent=None):
            return self._classify_image(j, type_prev, v, ms_curr, adj_curr, radius, matches, cp_idx, cp_types, pers_c,
                                        adjacent, eps)

        for _, r in ms_prev.critical.iterrows():
            v = int(r.vertex)
            j, rings = self._image_vertex(v, p2p_21, p2p_12, adj_prev)
            adjacent = None
            if r.type == "saddle" and r.pair_kind == "max":
                a, b = matches.get(int(r.pair_extremum)), matches.get(int(r.pair_killer))
                if a is not None and b is not None and a != b:
                    adjacent = sad_by_pair.get(frozenset((a, b)))
            row = {"Transition": f"T{fp} -> T{fc}", "Time_Step": fc, "vertex_prev": v, "type_prev": r.type,
                   "f_prev": float(r.f), "persistence_prev": float(r.persistence_rel), "image_vertex": j, "image_rings": rings,
                   "track_id": self._tid(fp, v) if r.type == "maximum" else -1}
            if j < 0:
                t_next, near, dnear, pers_next = "lost", -1, np.nan, np.nan
                row.update(type_next=t_next, nearest_cp_vertex=near, nearest_cp_distance=dnear, same_region=0,
                           delta_f_rel=np.nan, delta_rank=np.nan, delta_persistence=np.nan, normal_displacement=np.nan,
                           tangential_displacement=np.nan, displacement=np.nan, growth="unknown", meaning="no correspondence")
            else:
                t_next, near, dnear, pers_next = classify(v, j, r.type, adjacent)
                disp = vc[j] - vp[v]
                nd = float(disp @ Np[v]); td = float(np.linalg.norm(disp - nd * Np[v]))
                same = int(r.type == "maximum" and matches.get(v, -2) == int(ms_curr.label_max[j]))
                df_rel = float((ms_curr.scalar[j] - r.f) / max(frange_p, 1e-300))
                grow = "growing" if nd > self.growth_tolerance * scale else ("retracting" if nd < -self.growth_tolerance * scale else "stable")
                meaning = FATE_MEANING.get((r.type, t_next), "")
                if r.type == "maximum" and t_next == "maximum":
                    meaning += f" ({grow})"
                row.update(type_next=t_next, nearest_cp_vertex=near, nearest_cp_distance=dnear / scale if np.isfinite(dnear) else np.nan,
                           same_region=same, delta_f_rel=df_rel, delta_rank=float(rank_c[j] - rank_p[v]),
                           delta_persistence=(pers_next - float(r.persistence_rel)) if (np.isfinite(pers_next) and np.isfinite(r.persistence_rel)) else np.nan,
                           normal_displacement=nd / scale, tangential_displacement=td / scale,
                           displacement=float(np.linalg.norm(disp)) / scale, growth=grow, meaning=meaning)
                if r.type == "maximum":
                    growth.append(nd / scale)
            fate_counts[(r.type, t_next)] = fate_counts.get((r.type, t_next), 0) + 1
            self.fates.append(row)
            # trajectory bookkeeping
            key = (fp, v)
            if key not in self.traj_by_cp:
                self.traj_by_cp[key] = len(self.traj)
                self.traj.append({"origin_frame": fp, "origin_vertex": v, "origin_type": r.type,
                                  "track_id": row["track_id"], "frames": [fp], "vertices": [v], "material": [v],
                                  "types": [r.type], "f": [float(r.f)], "normal_disp": [], "active": True, "ghost": 0})
            tr = self.traj[self.traj_by_cp[key]]
            jm, _ = self._image_vertex(tr["material"][-1], p2p_21, p2p_12, adj_prev)   # material point of the origin
            self._extend_traj(tr, fc, j, t_next, near, float(ms_curr.scalar[j]) if j >= 0 else np.nan,
                              row.get("normal_displacement", np.nan), jm)
        # ghosts: trajectories not attached to a critical point of the previous frame follow the
        # material point of their origin (so that a protrusion tip that flattens and then sinks into a
        # pit is recognised as an inversion at the same location)
        for tr in self.traj:
            if tr["active"] and tr["frames"][-1] == fp and (fp, tr["vertices"][-1]) not in self.traj_by_cp:
                j, _ = self._image_vertex(tr["material"][-1], p2p_21, p2p_12, adj_prev)
                if j < 0:
                    tr["active"] = False; continue
                t_next, near, _, _ = classify(-1, j, "regular")
                self._extend_traj(tr, fc, j, t_next, near, float(ms_curr.scalar[j]),
                                  float((vc[j] - vp[tr["material"][-1]]) @ Np[tr["material"][-1]]) / scale, j)

        growth = np.asarray(growth)
        self.summary.append({"Transition": f"T{fp} -> T{fc}", "Time_Step": fc, **{f"n_{k}": val for k, val in n_ev.items()},
                             "n_regions_prev": len(ids_p), "n_regions_curr": len(ids_c),
                             "mean_IoU_matched": float(np.mean([IoU[ip[a], ic[b]] for a, b in matches.items()])) if matches else np.nan,
                             "n_matched": len(matches), "n_matched_tip": int(sum(k == "tip" for k in match_kind.values())),
                             "matched_fraction_prev": len(matches) / max(len(ids_p), 1),
                             "mean_IoU_normalised": float(np.mean([IoU[ip[a], ic[b]] / max(e_iou[ip[a], ic[b]], 1e-3)
                                                                   for a, b in matches.items()])) if matches else np.nan,
                             "map_error_rel": map_eps / scale if np.isfinite(map_eps) else np.nan,
                             "map_scatter_rel": scatter / scale if np.isfinite(scatter) else np.nan,
                             "region_shift_rel": shift / scale if np.isfinite(shift) else np.nan,
                             "soft_confidence_mean": float(np.mean(conf)) if conf is not None else np.nan,
                             "hard_soft_agreement": float(np.mean(soft == hard)) if soft is not None else np.nan,
                             "max_kept": fate_counts.get(("maximum", "maximum"), 0),
                             "max_to_saddle": fate_counts.get(("maximum", "saddle"), 0),
                             "max_to_min": fate_counts.get(("maximum", "minimum"), 0),
                             "max_flattened": fate_counts.get(("maximum", "regular"), 0),
                             "min_kept": fate_counts.get(("minimum", "minimum"), 0),
                             "min_to_saddle": fate_counts.get(("minimum", "saddle"), 0),
                             "min_to_max": fate_counts.get(("minimum", "maximum"), 0),
                             "saddle_kept": fate_counts.get(("saddle", "saddle"), 0),
                             "saddle_to_max": fate_counts.get(("saddle", "maximum"), 0),
                             "saddle_to_min": fate_counts.get(("saddle", "minimum"), 0),
                             "fraction_growing": float(np.mean(growth > self.growth_tolerance * scale)) if growth.size else np.nan,
                             "fraction_retracting": float(np.mean(growth < -self.growth_tolerance * scale)) if growth.size else np.nan,
                             "mean_normal_growth": float(growth.mean()) if growth.size else np.nan})
        if save:
            rows_fc = [r_ for r_ in self.fates if r_["Time_Step"] == fc]
            np.savez_compressed(self.d["tracking"] / f'mapped_T{fp:04d}_T{fc:04d}.npz',
                                hard_label=hard, hard_label_raw=hard_raw, soft_label=soft if soft is not None else np.array([]),
                                match_kind=np.array([match_kind.get(k_, "") for k_ in matches.keys()]),
                                match_threshold=np.array([thr[ip[a], ic[b]] for a, b in matches.items()]),
                                map_error=np.array(map_eps / scale if np.isfinite(map_eps) else np.nan),
                                adaptive=np.array(self.adaptive),
                                soft_confidence=conf if conf is not None else np.array([]),
                                prev_ids=np.array(ids_p), curr_ids=np.array(ids_c), IoU=IoU,
                                match_prev=np.array(list(matches.keys()), dtype=np.int64),
                                match_curr=np.array(list(matches.values()), dtype=np.int64),
                                image_vertex=np.array([r_["image_vertex"] for r_ in rows_fc], dtype=np.int64),
                                cp_vertex_prev=np.array([r_["vertex_prev"] for r_ in rows_fc], dtype=np.int64),
                                cp_type_prev=np.array([r_["type_prev"] for r_ in rows_fc]),
                                cp_type_next=np.array([r_["type_next"] for r_ in rows_fc]),
                                cp_growth=np.array([r_["growth"] for r_ in rows_fc]))
        return {"matches": matches, "IoU": IoU, "events": n_ev, "hard": hard, "soft": soft, "confidence": conf}

    def _extend_traj(self, tr, fc, j, t_next, near, f_val, nd, jm):
        tr["frames"].append(fc); tr["vertices"].append(j if near < 0 else near); tr["types"].append(t_next)
        tr["material"].append(jm if jm >= 0 else j)
        tr["f"].append(f_val); tr["normal_disp"].append(nd)
        if t_next in CP_TYPES and near >= 0:
            tr["ghost"] = 0
            self.traj_by_cp.setdefault((fc, near), self.traj.index(tr))
        else:
            tr["ghost"] += 1
            if t_next == "lost" or tr["ghost"] > self.ghost_horizon:
                tr["active"] = False

    # -- outputs -----------------------------------------------------------
    @staticmethod
    def _long_fate(types):
        """Summary of a type sequence: persistent / inverted / absorbed / flattened / split."""
        t0 = types[0]
        later = [t for t in types[1:] if t not in ("lost",)]
        if not later:
            return "lost"
        opposite = {"maximum": "minimum", "minimum": "maximum"}.get(t0)
        if opposite and opposite in later:
            return f"inverted ({t0} -> {opposite})"
        if t0 in ("maximum", "minimum") and "saddle" in later:
            return f"absorbed ({t0} -> saddle)"
        if t0 == "saddle" and ("maximum" in later or "minimum" in later):
            return "saddle -> extremum (region split / new pit)"
        if later[-1] == t0:
            return "persistent" if all(t == t0 for t in later) else "persistent (intermittent)"
        if later[-1] == "regular":
            return "flattened"
        return f"{t0} -> {later[-1]}"

    def save(self):
        d = self.d["tracking"]
        lineage = pd.DataFrame(self.lineage); fates = pd.DataFrame(self.fates); summary = pd.DataFrame(self.summary)
        from PynamicMesh.core.dyn_common import add_time_columns, find_frame_times
        ft = find_frame_times(self.d["root"])                         # Results/<scene>/frame_times.csv, if any
        if ft is not None and len(summary) and "Time_Step" in summary:   # events per time unit (irregular intervals)
            summary = add_time_columns(summary, self.d["root"], frame_col="Time_Step",
                                       rate_cols=[c for c in summary.columns if c.startswith("n_") and not c.startswith("n_regions")])
        lineage.to_csv(d / 'region_lineage.csv', index=False)
        fates.to_csv(d / 'critical_point_fates.csv', index=False)
        summary.to_csv(d / 'tracking_summary.csv', index=False)
        if len(fates):
            tm = pd.crosstab(fates.type_prev, fates.type_next).reindex(index=CP_TYPES, columns=FATE_LABELS, fill_value=0)
            tm.to_csv(d / 'fate_transition_counts.csv')
        if len(lineage):
            tracks = []
            for tid, g in lineage[lineage.curr_max_vertex >= 0].groupby("curr_track_id"):
                nd = fates[(fates.type_prev == "maximum") & (fates.track_id == tid)].normal_displacement
                tracks.append({"track_id": int(tid), "first_frame": int(g.Time_Step.min() - (0 if (g.event == "birth").any() else 1)),
                               "last_frame": int(g.Time_Step.max()), "n_frames": int(len(g)),
                               "mean_area": float(g.area_curr.mean()), "mean_IoU": float(g.IoU.mean()),
                               "n_splits": int((g.event == "split").sum()), "n_merges": int((g.event == "merge").sum()),
                               "born": int((g.event == "birth").any()),
                               "died": int((lineage[(lineage.track_id == tid) & (lineage.event.isin(["death", "merge"]))]).shape[0] > 0),
                               "mean_normal_growth": float(nd.mean()) if len(nd) else np.nan,
                               "persistence_start": float(g.persistence_curr.iloc[0]), "persistence_end": float(g.persistence_curr.iloc[-1])})
            tracks = pd.DataFrame(tracks).sort_values("track_id")
            if ft is not None and len(tracks) and tracks.last_frame.max() < len(ft):   # lifetimes in time units
                tracks["first_time"] = ft.t[tracks.first_frame.clip(lower=0).to_numpy()]
                tracks["last_time"] = ft.t[tracks.last_frame.to_numpy()]
                tracks["lifetime"] = tracks.last_time - tracks.first_time
                tracks["mean_normal_growth_rate"] = tracks.mean_normal_growth * (tracks.n_frames / tracks.lifetime.where(tracks.lifetime > 0))
                tracks["time_unit"] = ft.unit
            tracks.to_csv(d / 'region_tracks.csv', index=False)
        rows = []
        for tr in self.traj:
            rows.append({"origin_frame": tr["origin_frame"], "origin_vertex": tr["origin_vertex"], "origin_type": tr["origin_type"],
                         "track_id": tr["track_id"], "n_frames": len(tr["frames"]), "last_frame": tr["frames"][-1],
                         "type_sequence": ",".join(tr["types"]), "frames": ",".join(map(str, tr["frames"])),
                         "vertices": ",".join(map(str, tr["vertices"])), "material_vertices": ",".join(map(str, tr["material"])),
                         "f_sequence": ",".join(f"{x:.4g}" for x in tr["f"]),
                         "first_change_frame": next((fr for fr, t in zip(tr["frames"][1:], tr["types"][1:]) if t != tr["origin_type"]), -1),
                         "final_type": tr["types"][-1], "long_fate": self._long_fate(tr["types"]),
                         "cumulative_normal_disp": float(np.nansum(tr["normal_disp"])) if tr["normal_disp"] else np.nan})
        pd.DataFrame(rows).to_csv(d / 'critical_point_trajectories.csv', index=False)
        with open(d / 'fate_meanings.json', 'w') as fh:
            json.dump({f"{a} -> {b}": m for (a, b), m in FATE_MEANING.items()}, fh, indent=2)
        return str(d)


def tracking_report(target_folder, single_file=True):
    """Plots of the region tracking: fate transition matrix, event counts, protrusion growth and lineage."""
    d = ms_folders(target_folder)
    tdir = d["tracking"]
    if not (tdir / 'tracking_summary.csv').exists():
        print("No tracking results found.")
        return None
    summary = pd.read_csv(tdir / 'tracking_summary.csv')
    X_ = summary.Time if "Time" in summary else summary.Time_Step
    fates = pd.read_csv(tdir / 'critical_point_fates.csv')
    lineage = pd.read_csv(tdir / 'region_lineage.csv')
    sns.set_theme(style="whitegrid")

    # 1) fate transition matrix
    if len(fates):
        tm = pd.crosstab(fates.type_prev, fates.type_next).reindex(index=CP_TYPES, columns=FATE_LABELS, fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(tm, annot=True, fmt="d", cmap="rocket_r", cbar_kws={"label": "critical points"}, ax=ax)
        ax.set_xlabel("type at t+1 (fate)"); ax.set_ylabel("type at t")
        ax.set_title("Fate of the critical points through the functional map", fontweight="bold")
        fig.tight_layout(); fig.savefig(d["plots"] / 'fate_transition_matrix.png', dpi=200); plt.close(fig)

    # 2) events and fates over time
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    ax = axes[0]
    for col, color in (("n_continue", "tab:blue"), ("n_split", "tab:orange"), ("n_merge", "tab:green"),
                       ("n_birth", "tab:red"), ("n_death", "tab:gray")):
        ax.plot(X_, summary[col], marker="o", color=color, label=col[2:])
    ax.set_title("Region events per transition (protrusion lineage)", fontweight="bold"); ax.set_ylabel("regions"); ax.legend(ncol=5)
    ax = axes[1]
    for col, color, mk in (("max_kept", CP_COLORS["maximum"], "^"), ("max_to_saddle", CP_COLORS["saddle"], "x"),
                           ("max_to_min", CP_COLORS["minimum"], "v"), ("max_flattened", "grey", "."),
                           ("min_to_max", "tab:pink", "^"), ("saddle_kept", "tab:cyan", "s"), ("saddle_to_max", "tab:olive", "*")):
        ax.plot(X_, summary[col], marker=mk, color=color, label=col.replace("_", " "))
    ax.set_title("Fates of the extrema", fontweight="bold"); ax.set_ylabel("critical points"); ax.legend(ncol=3, fontsize=8)
    ax = axes[2]
    ax.bar(X_, summary.fraction_growing, color=CP_COLORS["maximum"], alpha=0.6, label="fraction growing")
    ax.bar(X_, -summary.fraction_retracting, color=CP_COLORS["minimum"], alpha=0.6, label="fraction retracting")
    ax2 = ax.twinx(); ax2.plot(X_, summary.mean_normal_growth, color="k", marker="d", label="mean normal growth of maxima")
    ax2.axhline(0, color="k", linewidth=0.5); ax2.set_ylabel("normal displacement / sqrt(area)")
    ax.set_ylim(-1, 1); ax.set_ylabel("fraction of protrusions"); ax.set_xlabel("Time step")
    ax.set_title("Protrusion growth (image displacement along the surface normal)", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(d["plots"] / 'region_events_evolution.png', dpi=200); plt.close(fig)

    # 3) lineage (Gantt-like): one line per track, colour = growth, size = area
    if len(lineage):
        ln = lineage[lineage.curr_max_vertex >= 0].copy()
        fig, ax = plt.subplots(figsize=(11, max(4, 0.18 * ln.curr_track_id.nunique() + 2)))
        gmap = fates[fates.type_prev == "maximum"].set_index(["Time_Step", "track_id"]).normal_displacement
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=float(np.nanmin(gmap)) if len(gmap) and np.nanmin(gmap) < 0 else -1e-3,
                                              vmax=float(np.nanmax(gmap)) if len(gmap) and np.nanmax(gmap) > 0 else 1e-3)
        cmap = plt.get_cmap("coolwarm")
        amax = max(ln.area_curr.max(), 1e-300)
        for _, r in ln.iterrows():
            g = gmap.get((r.Time_Step, r.track_id), np.nan) if len(gmap) else np.nan
            color = cmap(norm(g)) if np.isfinite(g) else (0.5, 0.5, 0.5, 1.0)
            ax.plot([r.Time_Step - 1, r.Time_Step], [r.track_id, r.curr_track_id], color=color, linewidth=1.5,
                    linestyle="-" if r.event in ("continue", "split", "merge") else ":")
            ax.scatter([r.Time_Step], [r.curr_track_id], s=10 + 120 * r.area_curr / amax, color=color, edgecolor="k", linewidth=0.3, zorder=3)
        births = ln[ln.event == "birth"]
        ax.scatter(births.Time_Step, births.curr_track_id, marker="*", s=90, color="gold", edgecolor="k", zorder=4, label="birth")
        deaths = lineage[lineage.event == "death"]
        ax.scatter(deaths.Time_Step - 1, deaths.track_id, marker="x", s=60, color="k", zorder=4, label="death")
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, label="normal growth of the maximum (/sqrt area)")
        ax.set_xlabel("Time step"); ax.set_ylabel("protrusion track id"); ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Protrusion lineage (marker size = region area; splits/merges connect tracks)", fontweight="bold")
        fig.tight_layout(); fig.savefig(d["plots"] / 'protrusion_lineage.png', dpi=200); plt.close(fig)
    # 4) critical point trajectories: type timeline of every followed critical point
    tfile = tdir / 'critical_point_trajectories.csv'
    if tfile.exists():
        tr = pd.read_csv(tfile)
        tr = tr[tr.n_frames >= 2].sort_values(["origin_type", "origin_frame"]).reset_index(drop=True)
        if len(tr):
            code = {"maximum": 3, "saddle": 2, "minimum": 1, "regular": 0, "lost": np.nan}
            frames_all = sorted({int(x) for s_ in tr.frames for x in s_.split(",")})
            grid = np.full((len(tr), len(frames_all)), np.nan)
            col = {f_: k for k, f_ in enumerate(frames_all)}
            for i_, r in tr.iterrows():
                for f_, t_ in zip(r.frames.split(","), r.type_sequence.split(",")):
                    grid[i_, col[int(f_)]] = code.get(t_, np.nan)
            cmap = matplotlib.colors.ListedColormap([CP_COLORS["regular"], CP_COLORS["minimum"], CP_COLORS["saddle"], CP_COLORS["maximum"]])
            nr = len(tr)
            H = min(60.0, max(4.0, 0.2 * nr + 2.0))                     # 0.2 inch per row (readable 6-pt labels)
            k = max(1, int(np.ceil(0.2 * nr / max(H - 2.0, 1e-9))))     # very long lists: every k-th label
            fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(frames_all) + 6), H))
            im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
            ax.set_xticks(range(len(frames_all))); ax.set_xticklabels(frames_all)
            rows_ = np.arange(nr)[::k]
            ax.set_yticks(rows_); ax.set_yticklabels([f"{t[:3]} v{v} (T{f_})" for t, v, f_ in
                                                      zip(tr.origin_type.iloc[::k], tr.origin_vertex.iloc[::k], tr.origin_frame.iloc[::k])], fontsize=6)
            # colorbar first (it takes its space from the axes), then the twin axis with the fate of every row, which
            # sits between the image and the colorbar
            cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], pad=0.16, fraction=0.025)
            cb.ax.set_yticklabels(["regular", "minimum", "saddle", "maximum"])
            ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(rows_)
            ax2.set_yticklabels(list(tr.long_fate.iloc[::k]), fontsize=6); ax2.grid(False); ax2.set_ylabel("fate", fontsize=8)
            ax.set_xlabel("Time step"); ax.set_title("Type of every critical point followed through the maps (rows: origin)", fontweight="bold")
            fig.savefig(d["plots"] / 'critical_point_trajectories.png', dpi=200, bbox_inches="tight"); plt.close(fig)
            counts = tr.groupby(["origin_type", "long_fate"]).size().unstack(fill_value=0)
            counts.to_csv(tdir / 'long_fate_counts.csv')
    if single_file:
        print(f"Tracking report saved under {d['plots']}")
    return str(d["plots"])



#  Sequence-level driver (standalone: from the files on disk)

def ms_graph_analysis(target_folder, graph_metrics='all', single_file=True):
    """Reuse the Reeb-graph analyses (graph_time_analysis, graph_similarity + plots) on the
    critical-point graphs stored in MSComplexAnalysis/MS_Graphs -> MSComplexAnalysis/Graph_analysis."""
    from PynamicMesh.core.reeb_graph import graph_time_analysis, plot_dynamic_graph_analysis
    from PynamicMesh.core.graph_sim import graph_similarity, plot_graph_similarity
    d = ms_folders(target_folder)
    if len(list(d["graphs"].glob('*.pkl'))) < 2:
        print("Not enough Morse–Smale graphs for the graph analyses.")
        return None
    csv_t = graph_time_analysis(str(d["graphs"]), single_file=single_file)
    if csv_t:
        plot_dynamic_graph_analysis(csv_t, single_file=single_file)
    csv_s = graph_similarity(reeb_folder_path=str(d["graphs"]), metrics_list=graph_metrics, single_file=single_file)
    if csv_s:
        plot_graph_similarity(csv_s, single_file=single_file)
    return csv_t, csv_s


def track_sequence_from_disk(mesh_folder, target_folder, loader, matrix_folder=None, **tracker_kwargs):
    """Region tracking from saved MS complexes + saved point-to-point maps (FMV_T####_T####.npy).
    ``loader(path)`` returns an object with .vertices/.faces (and .eigenvectors if available)."""
    from PynamicMesh.core.pipelines import natural_sort_key
    mesh_folder = Path(mesh_folder); target_folder = Path(target_folder)
    matrix_folder = Path(matrix_folder) if matrix_folder else target_folder / 'Transform_Matrices'
    files = sorted([f for f in mesh_folder.iterdir() if f.suffix in ('.obj', '.mat')], key=natural_sort_key)
    seq = {ms.frame: ms for ms in load_ms_sequence(target_folder)}
    tracker = RegionTracker(target_folder, **tracker_kwargs)
    prev = None
    for i, f in enumerate(files):
        if i not in seq:
            prev = None; continue
        mesh = loader(f)
        if prev is not None and (i - 1) in seq:
            p2p_file = matrix_folder / f'FMV_T{i:04d}_T{i-1:04d}.npy'
            fm_file = matrix_folder / f'FMC_T{i-1:04d}_T{i:04d}.npy'
            if p2p_file.exists():
                FM = np.load(fm_file) if fm_file.exists() else None
                tracker.track_pair(seq[i - 1], seq[i], prev.vertices, prev.faces, mesh.vertices, mesh.faces,
                                   np.load(p2p_file), FM_12=FM, mesh_prev=prev, mesh_curr=mesh)
        prev = mesh
    tracker.save()
    return tracker