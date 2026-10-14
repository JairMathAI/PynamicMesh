"""
topology_events.py  —  sequences whose surfaces split or merge: detection, lineage, necks and divisions
=========================================================================================================

The analyses of the pipeline assume ONE connected surface per frame. When a surface divides (a cell becoming two) or
two surfaces fuse, functional maps, trajectories, parametrization and everything built on them are no longer defined
across the event. This stage detects such events and turns the sequence into a LINEAGE of single-surface tracks:

1. Topology of every frame. Pieces are counted three ways: (i) connected components of the mesh, (ii) connected
   components of the SOLID — the surface is voxelized on one global grid and its interior filled (the inside/outside
   classification of Müller 2009, independent of the mesh connectivity: two cells touching at a vertex or through a
   self-intersection are two solids), (iii) components of the solid eroded by one voxel — thin bridges break, so a
   cell in the process of separating is seen before the split.
2. Component tracking. Components of consecutive frames are matched by the overlap of their solids on the common
   grid (centroid distance as a fallback); a component with several children DIVIDES, several components with one
   child FUSE. The result is a lineage tree whose branches (tracks) are single surfaces between events.
3. Necks. Before a division the cell constricts. The narrowest bottleneck of the surface is found by a spectral
   sweep cut (Fiedler vector of the mesh graph, minimal conductance — Cheeger) or, optionally, by Ricci flow on the
   mesh graph (GraphRicciCurvature: edges across a bottleneck are stretched by the flow); the neck radius comes from
   the exact contour of the cut, the neck ratio = neck radius / equivalent radius of the smaller part.
4. Divisions. The mother solid (last frame before the split) is partitioned by nearest daughter (distance transform
   on the grid); every part is closed with marching cubes into a VIRTUAL DAUGHTER (the mother's material that
   becomes each daughter). This gives: the partition of the mother surface, the new membrane (surface of a virtual
   daughter with no mother counterpart: the furrow), the division plane vs the mother's long axis (Hertwig's rule),
   volume / area ratios, and correspondences daughter -> mother (functional map virtual daughter -> daughter, or
   nearest points), in the spirit of label-preserving surface tracking (Heiss-Synak et al. 2024).
5. Lineage mode. Every track is exported as a single-surface mesh sequence and the whole pipeline runs on it
   (functional maps, Reeb, Morse–Smale, parametrization, trajectories, motion analysis, …): results per cell.

Outputs: Results/<scene>/TopologyEvents/ — topology_timeline.csv, components.csv, transitions.csv, lineage.csv,
necks.csv, divisions.csv, Divisions/division_<k>/ (partition, virtual daughters, maps), Lineage_meshes/<track>/,
Results/<track>/ (lineage mode), plots/, summary.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PynamicMesh.core.dyn_common import (logged_stage, progress, pbar, ensure_dir, load_frames, find_frame_times,
                                         face_normals_areas)

TOPOLOGY_DEFAULTS = {
    "voxels": 96,                 # voxels along the longest side of the global bounding box
    "min_volume_fraction": 0.01,  # solid pieces smaller than this fraction of the frame volume are ignored
    "erode": 1,                   # erosion (voxels) that breaks thin bridges: "separating" pieces
    "neck_method": "spectral",    # 'spectral' (Cheeger sweep of the Fiedler vector) | 'ricci' (GraphRicciCurvature flow)
    "neck_threshold": 0.5,        # neck ratio below which a single surface is reported as constricted
    "split_pinched": True,        # a surface pinched at its neck (two cells held by a vertex / a thread) counts as two
    "pinch_ratio": 0.08,          # neck ratio below which a surface is pinched
    "min_part_fraction": 0.15,    # both parts of a pinch must hold at least this fraction of the area (no tips)
    "conformal_index": True,      # conformal factor of every surface (discrete surface Ricci flow) and its neck index
    "conformal_method": "auto",   # 'auto' (Ricci flow, linearized when it does not converge) | 'ricci' | 'linear'
    "ricci_nodes": 600,           # 'ricci': size of the coarsened graph
    "ricci_iterations": 10,       # 'ricci': iterations of the Ricci flow
    "match_min_overlap": 0.05,    # overlap fraction (of the smaller solid) for two components to be linked
    "lineage": "auto",            # 'auto' (when events or several pieces exist) | True | False
    "min_track_frames": 3,        # shortest track analysed by the lineage pipeline
    "division_maps": "fm",        # daughter -> mother correspondences: 'fm' (functional map) | 'nn' (nearest points)
    "plots": True,
}


# --------------------------------------------------------------------------------------------------------------- #
#  helpers: GPU-aware ndimage, mesh components
# --------------------------------------------------------------------------------------------------------------- #
def _nd():
    """(array module, ndimage module): CuPy / cupyx on the GPU when available, NumPy / SciPy otherwise."""
    try:
        from PynamicMesh.core.dyn_common import GPU_AVAILABLE
        if GPU_AVAILABLE:
            import cupy as cp
            import cupyx.scipy.ndimage as cnd
            return cp, cnd
    except Exception:  # noqa: BLE001
        pass
    import scipy.ndimage as snd
    return np, snd


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def mesh_components(F, n):
    """Connected components of a triangle mesh (through shared vertices): label of every vertex (-1 = unused)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = np.asarray(F, dtype=np.int64)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    used = np.zeros(n, bool); used[F.ravel()] = True
    out = np.full(n, -1, dtype=np.int64)
    u = np.unique(lab[used]); remap = {int(c): i for i, c in enumerate(u)}
    out[used] = np.vectorize(remap.get)(lab[used])
    return out


def submesh(V, F, vmask):
    """Faces with all vertices in vmask, re-indexed (vertices, faces, original vertex indices)."""
    fm = vmask[F].all(axis=1)
    Fs = F[fm]
    keep = np.unique(Fs)
    idx = -np.ones(len(V), dtype=np.int64); idx[keep] = np.arange(len(keep))
    return V[keep], idx[Fs], keep


# --------------------------------------------------------------------------------------------------------------- #
#  1. solid voxelization on a global grid
# --------------------------------------------------------------------------------------------------------------- #
def global_grid(frames, voxels=96, margin=2):
    lo = np.min([f["V"].min(0) for f in frames], axis=0); hi = np.max([f["V"].max(0) for f in frames], axis=0)
    pitch = float(np.max(hi - lo)) / max(int(voxels) - 2 * margin, 4)
    origin = lo - margin * pitch
    shape = tuple((np.ceil((hi - lo) / pitch).astype(int) + 2 * margin + 1).tolist())
    return {"origin": origin, "pitch": pitch, "shape": shape}


def voxelize_solid(V, F, grid):
    """
    Solid of a surface on the grid: the triangles are rasterized (barycentric samples denser than the voxels) and the
    enclosed voxels are filled — inside / outside from the geometry only (Müller 2009), so pinches, shared vertices and
    self-intersections do not merge or split pieces artificially. Returns a boolean grid (host array).
    """
    xp, nd = _nd()
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    pitch, origin, shape = grid["pitch"], grid["origin"], grid["shape"]
    e = max(float(np.max(np.linalg.norm(V[F[:, [1, 2, 0]]] - V[F], axis=2))), 1e-12)
    s = int(min(max(np.ceil(2.0 * e / pitch), 1), 12))
    a, b = np.meshgrid(np.arange(s + 1), np.arange(s + 1), indexing="ij"); m = a + b <= s
    bc = np.stack([a[m], b[m], s - a[m] - b[m]], 1).astype(np.float64) / s          # (k, 3) barycentric samples
    P = xp.asarray(V)[xp.asarray(F)]                                                  # (F, 3, 3)
    pts = xp.einsum("kj,fjd->fkd", xp.asarray(bc), P).reshape(-1, 3)
    ijk = xp.floor((pts - xp.asarray(origin)) / pitch).astype(xp.int64)
    ijk = xp.clip(ijk, 0, xp.asarray(shape) - 1)
    surf = xp.zeros(shape, dtype=bool)
    surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    solid = nd.binary_fill_holes(surf)
    return _host(solid)


def solid_components(solid, min_voxels=1, erode=0):
    """Labels of the solid pieces (6-connectivity), small pieces removed; with erode > 0 the solid is first eroded."""
    xp, nd = _nd()
    S = xp.asarray(solid)
    if erode:
        S = nd.binary_erosion(S, iterations=int(erode))
    lab, n = nd.label(S)
    lab = _host(lab).astype(np.int32)
    if n:
        sizes = np.bincount(lab.ravel())
        small = np.flatnonzero(sizes < min_voxels); small = small[small > 0]
        if len(small):
            lab[np.isin(lab, small)] = 0
        u = np.unique(lab); u = u[u > 0]
        remap = np.zeros(lab.max() + 1, dtype=np.int32); remap[u] = np.arange(1, len(u) + 1)
        lab = remap[lab]
    return lab


def _vertex_labels_from_grid(V, lab, grid):
    """Solid label of every vertex: the label of the nearest labelled voxel (the surface lies on the solid's border)."""
    from scipy.spatial import cKDTree
    occ = np.argwhere(lab > 0)
    if not len(occ):
        return np.zeros(len(V), dtype=np.int32)
    centers = grid["origin"] + (occ + 0.5) * grid["pitch"]
    _, j = cKDTree(centers).query(np.asarray(V, float))
    return lab[tuple(occ[j].T)].astype(np.int32)


def _frame_task(args):
    """Topology of one frame (worker-friendly)."""
    V, F, grid, min_frac, erode = args
    n = len(V)
    mlab = mesh_components(F, n)
    solid = voxelize_solid(V, F, grid)
    vox_total = int(solid.sum())
    min_vox = max(1, int(min_frac * vox_total))
    lab = solid_components(solid, min_vox)
    lab_er = solid_components(solid, max(1, min_vox // 4), erode=erode) if erode else lab
    vlab = _vertex_labels_from_grid(V, lab, grid)
    return {"mesh_labels": mlab, "solid_labels": lab, "vertex_labels": vlab,
            "n_mesh": int(mlab.max() + 1) if (mlab >= 0).any() else 0,
            "n_solid": int(lab.max()), "n_eroded": int(lab_er.max()), "solid_voxels": vox_total}


# --------------------------------------------------------------------------------------------------------------- #
#  2. component tracking and lineage
# --------------------------------------------------------------------------------------------------------------- #
def _component_table(frame_idx, V, F, vlab, lab, grid, t):
    rows = []
    fa = face_normals_areas(V, F)[1]
    for c in range(1, int(lab.max()) + 1):
        vm = vlab == c
        if not vm.any():
            continue
        fm = vm[F].all(axis=1)
        vox = int((lab == c).sum())
        cen = V[vm].mean(0)
        rows.append({"frame": frame_idx, "time": float(t[frame_idx]), "component": c, "n_vertices": int(vm.sum()),
                     "area": float(fa[fm].sum()), "volume": vox * grid["pitch"] ** 3,
                     "cx": cen[0], "cy": cen[1], "cz": cen[2]})
    return rows


def match_components(lab_a, lab_b, comp_a, comp_b, min_overlap=0.05):
    """
    Links between the solid pieces of two consecutive frames: overlap of the solids on the common grid (a bincount of
    the paired labels), relative to the smaller piece; pieces without overlap are linked to the nearest centroid when
    it is closer than their size. Returns a list of (component a, component b, overlap fraction).
    """
    xp, _ = _nd()
    na, nb = int(lab_a.max()), int(lab_b.max())
    if na == 0 or nb == 0:
        return []
    A = xp.asarray(lab_a).ravel(); B = xp.asarray(lab_b).ravel()
    both = (A > 0) & (B > 0)
    O = _host(xp.bincount((A[both] * (nb + 1) + B[both]).astype(xp.int64), minlength=(na + 1) * (nb + 1)))
    O = O.reshape(na + 1, nb + 1)[1:, 1:].astype(float)
    sa = np.bincount(_host(A).astype(np.int64), minlength=na + 1)[1:].astype(float)
    sb = np.bincount(_host(B).astype(np.int64), minlength=nb + 1)[1:].astype(float)
    frac = O / np.maximum(np.minimum(sa[:, None], sb[None, :]), 1)
    links = [(i + 1, j + 1, float(frac[i, j])) for i, j in zip(*np.nonzero(frac >= min_overlap))]
    linked_b = {j for _, j, _ in links}; linked_a = {i for i, _, _ in links}
    ca = {r["component"]: np.array([r["cx"], r["cy"], r["cz"]]) for r in comp_a}
    cb = {r["component"]: np.array([r["cx"], r["cy"], r["cz"]]) for r in comp_b}
    va = {r["component"]: r["volume"] for r in comp_a}
    for j in range(1, nb + 1):                         # fallback: nearest centroid for pieces that moved apart
        if j in linked_b or j not in cb or not ca:
            continue
        i = min(ca, key=lambda k: np.linalg.norm(ca[k] - cb[j]))
        # a piece next to a piece that already has a child is its sibling (a daughter that moved out of the
        # mother's volume): a one-to-many link, i.e. a division
        if np.linalg.norm(ca[i] - cb[j]) < 2.0 * va[i] ** (1 / 3):
            links.append((i, j, 0.0))
    return links


def build_lineage(frames_components, links_per_step):
    """
    Tracks (single surfaces between events) from the per-frame components and the links between consecutive frames.
    Returns (component -> track map {(frame, comp): track}, lineage rows, transition rows).
    """
    track_of, rows, trans = {}, {}, []
    next_id = 0
    for c in frames_components[0]:
        track_of[(0, c["component"])] = next_id
        rows[next_id] = {"track": next_id, "parents": [], "children": [], "start_frame": 0, "end_frame": 0, "fate": "ongoing"}
        next_id += 1
    for k, links in enumerate(links_per_step):
        a_frame, b_frame = k, k + 1
        kids, pars = {}, {}
        for i, j, o in links:
            kids.setdefault(i, []).append(j); pars.setdefault(j, []).append(i)
        comps_b = [c["component"] for c in frames_components[b_frame]]
        for j in comps_b:
            ps = pars.get(j, [])
            if len(ps) == 1 and len(kids.get(ps[0], [])) == 1:              # continues
                tid = track_of[(a_frame, ps[0])]
                track_of[(b_frame, j)] = tid; rows[tid]["end_frame"] = b_frame
                continue
            tid = next_id; next_id += 1
            parent_tracks = sorted({track_of[(a_frame, p)] for p in ps})
            rows[tid] = {"track": tid, "parents": parent_tracks, "children": [], "start_frame": b_frame,
                         "end_frame": b_frame, "fate": "ongoing"}
            track_of[(b_frame, j)] = tid
            for pt in parent_tracks:
                rows[pt]["children"].append(tid)
        for i, js in kids.items():
            pt = track_of[(a_frame, i)]
            if len(js) > 1:
                rows[pt]["fate"] = "divided"
                trans.append({"frame": b_frame, "event": "division", "parent_tracks": str([pt]),
                              "child_tracks": str(sorted(track_of[(b_frame, j)] for j in js))})
        for j, ps in pars.items():
            if len(ps) > 1:
                tracks_ps = sorted(track_of[(a_frame, p)] for p in ps)
                for pt in tracks_ps:
                    rows[pt]["fate"] = "fused"
                trans.append({"frame": b_frame, "event": "fusion", "parent_tracks": str(tracks_ps),
                              "child_tracks": str([track_of[(b_frame, j)]])})
        for c in frames_components[a_frame]:
            if c["component"] not in kids:
                tid = track_of[(a_frame, c["component"])]
                if rows[tid]["end_frame"] == a_frame:
                    rows[tid]["fate"] = "disappeared"
                    trans.append({"frame": b_frame, "event": "disappearance", "parent_tracks": str([tid]), "child_tracks": "[]"})
        for j in comps_b:
            if j not in pars:
                trans.append({"frame": b_frame, "event": "appearance", "parent_tracks": "[]",
                              "child_tracks": str([track_of[(b_frame, j)]])})
    return track_of, list(rows.values()), trans


# --------------------------------------------------------------------------------------------------------------- #
#  3. necks: narrowest bottleneck of a surface
# --------------------------------------------------------------------------------------------------------------- #
def _cut_contour_length(V, F, lab, field=None, thr=None):
    """Length of the contour separating two vertex labels. With a scalar field (the cut is its iso-line at thr) the
    crossing points are interpolated along the edges (exact iso-line); otherwise the midpoints of the crossing edges
    are joined in every face (a zigzag, slightly longer)."""
    L = 0.0; pts = []
    fl = lab[F]
    mixed = (fl.min(1) != fl.max(1))
    for f in F[mixed]:
        mids = []
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if lab[a] != lab[b]:
                if field is not None and field[b] != field[a]:
                    u = float(np.clip((thr - field[a]) / (field[b] - field[a]), 0.0, 1.0))
                    mids.append((1 - u) * V[a] + u * V[b])
                else:
                    mids.append(0.5 * (V[a] + V[b]))
        if len(mids) == 2:
            L += float(np.linalg.norm(mids[0] - mids[1])); pts += mids
    return L, (np.asarray(pts) if pts else np.zeros((0, 3)))


def spectral_neck(V, F):
    """
    Narrowest bottleneck by a sweep cut of the Fiedler vector of the mesh graph (weights 1 / edge length): among the
    prefixes of the vertices sorted by the Fiedler vector, the one with minimal conductance (Cheeger) — the cut
    through the thinnest part of the surface. Returns the two-part vertex labelling.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh
    n = len(V)
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    w = 1.0 / np.maximum(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1), 1e-12)
    W = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    d = np.asarray(W.sum(1)).ravel()
    L = diags(d) - W
    try:
        vals, vecs = eigsh(L, k=2, sigma=-1e-6, which="LM")
        fv = vecs[:, np.argsort(vals)[1]]
    except Exception:  # noqa: BLE001
        fv = V[:, int(np.argmax(np.ptp(V, 0)))]
    order = np.argsort(fv); pos = np.empty(n, int); pos[order] = np.arange(n)
    # cut(k) for the prefix of the first k vertices: edges with one end inside -> difference array over positions
    lo, hi = np.minimum(pos[E[:, 0]], pos[E[:, 1]]), np.maximum(pos[E[:, 0]], pos[E[:, 1]])
    diff = np.zeros(n + 1); np.add.at(diff, lo + 1, w); np.add.at(diff, hi + 1, -w)
    cut = np.cumsum(diff)[1:n]                                      # cut after the first k = 1..n-1 vertices
    vol = np.cumsum(d[order])[:-1]; tot = d.sum()
    cond = cut / np.maximum(np.minimum(vol, tot - vol), 1e-300)
    lo_k, hi_k = int(0.05 * n), int(0.95 * n)                      # parts of at least 5 %
    k = lo_k + int(np.argmin(cond[lo_k:hi_k])) if hi_k > lo_k else n // 2
    lab = np.zeros(n, dtype=np.int64); lab[order[k + 1:]] = 1
    return lab, fv, 0.5 * (fv[order[k]] + fv[order[k + 1]])


def ricci_neck(V, F, nodes=600, iterations=10):
    """
    Bottleneck by Ricci flow (GraphRicciCurvature) on a coarsened mesh graph: the flow stretches the edges across a
    neck; the graph is cut at the stretched edges into its two largest communities, mapped back to the vertices.
    """
    import networkx as nx
    import logging, io, contextlib
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    logging.getLogger("GraphRicciCurvature").setLevel(logging.ERROR)
    n = len(V)
    # coarsening by vertex clustering on a grid (about `nodes` clusters)
    ext = np.ptp(V, 0); vol = float(np.prod(np.maximum(ext, 1e-9)))
    h = (vol / max(nodes, 10)) ** (1 / 3) * 0.6
    key = np.floor((V - V.min(0)) / h).astype(np.int64)
    _, cl = np.unique(key, axis=0, return_inverse=True); cl = cl.ravel()
    nc = int(cl.max() + 1)
    C = np.zeros((nc, 3)); np.add.at(C, cl, V); C /= np.bincount(cl, minlength=nc)[:, None]
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    ce = np.unique(np.sort(np.stack([cl[E[:, 0]], cl[E[:, 1]]], 1), 1), axis=0); ce = ce[ce[:, 0] != ce[:, 1]]
    G = nx.Graph(); G.add_weighted_edges_from([(int(a), int(b), float(np.linalg.norm(C[a] - C[b]))) for a, b in ce])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        orc = OllivierRicci(G, alpha=0.5, method="OTDSinkhornMix", verbose="ERROR")
        orc.compute_ricci_flow(iterations=int(iterations))
    Gf = orc.G
    edges = list(Gf.edges()); w = np.array([Gf[a][b]["weight"] for a, b in edges])
    H = Gf.copy(); comps = [set(Gf.nodes())]
    for i in np.argsort(-w):
        H.remove_edge(*edges[i])
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        if len(comps) >= 2 and len(comps[1]) >= 0.05 * nc:
            break
    part = {c: 0 for c in comps[0]}
    for c in comps[1]:
        part[c] = 1
    # small leftover communities: to the nearest of the two parts
    big = np.array([part.get(i, -1) for i in range(nc)])
    if (big < 0).any():
        from scipy.spatial import cKDTree
        ok = np.flatnonzero(big >= 0); _, j = cKDTree(C[ok]).query(C[big < 0]); big[big < 0] = big[ok[j]]
    return big[cl].astype(np.int64)


def surface_metrics(V, F):
    """
    Shape descriptors of one closed surface (discrete differential geometry, cotangent Laplacian):
      fiedler_norm      first non-zero eigenvalue of the cotangent Laplacian x area / 8 pi: 1 for a sphere, -> 0 when a
                        bottleneck forms (Cheeger inequality), independent of where the cut is;
      reduced_volume    V / volume of the sphere of the same area: 1 sphere, 1/sqrt 2 two touching equal spheres —
                        the control parameter of vesicle shape transitions (Seifert 1997);
      bending_energy    Helfrich bending energy  int (2H)^2 dA / 16 pi: 1 sphere, 2 two spheres (rises to division);
      solidity          V / V(convex hull): falls as a furrow forms.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh
    from scipy.spatial import ConvexHull
    V = np.asarray(V, float); F = np.asarray(F, np.int64); n = len(V)
    e0, e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 1]], V[F[:, 0]] - V[F[:, 2]]
    cr = np.cross(e0, -e2); fa = 0.5 * np.linalg.norm(cr, axis=1); A = float(fa.sum())
    vol = float(abs(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum()) / 6.0)

    def cot(a, b):
        return np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-300)
    c0 = cot(-e2, e0); c1 = cot(-e0, e1); c2 = cot(-e1, e2)          # angles at corners 0, 1, 2
    I = np.r_[F[:, 1], F[:, 2], F[:, 0]]; J = np.r_[F[:, 2], F[:, 0], F[:, 1]]; Wv = 0.5 * np.r_[c0, c1, c2]
    W = coo_matrix((np.r_[Wv, Wv], (np.r_[I, J], np.r_[J, I])), shape=(n, n)).tocsr()
    L = diags(np.asarray(W.sum(1)).ravel()) - W
    va = np.zeros(n); np.add.at(va, F.ravel(), np.repeat(fa / 3.0, 3)); va = np.maximum(va, 1e-300)
    out = {"area": A, "volume": vol,
           "reduced_volume": float(vol / ((4 * np.pi / 3) * (A / (4 * np.pi)) ** 1.5)) if A > 0 else np.nan}
    try:
        vals = eigsh(L.tocsc(), k=3, M=diags(va).tocsc(), sigma=-1e-8, which="LM", return_eigenvectors=False)
        out["fiedler_norm"] = float(np.sort(vals)[1] * A / (8 * np.pi))
    except Exception:  # noqa: BLE001
        out["fiedler_norm"] = np.nan
    Hn = (L @ V) / (2 * va[:, None])                               # mean-curvature normal (Laplace–Beltrami of x)
    out["bending_energy"] = float(np.sum(4 * np.einsum("ij,ij->i", Hn, Hn) * va) / (16 * np.pi))
    try:
        out["solidity"] = float(vol / ConvexHull(V).volume)
    except Exception:  # noqa: BLE001
        out["solidity"] = np.nan
    return out


def neck_measures(V, F, method="spectral", ricci_nodes=600, ricci_iterations=10):
    """Neck of a single surface: two-part cut, neck radius (contour length / 2 pi), neck ratio = neck radius /
    (sqrt 2 x equivalent radius of the smaller part: about 1 for a round surface, -> 0 when the neck closes), position along the long axis (0..1), plane of the cut, area of the parts."""
    field = thr = None
    if method == "ricci":
        try:
            lab = ricci_neck(V, F, ricci_nodes, ricci_iterations)
        except Exception:  # noqa: BLE001 - GraphRicciCurvature missing or failing: spectral cut
            lab, field, thr = spectral_neck(V, F); method = "spectral"
    else:
        lab, field, thr = spectral_neck(V, F)
    L, pts = _cut_contour_length(V, F, lab, field, thr)
    fa = face_normals_areas(V, F)[1]
    fl = lab[F]
    A = [float(fa[(fl == p).all(1)].sum()) for p in (0, 1)]
    r_neck = L / (2 * np.pi)
    # reference: a round surface cut through its middle has r_neck = R and halves of area 2 pi R² -> ratio 1
    r_eq = np.sqrt(2.0) * np.sqrt(max(min(A), 1e-300) / (4 * np.pi))
    c = V.mean(0); X = V - c
    axis = np.linalg.svd(X, full_matrices=False)[2][0]
    proj = X @ axis
    pos = float((pts.mean(0) - c) @ axis - proj.min()) / max(float(np.ptp(proj)), 1e-300) if len(pts) else np.nan
    normal = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][2] if len(pts) >= 3 else axis
    return {"labels": lab, "neck_radius": r_neck, "neck_ratio": float(r_neck / max(r_eq, 1e-300)),
            "neck_position": pos, "neck_center": pts.mean(0) if len(pts) else c, "cut_normal": normal,
            "long_axis": axis, "area_part_0": A[0], "area_part_1": A[1], "method": method}


# --------------------------------------------------------------------------------------------------------------- #
#  4. divisions: virtual daughters, partition of the mother, new membrane, correspondences
# --------------------------------------------------------------------------------------------------------------- #
def virtual_daughters(mother_solid, daughter_labels, grid, Vm):
    """
    The mother solid partitioned by nearest daughter (distance transform of the daughters' solids, with indices) and
    every part closed by marching cubes: the VIRTUAL DAUGHTERS (meshes at the mother's time). Returns (list of
    (V, F) per daughter, label of every mother vertex).
    """
    from skimage import measure
    xp, nd = _nd()
    D = xp.asarray(daughter_labels)
    _, ind = nd.distance_transform_edt(D == 0, return_indices=True)
    nearest = _host(D[tuple(ind)])
    meshes = []
    for d in range(1, int(_host(D).max()) + 1):
        part = mother_solid & (nearest == d)
        if part.sum() < 8:
            meshes.append((np.zeros((0, 3)), np.zeros((0, 3), int))); continue
        Vv, Fv, _, _ = measure.marching_cubes(np.pad(part, 1).astype(np.float32), 0.5)
        Vv = (Vv - 1 + 0.5) * grid["pitch"] + grid["origin"]
        meshes.append((Vv, Fv.astype(np.int64)))
    vlab = _vertex_labels_from_grid(Vm, np.where(mother_solid, nearest, 0).astype(np.int32), grid)
    return meshes, vlab


def analyse_fusion(k, parents, child, child_solid, parent_labels, grid, out):
    """
    A fusion, mirror of a division: the fused solid is partitioned by nearest parent (distance transform with indices
    on the grid); every vertex of the fused surface gets its parent and the nearest vertex of that parent; vertices
    whose distance to the parents is an outlier of the frame (beyond median + 4 MAD of the distances, and at least
    1.5 voxels — the parents are one frame earlier, so the whole surface has moved a little) are the BRIDGE (-1).
    Writes child_partition.npy (parent of every fused vertex, 1..), child_to_parent.npy (vertex of that parent, -1 =
    bridge) and returns the geometry of the fusion.
    """
    from scipy.spatial import cKDTree
    xp, nd = _nd()
    d = ensure_dir(Path(out) / f"fusion_{k:02d}")
    P = xp.asarray(parent_labels)
    _, ind = nd.distance_transform_edt(P == 0, return_indices=True)
    nearest = _host(P[tuple(ind)])
    lab = _vertex_labels_from_grid(child["V"], np.where(child_solid, nearest, 0).astype(np.int32), grid)
    c2p = np.full(len(child["V"]), -1, dtype=np.int64)
    dists = np.zeros(len(child["V"]))
    for i, pr in enumerate(parents):
        m = lab == i + 1
        if m.any():
            dist, j = cKDTree(pr["V"]).query(child["V"][m])
            c2p[np.flatnonzero(m)] = j; dists[m] = dist
    # the parents are one frame earlier: the surface has also moved, so the bridge is where the distance to the
    # parents is an OUTLIER of the frame (median + 4 MAD), and at least 1.5 voxels
    med = np.median(dists); mad = np.median(np.abs(dists - med)) * 1.4826
    c2p[dists > max(1.5 * grid["pitch"], med + 4 * mad)] = -1
    np.save(d / "child_partition.npy", lab.astype(np.int16)); np.save(d / "child_to_parent.npy", c2p)
    vols = [pr["volume"] for pr in parents]; areas = [pr["area"] for pr in parents]
    row = {"fusion": k, "frame": child["frame"], "parent_tracks": str([pr["track"] for pr in parents]),
           "child_track": child["track"], "n_parents": len(parents),
           "volume_ratio_parents": float(min(vols) / max(max(vols), 1e-300)),
           "volume_conservation": float(child["volume"] / max(sum(vols), 1e-300)),
           "area_change_fraction": float((child["area"] - sum(areas)) / max(sum(areas), 1e-300)),
           "bridge_fraction": float(np.mean(c2p < 0))}
    if len(parents) == 2:
        c = child["V"].mean(0); axis = np.linalg.svd(child["V"] - c, full_matrices=False)[2][0]
        nvec = parents[1]["V"].mean(0) - parents[0]["V"].mean(0); nvec /= max(np.linalg.norm(nvec), 1e-300)
        row["angle_parents_axis_long_axis_deg"] = float(np.degrees(np.arccos(min(1.0, abs(float(nvec @ axis))))))
    return row


def cobordism_events(tl, tr):
    """
    The sequence as a cobordism (space-time): every change of the surfaces is a critical point of time, of Morse index
    0 (a sphere is born), 1 (a 1-handle: a bridge between two surfaces = fusion, or a new tunnel), 2 (a 2-handle: a
    neck pinch = division, or a tunnel closing) or 3 (a sphere disappears). The changes of the number of surfaces
    (Betti-0) and of the Euler characteristic between frames give them, checked with the Euler–Poincaré balance
    Delta chi = 2 (n0 - n1 + n2 - n3).
    """
    rows = []
    for i in range(1, len(tl)):
        db0 = int(tl.pieces_tracked.iloc[i] - tl.pieces_tracked.iloc[i - 1])
        dchi = int(tl.euler.iloc[i] - tl.euler.iloc[i - 1])
        ev = tr[tr.frame == tl.frame.iloc[i]] if len(tr) else tr
        n_div = int(sum(len(eval(r.child_tracks)) - 1 for r in ev.itertuples() if r.event == "division"))
        n_fus = int(sum(len(eval(r.parent_tracks)) - 1 for r in ev.itertuples() if r.event == "fusion"))
        n_birth = int((ev.event == "appearance").sum()) if len(ev) else 0
        n_death = int((ev.event == "disappearance").sum()) if len(ev) else 0
        dg = int(tl.genus.iloc[i] - tl.genus.iloc[i - 1])
        n1 = n_fus + max(dg, 0); n2 = n_div + max(-dg, 0)
        if not (db0 or dchi or n_div or n_fus or n_birth or n_death):
            continue
        kinds = ([f"neck pinch x{n_div} (index 2, division)"] if n_div else []) + \
                ([f"bridge x{n_fus} (index 1, fusion)"] if n_fus else []) + \
                ([f"handle created x{dg} (index 1)"] if dg > 0 else []) + ([f"handle closed x{-dg} (index 2)"] if dg < 0 else []) + \
                ([f"sphere born x{n_birth} (index 0)"] if n_birth else []) + ([f"sphere vanished x{n_death} (index 3)"] if n_death else [])
        rows.append({"frame": int(tl.frame.iloc[i]), "time": float(tl.time.iloc[i]), "delta_betti0": db0,
                     "delta_euler": dchi, "delta_genus": dg, "n_index0": n_birth, "n_index1": n1, "n_index2": n2,
                     "n_index3": n_death, "euler_balance_ok": bool(dchi == 2 * (n_birth - n1 + n2 - n_death)),
                     "surgery": "; ".join(kinds) or "unexplained change"})
    return pd.DataFrame(rows)


def pinch_fit(necks, divisions, transitions):
    """
    Neck-pinch singularity of every division: the neck radius of the mother before the split is fitted as
    r = c (T - t)^alpha (T searched between the last frame before the split and the split). The exponent indicates the
    mechanism of the thinning: ~1/2 curvature-driven (mean curvature flow / surface tension, Huisken; Angenent &
    Velazquez), ~2/3 inertial-capillary, ~1 viscous (Stokes) pinch-off (Eggers 1997), ~1/4 surface diffusion.
    """
    rows = []
    regimes = {0.25: "surface diffusion", 0.5: "curvature-driven (mean curvature flow)", 2 / 3: "inertial-capillary",
               1.0: "viscous (Stokes)"}
    for r in divisions.itertuples():
        mt = int(r.mother_track)
        g = necks[necks.track == mt].sort_values("time")
        tsplit = float(transitions[(transitions.event == "division") & (transitions.frame == r.frame)].time.iloc[0])
        g = g[g.time < tsplit]
        g = g[g.neck_ratio < 0.95].tail(10)
        if len(g) < 3:
            continue
        t, rad = g.time.to_numpy(float), g.neck_radius.to_numpy(float)
        best = None
        for T in np.linspace(t[-1] + 1e-3 * (tsplit - t[-1] + 1e-9), tsplit, 200):
            x = np.log(np.maximum(T - t, 1e-12)); y = np.log(np.maximum(rad, 1e-12))
            A = np.vstack([x, np.ones_like(x)]).T
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - (np.sum((A @ coef - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300))
            if best is None or r2 > best[0]:
                best = (r2, T, coef[0], np.exp(coef[1]))
        r2, T, alpha, c = best
        reg = min(regimes, key=lambda a: abs(a - alpha))
        rows.append({"division": int(r.division), "mother_track": mt, "n_frames_fitted": len(g), "alpha": float(alpha),
                     "predicted_pinch_time": float(T), "observed_split_time": tsplit, "r2": float(r2),
                     "prefactor": float(c), "closest_regime": regimes[reg]})
    return pd.DataFrame(rows)


def _fm_map(Vs, Fs, Vd, Fd, fm_params):
    """Point-to-point map destination -> source with the project's functional maps (fallback: nearest points after
    centring)."""
    try:
        from pyFM.mesh import TriMesh
        from PynamicMesh.core.custom_fm import CustomFunctionalMapping
        import io, contextlib
        m1, m2 = TriMesh(np.asarray(Vs, float), np.asarray(Fs, int)), TriMesh(np.asarray(Vd, float), np.asarray(Fd, int))
        for m in (m1, m2):
            m.process(k=100, verbose=False)
        fp = dict(fm_params or {})
        M = CustomFunctionalMapping(m1, m2)
        with contextlib.redirect_stdout(io.StringIO()):
            M.preprocess(n_ev=(30, 30), n_descr=100, descr_type="WKS", landmarks=None, subsample_step=4, k_process=100,
                         symmetry_mode=fp.get("symmetry_mode", "orientation+extrinsic"),
                         landmark_params=fp.get("landmark_params"), descr_params=fp.get("descr_params"), verbose=False)
            M.fit(**dict(fp.get("fit_params") or {}))
            FM = np.asarray(M.zoomout_refine(nit=20, step=2))
        return np.asarray(M.get_p2p_from_FM(FM)), "fm"
    except Exception:  # noqa: BLE001
        from scipy.spatial import cKDTree
        _, j = cKDTree(np.asarray(Vs) - np.mean(Vs, 0)).query(np.asarray(Vd) - np.mean(Vd, 0))
        return j, "nn"


def analyse_division(k, mother, daughters, mother_solid, daughter_labels, grid, out, maps="fm", fm_params=None):
    """Geometry of one division and the correspondences daughters -> mother (see the module docstring)."""
    from scipy.spatial import cKDTree
    d = ensure_dir(Path(out) / f"division_{k:02d}")
    Vm, Fm = mother["V"], mother["F"]
    meshes, mlab = virtual_daughters(mother_solid, daughter_labels, grid, Vm)
    np.save(d / "mother_partition.npy", mlab.astype(np.int16))
    c = Vm.mean(0); axis = np.linalg.svd(Vm - c, full_matrices=False)[2][0]
    cents = [dd["V"].mean(0) for dd in daughters]
    vols = [dd["volume"] for dd in daughters]; areas = [dd["area"] for dd in daughters]
    row = {"division": k, "frame": mother["frame"] + 1, "mother_track": mother["track"],
           "daughter_tracks": str([dd["track"] for dd in daughters]), "n_daughters": len(daughters),
           "mother_volume": mother["volume"], "mother_area": mother["area"],
           "volume_ratio": float(min(vols) / max(max(vols), 1e-300)), "area_ratio": float(min(areas) / max(max(areas), 1e-300)),
           "volume_conservation": float(sum(vols) / max(mother["volume"], 1e-300)),
           "new_area_fraction": float((sum(areas) - mother["area"]) / max(mother["area"], 1e-300))}
    if len(daughters) == 2:
        nvec = cents[1] - cents[0]; nvec /= max(np.linalg.norm(nvec), 1e-300)
        # Hertwig's rule: the division plane cuts the long axis -> its normal (daughter-daughter) along the long axis
        row["angle_plane_normal_long_axis_deg"] = float(np.degrees(np.arccos(min(1.0, abs(float(nvec @ axis))))))
    mother_tree = cKDTree(Vm)
    for i, ((Vv, Fv), dd) in enumerate(zip(meshes, daughters)):
        if not len(Vv):
            continue
        dist, vj = mother_tree.query(Vv)
        new_mem = dist > 1.5 * grid["pitch"]                  # surface of the virtual daughter with no mother counterpart
        np.savez_compressed(d / f"virtual_daughter_{i}.npz", V=Vv, F=Fv, new_membrane=new_mem, mother_vertex=vj)
        row[f"new_membrane_fraction_{i}"] = float(new_mem.mean())
        if maps == "fm":
            p2p, how = _fm_map(Vv, Fv, dd["V"], dd["F"], fm_params)
        else:                                             # nearest points after centring both surfaces
            p2p, how = cKDTree(Vv - Vv.mean(0)).query(dd["V"] - dd["V"].mean(0))[1], "nn"
        # daughter vertex -> virtual daughter vertex -> mother vertex (-1: new membrane)
        mv = np.where(new_mem[p2p], -1, vj[p2p])
        np.save(d / f"daughter_{i}_to_mother.npy", mv.astype(np.int64))
        row[f"map_{i}"] = how
        row[f"daughter_{i}_from_new_membrane"] = float(np.mean(mv < 0))
    return row


# --------------------------------------------------------------------------------------------------------------- #
#  stage driver
# --------------------------------------------------------------------------------------------------------------- #
def _raw_vertices(frame, loader):
    """Vertices of a frame in the coordinates of its FILE (the inverse of the loader's alignment), same order."""
    try:
        from PynamicMesh.utils.tools import load_aligned_mesh, mesh_mat2object
        if loader is load_aligned_mesh or getattr(loader, "__name__", "") == "load_aligned_mesh":
            raw = np.asarray(mesh_mat2object(frame["path"]).vertices, dtype=float)
            if raw.shape == np.asarray(frame["V"]).shape:
                return raw
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(frame["V"], dtype=float)


def _write_obj(path, V, F):
    with open(path, "w") as fh:
        fh.write("".join(f"v {x:.7g} {y:.7g} {z:.7g}\n" for x, y, z in V))
        fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F))


@logged_stage("TopologyEvents", (1, "target_folder"), "TopologyEvents")
def compute_topology_events(mesh_folder, target_folder, loader=None, frames=None, times=None, fm_params=None,
                            verbose=True, **params):
    """Stage driver (see the module docstring); `params` override TOPOLOGY_DEFAULTS. Returns the summary dict, with
    'needs_lineage' telling whether the sequence changes topology (or holds several surfaces)."""
    p = {**TOPOLOGY_DEFAULTS, **params}
    target_folder = Path(target_folder); out = ensure_dir(target_folder / "TopologyEvents"); ensure_dir(out / "plots")
    frames = frames if frames is not None else load_frames(mesh_folder, loader)
    T = len(frames)
    ft = find_frame_times(mesh_folder) if times is None else None
    t = np.asarray(getattr(times, "t", times), float) if times is not None else (
        np.asarray(ft.t, float) if ft is not None and getattr(ft, "t", None) is not None and len(ft.t) == T else np.arange(T, dtype=float))
    progress(total=6, advance=0)
    # 1. topology of every frame (worker pool)
    grid = global_grid(frames, p["voxels"])
    from PynamicMesh.core import accel
    res = accel.parallel_map(_frame_task, [(f["V"], f["F"], grid, p["min_volume_fraction"], p["erode"]) for f in frames],
                             desc="Topology of the frames")
    # necks of every surface; a surface pinched at its neck (two cells still held by a vertex or a thin thread) is
    # split into its two parts before tracking, so the division is dated at the pinch
    from scipy.spatial import cKDTree
    neck_at = {}
    conf = {k: np.full(len(frames[k]["V"]), np.nan, dtype=np.float32) for k in range(T)} if p["conformal_index"] else None
    for k in pbar(range(T), desc="Necks and pinches"):
        r = res[k]; r["pinches"] = 0
        queue = [c for c in range(1, int(r["solid_labels"].max()) + 1)]
        while queue:
            c = queue.pop(0)
            vm = r["vertex_labels"] == c
            Vs, Fs, keep = submesh(frames[k]["V"], frames[k]["F"], vm)
            if len(Fs) < 50:
                continue
            nm = neck_measures(Vs, Fs, p["neck_method"], p["ricci_nodes"], p["ricci_iterations"])
            small = min(nm["area_part_0"], nm["area_part_1"]) / max(nm["area_part_0"] + nm["area_part_1"], 1e-300)
            if p["split_pinched"] and nm["neck_ratio"] < p["pinch_ratio"] and small >= p["min_part_fraction"]:
                new = int(r["solid_labels"].max()) + 1
                r["vertex_labels"][keep[nm["labels"] == 1]] = new
                occ = np.argwhere(r["solid_labels"] == c)          # its voxels: to the nearer half
                centers = grid["origin"] + (occ + 0.5) * grid["pitch"]
                _, j = cKDTree(Vs).query(centers)
                moved = occ[nm["labels"][j] == 1]
                r["solid_labels"][tuple(moved.T)] = new
                r["pinches"] += 1
                queue += [c, new]                                  # each half may still be pinched
                continue
            if conf is not None:
                # conformal factor of the uniformization (surface Ricci flow): u rises at a neck that closes
                from PynamicMesh.core.ricci_flow import conformal_factor
                try:
                    u, info = conformal_factor(Vs, Fs, method=p["conformal_method"], return_info=True)
                    lab = nm["labels"]
                    e = np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]])
                    cut = np.unique(e[lab[e[:, 0]] != lab[e[:, 1]]])
                    nm["conformal_neck_index"] = float(np.mean(u[cut]) - np.median(u)) if len(cut) else np.nan
                    nm["conformal_method"] = info["method"]
                    conf[k][keep] = u.astype(np.float32)
                except Exception as exc:  # noqa: BLE001
                    nm["conformal_neck_index"] = np.nan; nm["conformal_method"] = f"failed: {exc}"[:60]
            try:
                nm["metrics"] = surface_metrics(Vs, Fs)
            except Exception:  # noqa: BLE001
                nm["metrics"] = {}
            neck_at[(k, c)] = nm
    comps, tl = [], []
    for k, (f, r) in enumerate(zip(frames, res)):
        rows = _component_table(k, f["V"], f["F"], r["vertex_labels"], r["solid_labels"], grid, t)
        comps.append(rows)
        chi = 0
        for rr in rows:
            Vs_, Fs_, _ = submesh(f["V"], f["F"], r["vertex_labels"] == rr["component"])
            if len(Fs_):
                ee = np.unique(np.sort(np.concatenate([Fs_[:, [0, 1]], Fs_[:, [1, 2]], Fs_[:, [2, 0]]]), 1), axis=0)
                chi += len(np.unique(Fs_)) - len(ee) + len(Fs_)
        tl.append({"frame": k, "time": float(t[k]), "name": f["name"], "euler": int(chi),
                   "genus": int(round((2 * len(rows) - chi) / 2)), "mesh_components": r["n_mesh"],
                   "solid_components": r["n_solid"], "separating_components": r["n_eroded"],
                   "pinches_split": r["pinches"], "pieces_tracked": len(rows)})
    progress("topology of the frames")
    # 2. tracking and lineage
    links = [match_components(res[k]["solid_labels"], res[k + 1]["solid_labels"], comps[k], comps[k + 1], p["match_min_overlap"])
             for k in pbar(range(T - 1), desc="Matching components")]
    track_of, lineage, trans = build_lineage(comps, links)
    for k, rows in enumerate(comps):
        for r in rows:
            r["track"] = track_of[(k, r["component"])]
    comp_df = pd.DataFrame([r for rows in comps for r in rows])
    comp_df.to_csv(out / "components.csv", index=False)
    lin_df = pd.DataFrame(lineage)
    if len(lin_df):
        lin_df["n_frames"] = lin_df.end_frame - lin_df.start_frame + 1
        lin_df["start_time"] = t[lin_df.start_frame]; lin_df["end_time"] = t[lin_df.end_frame]
        lin_df["parents"] = lin_df.parents.astype(str); lin_df["children"] = lin_df.children.astype(str)
    lin_df.to_csv(out / "lineage.csv", index=False)
    tr_df = pd.DataFrame(trans, columns=["frame", "event", "parent_tracks", "child_tracks"])
    if len(tr_df):
        tr_df["time"] = t[tr_df.frame.to_numpy(int)]
    tr_df.to_csv(out / "transitions.csv", index=False)
    tl_df = pd.DataFrame(tl)
    ev_at = tr_df.groupby("frame").event.apply(lambda s: ";".join(sorted(set(s)))) if len(tr_df) else pd.Series(dtype=str)
    tl_df["events"] = tl_df.frame.map(ev_at).fillna("")
    tl_df.to_csv(out / "topology_timeline.csv", index=False)
    progress("component tracking and lineage")
    # 3. necks of every single surface (per track and frame)
    neck_rows = []
    for (k, c), tid in sorted(track_of.items()):
        nm = neck_at.get((k, c))
        if nm is None:
            continue
        neck_rows.append({"frame": k, "time": float(t[k]), "track": tid, "neck_radius": nm["neck_radius"],
                          "neck_ratio": nm["neck_ratio"], "neck_position": nm["neck_position"],
                          "nx": nm["cut_normal"][0], "ny": nm["cut_normal"][1], "nz": nm["cut_normal"][2],
                          "cx": nm["neck_center"][0], "cy": nm["neck_center"][1], "cz": nm["neck_center"][2],
                          "constricted": bool(nm["neck_ratio"] < p["neck_threshold"]), "method": nm["method"],
                          "conformal_neck_index": nm.get("conformal_neck_index", np.nan),
                          "conformal_method": nm.get("conformal_method", "")})
    neck_df = pd.DataFrame(neck_rows); neck_df.to_csv(out / "necks.csv", index=False)
    srf = [{"frame": k, "time": float(t[k]), "track": tid, **neck_at[(k, c)].get("metrics", {})}
           for (k, c), tid in sorted(track_of.items()) if (k, c) in neck_at]
    pd.DataFrame(srf).to_csv(out / "surfaces.csv", index=False)
    progress("necks")
    # 4. divisions
    div_rows = []
    dsub = ensure_dir(out / "Divisions")
    divisions = tr_df[tr_df.event == "division"] if len(tr_df) else tr_df
    for k_div, r in enumerate(pbar(list(divisions.itertuples()), desc="Divisions")):
        fb = int(r.frame); fa = fb - 1
        mt = int(eval(r.parent_tracks)[0]); kids = eval(r.child_tracks)
        mc = [c for (fr, c), tid in track_of.items() if fr == fa and tid == mt][0]
        crow = comp_df[(comp_df.frame == fa) & (comp_df.component == mc)].iloc[0]
        vm = res[fa]["vertex_labels"] == mc
        Vm, Fm, _ = submesh(frames[fa]["V"], frames[fa]["F"], vm)
        mother = {"V": Vm, "F": Fm, "frame": fa, "track": mt, "volume": float(crow.volume), "area": float(crow.area)}
        mother_solid = res[fa]["solid_labels"] == mc
        dl = np.zeros_like(res[fb]["solid_labels"]); daughters = []
        for i, kt in enumerate(kids):
            dc = [c for (fr, c), tid in track_of.items() if fr == fb and tid == kt][0]
            drow = comp_df[(comp_df.frame == fb) & (comp_df.component == dc)].iloc[0]
            dl[res[fb]["solid_labels"] == dc] = i + 1
            Vd, Fd, _ = submesh(frames[fb]["V"], frames[fb]["F"], res[fb]["vertex_labels"] == dc)
            daughters.append({"V": Vd, "F": Fd, "track": kt, "volume": float(drow.volume), "area": float(drow.area)})
        div_rows.append(analyse_division(k_div, mother, daughters, mother_solid, dl, grid, dsub, p["division_maps"], fm_params))
    div_df = pd.DataFrame(div_rows); div_df.to_csv(out / "divisions.csv", index=False)
    fus_rows = []
    fsub = ensure_dir(out / "Fusions")
    fusions = tr_df[tr_df.event == "fusion"] if len(tr_df) else tr_df
    for k_f, r in enumerate(pbar(list(fusions.itertuples()), desc="Fusions")):
        fb = int(r.frame); fa = fb - 1
        ct = int(eval(r.child_tracks)[0]); pts = eval(r.parent_tracks)
        cc = [c for (fr, c), tid in track_of.items() if fr == fb and tid == ct][0]
        crow = comp_df[(comp_df.frame == fb) & (comp_df.component == cc)].iloc[0]
        Vc, Fc, _ = submesh(frames[fb]["V"], frames[fb]["F"], res[fb]["vertex_labels"] == cc)
        child = {"V": Vc, "F": Fc, "frame": fb, "track": ct, "volume": float(crow.volume), "area": float(crow.area)}
        pl = np.zeros_like(res[fa]["solid_labels"]); parents = []
        for i, pt in enumerate(pts):
            pc = [c for (fr, c), tid in track_of.items() if fr == fa and tid == pt][0]
            prow = comp_df[(comp_df.frame == fa) & (comp_df.component == pc)].iloc[0]
            pl[res[fa]["solid_labels"] == pc] = i + 1
            Vp, Fp, _ = submesh(frames[fa]["V"], frames[fa]["F"], res[fa]["vertex_labels"] == pc)
            parents.append({"V": Vp, "F": Fp, "track": pt, "volume": float(prow.volume), "area": float(prow.area)})
        fus_rows.append(analyse_fusion(k_f, parents, child, res[fb]["solid_labels"] == cc, pl, grid, fsub))
    pd.DataFrame(fus_rows).to_csv(out / "fusions.csv", index=False)
    cob = cobordism_events(tl_df, tr_df); cob.to_csv(out / "cobordism.csv", index=False)
    pin = pinch_fit(neck_df, div_df, tr_df) if len(div_df) and len(neck_df) else pd.DataFrame()
    pin.to_csv(out / "pinch_singularity.csv", index=False)
    progress("divisions")
    # 5. lineage meshes: every track as a single-surface sequence
    multi = bool(len(tr_df)) or int(tl_df.pieces_tracked.max()) > 1
    needs = multi if p["lineage"] == "auto" else bool(p["lineage"])
    lm = out / "Lineage_meshes"
    exported = []
    if needs and len(lin_df):
        scene = Path(mesh_folder).name
        for row in lin_df.itertuples():
            if row.n_frames < 1:
                continue
            d = ensure_dir(lm / f"{scene}_track{row.track:02d}")
            tt = []
            for k in range(int(row.start_frame), int(row.end_frame) + 1):
                cs = [c for (fr, c), tid in track_of.items() if fr == k and tid == row.track]
                if not cs:
                    continue
                Vs, Fs, keep_ = submesh(frames[k]["V"], frames[k]["F"], res[k]["vertex_labels"] == cs[0])
                # raw (file) coordinates: the pipeline re-applies its loader to these meshes, so the per-track
                # results live in the same coordinates as the scene
                _write_obj(d / f"{scene}_track{row.track:02d}_{k:04d}.obj", _raw_vertices(frames[k], loader)[keep_], Fs)
                tt.append((k, float(t[k])))
            if tt:
                pd.DataFrame({"frame": range(len(tt)), "file": [f"{scene}_track{row.track:02d}_{k:04d}.obj" for k, _ in tt],
                              "time": [x for _, x in tt]}).to_csv(d / "frame_times.csv", index=False)
                exported.append({"track": int(row.track), "folder": str(d), "n_frames": len(tt)})
    progress("lineage meshes")
    np.savez_compressed(out / "grid.npz", origin=grid["origin"], pitch=grid["pitch"], shape=np.asarray(grid["shape"]))
    np.savez_compressed(out / "vertex_labels.npz", **{f"frame_{k:04d}": res[k]["vertex_labels"] for k in range(T)})
    if conf is not None:
        np.savez_compressed(out / "conformal_factor.npz", **{f"frame_{k:04d}": conf[k] for k in range(T)})
    summary = {"frames": T, "needs_lineage": bool(needs), "n_tracks": int(len(lin_df)),
               "events": tr_df.event.value_counts().to_dict() if len(tr_df) else {},
               "max_pieces": int(tl_df.pieces_tracked.max()) if len(tl_df) else 0,
               "constricted_frames": int(neck_df.constricted.sum()) if len(neck_df) else 0,
               "exported_tracks": exported, "voxel_pitch": grid["pitch"], "neck_method": p["neck_method"]}
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if p["plots"]:
        _plots(out, tl_df, lin_df, tr_df, neck_df, div_df, p["neck_threshold"])
    progress("summary and plots")
    if verbose:
        print(f"Topology events: {summary['events'] or 'none'} | tracks {summary['n_tracks']} | lineage mode: {needs}")
    return summary


def run_lineage(target_folder, pipeline_params, min_frames=3):
    """Lineage mode: the whole pipeline on every exported track (Results/<scene>/TopologyEvents/Results/<track>/),
    then the comparison of the tracks (sister cells) with the usual scene comparison."""
    from PynamicMesh.core.pipelines import run_pipeline
    lm = Path(target_folder) / "TopologyEvents" / "Lineage_meshes"
    if not lm.is_dir():
        return None
    short = [d for d in lm.iterdir() if d.is_dir() and len(list(d.glob("*.obj"))) < int(min_frames)]
    hold = lm.parent / "Lineage_meshes_short"
    for d in short:                                    # tracks too short for the dynamic analyses are set aside
        ensure_dir(hold); os.replace(d, hold / d.name)
    if not any(d.is_dir() for d in lm.iterdir()):
        return None
    params = {k: v for k, v in dict(pipeline_params).items()
              if k not in ("compute_topology_events", "topology_params", "lineage_mode")}
    params["compute_topology_events"] = False
    # the tracks are pieces of ONE scene (different stages of its lineage), not scenes to compare with each other
    params["compare_scenes_flag"] = False
    return run_pipeline(str(lm), **params)


# --------------------------------------------------------------------------------------------------------------- #
#  plots
# --------------------------------------------------------------------------------------------------------------- #
def _plots(out, tl, lin, tr, necks, divs, thr):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(tl.time, tl.mesh_components, where="mid", label="mesh components")
    ax.step(tl.time, tl.solid_components, where="mid", ls="--", label="solid components")
    ax.step(tl.time, tl.separating_components, where="mid", ls=":", label="components after erosion (separating)")
    for r in (tr.itertuples() if len(tr) else []):
        ax.axvline(r.time, color={"division": "tab:red", "fusion": "tab:purple"}.get(r.event, "gray"), alpha=0.4)
    ax.set_xlabel("time"); ax.set_ylabel("pieces"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("topology of the sequence (red: divisions, purple: fusions)")
    fig.tight_layout(); fig.savefig(Path(out) / "plots" / "1_topology_timeline.png", dpi=200); plt.close(fig)
    if len(lin):
        fig, ax = plt.subplots(figsize=(12, 1.5 + 0.45 * len(lin)))
        ypos = {}
        def place(tid, y=[0]):
            if tid in ypos:
                return ypos[tid]
            ypos[tid] = y[0]; y[0] += 1
            for ch in eval(lin.set_index("track").loc[tid, "children"]):
                place(ch)
            return ypos[tid]
        for tid in lin[lin.parents == "[]"].track:
            place(int(tid))
        for tid in lin.track:
            place(int(tid))
        for r in lin.itertuples():
            y = ypos[r.track]
            ax.plot([r.start_time, r.end_time], [y, y], lw=4, solid_capstyle="round")
            ax.text(r.end_time, y, f"  {r.track} ({r.fate})", va="center", fontsize=7)
            for ch in eval(r.children):
                cr = lin.set_index("track").loc[ch]
                ax.plot([r.end_time, cr.start_time], [y, ypos[ch]], color="gray", lw=1)
        ax.set_yticks([]); ax.set_xlabel("time"); ax.grid(alpha=0.3, axis="x")
        ax.set_title("lineage tree (one line per surface between events)")
        fig.tight_layout(); fig.savefig(Path(out) / "plots" / "2_lineage_tree.png", dpi=200); plt.close(fig)
    if necks is not None and len(necks):
        has_c = "conformal_neck_index" in necks and necks.conformal_neck_index.notna().any()
        fig, axs = plt.subplots(2 if has_c else 1, 1, figsize=(12, 7.5 if has_c else 4.2), sharex=True, squeeze=False)
        ax = axs[0, 0]
        for tid, g in necks.groupby("track"):
            ax.plot(g.time, g.neck_ratio, "o-", ms=3, label=f"track {tid}")
        ax.axhline(thr, color="gray", ls="--", lw=0.8, label="constriction threshold")
        ax.set_ylim(0, None); ax.set_ylabel("neck radius / equivalent radius")
        ax.set_title("neck of every surface (narrowest bottleneck): falls towards 0 before a division")
        ax.legend(fontsize=7, ncol=4); ax.grid(alpha=0.3)
        if has_c:
            a2 = axs[1, 0]
            for tid, g in necks.groupby("track"):
                a2.plot(g.time, g.conformal_neck_index, "o-", ms=3)
            a2.axhline(0, color="gray", lw=0.6)
            a2.set_ylabel("conformal factor at the neck - median")
            a2.set_title("conformal neck index (surface Ricci flow): rises as the neck closes")
            a2.grid(alpha=0.3)
        axs[-1, 0].set_xlabel("time")
        fig.tight_layout(); fig.savefig(Path(out) / "plots" / "3_necks.png", dpi=200); plt.close(fig)
    if divs is not None and len(divs):
        cols = [c for c in ("volume_ratio", "area_ratio", "volume_conservation", "new_area_fraction") if c in divs]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        divs[cols].plot.bar(ax=ax[0]); ax[0].set_xlabel("division"); ax[0].set_title("division ratios"); ax[0].grid(alpha=0.3)
        if "angle_plane_normal_long_axis_deg" in divs:
            ax[1].bar(divs.division, divs.angle_plane_normal_long_axis_deg, color="tab:orange")
            ax[1].set_ylim(0, 90); ax[1].set_xlabel("division")
            ax[1].set_title("angle daughters' axis / mother's long axis (0 = Hertwig's rule)")
        fig.tight_layout(); fig.savefig(Path(out) / "plots" / "4_divisions.png", dpi=200); plt.close(fig)
