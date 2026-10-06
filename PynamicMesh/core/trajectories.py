"""
trajectories.py  —  vertex trajectories and their interpolation
=========================================================================

1. Consistent trajectories.  The frames have different vertex counts, so a trajectory is only defined
   through correspondences (p2p_i[j] = vertex of frame i-1 matched to vertex j of frame i).  Output:
   X ∈ R^{T × n₀ × 3} with the connectivity F₀ of the reference mesh — a *single* deforming mesh.
   trajectory_method='arap_tracking' (default, track_reference_mesh): the reference mesh is registered to every
   frame in turn — one-step prediction through the FM displacement field, robust (re-weighted) ARAP fit with an
   elastic memory of the reference shape, point-to-plane refinement on the true surface — so the triangulation keeps
   its integrity where the correspondences are poor (thin, strongly moving parts).
   Then an as-rigid-as-possible projection (Sorkine & Alexa 2007, rigidity_projection) restores the local shape of the
   reference mesh in every cell of every key frame and of every dense interpolation (rigidity_params).
   trajectory_method='chain' (build_trajectories, previous behaviour): the maps are composed backwards to frame 0 and
   every reference vertex takes the mean of the frame-i vertices pulled onto it (harmonic filling of the others);
   correspondence errors then accumulate and collapse / flip triangles on the thin, strongly moving parts.

2. Interpolation of the trajectories (continuous time):
      - cubic Hermite with Catmull–Rom / finite-difference / Kochanek–Bartels (tension, bias) tangents,
      - natural cubic splines and B-splines (scipy, C² curves through the samples),
      - **as-rigid-as-possible shape interpolation**: per-vertex cells (Sorkine & Alexa 2007, eq. 5–6)
        give the best rotation R_i and, by polar decomposition of the cell affine map A_i = R_i S_i,
        the stretch S_i between two consecutive frames.  Rotations are interpolated on SO(3) (quaternion
        slerp), stretches linearly (Alexa, Cohen-Or & Levin 2000), and the intermediate mesh is the
        solution of the Poisson system  L p = b  (eq. 8–9 of the paper, cotangent weights) with the
        Hermite trajectory as a weak positional prior.  The system matrix is factorised once.
   The ARAP stage is made robust to noisy / partly wrong correspondences: positive bounded weights (the
   cotangents of the clean reference frame), a normal-completed cell fit (no spurious stretch along the
   normal), rotation and stretch from the same polar decomposition (log-Euclidean stretch interpolation),
   harmonic filling of unreliable cells and an end-point correction that makes the curve pass exactly
   through every observed frame (see ARAPInterpolator).
   A leave-one-frame-out validation quantifies which scheme reproduces the observed frames best.

3. Kinematics (per vertex, per frame): velocity, acceleration, speed, Frenet curvature and torsion of the
   trajectories, path length / tortuosity, deformation gradients of the cells, Green–Lagrange strain,
   local ARAP rigidity energy E_i (eq. 3, how far each cell is from a rigid motion), kinetic energy.

Results/<scene>/Trajectories/
    trajectories.npy             (T, n0, 3)   reference-connectivity trajectories
    reference_faces.npy, coverage.csv, trajectories.vtp (polylines coloured by mean speed)
    tracking.csv                 per frame: FM inlier fraction, ICP acceptance, distance to the frame surface (tracking)
    interpolation/<scheme>/frame_####.obj   dense interpolated sequence (obj) + <scheme>.npy
    interpolation/<arap scheme>_diagnostics.csv   unreliable cells per segment, key-frame residuals
    interpolation/validation.csv           leave-one-out errors of every scheme
    kinematics/*.npy, kinematics/summary.csv, kinematics/arap_energy.csv
    plots/
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.spatial.transform import Rotation, Slerp
from tqdm.auto import tqdm

from PynamicMesh.core.dyn_common import pbar
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots
from PynamicMesh.core.dyn_common import (
    logged_stage, progress, vertex_normals,
    xp, GPU_AVAILABLE, to_gpu, to_cpu, as_faces, ensure_dir, load_frames, load_p2p_chain,
    cotan_laplacian, cotan_weights, vertex_areas, harmonic_inpaint, write_obj, save_vtp, smooth_derivative
)


# ----------------------------------------------------------------------------- #
#  1. Consistent trajectories through the p2p chain
# ----------------------------------------------------------------------------- #

def build_trajectories(frames, p2p, verbose=True):
    """
    frames: list of dicts with 'V', 'F' (dyn_common.load_frames); p2p: list from load_p2p_chain.
    Returns X (T, n0, 3), F0, coverage DataFrame.
    """
    T = len(frames)
    V0, F0 = frames[0]["V"], frames[0]["F"]
    n0 = V0.shape[0]
    W0, M0, _ = cotan_laplacian(V0, F0)
    X = np.zeros((T, n0, 3)); X[0] = V0
    chain = np.arange(n0)                                   # frame-0 index of every vertex of frame i
    coverage = [{"frame": 0, "covered_fraction": 1.0, "samples_per_ref_vertex": 1.0, "inpainted": 0}]
    for i in range(1, T):
        Vi, Fi = frames[i]["V"], frames[i]["F"]
        if p2p[i] is None or p2p[i].shape[0] != Vi.shape[0]:
            raise ValueError(f"Missing or inconsistent p2p map for frame {i}; run the functional-map stage first.")
        chain = chain[p2p[i]]                               # compose: vertex j of frame i → frame 0
        A = vertex_areas(Vi, Fi)
        pos = np.zeros((n0, 3)); wsum = np.zeros(n0)
        np.add.at(pos, chain, Vi * A[:, None]); np.add.at(wsum, chain, A)
        known = wsum > 0
        pos[known] /= wsum[known, None]
        X[i] = harmonic_inpaint(W0, pos, known) if not known.all() else pos
        coverage.append({"frame": i, "covered_fraction": float(known.mean()),
                         "samples_per_ref_vertex": float(Vi.shape[0] / n0), "inpainted": int((~known).sum())})
    cov = pd.DataFrame(coverage)
    if verbose:
        print(f"Trajectories: {T} frames on {n0} reference vertices, mean coverage {cov['covered_fraction'].mean():.3f}")
    return X, F0, cov


TRACKING_DEFAULTS = {
    "arap_iters": 6,            # local-global ARAP iterations per solve
    "robust_rounds": 3,         # robust re-weighting rounds of the fit to the FM prediction
    "icp_rounds": 3,            # refinement rounds towards the true surface of the frame
    "data_weight": 0.5,         # weight of the predicted positions (relative to the ARAP term, dimensionless)
    "icp_weight": 1.0,          # weight of the closest-point (point-to-plane) targets in the refinement
    "robust_scale": 2.5,        # Cauchy scale = robust_scale x 1.4826 x median residual
    "inpainted_confidence": 0.25,  # confidence of predictions from mesh vertices the FM did not reach
    "normal_threshold": 0.3,    # closest points whose normal disagrees more than this (cosine) are ignored
    "max_icp_distance": 4.0,    # ... and those farther than this x the mean edge length
    "reference_rigidity": 0.5,  # elastic memory β: ARAP w.r.t. (1-β)·previous state + β·reference shape. 0 = purely
                                # incremental (genuine non-isometric change is free, but the distortion accepted at every
                                # step accumulates); 1 = rigid w.r.t. the reference (resists growth / real stretching)
}


def _forward_field(Vp, Fp, Vi, m):
    """Displacement of every vertex of mesh i-1 towards mesh i from the p2p map m (mesh i -> mesh i-1):
    mean of the matched vertices, harmonically inpainted where no vertex of mesh i lands. Returns (fwd, hit)."""
    pos = np.zeros((Vp.shape[0], 3)); wsum = np.zeros(Vp.shape[0])
    np.add.at(pos, m, Vi); np.add.at(wsum, m, 1.0)
    hit = wsum > 0
    pos[hit] /= wsum[hit, None]
    if not hit.all():
        Wp, _, _ = cotan_laplacian(Vp, Fp)
        pos = harmonic_inpaint(Wp, pos, hit)
    return pos - Vp, hit


def _arap_rotations(i, j, w, e_rest, Y, n, F=None, N_rest=None, wn=None):
    """Best rotation of every one-ring (ARAP local step): R_v = argmax tr(Rᵀ S_v), S_v = Σ_j w_vj e'_vj e_vjᵀ, det +1,
    completed by the normals (+ wn_v n'_v n_vᵀ) so nearly flat one-rings have a well-defined rotation about their
    normal. Covariances by bincount, one batched SVD (GPU through xp when available)."""
    ep = Y[i] - Y[j]
    S = np.zeros((n, 3, 3))
    for r_ in range(3):
        for c_ in range(3):
            S[:, r_, c_] = np.bincount(i, w * ep[:, r_] * e_rest[:, c_], minlength=n)
    if F is not None and N_rest is not None and wn is not None:
        N_cur = vertex_normals(Y, F)
        S += wn[:, None, None] * N_cur[:, :, None] * N_rest[:, None, :]
    U, _, Vt = _batched_svd(S)
    R = U @ Vt
    neg = np.linalg.det(R) < 0
    if neg.any():
        U[neg, :, -1] *= -1
        R[neg] = U[neg] @ Vt[neg]
    return R


def _arap_solve(rests, betas, Y0, F, W, L, target, weight, iters, mu):
    """
    min_Y  Σ_r β_r Σ_ij w_ij ||(y_i − y_j) − R^r_i (x^r_i − x^r_j)||² + μ Σ_v weight_v ||y_v − target_v||²
    (several rest shapes x^r with Σ β_r = 1: the ARAP energy w.r.t. the previous state and w.r.t. the reference) by
    local (per-vertex best rotation for every rest shape) / global (one sparse Poisson solve: all the ARAP terms share
    the matrix L) iterations, started at Y0.
    """
    Wc = W.tocoo(); i, j, w = Wc.row, Wc.col, Wc.data
    n = Y0.shape[0]
    edges = [x[i] - x[j] for x in rests]
    normals = [vertex_normals(x, F) for x in rests]
    wns = [np.bincount(i, w * np.sum(e ** 2, axis=1), minlength=n) for e in edges]   # Σ w |e|² of the one-ring
    D = sparse.diags(mu * weight)
    lu = splinalg.splu((L + D).tocsc())
    rhs_data = (mu * weight)[:, None] * target
    Y = Y0.copy()
    for _ in range(max(int(iters), 1)):
        b = np.zeros((n, 3))
        for x, e, beta, Nr, wn in zip(rests, edges, betas, normals, wns):
            if beta <= 0:
                continue
            R = _arap_rotations(i, j, w, e, Y, n, F, Nr, wn)
            Te = 0.5 * np.einsum("nij,nj->ni", R[i] + R[j], e)
            b += beta * np.stack([np.bincount(i, w * Te[:, c], minlength=n) for c in range(3)], axis=1)
        Y = lu.solve(b + rhs_data)
    return Y


def track_reference_mesh(frames, p2p, params=None, verbose=True):
    """
    Trajectories of the reference mesh by ROBUST ARAP TRACKING (replaces the composition of the p2p chain).

    Why: build_trajectories composes the maps back to frame 0 and averages the vertices pulled onto every reference
    vertex. Functional maps are least accurate on thin, strongly moving parts (legs, tails, protrusions: the
    low-frequency basis cannot tell positions along a thin part apart, symmetric parts get confused); the composed
    errors accumulate frame after frame, fewer and fewer reference vertices are reached (coverage decays), and the
    averaging / harmonic filling of the rest produces collapsed, sheared and flipped triangles exactly there.
    Every interpolation scheme then interpolates these damaged key frames.

    Here the reference mesh is REGISTERED to every frame in turn, starting from its registration to the previous one:
      1. prediction: every reference vertex (on mesh i-1) is moved by the FM displacement field of that single
         transition (no chain composition, errors do not compound);
      2. robust ARAP fit to the prediction: the reference mesh deforms as-rigidly-as-possible w.r.t. a blend of its
         previous state and the reference shape (elastic memory, reference_rigidity β: the previous state lets genuine
         non-isometric change such as growth accumulate, the reference term stops the small distortions accepted at
         every step from accumulating), and wrong matches are down-weighted by iteratively re-weighted Cauchy weights
         (they would need large distortion);
      3. refinement (ICP): point-to-plane attraction to the true surface of frame i where the normals agree, so the
         result fits the actual geometry and not a noisy average.
    The connectivity F0 is never changed and every cell stays close to a rigid motion of its previous state, so the
    triangulation keeps its integrity. Returns X (T, n0, 3), F0 and a diagnostics DataFrame (one row per frame).
    """
    p = {**TRACKING_DEFAULTS, **(params or {})}
    T = len(frames)
    V0, F0 = np.asarray(frames[0]["V"], float), as_faces(frames[0]["F"])
    n0 = V0.shape[0]
    W = robust_edge_weights(V0, F0)
    L = graph_laplacian(W)
    mu_scale = float(L.diagonal().mean())
    h = float(np.mean(np.linalg.norm(V0[F0[:, 0]] - V0[F0[:, 1]], axis=1)))
    X = np.zeros((T, n0, 3)); X[0] = V0
    rows = [{"frame": 0, "covered_fraction": 1.0, "reliable_fraction": 1.0, "fm_inlier_fraction": 1.0,
             "icp_accepted_fraction": np.nan, "mean_surface_distance": 0.0, "p95_surface_distance": 0.0,
             "mean_prediction_residual": 0.0, "samples_per_ref_vertex": 1.0, "inpainted": 0}]
    from scipy.spatial import cKDTree
    for k in pbar(range(1, T), desc="ARAP tracking"):
        Vp, Fp = np.asarray(frames[k - 1]["V"], float), as_faces(frames[k - 1]["F"])
        Vi, Fi = np.asarray(frames[k]["V"], float), as_faces(frames[k]["F"])
        m = p2p[k]
        if m is None or m.shape[0] != Vi.shape[0]:
            raise ValueError(f"Missing or inconsistent p2p map for frame {k}; run the functional-map stage first.")
        Yp = X[k - 1]
        # 1. one-step prediction through the FM displacement field of mesh k-1
        fwd, hit = _forward_field(Vp, Fp, Vi, np.asarray(m, dtype=np.int64))
        _, u = cKDTree(Vp).query(Yp)
        pred = Yp + fwd[u]
        conf = np.where(hit[u], 1.0, float(p["inpainted_confidence"]))
        # 2. robust ARAP fit (rest = previous registered state)
        mu = float(p["data_weight"]) * mu_scale
        beta = float(np.clip(p["reference_rigidity"], 0.0, 1.0))
        rests, betas = (Yp, V0), (1.0 - beta, beta)
        wgt = conf.copy(); Y = Yp.copy()
        for _ in range(int(p["robust_rounds"])):
            Y = _arap_solve(rests, betas, Y, F0, W, L, pred, wgt, p["arap_iters"], mu)
            r = np.linalg.norm(Y - pred, axis=1)
            sig = max(float(p["robust_scale"]) * 1.4826 * float(np.median(r)), 0.05 * h)
            wgt = conf * sig ** 2 / (sig ** 2 + r ** 2)
        inlier = float(np.mean(wgt > 0.5 * conf))
        # 3. refinement towards the true surface of frame k (point-to-plane, normal-compatible closest points)
        tree = cKDTree(Vi); Ni = vertex_normals(Vi, Fi)
        acc_frac = np.nan
        for _ in range(int(p["icp_rounds"])):
            NY = vertex_normals(Y, F0)
            d, q = tree.query(Y)
            proj = Y + np.einsum("ij,ij->i", Vi[q] - Y, Ni[q])[:, None] * Ni[q]
            acc = (np.einsum("ij,ij->i", NY, Ni[q]) > float(p["normal_threshold"])) & (d < float(p["max_icp_distance"]) * h)
            acc_frac = float(acc.mean())
            w_icp = float(p["icp_weight"]) * acc
            wt = wgt + w_icp
            tgt = (wgt[:, None] * pred + w_icp[:, None] * proj) / np.maximum(wt, 1e-12)[:, None]
            Y = _arap_solve(rests, betas, Y, F0, W, L, tgt, np.maximum(wt, 1e-6), p["arap_iters"], mu)
        X[k] = Y
        d_fin, _ = tree.query(Y)
        rows.append({"frame": k, "covered_fraction": float(np.mean(hit[u])), "reliable_fraction": float(np.mean(hit[u])),
                     "fm_inlier_fraction": inlier, "icp_accepted_fraction": acc_frac,
                     "mean_surface_distance": float(d_fin.mean()), "p95_surface_distance": float(np.percentile(d_fin, 95)),
                     "mean_prediction_residual": float(np.linalg.norm(Y - pred, axis=1).mean()),
                     "samples_per_ref_vertex": float(Vi.shape[0] / n0), "inpainted": int((~hit[u]).sum())})
    diag = pd.DataFrame(rows)
    if verbose:
        print(f"Trajectories (ARAP tracking): {T} frames on {n0} reference vertices, FM inliers {diag['fm_inlier_fraction'].iloc[1:].mean():.3f}, "
              f"mean distance to the frame surfaces {diag['mean_surface_distance'].iloc[1:].mean():.3g} (mean edge {h:.3g})")
    return X, F0, diag


RIGIDITY_DEFAULTS = {
    "enabled": True,            # as-rigid-as-possible projection of the trajectories (Sorkine & Alexa 2007)
    "mode": "asap",             # 'asap' (as-similar-as-possible: every one-ring may rotate AND scale uniformly - follows
                                # growth / shrinking, forbids shear, sliding and collapse) | 'arap' (rotations only, as in
                                # the paper - best for near-isometric motion such as walking animals)
    "data_weight": 0.005,       # pull towards the trajectory positions (relative to the rigidity term): lower = more rigid
                                # (cleaner triangles, but genuine stretching is followed less), higher = closer to the input
    "iterations": 15,           # local / global iterations per frame (the system matrix is factored once)
    "keyframes": True,          # project the trajectories of the observed frames (before the interpolation)
    "interpolations": True,     # project every dense interpolation (the raw ones are kept in interpolation/raw/)
}


def rigidity_projection(V0, F0, frames_xyz, mode="asap", data_weight=0.005, iterations=15, verbose=False):
    """
    As-rigid-as-possible projection of a sequence of positions of the reference mesh (Sorkine & Alexa, "As-rigid-as-
    possible surface modeling", SGP 2007) with SOFT positional constraints - every frame X_t is replaced by
        argmin_P  Σ_i Σ_{j∈N(i)} w_ij ||(p_i − p_j) − s_i R_i (v_i − v_j)||²  +  μ Σ_i ||p_i − x_i(t)||²
    v = reference mesh (rest shape), w_ij its cotangent weights, R_i the best rotation of every one-ring (local step,
    eq. 5-6 of the paper, normal-completed), s_i = 1 ('arap') or the best uniform scale of the one-ring ('asap': the
    cell may grow / shrink but not shear), μ = data_weight · mean(diag L).
    Global step: (L + μ I) P = b(R) + μ X - the matrix does not depend on the frame or the iteration, so it is
    factored ONCE for the whole sequence and every iteration costs one back-substitution (paper, Sec. 3).
    Why: the trajectories come from noisy correspondences, so the vertices slide along the surface and the triangles of
    the strongly moving parts get sheared / collapsed; per-vertex interpolation (Hermite, splines) then also shortens
    the parts that rotate between two frames. The projection keeps the motion (data term) but restores the local
    shape of the reference mesh in every cell (rigidity term). Returns the projected sequence (same shape).
    """
    V0 = np.asarray(V0, dtype=np.float64); F0 = as_faces(F0)
    Xs = np.asarray(frames_xyz, dtype=np.float64)
    n = V0.shape[0]
    if mode not in ("arap", "asap"):
        raise ValueError("rigidity mode must be 'arap' or 'asap'")
    W = robust_edge_weights(V0, F0); L = graph_laplacian(W)
    mu = float(data_weight) * float(L.diagonal().mean())
    lu = splinalg.splu((L + sparse.identity(n, format="csr") * mu).tocsc())
    Wc = W.tocoo(); i, j, w = Wc.row, Wc.col, Wc.data
    e = V0[i] - V0[j]
    N_rest = vertex_normals(V0, F0)
    wn = np.bincount(i, w * np.sum(e ** 2, axis=1), minlength=n)
    den = np.maximum(wn, 1e-300)
    out = np.empty_like(Xs)
    it = pbar(range(Xs.shape[0]), desc="Rigidity projection")
    for f in it:
        X = Xs[f]; Y = X.copy()
        for _ in range(max(int(iterations), 1)):
            R = _arap_rotations(i, j, w, e, Y, n, F0, N_rest, wn)
            if mode == "asap":
                Re = np.einsum("nij,nj->ni", R[i], e)
                sc = np.clip(np.bincount(i, w * np.sum((Y[i] - Y[j]) * Re, axis=1), minlength=n) / den, 0.2, 5.0)
                R = R * sc[:, None, None]
            Te = 0.5 * np.einsum("nij,nj->ni", R[i] + R[j], e)
            b = np.stack([np.bincount(i, w * Te[:, c], minlength=n) for c in range(3)], axis=1)
            Y = lu.solve(b + mu * X)
        out[f] = Y
    return out


def mesh_integrity(V0, F0, frames_xyz):
    """
    Integrity of the deforming mesh w.r.t. the reference, per frame (no ground truth needed):
      edge_length_change   RMS of (L − L0) / mean(L0) over the edges (includes genuine stretching / growth)
      shape_change         RMS of log(L / L0) after removing the frame's median scale (shear / sliding, growth-free)
      degraded_faces       fraction of triangles whose shape quality fell below 40 % of the reference one or whose area
                           changed more than 4x w.r.t. the frame's median scale - the visible 'broken triangulation'
    """
    V0 = np.asarray(V0, float); F0 = as_faces(F0)
    E = _unique_edges(F0); L0 = np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1)

    def quality(P):
        a, b = P[F0[:, 1]] - P[F0[:, 0]], P[F0[:, 2]] - P[F0[:, 0]]
        A = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
        l2 = np.sum(a ** 2, 1) + np.sum(b ** 2, 1) + np.sum((a - b) ** 2, 1)
        return 4 * np.sqrt(3) * A / np.maximum(l2, 1e-300), A
    q0, A0 = quality(V0)
    rows = []
    for P in np.asarray(frames_xyz, float):
        L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
        q, A = quality(P)
        lr = np.log(np.maximum(L, 1e-300) / np.maximum(L0, 1e-300)); lr -= np.median(lr)
        ratio = A / np.maximum(A0, 1e-300); s2 = np.median(ratio)
        bad = (q < 0.4 * q0) | (ratio < 0.25 * s2) | (ratio > 4 * s2)
        rows.append({"edge_length_change": float(np.sqrt(np.mean(((L - L0) / L0.mean()) ** 2))),
                     "shape_change": float(np.sqrt(np.mean(lr ** 2))), "degraded_faces": float(bad.mean())})
    return pd.DataFrame(rows)


def mesh_to_mesh_displacements(frames, p2p, out_dir, verbose=True):
    """
    Vertex-to-vertex displacement fields between CONSECUTIVE MESHES, for all vertices of each mesh,
    straight from the functional-map assignment (no reference mesh):

      backward_T{i}_T{i-1}.npy  (n_i, 3)     d_j = V_{i-1}[p2p_i[j]] − V_i[j]      exact for every vertex j of mesh i
      forward_T{i-1}_T{i}.npy   (n_{i-1}, 3) d_u = mean{ V_i[j] : p2p_i[j] = u } − V_{i-1}[u]
                                              (vertices of mesh i−1 hit by no vertex of mesh i are harmonically
                                              inpainted; forward_hit_T{i-1}_T{i}.npy marks the directly matched ones)

    The forward field is the surface velocity × dt of mesh i−1: its integral curve through every vertex is the
    vertex trajectory, and the family of all of them is the trajectory of the surface itself (trajectories.npy
    is exactly the composition of these steps from the reference frame).  surface_trajectory.csv summarises each
    transition (mean / max displacement, normal and tangential parts, direct-match fraction).
    """
    out = ensure_dir(Path(out_dir) / "displacements")
    rows = []
    T = len(frames)
    for i in range(1, T):
        Vp, Fp = frames[i - 1]["V"], frames[i - 1]["F"]; Vi, Fi = frames[i]["V"], frames[i]["F"]
        m = p2p[i]
        if m is None or m.shape[0] != Vi.shape[0]:
            continue
        back = Vp[m] - Vi
        np.save(out / f"backward_T{i:04d}_T{i - 1:04d}.npy", back)
        A = vertex_areas(Vi, Fi)
        pos = np.zeros((Vp.shape[0], 3)); wsum = np.zeros(Vp.shape[0])
        np.add.at(pos, m, Vi * A[:, None]); np.add.at(wsum, m, A)
        hit = wsum > 0; pos[hit] /= wsum[hit, None]
        if not hit.all():
            Wp, _, _ = cotan_laplacian(Vp, Fp)
            pos = harmonic_inpaint(Wp, pos, hit)
        fwd = pos - Vp
        np.save(out / f"forward_T{i - 1:04d}_T{i:04d}.npy", fwd); np.save(out / f"forward_hit_T{i - 1:04d}_T{i:04d}.npy", hit)
        N = vertex_normals(Vp, Fp); dn = np.einsum("ij,ij->i", fwd, N); dt_ = np.linalg.norm(fwd - dn[:, None] * N, axis=1)
        Ap = vertex_areas(Vp, Fp); w = Ap / Ap.sum()
        mag = np.linalg.norm(fwd, axis=1)
        rows.append(dict(transition=f"T{i - 1:04d}->T{i:04d}", n_source_vertices=int(Vp.shape[0]), n_target_vertices=int(Vi.shape[0]),
                         direct_match_fraction=float(hit.mean()), mean_displacement=float(w @ mag), max_displacement=float(mag.max()),
                         mean_normal_displacement=float(w @ dn), mean_abs_normal=float(w @ np.abs(dn)), mean_tangential=float(w @ dt_),
                         normal_share=float((w @ dn ** 2) / max(w @ mag ** 2, 1e-300)),
                         net_translation=float(np.linalg.norm(w @ fwd))))
    df = pd.DataFrame(rows); df.to_csv(out / "surface_trajectory.csv", index=False)
    if verbose and len(df):
        print(f"Mesh-to-mesh displacements: {len(df)} transitions, mean direct-match fraction {df['direct_match_fraction'].mean():.3f}, "
              f"mean displacement {df['mean_displacement'].mean():.4g} (normal share {df['normal_share'].mean():.2f})")
    return df


# ----------------------------------------------------------------------------- #
#  2. Interpolators
# ----------------------------------------------------------------------------- #

def hermite_tangents(X, t, method="catmull_rom", tension=0.0, bias=0.0):
    """
    Tangents m_k of the cubic Hermite curve through (t_k, X_k).
    catmull_rom       : m_k = (X_{k+1} − X_{k−1}) / (t_{k+1} − t_{k−1})
    finite_difference : average of the one-sided slopes
    kochanek_bartels  : (1−c)(1+b)/2 · in-slope + (1−c)(1−b)/2 · out-slope with tension c and bias b
    """
    T = X.shape[0]
    m = np.zeros_like(X)
    dX = np.diff(X, axis=0); dt = np.diff(t)[:, None, None] if X.ndim == 3 else np.diff(t)[:, None]
    slopes = dX / dt
    m[0] = slopes[0]; m[-1] = slopes[-1]
    if method == "catmull_rom":
        for k in range(1, T - 1):
            m[k] = (X[k + 1] - X[k - 1]) / (t[k + 1] - t[k - 1])
    elif method == "finite_difference":
        m[1:-1] = 0.5 * (slopes[:-1] + slopes[1:])
    elif method == "kochanek_bartels":
        c, b = tension, bias
        m[1:-1] = (1 - c) * (1 + b) / 2 * slopes[:-1] + (1 - c) * (1 - b) / 2 * slopes[1:]
    else:
        raise ValueError(f"Unknown tangent method '{method}'.")
    return m


def hermite_interpolate(X, t, tq, method="catmull_rom", tension=0.0, bias=0.0, derivative=0):
    """Cubic Hermite interpolation of the (T, ...) samples at query times tq (also 1st / 2nd derivatives)."""
    X = np.asarray(X, dtype=np.float64); t = np.asarray(t, dtype=np.float64); tq = np.atleast_1d(tq)
    m = hermite_tangents(X, t, method, tension, bias)
    out = np.zeros((len(tq),) + X.shape[1:])
    for q, tv in enumerate(tq):
        k = int(np.clip(np.searchsorted(t, tv, side="right") - 1, 0, len(t) - 2))
        h = t[k + 1] - t[k]; s = (tv - t[k]) / h
        if derivative == 0:
            h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s
            h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
            out[q] = h00 * X[k] + h10 * h * m[k] + h01 * X[k + 1] + h11 * h * m[k + 1]
        elif derivative == 1:
            d00 = (6 * s ** 2 - 6 * s) / h; d10 = 3 * s ** 2 - 4 * s + 1
            d01 = (-6 * s ** 2 + 6 * s) / h; d11 = 3 * s ** 2 - 2 * s
            out[q] = d00 * X[k] + d10 * m[k] + d01 * X[k + 1] + d11 * m[k + 1]
        else:
            e00 = (12 * s - 6) / h ** 2; e10 = (6 * s - 4) / h; e01 = (-12 * s + 6) / h ** 2; e11 = (6 * s - 2) / h
            out[q] = e00 * X[k] + e10 * m[k] + e01 * X[k + 1] + e11 * m[k + 1]
    return out


def spline_interpolate(X, t, tq, kind="natural_cubic", k=3, derivative=0):
    """C² natural cubic spline (scipy CubicSpline) or B-spline of order k through the samples."""
    X = np.asarray(X, dtype=np.float64); shape = X.shape[1:]
    flat = X.reshape(X.shape[0], -1)
    if kind == "natural_cubic":
        cs = CubicSpline(t, flat, axis=0, bc_type="natural")
        val = cs(tq, nu=derivative)
    elif kind == "bspline":
        k = int(min(k, len(t) - 1))
        bs = make_interp_spline(t, flat, k=k, axis=0)
        val = bs(tq, nu=derivative) if derivative else bs(tq)
    else:
        raise ValueError(f"Unknown spline kind '{kind}'.")
    return np.asarray(val).reshape((len(np.atleast_1d(tq)),) + shape)


# ----------------------------------------------------------------------------- #
#  ARAP building blocks: robust edge weights, cell fits, regularised fields
# ----------------------------------------------------------------------------- #

def _unique_edges(F):
    """Undirected edges (m, 2) of a triangle mesh, i < j."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    E.sort(axis=1)
    return np.unique(E, axis=0)


def robust_edge_weights(V, F, kind="cotan", bounds=(1e-2, 50.0)):
    """
    Symmetric, strictly positive and bounded edge weights (sparse n × n, zero diagonal).

    kind='cotan'  : w_ij = ½ (cot α_ij + cot β_ij); corners of degenerate triangles contribute 0, and
                    the result is clamped to [bounds[0], bounds[1]] × median(w > 0).  Clamping from below
                    removes the negative weights of obtuse / flipped triangles (they make the Laplacian
                    indefinite), clamping from above removes the 10⁶–10¹⁰ weights of collapsed triangles.
    kind='uniform': w_ij = 1 (graph Laplacian, the most robust choice for very noisy meshes).
    Every edge of F is present in the pattern, so the one-ring graph is never disconnected.
    """
    V = np.asarray(V, dtype=np.float64); F = as_faces(F); n = V.shape[0]
    edges = _unique_edges(F)
    if kind == "uniform":
        w = np.ones(len(edges))
    elif kind == "cotan":
        rows, cols, vals = [], [], []
        for c in range(3):
            o, a, b = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
            u, v = V[a] - V[o], V[b] - V[o]
            cr = np.linalg.norm(np.cross(u, v), axis=1)
            nn = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            ok = cr > 1e-8 * np.maximum(nn, 1e-300)
            cot = np.where(ok, np.einsum("ij,ij->i", u, v) / np.where(ok, cr, 1.0), 0.0)
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            rows.append(lo); cols.append(hi); vals.append(0.5 * cot)
        Wt = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        w = np.asarray(Wt[edges[:, 0], edges[:, 1]]).ravel()
        pos = w[w > 0]
        med = np.median(pos) if pos.size else 1.0
        w = np.clip(w, bounds[0] * med, bounds[1] * med)
    else:
        raise ValueError(f"Unknown weight kind '{kind}'.")
    W = sparse.coo_matrix((np.concatenate([w, w]), (np.concatenate([edges[:, 0], edges[:, 1]]),
                                                    np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n))
    return W.tocsr()


def _sanitize_weights(Wij, bounds=(1e-2, 50.0)):
    """Symmetric, positive, bounded copy of an externally computed weight matrix (same sparsity pattern)."""
    W = sparse.csr_matrix(Wij, dtype=np.float64, copy=True)
    W = ((W + W.T) * 0.5).tocsr()
    W.setdiag(0.0); W.eliminate_zeros()
    pos = W.data[W.data > 0]
    med = np.median(pos) if pos.size else 1.0
    W.data = np.clip(W.data, bounds[0] * med, bounds[1] * med)
    return W


def graph_laplacian(W):
    """L = diag(W 1) − W  (positive semi-definite for positive weights)."""
    return (sparse.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsr()


def _batched_svd(A):
    U, s, Vt = xp.linalg.svd(to_gpu(np.ascontiguousarray(A)))
    return to_cpu(U), to_cpu(s), to_cpu(Vt)


def fit_cells(V_src, V_dst, F, W, reg=1e-6):
    """
    Per-vertex (one-ring) cell fit between two embeddings with the same connectivity and positive weights W.

    The one-ring of a smooth surface is almost planar, so the covariance Σ w e eᵀ is nearly rank 2 and a plain
    affine fit A = (Σ w e' eᵀ)(Σ w e eᵀ)⁻¹ is ill-posed along the normal: noise of the neighbours in the
    normal direction is divided by a ~0 eigenvalue and produces enormous normal stretches (the radial spikes).
    Here the cell is completed by its normal as a 4th "edge" of length ℓ (the cell's RMS edge length) and
    weight Σ_j w_ij, mapped from n_src to n_dst, which makes the fit well posed and fixes the normal stretch
    to 1 (a surface carries no thickness information).

    Returns a dict with
      R_arap  (n,3,3)  Procrustes rotation of the covariance, eq. 5–6 (det > 0)
      R_polar (n,3,3), sigma (n,3), axes (n,3,3)  polar decomposition A = R_polar · axes diag(sigma) axesᵀ
      reflected (n,)   det A ≤ 0 (inverted cell)
      E (n,)           ARAP energy Σ_j w_ij ||e'_ij − R_arap e_ij||²  (eq. 3)
      residual (n,)    relative residual of the affine fit (non-affine part: correspondence errors, noise)
    """
    V_src = np.asarray(V_src, dtype=np.float64); V_dst = np.asarray(V_dst, dtype=np.float64)
    n = V_src.shape[0]
    Wc = W.tocoo(); i, j, w = Wc.row, Wc.col, Wc.data
    e = V_src[i] - V_src[j]; ep = V_dst[i] - V_dst[j]
    wsum = np.bincount(i, w, minlength=n)
    l2 = np.bincount(i, w * np.sum(e ** 2, axis=1), minlength=n) / np.maximum(wsum, 1e-300)
    N0 = vertex_normals(V_src, F); N1 = vertex_normals(V_dst, F)
    wn = wsum * l2                                             # weight × ℓ² of the normal "edge"

    def outer_sum(a, b, extra):
        C = np.zeros((n, 3, 3))
        for r in range(3):
            for c in range(3):
                C[:, r, c] = np.bincount(i, w * a[:, r] * b[:, c], minlength=n)
        return C + extra

    Cee = outer_sum(e, e, wn[:, None, None] * N0[:, :, None] * N0[:, None, :])
    Cpe = outer_sum(ep, e, wn[:, None, None] * N1[:, :, None] * N0[:, None, :])     # A = Cpe Cee⁻¹  (A e ≈ e')
    tr_ = np.trace(Cee, axis1=1, axis2=2)
    tr_ = np.where(tr_ > 0, tr_, 1.0)
    A = Cpe @ np.linalg.inv(Cee + (reg * tr_ / 3.0)[:, None, None] * np.eye(3)[None])

    # polar decomposition A = R S with R ∈ SO(3)
    U, sig, Vt = _batched_svd(A)
    d = np.sign(np.linalg.det(U @ Vt)); d[d == 0] = 1.0
    U[:, :, 2] *= d[:, None]; sig[:, 2] *= d
    R_polar = U @ Vt
    axes = np.transpose(Vt, (0, 2, 1))                         # columns = principal stretch directions
    reflected = sig[:, 2] <= 0

    # Procrustes rotation of the covariance Σ w e e'ᵀ (eq. 5–6), same normal completion
    Uc, _, Vtc = _batched_svd(np.transpose(Cpe, (0, 2, 1)))
    R_arap = np.transpose(Vtc, (0, 2, 1)) @ np.transpose(Uc, (0, 2, 1))
    neg = np.linalg.det(R_arap) < 0
    if neg.any():
        Uc[neg, :, -1] *= -1
        R_arap[neg] = np.transpose(Vtc[neg], (0, 2, 1)) @ np.transpose(Uc[neg], (0, 2, 1))

    E = np.bincount(i, w * np.sum((ep - np.einsum("nij,nj->ni", R_arap[i], e)) ** 2, axis=1), minlength=n)
    Ae = np.einsum("nij,nj->ni", A[i], e)
    num = np.bincount(i, w * np.sum((Ae - ep) ** 2, axis=1), minlength=n)
    den = np.bincount(i, w * 0.5 * (np.sum(e ** 2, axis=1) + np.sum(ep ** 2, axis=1)), minlength=n)
    residual = num / np.maximum(den, 1e-300)
    residual[wsum <= 0] = np.inf
    return {"R_arap": R_arap, "R_polar": R_polar, "sigma": sig, "axes": axes, "reflected": reflected,
            "E": E, "residual": residual}


def cell_transforms(V_src, V_dst, F, Wij=None, with_stretch=True, reg=1e-6):
    """
    Per-vertex cell transformations between two embeddings with the same connectivity.
    R_i is the Procrustes rotation of S_i = Σ_j w_ij e_ij e'_ijᵀ (eq. 5–6, det > 0); with `with_stretch` the
    normal-completed affine fit of the cell (see fit_cells) is polar-decomposed A_i = R̃_i S_i and S_i
    (symmetric positive definite) is returned.  Weights are made symmetric, positive and bounded first.
    Returns R (n,3,3), S (n,3,3) or None, E (n,) the per-cell ARAP energy (eq. 3).
    """
    W = robust_edge_weights(V_src, F) if Wij is None else _sanitize_weights(Wij)
    fit = fit_cells(V_src, V_dst, F, W, reg=reg)
    S = None
    if with_stretch:
        s = np.maximum(np.abs(fit["sigma"]), 1e-6)
        S = fit["axes"] @ (s[:, :, None] * np.transpose(fit["axes"], (0, 2, 1)))
    return fit["R_arap"], S, fit["E"]


def _sym_to_vec(M):
    return np.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]], axis=1)


def _vec_to_sym(v):
    M = np.zeros((v.shape[0], 3, 3))
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = v[:, 0], v[:, 1], v[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = v[:, 3]; M[:, 0, 2] = M[:, 2, 0] = v[:, 4]; M[:, 1, 2] = M[:, 2, 1] = v[:, 5]
    return M


class ARAPInterpolator:
    """
    Robust as-rigid-as-possible interpolation of a consistent mesh sequence X (T, n, 3) with connectivity F.

    For t ∈ [t_k, t_{k+1}], τ = (t − t_k)/(t_{k+1} − t_k):
        T_i(τ) = exp(τ log R_i) · exp(τ log S_i)      (mode='polar', Alexa et al. 2000, log-Euclidean stretch)
        T_i(τ) = exp(τ log R_i)                        (mode='rigid')
    and the vertices solve  (L + μ M) p = b(T) + μ M p_prior,  b_i = Σ_j w_ij/2 (T_i + T_j) e_ij  (eq. 8–9),
    p_prior the Hermite trajectory and μ = prior_weight × (lowest non-constant eigenvalue of L), i.e. the prior
    only fixes the translation and barely touches the shape whatever the resolution of the mesh,
    with L = diag(W1) − W built from the SAME positive weights W as b (identity transforms reproduce X_k exactly).

    Robustness measures (all of them were missing and together caused the spikes):
      * normal-completed cell fit → no spurious stretch along the surface normal (fit_cells),
      * rotation and stretch from the SAME polar decomposition, so T_i(1) = A_i,
      * positive, bounded weights (default: cotangents of the reference frame X_0, the original clean mesh),
        so the Poisson matrix is an SPD M-matrix and cannot amplify errors of b,
      * unreliable cells (inverted, rotation > max_rotation_deg — where the axis of the log map is ambiguous —,
        stretch outside stretch_bounds, non-affine residual outliers) get their rotation / log-stretch from
        their neighbours by a screened harmonic fill; field_smoothing additionally denoises both fields,
      * endpoint correction p(τ) += (1−τ)(X_k − P_k(0)) + τ (X_{k+1} − P_k(1)): the curve passes exactly through
        every observed frame (no "pop" at the key frames); the non-affine part of the motion is blended linearly.

    diagnostics: list of per-segment dicts (fraction of unreliable cells, reasons, key-frame residual).
    """

    def __init__(self, X, F, t=None, mode="polar", prior_weight=1e-2, prior_method="catmull_rom",
                 weights="reference", weight_kind="cotan", weight_bounds=(1e-2, 50.0),
                 stretch_bounds=(1 / 3, 3.0), max_rotation_deg=90.0, residual_outlier=4.0,
                 residual_floor=0.05, field_smoothing=1.0, endpoint_correction=True,
                 clip_negative=None, verbose=False):
        if clip_negative is not None:
            warnings.warn("ARAPInterpolator: 'clip_negative' is obsolete, weights are always positive and bounded.")
        if mode not in ("polar", "rigid"):
            raise ValueError(f"Unknown ARAP mode '{mode}'.")
        if weights not in ("reference", "segment"):
            raise ValueError("weights must be 'reference' or 'segment'.")
        self.X = np.asarray(X, dtype=np.float64); self.F = as_faces(F)
        self.T, self.n, _ = self.X.shape
        if self.T < 2:
            raise ValueError("ARAP interpolation needs at least two frames.")
        self.t = np.arange(self.T, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
        self.mode, self.prior_method, self.mu = mode, prior_method, prior_weight
        self.weights, self.weight_kind, self.weight_bounds = weights, weight_kind, weight_bounds
        self.log_bounds = (np.log(stretch_bounds[0]), np.log(stretch_bounds[1]))
        self.max_angle = np.radians(max_rotation_deg)
        self.residual_outlier, self.residual_floor = residual_outlier, residual_floor
        self.field_smoothing, self.endpoint_correction = field_smoothing, endpoint_correction

        A0 = vertex_areas(self.X[0], self.F)
        self.Mn = np.maximum(A0 / max(A0.mean(), 1e-300), 1e-3)          # normalised lumped mass
        self._W, self._L, self._lu = {}, {}, {}
        self.rotvec, self.log_eval, self.log_axes, self.E, self.reliable, self.diagnostics = [], [], [], [], [], []
        for k in range(self.T - 1):
            self._segment_fit(k)
        self._r0, self._r1 = {}, {}
        frac = np.array([d["unreliable_fraction"] for d in self.diagnostics])
        if verbose:
            print(f"ARAP ({mode}): unreliable cells per segment {frac.mean():.2%} on average (max {frac.max():.2%}, "
                  f"segment {int(frac.argmax())})")
        if frac.max() > 0.3:
            warnings.warn(f"ARAP ({mode}): up to {frac.max():.0%} of the cells are unreliable (segment {int(frac.argmax())}); "
                          "the correspondences are very noisy there and the interpolation falls back towards linear "
                          "blending. Consider weight_kind='uniform', a larger field_smoothing or wider stretch_bounds.")

    # -- weights / systems ---------------------------------------------------- #
    def _weights(self, k):
        key = 0 if self.weights == "reference" else k
        if key not in self._W:
            W = robust_edge_weights(self.X[key], self.F, kind=self.weight_kind, bounds=self.weight_bounds)
            self._W[key] = W; self._L[key] = graph_laplacian(W)
        return self._W[key], self._L[key]

    def _system(self, k):
        key = 0 if self.weights == "reference" else k
        if key not in self._lu:
            _, L = self._weights(k)
            # prior strength relative to the lowest non-constant mode of L p = λ M p (≈ 8π / n for a closed
            # surface with normalised mass), so its effect does not grow with the mesh resolution
            lam1 = 8.0 * np.pi / self.n * (L.diagonal().mean() / 3.5)
            prior_op = sparse.diags(self.mu * lam1 * self.Mn)
            self._lu[key] = (splinalg.splu((L + prior_op).tocsc()), prior_op)
        return self._lu[key]

    # -- transforms ----------------------------------------------------------- #
    def _segment_fit(self, k):
        W, L = self._weights(k)
        fit = fit_cells(self.X[k], self.X[k + 1], self.F, W)
        R = fit["R_polar"] if self.mode == "polar" else fit["R_arap"]
        rv = Rotation.from_matrix(R).as_rotvec()
        angle = np.linalg.norm(rv, axis=1)
        sig = fit["sigma"]
        with np.errstate(divide="ignore", invalid="ignore"):
            lsig = np.log(np.abs(sig))
        lsig = np.where(np.isfinite(lsig), lsig, self.log_bounds[0])
        res = fit["residual"]
        finite = np.isfinite(res)
        med = np.median(res[finite]) if finite.any() else 0.0
        mad = 1.4826 * np.median(np.abs(res[finite] - med)) if finite.any() else 0.0
        bad_res = ~finite | (res > max(med + self.residual_outlier * mad, self.residual_floor))
        bad_rot = angle > self.max_angle
        bad_str = (lsig < self.log_bounds[0]).any(axis=1) | (lsig > self.log_bounds[1]).any(axis=1)
        bad = fit["reflected"] | bad_rot | bad_res
        if self.mode == "polar":
            bad |= bad_str
        lsig = np.clip(lsig, *self.log_bounds)
        logS = fit["axes"] @ (lsig[:, :, None] * np.transpose(fit["axes"], (0, 2, 1)))

        # screened harmonic fill (unreliable cells) + smoothing (all cells) of rotation / log-stretch fields
        Y0 = np.concatenate([rv, _sym_to_vec(logS)], axis=1) if self.mode == "polar" else rv
        c = (~bad).astype(np.float64) * self.Mn
        lam = max(self.field_smoothing, 1e-2) / max(L.diagonal().mean(), 1e-300)
        if bad.any() or self.field_smoothing > 0:
            Asys = (sparse.diags(c + 1e-8 * self.Mn) + lam * L).tocsc()
            Y = splinalg.splu(Asys).solve(c[:, None] * Y0)
        else:
            Y = Y0
        rv = Y[:, :3]
        ang = np.linalg.norm(rv, axis=1)
        too_big = ang > self.max_angle                               # keep the log map unambiguous
        rv[too_big] *= (self.max_angle / ang[too_big])[:, None]
        if self.mode == "polar":
            ev, evec = np.linalg.eigh(_vec_to_sym(Y[:, 3:]))
            self.log_eval.append(np.clip(ev, *self.log_bounds)); self.log_axes.append(evec)
        else:
            self.log_eval.append(None); self.log_axes.append(None)
        self.rotvec.append(rv); self.E.append(fit["E"]); self.reliable.append(~bad)
        self.diagnostics.append({"segment": k, "t0": float(self.t[k]), "t1": float(self.t[k + 1]),
                                 "unreliable_fraction": float(bad.mean()),
                                 "reflected_cells": int(fit["reflected"].sum()),
                                 "rotation_over_max": int(bad_rot.sum()),
                                 "stretch_out_of_bounds": int(bad_str.sum()),
                                 "residual_outliers": int(bad_res.sum()),
                                 "median_rotation_deg": float(np.degrees(np.median(angle))),
                                 "keyframe_residual_rel": np.nan})

    def _transforms(self, k, tau):
        Rt = Rotation.from_rotvec(tau * self.rotvec[k]).as_matrix()
        if self.mode == "rigid":
            return Rt
        V = self.log_axes[k]
        St = V @ (np.exp(tau * self.log_eval[k])[:, :, None] * np.transpose(V, (0, 2, 1)))
        return Rt @ St

    def _poisson(self, k, tau, prior):
        W, _ = self._weights(k)
        Wc = W.tocoo(); i, j, w = Wc.row, Wc.col, Wc.data
        Tt = self._transforms(k, tau)
        e = self.X[k][i] - self.X[k][j]
        Te = 0.5 * np.einsum("nij,nj->ni", Tt[i] + Tt[j], e)
        b = np.stack([np.bincount(i, w * Te[:, c], minlength=self.n) for c in range(3)], axis=1)
        lu, prior_op = self._system(k)
        return lu.solve(b + prior_op @ prior)

    def _endpoint_residuals(self, k):
        if k not in self._r1:
            self._r0[k] = self.X[k] - self._poisson(k, 0.0, self.X[k])
            self._r1[k] = self.X[k + 1] - self._poisson(k, 1.0, self.X[k + 1])
            diam = np.linalg.norm(self.X[k].max(0) - self.X[k].min(0)) + 1e-300
            self.diagnostics[k]["keyframe_residual_rel"] = float(np.sqrt(np.mean(np.sum(self._r1[k] ** 2, axis=1))) / diam)
        return self._r0[k], self._r1[k]

    def __call__(self, tq):
        tq = np.atleast_1d(np.asarray(tq, dtype=np.float64))
        out = np.zeros((len(tq), self.n, 3))
        prior = hermite_interpolate(self.X, self.t, tq, method=self.prior_method)
        for q, tv in enumerate(tq):
            k = int(np.clip(np.searchsorted(self.t, tv, side="right") - 1, 0, self.T - 2))
            tau = (tv - self.t[k]) / (self.t[k + 1] - self.t[k])
            if tau <= 1e-12:
                out[q] = self.X[k]; continue
            if tau >= 1 - 1e-12:
                out[q] = self.X[k + 1]; continue
            p = self._poisson(k, tau, prior[q])
            if self.endpoint_correction:
                r0, r1 = self._endpoint_residuals(k)
                p = p + (1 - tau) * r0 + tau * r1
            out[q] = p
        return out

    def rigidity_energy(self):
        """Per-transition per-vertex ARAP energy fields (eq. 3), shape (T-1, n)."""
        return np.stack(self.E)


INTERPOLATORS = ("linear", "hermite_catmull_rom", "hermite_finite_difference", "kochanek_bartels",
                 "natural_cubic", "bspline", "arap_rigid", "arap_polar")

_ARAP_KW = ("prior_weight", "prior_method", "weights", "weight_kind", "weight_bounds", "stretch_bounds",
            "max_rotation_deg", "residual_outlier", "residual_floor", "field_smoothing", "endpoint_correction",
            "verbose")


def make_arap_interpolator(X, F, scheme, t=None, **kw):
    """ARAPInterpolator for 'arap_rigid' / 'arap_polar'; unknown keywords are ignored."""
    return ARAPInterpolator(X, F, t, mode="rigid" if scheme == "arap_rigid" else "polar",
                            **{k: v for k, v in kw.items() if k in _ARAP_KW})


def interpolate_sequence(X, F, scheme, tq, t=None, **kw):
    t = np.arange(X.shape[0], dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
    if scheme == "linear":
        return _linear(X, t, tq)
    if scheme == "hermite_catmull_rom":
        return hermite_interpolate(X, t, tq, method="catmull_rom")
    if scheme == "hermite_finite_difference":
        return hermite_interpolate(X, t, tq, method="finite_difference")
    if scheme == "kochanek_bartels":
        return hermite_interpolate(X, t, tq, method="kochanek_bartels", tension=kw.get("tension", 0.3), bias=kw.get("bias", 0.0))
    if scheme == "natural_cubic":
        return spline_interpolate(X, t, tq, kind="natural_cubic")
    if scheme == "bspline":
        return spline_interpolate(X, t, tq, kind="bspline", k=kw.get("k", 3))
    if scheme in ("arap_rigid", "arap_polar"):
        return make_arap_interpolator(X, F, scheme, t, **kw)(tq)
    raise ValueError(f"Unknown interpolation scheme '{scheme}'.")

def _linear(X, t, tq):
    out = np.zeros((len(tq),) + X.shape[1:])
    for q, tv in enumerate(tq):
        k = int(np.clip(np.searchsorted(t, tv, side="right") - 1, 0, len(t) - 2))
        s = (tv - t[k]) / (t[k + 1] - t[k])
        out[q] = (1 - s) * X[k] + s * X[k + 1]
    return out


def leave_one_out_validation(X, F, schemes=INTERPOLATORS, t=None, areas=None, verbose=True, arap_params=None):
    """
    For every interior frame k, predict X_k from the other frames with each scheme and measure the
    area-weighted RMS error relative to the mesh scale.  Returns a DataFrame (scheme × frame).
    """
    T = X.shape[0]
    t = np.arange(T, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
    A = vertex_areas(X[0], F) if areas is None else areas; A = A / A.sum()
    scale = np.sqrt(np.sum(A * np.sum((X[0] - (A[:, None] * X[0]).sum(0)) ** 2, axis=1)))
    rows = []
    for scheme in schemes:
        for k in pbar(range(1, T - 1), desc=f"Validation {scheme}"):
            keep = np.array([i for i in range(T) if i != k])
            try:
                if scheme.startswith("arap"):
                    Xs = X[[k - 1, k + 1]]
                    pred = interpolate_sequence(Xs, F, scheme, np.array([0.5]), t=np.array([0.0, 1.0]),
                                                **(arap_params or {}))[0]
                else:
                    pred = interpolate_sequence(X[keep], F, scheme, np.array([t[k]]), t=t[keep])[0]
                err = np.sqrt(np.sum(A * np.sum((pred - X[k]) ** 2, axis=1)))
                rows.append({"scheme": scheme, "frame": k, "rms_error": err, "rel_error": err / scale})
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"LOO validation failed for {scheme} at frame {k}: {exc}")
    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        print("Leave-one-out relative error (median over frames):")
        print(df.groupby("scheme")["rel_error"].median().sort_values().to_string())
    return df


# ----------------------------------------------------------------------------- #
#  3. Kinematics
# ----------------------------------------------------------------------------- #

def trajectory_kinematics(X, F, t=None, dt=1.0):
    """
    Velocity / acceleration (spline derivatives), speed, Frenet curvature κ = |v×a|/|v|³ and torsion
    τ = (v×a)·ȷ / |v×a|², path length, tortuosity, kinetic energy (area-weighted), per-frame ARAP energy,
    Green–Lagrange strain invariants of the cells.
    """
    T, n, _ = X.shape
    t = np.arange(T, dtype=np.float64) * dt if t is None else np.asarray(t, dtype=np.float64)
    if T >= 4:
        vel = spline_interpolate(X, t, t, kind="natural_cubic", derivative=1)
        acc = spline_interpolate(X, t, t, kind="natural_cubic", derivative=2)
        jerk = spline_interpolate(X, t, t, kind="natural_cubic", derivative=3)
    else:
        vel = np.gradient(X, t, axis=0); acc = np.gradient(vel, t, axis=0); jerk = np.gradient(acc, t, axis=0)
    speed = np.linalg.norm(vel, axis=2)
    vxa = np.cross(vel, acc)
    curvature = np.linalg.norm(vxa, axis=2) / np.maximum(speed ** 3, 1e-300)
    torsion = np.einsum("tnc,tnc->tn", vxa, jerk) / np.maximum(np.sum(vxa ** 2, axis=2), 1e-300)
    seg = np.linalg.norm(np.diff(X, axis=0), axis=2)
    path_length = seg.sum(axis=0)
    net = np.linalg.norm(X[-1] - X[0], axis=1)
    tortuosity = path_length / np.maximum(net, 1e-12)
    A = vertex_areas(X[0], F)
    kinetic = 0.5 * np.sum(A[None, :] * speed ** 2, axis=1)
    # deformation of the cells relative to the reference frame
    strain_rows, arap_rows = [], []
    E_fields = np.zeros((T - 1, n)); J_fields = np.zeros((T, n)); J_fields[0] = 1.0
    Wij0 = cotan_weights(X[0], F)
    for k in range(1, T):
        R, S, E = cell_transforms(X[k - 1], X[k], F, cotan_weights(X[k - 1], F), with_stretch=False)
        E_fields[k - 1] = E
        Rr, Sr, _ = cell_transforms(X[0], X[k], F, Wij0, with_stretch=True)
        C = Sr @ Sr                                                # right Cauchy–Green (Sᵀ S, S symmetric)
        Egl = 0.5 * (C - np.eye(3)[None])
        J = np.linalg.det(Sr); J_fields[k] = J
        strain_rows.append({"frame": k, "mean_volumetric_J": float(J.mean()),
                            "mean_GreenLagrange_norm": float(np.linalg.norm(Egl, axis=(1, 2)).mean()),
                            "max_principal_stretch_mean": float(np.linalg.eigvalsh(Sr)[:, -1].mean()),
                            "min_principal_stretch_mean": float(np.linalg.eigvalsh(Sr)[:, 0].mean())})
        arap_rows.append({"transition": f"T{k-1} -> T{k}", "frame": k - 1, "arap_energy_total": float(E.sum()),
                          "arap_energy_area_normalised": float(E.sum() / A.sum()),
                          "arap_energy_max_cell": float(E.max())})
    summary = pd.DataFrame({"frame": np.arange(T), "time": t, "mean_speed": speed.mean(axis=1),
                            "max_speed": speed.max(axis=1), "mean_acceleration": np.linalg.norm(acc, axis=2).mean(axis=1),
                            "mean_curvature_traj": np.nanmean(np.where(speed > 1e-9, curvature, np.nan), axis=1),
                            "kinetic_energy": kinetic,
                            "mean_normal_speed": _normal_speed(X, F, vel).mean(axis=1)})
    return {"velocity": vel, "acceleration": acc, "speed": speed, "curvature": curvature, "torsion": torsion,
            "path_length": path_length, "tortuosity": tortuosity, "arap_energy_field": E_fields,
            "volumetric_J": J_fields, "summary": summary, "strain": pd.DataFrame(strain_rows),
            "arap": pd.DataFrame(arap_rows)}


def _normal_speed(X, F, vel):
    from PynamicMesh.core.dyn_common import vertex_normals
    return np.stack([np.einsum("ij,ij->i", vel[k], vertex_normals(X[k], F)) for k in range(X.shape[0])])


# ----------------------------------------------------------------------------- #
#  4. Sequence driver
# ----------------------------------------------------------------------------- #

def query_times(t, substeps):
    """Dense query times: `substeps` equal sub-intervals inside EVERY observed interval (the observed times are always
    among them, also for irregular time-lapses); identical to the uniform grid when the frames are equally spaced."""
    t = np.asarray(t, dtype=np.float64); T = t.size
    steps = np.diff(t)
    if T < 2 or np.ptp(steps) <= 1e-9 * max(float(np.abs(steps).max()), 1e-300):
        return np.linspace(t[0], t[-1], (T - 1) * substeps + 1)
    return np.concatenate([np.linspace(t[k], t[k + 1], substeps + 1)[:-1] for k in range(T - 1)] + [t[-1:]])


@logged_stage("Trajectories", (1, "target_folder"), "Trajectories")
def compute_trajectories(mesh_folder, target_folder, loader=None, dt=1.0, schemes=("hermite_catmull_rom", "natural_cubic", "arap_polar"),
                         substeps=4, validate=True, validation_schemes=INTERPOLATORS, kinematics=True,
                         export_obj=True, plots=True, verbose=True, frames=None, arap_params=None,
                         trajectory_method="arap_tracking", tracking_params=None, rigidity_params=None, times=None):
    """
    Returns dict with X, F0, t and paths.
    trajectory_method: 'arap_tracking' (default: robust ARAP registration of the reference mesh to every frame, keeps
        the triangulation intact) or 'chain' (composition of the p2p chain, previous behaviour).
    tracking_params: options of track_reference_mesh (TRACKING_DEFAULTS), e.g. {"reference_rigidity": 0.3} for
        strongly growing cells, {"icp_rounds": 0} to follow the functional maps only.
    times: acquisition times of the frames (FrameTimes or array, dyn_common.resolve_frame_times). When given they
        replace t = k * dt everywhere (key-frame times, interpolation query times - also for irregular intervals -,
        kinematics in physical units, validation); time.npy then holds the real times, which the dynamic analysis, the
        graph animation and the viewers read.
    rigidity_params: as-rigid-as-possible projection (RIGIDITY_DEFAULTS, Sorkine & Alexa 2007) of the key-frame
        trajectories and of every dense interpolation: keeps the local shape of the reference mesh (no sliding, shear or
        collapsed triangles) while following the motion. {"enabled": False} = previous behaviour; {"mode": "arap"}
        for near-isometric motion; a lower data_weight = more rigid. integrity.csv / plots/6_mesh_integrity.png
        compare the meshes before and after.
    arap_params: optional dict of ARAPInterpolator options for the 'arap_*' schemes, e.g.
        {"field_smoothing": 2.0, "stretch_bounds": (0.5, 2.0), "weight_kind": "uniform", "verbose": True}
    """
    arap_params = dict(arap_params or {})
    target_folder = Path(target_folder)
    out = ensure_dir(target_folder / "Trajectories")
    d_interp = ensure_dir(out / "interpolation"); d_kin = ensure_dir(out / "kinematics"); d_plots = ensure_dir(out / "plots")
    if frames is None:
        frames = load_frames(mesh_folder, loader)
    T = len(frames)
    p2p, _ = load_p2p_chain(target_folder / "Transform_Matrices", T)
    progress(total=3 + len(schemes) + int(bool(validate and T >= 3)) + int(bool(plots)), advance=0)
    if trajectory_method == "chain":
        X, F0, cov = build_trajectories(frames, p2p, verbose=verbose)
    elif trajectory_method == "arap_tracking":
        X, F0, cov = track_reference_mesh(frames, p2p, params=tracking_params, verbose=verbose)
        cov.to_csv(out / "tracking.csv", index=False)
    else:
        raise ValueError(f"Unknown trajectory_method '{trajectory_method}' (use 'arap_tracking' or 'chain').")
    progress("trajectories of the reference mesh")
    rig = {**RIGIDITY_DEFAULTS, **(rigidity_params or {})}
    V_ref = X[0].copy()                                   # the reference mesh (frame 0 of the trajectories)
    integrity_rows = []
    if rig["enabled"] and rig["keyframes"]:
        np.save(out / "trajectories_raw.npy", X)
        X_raw = X
        X = rigidity_projection(V_ref, F0, X, rig["mode"], rig["data_weight"], rig["iterations"], verbose=verbose)
        for tag, seq in (("raw", X_raw), ("projected", X)):
            integrity_rows.append(mesh_integrity(V_ref, F0, seq).assign(sequence="key frames", version=tag,
                                                                       step=np.arange(T)))
        progress("as-rigid-as-possible projection of the key frames")
    mesh_to_mesh_displacements(frames, p2p, out, verbose=verbose)
    progress("mesh-to-mesh displacement fields")
    (out / "frame_names.txt").write_text("\n".join(fr["name"] for fr in frames), encoding="utf-8")
    t = np.arange(T, dtype=np.float64) * dt
    time_unit = "frame" if dt == 1.0 else "time"
    if times is not None:
        t_given = np.asarray(getattr(times, "t", times), dtype=np.float64)
        if t_given.size == T:
            t = t_given; time_unit = getattr(times, "unit", "s")
        else:
            warnings.warn(f"{t_given.size} frame times for {T} frames: t = k * dt is used")
    np.save(out / "trajectories.npy", X); np.save(out / "reference_faces.npy", F0); np.save(out / "time.npy", t)
    cov.to_csv(out / "coverage.csv", index=False)

    kin = None
    if kinematics:
        kin = trajectory_kinematics(X, F0, t)
        for key in ("velocity", "acceleration", "speed", "curvature", "torsion", "path_length", "tortuosity",
                    "arap_energy_field", "volumetric_J"):
            np.save(d_kin / f"{key}.npy", kin[key])
        kin["summary"].to_csv(d_kin / "summary.csv", index=False)
        kin["strain"].to_csv(d_kin / "strain.csv", index=False)
        kin["arap"].to_csv(d_kin / "arap_energy.csv", index=False)
        _save_trajectory_polylines(out / "trajectories.vtp", X, kin["speed"].mean(axis=0))
        for k in pbar(range(T), desc="Kinematics export"):
            save_vtp(d_kin / f"kinematics_T{k:04d}.vtp", X[k], F0,
                     {"speed": kin["speed"][k], "curvature": kin["curvature"][k],
                      "arap_energy": kin["arap_energy_field"][min(k, T - 2)], "volumetric_J": kin["volumetric_J"][k],
                      "path_length": kin["path_length"], "tortuosity": kin["tortuosity"]})

    progress("kinematics" if kinematics else "kinematics skipped")
    tq = query_times(t, substeps)
    np.save(d_interp / "query_times.npy", tq)
    for scheme in pbar(list(schemes), desc="Interpolation schemes"):
        if scheme.startswith("arap"):
            interp = make_arap_interpolator(X, F0, scheme, t, **{"verbose": verbose, **arap_params})
            Xi = interp(tq)
            pd.DataFrame(interp.diagnostics).to_csv(d_interp / f"{scheme}_diagnostics.csv", index=False)
        else:
            Xi = interpolate_sequence(X, F0, scheme, tq, t=t)
        if rig["enabled"] and rig["interpolations"]:
            np.save(ensure_dir(d_interp / "raw") / f"{scheme}.npy", Xi)
            Xr = Xi
            Xi = rigidity_projection(V_ref, F0, Xi, rig["mode"], rig["data_weight"], rig["iterations"], verbose=verbose)
            for tag, seq in (("raw", Xr), ("projected", Xi)):
                integrity_rows.append(mesh_integrity(V_ref, F0, seq).assign(sequence=scheme, version=tag,
                                                                            step=np.arange(len(tq))))
        np.save(d_interp / f"{scheme}.npy", Xi)
        progress(f"interpolation {scheme}")
        if export_obj:
            d_s = ensure_dir(d_interp / scheme)
            for q in pbar(range(Xi.shape[0]), desc=f"OBJ export {scheme}"):
                write_obj(d_s / f"frame_{q:04d}.obj", Xi[q], F0)
    val = None
    if validate and T >= 3:
        val = leave_one_out_validation(X, F0, schemes=validation_schemes, t=t, verbose=verbose, arap_params=arap_params)
        val.to_csv(d_interp / "validation.csv", index=False)
        progress("leave-one-frame-out validation")
    integ = pd.concat(integrity_rows, ignore_index=True) if integrity_rows else None
    if integ is not None:
        integ.to_csv(out / "integrity.csv", index=False)
    if plots:
        _plots(out, cov, kin, val)
        if integ is not None:
            _plot_integrity(out / "plots" / "6_mesh_integrity.png", integ, rig)
        progress("plots")
    with open(out / "info.json", "w", encoding="utf-8") as fh:
        json.dump({"n_frames": T, "n_reference_vertices": int(X.shape[1]), "dt": dt, "schemes": list(schemes),
                   "times": [float(v) for v in t], "time_unit": time_unit,
                   "trajectory_method": trajectory_method, "rigidity_params": rig,
                   "tracking_params": {**TRACKING_DEFAULTS, **(tracking_params or {})} if trajectory_method == "arap_tracking" else None,
                   "substeps": substeps, "arap_params": {k: (list(v) if isinstance(v, tuple) else v)
                                                          for k, v in arap_params.items()}}, fh, indent=2)
    if verbose:
        print(f"Trajectories: results in {out}")
    return {"X": X, "F0": F0, "t": t, "folder": str(out), "kinematics": kin, "validation": val}


def _plot_integrity(path, integ, rig):
    """Integrity of the meshes before / after the as-rigid-as-possible projection (key frames and every scheme)."""
    seqs = list(dict.fromkeys(integ["sequence"]))
    fig, axes = plt.subplots(len(seqs), 2, figsize=(13, 3.3 * len(seqs)), squeeze=False)
    for r, sq in enumerate(seqs):
        for c, (col, lab) in enumerate((("degraded_faces", "degraded triangles (fraction)"),
                                        ("shape_change", "local shape change (RMS log edge ratio, scale-free)"))):
            ax = axes[r, c]
            for ver, sty in (("raw", dict(color="tab:red", ls="--", marker="x")), ("projected", dict(color="tab:green", marker="o", ms=3))):
                d = integ[(integ.sequence == sq) & (integ.version == ver)]
                if sq == "key frames":
                    lab_ = "trajectories (tracking)" if ver == "raw" else "after the projection"
                else:                                      # the schemes interpolate the already projected key frames
                    lab_ = "interpolated from projected key frames" if ver == "raw" else "after the projection"
                ax.plot(d.step, d[col], label=lab_, **sty)
            ax.set_title(f"{sq}: {lab}", fontsize=10, fontweight="bold"); ax.set_xlabel("step"); ax.legend(fontsize=8)
    fig.suptitle(f"Mesh integrity before / after the {rig['mode'].upper()} projection (data weight {rig['data_weight']}); "
                 "integrity.csv has the values", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _save_trajectory_polylines(path, X, color):
    try:
        import pyvista as pv
        T, n, _ = X.shape
        pts = X.transpose(1, 0, 2).reshape(-1, 3)
        lines = np.hstack([np.concatenate([[T], np.arange(v * T, (v + 1) * T)]) for v in range(n)])
        poly = pv.PolyData(pts, lines=lines)
        poly.point_data["mean_speed"] = np.repeat(color, T)
        poly.point_data["time_index"] = np.tile(np.arange(T), n)
        poly.save(str(path))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not save trajectory polylines: {exc}")


def _plots(out, cov, kin, val):
    d = Path(out) / "plots"
    if kin is not None:
        s = kin["summary"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(s["time"], s["mean_speed"], marker="o", label="mean"); axes[0, 0].plot(s["time"], s["max_speed"], marker="s", label="max")
        axes[0, 0].set_title("vertex speed"); axes[0, 0].legend()
        axes[0, 1].plot(s["time"], s["kinetic_energy"], marker="o", color="tab:red"); axes[0, 1].set_title("kinetic energy ½∫|v|² dA")
        axes[1, 0].plot(kin["arap"]["frame"], kin["arap"]["arap_energy_area_normalised"], marker="^", color="tab:purple")
        axes[1, 0].set_title("ARAP rigidity energy per transition (eq. 7, area normalised)")
        axes[1, 1].plot(s["time"], s["mean_normal_speed"], marker="d", color="tab:green"); axes[1, 1].set_title("mean normal speed (growth / shrinkage)")
        for a in axes.ravel():
            a.set_xlabel("time")
        fig.tight_layout(); fig.savefig(d / "1_kinematics.png", dpi=200); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(kin["path_length"], bins=40, color="tab:blue"); axes[0].set_title("path length per vertex")
        axes[1].hist(np.clip(kin["tortuosity"], 0, np.percentile(kin["tortuosity"], 99)), bins=40, color="tab:orange"); axes[1].set_title("tortuosity")
        fig.tight_layout(); fig.savefig(d / "2_path_statistics.png", dpi=200); plt.close(fig)
        st = kin["strain"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(st["frame"], st["mean_volumetric_J"], marker="o", label="mean det S (volumetric)")
        ax.plot(st["frame"], st["max_principal_stretch_mean"], marker="^", label="mean λ_max")
        ax.plot(st["frame"], st["min_principal_stretch_mean"], marker="v", label="mean λ_min")
        ax.set_xlabel("frame"); ax.set_title("cell stretch relative to the reference frame"); ax.legend()
        fig.tight_layout(); fig.savefig(d / "3_strain.png", dpi=200); plt.close(fig)
    if val is not None and not val.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        order = list(val.groupby("scheme")["rel_error"].median().sort_values().index)
        ax.boxplot([val.loc[val["scheme"] == s_, "rel_error"].values for s_ in order])
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel("relative RMS error"); ax.set_title("leave-one-frame-out validation"); ax.set_yscale("log")
        fig.tight_layout(); fig.savefig(d / "4_interpolation_validation.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    if "fm_inlier_fraction" in cov:                       # ARAP tracking diagnostics
        ax.plot(cov["frame"], cov["reliable_fraction"], marker="o", label="predictions from directly matched vertices")
        ax.plot(cov["frame"], cov["fm_inlier_fraction"], marker="s", label="FM predictions kept by the robust fit")
        ax.plot(cov["frame"], cov["icp_accepted_fraction"], marker="^", label="closest points accepted (refinement)")
        ax.set_ylim(0, 1.05); ax.set_xlabel("frame"); ax.set_ylabel("fraction of reference vertices"); ax.legend(fontsize=8, loc="lower left")
        ax2 = ax.twinx(); ax2.plot(cov["frame"], cov["mean_surface_distance"], color="k", ls="--", marker="d", label="mean distance to the frame surface")
        ax2.set_ylabel("distance to the frame surface"); ax2.legend(fontsize=8, loc="lower right")
        ax.set_title("ARAP tracking of the reference mesh")
    else:
        ax.plot(cov["frame"], cov["covered_fraction"], marker="o"); ax.set_ylim(0, 1.05)
        ax.set_xlabel("frame"); ax.set_ylabel("fraction of reference vertices hit by the p2p chain"); ax.set_title("correspondence coverage")
    fig.tight_layout(); fig.savefig(d / "5_coverage.png", dpi=200); plt.close(fig)