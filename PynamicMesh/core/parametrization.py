"""
parametrization.py  —   explicit parametrizations of the mesh surfaces
================================================================================

A triangle mesh is only an *implicit* sampling of a surface.  An explicit parametrization is a map
x : Ω → R³ from a canonical domain Ω onto the surface.  What is achievable depends on the topology:

  * closed genus-0 surfaces (biological cells, most organic shapes) ............ Ω = S²
        1. a bijective **spherical parametrization** u : S → S²: a discrete conformal map (stereographic
           harmonic maps + Schwarz sweeps with Möbius normalisation) or a harmonic map (Gu, Wang, Chan,
           Thompson & Yau 2004), followed by a spherical **stretch-energy area correction** (Yueh et al. 2019)
           that removes the exponential area compression of protrusions; this compression, not the
           harmonic expansion, is what limits the SPHARM accuracy and creates the 'knots' at the poles;
        2. a **spherical-harmonic (SPHARM) expansion** (Brechbühler et al. 1995, Styner et al. 2006)
                x(θ, φ) = Σ_{l≤L} Σ_{|m|≤l} c_lm Y_lm(θ, φ),   c_lm ∈ R³,
           fitted by area-weighted least squares with a Tikhonov (bi-Laplacian) penalty.
           This is a *closed-form* smooth surface x(θ,φ); the coefficients are exported and the
           truncated series is also produced as a SymPy expression (LaTeX / lambdify ready).
        3. rotation-invariant descriptors: the per-degree energies E_l = Σ_m ||c_lm||².

  * any topology (also as a spectral compression) ....................... Ω = the surface itself
        4. **manifold-harmonics reconstruction** (Vallet & Lévy 2008):  x ≈ Σ_k α_k φ_k with the
           Laplace–Beltrami eigenbasis, α = Φᵀ M x (explicit in the intrinsic coordinates φ_k).

  * disk topology .......................................................... Ω ⊂ R²
        5. **ARAP planar parametrization** (Liu, Zhang, Gotsman & Gortler 2008), i.e. the
           local/global scheme of Sorkine & Alexa 2007 applied to per-triangle Jacobians, started from
           a Tutte embedding.

Time consistency.  The functional maps computed by the pipeline give a point-to-point map between
consecutive frames.  Frame t's sphere is rotated (Kabsch) so that corresponding vertices land at the
same spherical location as in frame t-1: SPHARM coefficient vectors of different frames then live in the
same coordinate system and their time series c_lm(t) is a low-dimensional, continuous encoding of the
whole deformation (input of the shape-space, reduced-coordinate and change-point analyses of dynamic_analysis.py).

Results/<scene>/Parametrization/
    Spherical/sphere_T####.npy            unit vectors u_i (n,3) of the harmonic map (aligned)
    Spherical/theta_phi_T####.npy         (n,2) spherical coordinates
    SPHARM/coeffs_T####.npy               ((L+1)², 3) real SPHARM coefficients in the aligned frame
    SPHARM/coeffs_consistent.npy          (T, (L+1)², 3) stacked coefficients (dynamics input)
    SPHARM/reconstruction_T####_L##.obj   SPHARM surfaces synthesised on an icosphere
    SPHARM/spharm_degree_energy.csv       E_l(t) rotation invariants
    SPHARM/reconstruction_error.csv       RMS error vs degree, per frame
    ManifoldHarmonics/mh_coeffs_T####.npy, mh_reconstruction_error.csv
    Planar/uv_T####.npy                   (disk topology only)
    symbolic/parametrization_T####.tex / .txt   truncated symbolic series
    quality.csv                           topology, fold-overs, area/angle distortion of the sphere map
    plots/                                summary figures
"""
import os
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
from tqdm.auto import tqdm

from PynamicMesh.core.dyn_common import pbar
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots
from PynamicMesh.core.dyn_common import (
    logged_stage, progress,
    xp, GPU_AVAILABLE, to_gpu, to_cpu, as_faces, vf, ensure_dir, load_frames, load_p2p_chain,
    cotan_weights, cotan_laplacian, vertex_areas, face_normals_areas, mesh_topology,
    laplacian_eigenbasis, kabsch_rotation, icosphere, write_obj, save_vtp, mesh_edges, face_gradient_operator
)


# ----------------------------------------------------------------------------- #
#  1. Spherical parametrization (discrete harmonic map)
# ----------------------------------------------------------------------------- #

def _sphere_orientation(U, F):
    """Sign of the oriented spherical triangles: det[u_a, u_b, u_c]."""
    tri = U[F]
    return np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))


def count_foldovers(U, F):
    s = _sphere_orientation(U, F)
    majority = np.sign(np.median(s)) or 1.0
    return int(np.sum(s * majority <= 0.0))


def _mobius_normalize(U, A, max_iter=20):
    """
    Area balancing by radial re-projection:  u ↦ (u − c)/|u − c|,  c = area-weighted centre of mass, iterated.
    This is a diffeomorphism of S² (bijective) but NOT conformal, so it is used only by the area-oriented
    stages (gradient descent, area correction).  The conformal stage uses the true Möbius centring
    `_mobius_center`, which keeps the map conformal.
    """
    for _ in range(max_iter):
        c = (U * A[:, None]).sum(axis=0) / A.sum()
        if np.linalg.norm(c) < 1e-9:
            break
        U = U - c
        U /= np.linalg.norm(U, axis=1)[:, None]
    return U


def _mobius_ball(U, a):
    """Conformal automorphism of S² induced by the hyperbolic isometry of the unit ball taking a (|a| < 1) to 0:
        x ↦ ((1 − |a|²)(x − a) − |x − a|² a) / |x − a|²."""
    d = U - a[None]
    d2 = np.maximum(np.sum(d ** 2, axis=1), 1e-300)[:, None]
    Y = ((1.0 - float(a @ a)) * d - d2 * a[None]) / d2
    return Y / np.maximum(np.linalg.norm(Y, axis=1), 1e-300)[:, None]


def _mobius_center(U, A, max_iter=60, tol=1e-9):
    """
    Möbius normalisation of a conformal sphere map: composes it with conformal automorphisms of S² until the
    area-weighted centre of mass is at the origin (damped fixed-point iteration, Baden–Crane–Kazhdan 2018).
    Unlike the radial re-projection this keeps the map conformal (bumpy cell: K 1.57 → 1.04).
    """
    U = np.asarray(U, dtype=np.float64)
    for _ in range(max_iter):
        c = (U * A[:, None]).sum(axis=0) / A.sum()
        if np.linalg.norm(c) < tol:
            break
        a = 0.5 * c
        if np.linalg.norm(a) > 0.9:
            a *= 0.9 / np.linalg.norm(a)
        U = _mobius_ball(U, a)
    return U


def spherical_parametrization(V, F, n_iter=500, step=0.5, tol=1e-7, clip_negative=True,
                              init=None, untangle_iters=50, verbose=False, pull_target=None, pull_weight=0.0):
    """
    Harmonic map S → S² by projected gradient descent of the Dirichlet energy
        E(u) = ½ Σ_ij w_ij ||u_i − u_j||²   on  |u_i| = 1,
    i.e. u_i ← normalise(u_i + step · (Δ_w u)_i^tangential), followed by Möbius normalisation
    (Gu et al. 2004).  Cotangent weights are clipped at zero by default (convex combination maps,
    fewer fold-overs); the remaining fold-overs are removed with Tutte-type (uniform) relaxations of
    the offending vertices.  Returns (U, info).
    """
    V = np.asarray(V, dtype=np.float64); F = as_faces(F)
    n = V.shape[0]
    A = vertex_areas(V, F)
    Wij = cotan_weights(V, F, clip_negative=clip_negative)
    deg = np.asarray(Wij.sum(axis=1)).ravel()
    deg[deg <= 1e-14] = 1.0
    Lnorm = sparse.diags(1.0 / deg) @ Wij                      # row-stochastic averaging operator

    if init is None:
        c = (V * A[:, None]).sum(axis=0) / A.sum()
        U = V - c
    else:
        U = np.array(init, dtype=np.float64, copy=True)
    U /= np.maximum(np.linalg.norm(U, axis=1), 1e-300)[:, None]

    Lg = to_gpu(Lnorm.toarray()) if (GPU_AVAILABLE and n <= 6000) else None
    Ug = to_gpu(U)
    energy_hist = []
    it = 0
    for it in range(n_iter):
        avg = (Lg @ Ug) if Lg is not None else to_gpu(Lnorm @ to_cpu(Ug))
        lap = avg - Ug
        if pull_target is not None and pull_weight > 0:              # time consistency: pull towards the previous frame
            lap = lap + pull_weight * (to_gpu(np.asarray(pull_target)) - Ug)
        lap = lap - xp.sum(lap * Ug, axis=1)[:, None] * Ug          # tangential component
        res = float(to_cpu(xp.sqrt(xp.mean(xp.sum(lap ** 2, axis=1)))))
        energy_hist.append(res)
        Ug = Ug + step * lap
        Ug = Ug / xp.maximum(xp.linalg.norm(Ug, axis=1), 1e-300)[:, None]
        if it % 10 == 0:
            Ug = to_gpu(_mobius_normalize(to_cpu(Ug), A))
        if res < tol:
            break
    U = _mobius_normalize(to_cpu(Ug), A)

    # fold-over removal: uniform-weight relaxation of vertices touching inverted triangles
    n_fold = count_foldovers(U, F)
    adj = sparse.coo_matrix((np.ones(3 * F.shape[0]), (np.concatenate([F[:, 0], F[:, 1], F[:, 2]]),
                                                        np.concatenate([F[:, 1], F[:, 2], F[:, 0]]))), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.float64)
    adj_deg = np.asarray(adj.sum(axis=1)).ravel(); adj_deg[adj_deg == 0] = 1
    Lu = sparse.diags(1.0 / adj_deg) @ adj
    k = 0
    while n_fold > 0 and k < untangle_iters:
        s = _sphere_orientation(U, F)
        maj = np.sign(np.median(s)) or 1.0
        bad_v = np.unique(F[s * maj <= 0.0].ravel())
        target = Lu @ U
        U[bad_v] = target[bad_v]
        U /= np.maximum(np.linalg.norm(U, axis=1), 1e-300)[:, None]
        U = _mobius_normalize(U, A)
        n_fold = count_foldovers(U, F)
        k += 1

    info = {"iterations": it + 1, "final_residual": float(energy_hist[-1]) if energy_hist else 0.0,
            "foldovers": int(n_fold), "untangle_passes": k, "residual_history": np.asarray(energy_hist)}
    info.update(sphere_map_distortion(V, F, U))
    if verbose:
        print(f"  spherical map: {info['iterations']} it, residual {info['final_residual']:.2e}, fold-overs {n_fold}")
    return U, info


def _harmonic_dirichlet(W, fixed_idx, fixed_val, n):
    """Solves the discrete Laplace equation W u = 0 on the free vertices with Dirichlet data on fixed_idx."""
    free = np.setdiff1d(np.arange(n), fixed_idx)
    W = W.tocsr()
    A = W[free][:, free].tocsc(); B = W[free][:, fixed_idx]
    rhs = -B @ fixed_val
    sol = splinalg.spsolve(A, rhs)
    out = np.zeros((n, fixed_val.shape[1]))
    out[fixed_idx] = fixed_val; out[free] = np.asarray(sol).reshape(len(free), -1)
    return out


def _rotation_a_to_b(a, b):
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b)); s_ = np.linalg.norm(v)
    if s_ < 1e-12:
        if c > 0:
            return np.eye(3)
        k = np.cross(a, [1.0, 0.0, 0.0])                       # antipodal: 180° rotation about an axis ⟂ a
        if np.linalg.norm(k) < 1e-6:
            k = np.cross(a, [0.0, 1.0, 0.0])
        k /= np.linalg.norm(k)
        return 2.0 * np.outer(k, k) - np.eye(3)                 # proper rotation (det = +1)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s_ ** 2)


def _inverse_stereographic(z, from_north=True):
    """C → S²:  (x, y) ↦ (2x, 2y, |z|²−1)/(|z|²+1)  (the point at infinity goes to the north pole)."""
    r2 = np.sum(z ** 2, axis=1)
    U = np.column_stack([2 * z[:, 0], 2 * z[:, 1], r2 - 1.0]) / (r2 + 1.0)[:, None]
    return U if from_north else U * np.array([1.0, 1.0, -1.0])


def _stereographic(U, from_north=True):
    """S² → C, projecting from the north pole (or the south pole with from_north=False)."""
    z = U[:, 2] if from_north else -U[:, 2]
    d = np.maximum(1.0 - z, 1e-12)
    return np.column_stack([U[:, 0] / d, U[:, 1] / d])


def spherical_conformal_map(V, F, pole=None, cap_fraction=0.05, max_sweeps=40, tol=1e-3, verbose=False):
    """
    Fast discrete conformal map S → S² by stereographic harmonic maps.

    Initialisation (Angenent–Haker–Tannenbaum 1999 / FLASH, Choi–Lam–Lui 2015): a 'pole' vertex p and its
    1-ring are removed, the remaining disk is mapped harmonically (full cotangent weights, Dirichlet data on
    the ring placed on a circle) into the plane, inverse-stereographically projected onto the sphere with p
    at the pole, and Möbius-normalised (area-weighted centre of mass at the origin).

    Möbius normalisation uses true conformal automorphisms of S² (`_mobius_center`); the former radial
    re-projection destroyed the conformality after every sweep and the sweeps plateaued far from K = 1.

    Refinement (alternating stereographic Schwarz sweeps): the map is alternately expressed in the charts of
    the north and the south pole and, in each chart, the hemisphere around the chart origin is re-solved
    harmonically with the other hemisphere as Dirichlet data.  A conformal map composed with a conformal
    chart has harmonic coordinates, so the fixed point of the sweeps is the discrete conformal map; each
    sweep is one sparse solve (~0.03 s for 2.5k vertices) and the quasi-conformal distortion decreases
    monotonically (bumpy sphere: 1.49 → 1.02 in one sweep; 2.4:1 elongated cell: 1.38 → 1.08).  This
    replaces the projected-gradient descent of Gu et al. 2004, which needs clipped weights to stay
    bijective and therefore plateaus at K ≈ 1.3 on elongated shapes.

    Returns U (n, 3) on the unit sphere and an info dict (sweeps, distortion history, fold-overs).
    """
    V = np.asarray(V, dtype=np.float64); F = as_faces(F); n = V.shape[0]
    A = vertex_areas(V, F)
    W, _, _ = cotan_laplacian(V, F)                                  # PSD stiffness, unclipped (conformal)
    Wr = (W + 1e-12 * sparse.identity(n)).tocsr()
    # ---- initialisation --------------------------------------------------------------------------------
    if pole is None:
        ring_area = np.zeros(n); _, af = face_normals_areas(V, F)
        for k in range(3):
            np.add.at(ring_area, F[:, k], af)
        pole = int(np.argmax(ring_area))
    else:
        _, af = face_normals_areas(V, F)
    ring_faces = np.any(F == pole, axis=1)
    ring = np.unique(F[ring_faces].ravel()); ring = ring[ring != pole]
    npole = np.zeros(3)
    for f in F[ring_faces]:
        npole += np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
    npole /= max(np.linalg.norm(npole), 1e-300)
    e1 = np.cross(npole, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(npole, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(npole, e1)
    d = V[ring] - V[pole]
    ang = np.arctan2(d @ e2, d @ e1); order = np.argsort(ang); ring = ring[order]; ang = ang[order]
    f_cap = max(float(af[ring_faces].sum() / af.sum()), cap_fraction)
    z_c = 1.0 - 2.0 * f_cap; R = np.sqrt((1.0 + z_c) / max(1.0 - z_c, 1e-9))
    Wd, _, _ = cotan_laplacian(V, F[~ring_faces])
    fixed = np.concatenate([ring, [pole]])
    fixed_val = np.vstack([R * np.column_stack([np.cos(ang), np.sin(ang)]), [[0.0, 0.0]]])
    Z = _harmonic_dirichlet(Wd + 1e-12 * sparse.identity(n), fixed, fixed_val, n)
    U = _inverse_stereographic(Z); U[pole] = [0.0, 0.0, 1.0]
    U = _mobius_center(U, A)
    # ---- alternating stereographic Schwarz sweeps ----------------------------------------------------
    hist = [sphere_map_distortion(V, F, U)["qc_distortion_mean"]]
    for sweep in range(max_sweeps):
        for from_north in (True, False):
            Zc = _stereographic(U, from_north)
            far = (U[:, 2] if from_north else -U[:, 2]) >= 0.0           # hemisphere near the projection pole = fixed
            fixed = np.where(far)[0]
            if fixed.size == 0 or fixed.size == n:
                continue
            Zc2 = _harmonic_dirichlet(Wr, fixed, Zc[fixed], n)
            Un = _inverse_stereographic(Zc2, from_north)
            Un /= np.maximum(np.linalg.norm(Un, axis=1), 1e-300)[:, None]
            Un = _mobius_center(Un, A)
            if count_foldovers(Un, F) <= count_foldovers(U, F):
                U = Un
        hist.append(sphere_map_distortion(V, F, U)["qc_distortion_mean"])
        if hist[-2] - hist[-1] < tol:
            break
    info = {"method": "stereographic harmonic map + alternating Schwarz sweeps", "pole_vertex": int(pole),
            "cap_fraction": float(f_cap), "sweeps": len(hist) - 1, "qc_history": np.asarray(hist),
            "foldovers": count_foldovers(U, F)}
    if verbose:
        print(f"  conformal map: {len(hist) - 1} sweeps, K {hist[0]:.3f} -> {hist[-1]:.3f}, fold-overs {info['foldovers']}")
    return U, info


def _sphere_vertex_gradient(U, F, phi):
    """Area-averaged, tangent-projected vertex gradient of a vertex function on the sphere mesh."""
    G, _, _ = face_gradient_operator(U, F)
    gf = (G @ phi).reshape(-1, 3)
    _, af = face_normals_areas(U, F)
    n = U.shape[0]; g = np.zeros((n, 3)); w = np.zeros(n)
    for k in range(3):
        np.add.at(g, F[:, k], gf * af[:, None]); np.add.at(w, F[:, k], af)
    g /= np.maximum(w, 1e-300)[:, None]
    g -= np.einsum("ij,ij->i", g, U)[:, None] * U
    return g


def spherical_triangle_areas(U, F):
    """Exact areas of the spherical triangles (Van Oosterom–Strackee), total 4π for a bijective map."""
    a, b, c = U[F[:, 0]], U[F[:, 1]], U[F[:, 2]]
    num = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    den = 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return 2.0 * np.arctan2(num, den)


def _face_qc(V, F, U, src=None):
    """Per-face quasi-conformal distortion K = σ1/σ2 of the piecewise-linear map V → U (1 = conformal).
    `src` = cached local2d frames of V (see _local_frames)."""
    l1, x2, y2 = _local_frames(V, F) if src is None else src
    L1, X2, Y2 = _local_frames(U, F)
    a = L1 / np.maximum(l1, 1e-300); b = (X2 - a * x2) / np.maximum(y2, 1e-300); dd = Y2 / np.maximum(y2, 1e-300)
    s1 = a ** 2 + b ** 2 + dd ** 2; det = a * dd
    disc = np.sqrt(np.maximum(s1 ** 2 - 4 * det ** 2, 0.0))
    return np.sqrt(np.maximum((s1 + disc) / 2, 1e-300)) / np.sqrt(np.maximum((s1 - disc) / 2, 1e-300))


def _local_frames(P, F):
    """Per-face isometric 2-D frame (|e1|, e1·e2/|e1|, height) of the triangles P[F]."""
    tri = P[F]; e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
    l1 = np.linalg.norm(e1, axis=1); x2 = np.einsum("ij,ij->i", e1, e2) / np.maximum(l1, 1e-300)
    y2 = np.sqrt(np.maximum(np.einsum("ij,ij->i", e2, e2) - x2 ** 2, 0.0))
    return l1, x2, y2


def _chart_cotan_weights(g, F, face_scale, floor=1e-3):
    """Cotangent weights of the planar chart triangles g[F], clipped at `floor` (positive ⇒ Tutte-type, bijective
    solves) and multiplied per face by `face_scale` (the stretch factor of the stretch energy)."""
    n = g.shape[0]; rows, cols, vals = [], [], []
    for k in range(3):
        o, a, b = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u = g[a] - g[o]; v = g[b] - g[o]
        cr = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        cot = np.einsum("ij,ij->i", u, v) / np.maximum(cr, 1e-300)
        w = 0.5 * np.maximum(cot, floor) * face_scale
        rows += [a, b]; cols += [b, a]; vals += [w, w]
    return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()


def sphere_area_correction(V, F, U, area_weight=0.5, max_iter=120, step=1.0, patience=15, cot_floor=1e-3,
                           seed=0, verbose=False):
    """
    Area correction of a bijective sphere map by spherical STRETCH-ENERGY MINIMISATION (Yueh, Lin, Wu & Yau 2019,
    spherical variant with stereographic hemisphere charts).

    Why: a conformal map compresses protrusions exponentially (a pseudopod or the tips of an elongated cell end
    up in a spherical cap 10²–10³ times too small).  A truncated SPHARM series cannot resolve such a cap, and the
    reconstruction on a uniform icosphere shows a pinched, wrinkled 'balloon knot' there — typically near the
    poles, where the stereographic initialisation concentrates the area.  The former density-equalising flow
    capped every move at half a MEAN edge, so on these maps each step flipped triangles and it stopped after 1
    step without correcting anything.

    Each sweep: pick a random hemisphere, express the map in the stereographic chart centred on it and re-solve
    the hemisphere with the weighted Laplacian  w_ij = ½ Σ_τ∋ij max(cot θ_ij^τ, floor) · σ_τ,
    σ_τ = |f(τ)|_{S²} / |τ|_S (area-normalised stretch): over-expanded triangles get stiffer springs and shrink,
    compressed ones relax and grow; an equiareal map is a fixed point.  Positive weights make every solve a
    convex-combination (Tutte-type) embedding, the fixed hemisphere is the Dirichlet data, and an area-balancing
    re-centring follows.  A sweep is ACCEPTED only if it creates no fold-over and lowers the blended energy
        E = area_weight · D_area + (1 − area_weight) · (K̄ − 1)
    (D_area = area-weighted std of log(sphere/surface area), K̄ = area-weighted mean quasi-conformal distortion),
    otherwise it is halved (step, step/2, step/4) or discarded, so E decreases monotonically and the trade-off
    between angle and area preservation is controlled by `area_weight` (0 = keep the conformal map, 1 = aim at an
    equiareal map, SPHARM-PDM-like).  Stops after `patience` consecutive rejected sweeps or `max_iter` sweeps.
    """
    V = np.asarray(V, dtype=np.float64); F = as_faces(F); U = np.asarray(U, dtype=np.float64)
    if area_weight <= 0:
        return U, {"area_steps": 0, "area_accepted": 0}
    A = vertex_areas(V, F)
    _, am = face_normals_areas(V, F)
    wf = am / am.sum(); am4 = wf * 4.0 * np.pi
    src = _local_frames(V, F)

    def energy(P):
        r = np.log(np.maximum(spherical_triangle_areas(P, F), 1e-300) / np.maximum(am4, 1e-300))
        d_area = float(np.sqrt(np.sum(wf * (r - np.sum(wf * r)) ** 2)))
        k_mean = float(np.sum(wf * _face_qc(V, F, P, src)))
        return area_weight * d_area + (1.0 - area_weight) * (k_mean - 1.0), d_area, k_mean

    rng = np.random.default_rng(seed)
    cur = U.copy(); E_cur, d0, k0 = energy(cur); f_cur = count_foldovers(cur, F)
    hist = [E_cur]; accepted = 0; stall = 0; it = 0
    north = np.array([0.0, 0.0, 1.0])
    for it in range(max_iter):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        R = _rotation_a_to_b(d, north)                          # hemisphere around −d is re-solved
        Ur = cur @ R.T
        free = np.where(Ur[:, 2] < 0.0)[0]; fixed = np.where(Ur[:, 2] >= 0.0)[0]
        if free.size == 0 or fixed.size == 0:
            continue
        g = _stereographic(Ur, True)
        sigma = np.clip(spherical_triangle_areas(cur, F) / np.maximum(am4, 1e-300), 1e-4, 1e4)
        Wc = _chart_cotan_weights(g, F, sigma, floor=cot_floor)
        Lc = (sparse.diags(np.asarray(Wc.sum(axis=1)).ravel()) - Wc).tocsr()
        try:
            sol = splinalg.spsolve(Lc[free][:, free].tocsc(), -(Lc[free][:, fixed] @ g[fixed]))
        except Exception:  # noqa: BLE001
            stall += 1; continue
        gn = g.copy(); gn[free] = np.asarray(sol).reshape(len(free), 2)
        if not np.all(np.isfinite(gn)):
            stall += 1; continue
        ok = False
        for t in (step, 0.5 * step, 0.25 * step):
            trial = _inverse_stereographic(g + t * (gn - g), True) @ R
            trial /= np.maximum(np.linalg.norm(trial, axis=1), 1e-300)[:, None]
            trial = _mobius_normalize(trial, A, max_iter=4)
            f_t = count_foldovers(trial, F)
            if f_t > f_cur:
                continue
            E_t, _, _ = energy(trial)
            if E_t < E_cur:
                cur, E_cur, f_cur, ok = trial, E_t, f_t, True
                break
        hist.append(E_cur)
        if ok:
            accepted += 1; stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    _, d1, k1 = energy(cur)
    info = {"area_steps": it + 1, "area_accepted": accepted, "area_energy_history": np.asarray(hist),
            "area_std_before": d0, "area_std_after": d1, "qc_before": k0, "qc_after": k1}
    if verbose:
        print(f"  area correction (SEM): {accepted}/{it + 1} sweeps accepted, log-area std {d0:.3f} -> {d1:.3f}, "
              f"K {k0:.3f} -> {k1:.3f}, blended energy {hist[0]:.3f} -> {E_cur:.3f}")
    return cur, info


def spherical_parametrization_v2(V, F, method="conformal", area_weight=0.7, polish_iters=40, temporal_target=None,
                                 temporal_weight=0.0, n_iter=500, verbose=False):
    """
    Improved spherical parametrization:
      initialisation — method='conformal' ('flash'): stereographic harmonic map + Schwarz sweeps with true Möbius
      normalisation; method='harmonic': projected-gradient descent of the Dirichlet energy (Gu et al. 2004), which
      falls back to the conformal initialisation if it cannot be untangled (fold-overs remain) —
      → fold-over untangling (and an optional pull towards `temporal_target`, the sphere positions of the
      corresponding vertices of the previous frame, weight `temporal_weight`, for time-consistent coefficients)
      → spherical stretch-energy area correction with the angle/area trade-off `area_weight`
      (sphere_area_correction; 0 keeps the initial map).
    Returns U, info (distortion metrics of every stage).
    """
    V = np.asarray(V, dtype=np.float64); F = as_faces(F)
    if method in ("conformal", "flash"):
        U0, info = spherical_conformal_map(V, F, verbose=verbose)
    elif method == "harmonic":
        U0, info = spherical_parametrization(V, F, n_iter=n_iter, verbose=verbose)
        info = {"method": "harmonic (projected gradient)", **info}
        if info["foldovers"] > 0:
            if verbose:
                print(f"  harmonic map kept {info['foldovers']} fold-overs -> conformal initialisation instead")
            U0, cinfo = spherical_conformal_map(V, F, verbose=verbose)
            info = {**cinfo, "method": "harmonic -> conformal fallback", "harmonic_foldovers": info["foldovers"]}
    else:
        raise ValueError(f"Unknown sphere method '{method}' (use 'conformal' or 'harmonic').")
    info = {k: v for k, v in info.items() if k not in ("residual_history",)}
    d0 = sphere_map_distortion(V, F, U0); info.update({f"initial_{k}": v for k, v in d0.items()})
    U = U0
    if temporal_target is not None and temporal_weight > 0:
        Rt = kabsch_rotation(U, temporal_target); U = U @ Rt.T
    # polishing: only the fold-over untangling unless a temporal pull is requested (the clipped-weight
    # gradient descent would degrade the conformality of the Schwarz solution)
    n_pol = polish_iters if (temporal_target is not None and temporal_weight > 0) else 0
    U, pinfo = spherical_parametrization(V, F, n_iter=n_pol, init=U, verbose=False,
                                         pull_target=temporal_target if temporal_weight > 0 else None, pull_weight=temporal_weight)
    info["polish_foldovers"] = pinfo["foldovers"]
    if area_weight > 0:
        U, ainfo = sphere_area_correction(V, F, U, area_weight=area_weight, verbose=verbose)
        info.update({k: v for k, v in ainfo.items() if k != "area_energy_history"})
        info["area_energy_history"] = ainfo.get("area_energy_history")
        if temporal_target is not None and temporal_weight > 0:
            Rt = kabsch_rotation(U, temporal_target); U = U @ Rt.T
    info["foldovers"] = count_foldovers(U, F)
    info.update(sphere_map_distortion(V, F, U))
    info["area_weight"] = float(area_weight); info["temporal_weight"] = float(temporal_weight)
    return U, info


def sphere_map_distortion(V, F, U):
    """Area distortion (normalised-area ratio) and angle distortion of the map S → S² (per-face, summarised)."""
    _, a_mesh = face_normals_areas(V, F)
    _, a_sph = face_normals_areas(U, F)
    a_mesh = a_mesh / a_mesh.sum(); a_sph = a_sph / max(a_sph.sum(), 1e-300)
    ratio = np.log(np.maximum(a_sph, 1e-15) / np.maximum(a_mesh, 1e-15))

    def angles(P):
        tri = P[F]; out = []
        for k in range(3):
            u = tri[:, (k + 1) % 3] - tri[:, k]; w = tri[:, (k + 2) % 3] - tri[:, k]
            cosang = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300)
            out.append(np.arccos(np.clip(cosang, -1, 1)))
        return np.stack(out, axis=1)
    dang = np.abs(angles(V) - angles(U))
    # quasi-conformal distortion K = σ1/σ2 of the per-face Jacobian (1 = conformal), in local 2-D frames
    def local2d(P):
        tri = P[F]; e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
        l1 = np.linalg.norm(e1, axis=1); x2 = np.einsum("ij,ij->i", e1, e2) / np.maximum(l1, 1e-300)
        y2 = np.sqrt(np.maximum(np.einsum("ij,ij->i", e2, e2) - x2 ** 2, 0.0))
        return l1, x2, y2
    l1, x2, y2 = local2d(V); L1, X2, Y2 = local2d(U)
    # J = [[L1, X2],[0, Y2]] · [[l1, x2],[0, y2]]^{-1}
    a = L1 / np.maximum(l1, 1e-300); b = (X2 - a * x2) / np.maximum(y2, 1e-300); dd = Y2 / np.maximum(y2, 1e-300)
    s1 = a ** 2 + b ** 2 + dd ** 2; det = a * dd
    disc = np.sqrt(np.maximum(s1 ** 2 - 4 * det ** 2, 0.0))
    sig1 = np.sqrt(np.maximum((s1 + disc) / 2, 1e-300)); sig2 = np.sqrt(np.maximum((s1 - disc) / 2, 1e-300))
    K = sig1 / sig2; w = a_mesh
    return {"area_distortion_L2": float(np.sqrt(np.mean(ratio ** 2))),
            "area_distortion_max": float(np.max(np.abs(ratio))),
            "angle_distortion_mean_deg": float(np.degrees(dang.mean())),
            "angle_distortion_max_deg": float(np.degrees(dang.max())),
            "qc_distortion_mean": float(np.sum(w * K)), "qc_distortion_p95": float(np.percentile(K, 95)),
            "log_area_ratio_std": float(np.sqrt(np.sum(w * (ratio - np.sum(w * ratio)) ** 2)))}


def align_spheres(U_curr, U_prev, p2p):
    """Rotate frame t's sphere so that vertex j lands on the sphere position of its match p2p[j] in frame t-1."""
    P = U_curr; Q = U_prev[p2p]
    R = kabsch_rotation(P, Q)
    Ua = U_curr @ R.T
    err = float(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", Ua, Q), -1, 1))).mean())
    return Ua, R, err


def sphere_to_angles(U):
    """θ ∈ [0, π] polar angle from +z, φ ∈ (−π, π] azimuth."""
    U = U / np.maximum(np.linalg.norm(U, axis=1), 1e-300)[:, None]
    theta = np.arccos(np.clip(U[:, 2], -1.0, 1.0))
    phi = np.arctan2(U[:, 1], U[:, 0])
    return theta, phi


# ----------------------------------------------------------------------------- #
#  2. Real spherical harmonics and SPHARM fitting
# ----------------------------------------------------------------------------- #

def _complex_sph_harm(l, m, theta, phi):
    """scipy compatibility: sph_harm_y (≥1.15, args l,m,θ,φ) or the legacy sph_harm(m,l,φ,θ)."""
    from scipy import special
    if hasattr(special, "sph_harm_y"):
        return special.sph_harm_y(l, m, theta, phi)
    return special.sph_harm(m, l, phi, theta)


def real_sh_basis(theta, phi, L):
    """
    Real orthonormal spherical harmonics up to degree L, ordered (l, m) with m = −l..l:
        Y_l0 = Y_l^0,  Y_lm = √2 Re Y_l^m (m>0),  Y_l,−m = √2 Im Y_l^m (m>0).
    Returns B (n, (L+1)²) and the list of (l, m).
    """
    theta = np.asarray(theta); phi = np.asarray(phi)
    cols, lm = [], []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y = _complex_sph_harm(l, abs(m), theta, phi)
            if m == 0:
                cols.append(np.real(Y))
            elif m > 0:
                cols.append(np.sqrt(2.0) * np.real(Y))
            else:
                cols.append(np.sqrt(2.0) * np.imag(Y))
            lm.append((l, m))
    return np.stack(cols, axis=1), lm


def fit_spharm(V, theta, phi, L, areas=None, reg=1e-6):
    """
    Area-weighted least squares  min Σ_i A_i ||x_i − B_i c||² + reg Σ_lm (l(l+1))² ||c_lm||²
    (the penalty is the squared bi-Laplacian on S², i.e. a thin-plate / Willmore-type smoother that
    tames high degrees where vertices are sparse).  Returns c ((L+1)², 3), B, lm.
    """
    B, lm = real_sh_basis(theta, phi, L)
    A = np.ones(V.shape[0]) if areas is None else np.asarray(areas)
    A = A / A.mean()
    pen = np.array([(l * (l + 1)) ** 2 for l, _ in lm], dtype=np.float64)
    Bg, Ag, Vg = to_gpu(B), to_gpu(A), to_gpu(np.asarray(V, dtype=np.float64))
    BtA = Bg.T * Ag[None, :]
    lhs = BtA @ Bg
    di = xp.arange(lhs.shape[0])
    lhs[di, di] += to_gpu(reg * pen)                          # regulariser on the diagonal (no dense N x N matrix)
    rhs = BtA @ Vg
    del BtA, Bg
    c = to_cpu(xp.linalg.solve(lhs, rhs))
    del lhs, rhs, Ag, Vg, di
    return c, B, lm


def spharm_reconstruction_errors(V, B, c, lm, areas=None):
    """RMS (area weighted) reconstruction error using degrees ≤ l, for l = 0..L."""
    A = np.ones(V.shape[0]) if areas is None else np.asarray(areas); A = A / A.sum()
    degs = np.array([l for l, _ in lm])
    scale = np.sqrt(np.sum(A * np.sum((V - (A[:, None] * V).sum(0)) ** 2, axis=1)))
    out = []
    for l in range(degs.max() + 1):
        sel = degs <= l
        rec = B[:, sel] @ c[sel]
        rms = np.sqrt(np.sum(A * np.sum((V - rec) ** 2, axis=1)))
        out.append((l, rms, rms / max(scale, 1e-300)))
    return out


def spharm_degree_energy(c, lm):
    degs = np.array([l for l, _ in lm])
    return np.array([np.sum(c[degs == l] ** 2) for l in range(degs.max() + 1)])


def spharm_synthesize(c, lm, L=None, sphere_subdiv=4, template=None):
    """Evaluate the explicit surface x(θ,φ) on an icosphere template (or on given unit vectors)."""
    if template is None:
        Us, Fs = icosphere(sphere_subdiv)
    else:
        Us, Fs = template
    theta, phi = sphere_to_angles(Us)
    Lmax = max(l for l, _ in lm) if L is None else L
    B, lm_t = real_sh_basis(theta, phi, Lmax)
    degs = np.array([l for l, _ in lm])
    return B @ c[degs <= Lmax], Fs


_SYM_BASIS_CACHE = {}


def _symbolic_real_sh(l, m):
    """Real spherical harmonic (l, m) as a SymPy expression in θ, φ — cached (identical for every frame)."""
    import sympy as sp
    key = (l, m)
    if key not in _SYM_BASIS_CACHE:
        theta, phi = sp.symbols("theta phi", real=True)
        Y = sp.Ynm(l, abs(m), theta, phi).expand(func=True)
        if m == 0:
            basis = sp.re(Y)
        elif m > 0:
            basis = sp.sqrt(2) * sp.re(Y)
        else:
            basis = sp.sqrt(2) * sp.im(Y)
        _SYM_BASIS_CACHE[key] = sp.trigsimp(sp.expand(basis, complex=True))
    return _SYM_BASIS_CACHE[key]


def spharm_symbolic(c, lm, L_sym=3, precision=6):
    """
    Truncated closed-form series as SymPy expressions x(θ,φ), y(θ,φ), z(θ,φ) (real basis), with the
    coefficients rounded to `precision` significant digits.  Returns dict with 'expr' (3 sympy
    expressions), 'latex' and 'text'.  Higher degrees are kept numerically in coeffs_T####.npy.
    """
    import sympy as sp
    theta, phi = sp.symbols("theta phi", real=True)
    exprs = [sp.Integer(0)] * 3
    for (l, m), coef in zip(lm, c):
        if l > L_sym:
            continue
        basis = _symbolic_real_sh(l, m)
        for k in range(3):
            if abs(coef[k]) > 10 ** (-precision - 2):
                exprs[k] = exprs[k] + sp.Float(coef[k], precision) * basis
    names = ["x", "y", "z"]
    latex = "\n".join(rf"{nm}(\theta,\varphi) = {sp.latex(e)}" for nm, e in zip(names, exprs))
    text = "\n".join(f"{nm}(theta, phi) = {sp.sstr(e)}" for nm, e in zip(names, exprs))
    return {"expr": exprs, "latex": latex, "text": text, "symbols": (theta, phi)}


# ----------------------------------------------------------------------------- #
#  3. Manifold-harmonics reconstruction (any topology)
# ----------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------- #
#  Automatic degree / number of modes: geometric complexity vs available memory
# ----------------------------------------------------------------------------- #

def spharm_memory_bytes(n, L):
    """Peak memory of fit_spharm (float64): basis on the host (n·N), on the device the basis, its weighted transpose
    and the normal equations with their factorisation (2·n·N + 2·N²), N = (L+1)²."""
    N = (int(L) + 1) ** 2
    return {"host": 8 * n * N, "device": 8 * (2 * n * N + 2 * N * N)}


def memory_budget(fraction=0.6):
    """Bytes the parametrization may use: `fraction` of the free memory of the device running the dense algebra
    (GPU when CuPy is used) and of the host."""
    from PynamicMesh.core.accel import available_memory, free_memory
    free_memory()
    mem = available_memory()
    host = fraction * mem["cpu"]
    dev = fraction * mem["gpu"] if (GPU_AVAILABLE and mem.get("gpu")) else host
    return {"host": float(host), "device": float(dev), "gpu": bool(GPU_AVAILABLE and mem.get("gpu"))}


def max_degree_for_memory(n, budget):
    """Largest SPHARM degree whose least-squares fit fits the memory budget (and the samples: N <= n / 2)."""
    L = 1
    while True:
        m = spharm_memory_bytes(n, L + 1)
        if budget.get("gpu"):
            too_big = m["device"] > budget["device"] or m["host"] > budget["host"]
        else:                                                # CPU only: everything lives in host memory
            too_big = m["device"] + m["host"] > budget["host"]
        if too_big or (L + 2) ** 2 > n / 2:
            return L
        L += 1
        if L > 4000:
            return L


def needed_spharm_degree(V, theta, phi, areas, target=0.02, L_cap=60, reg=1e-6, L_min=4):
    """
    Degree the GEOMETRY needs, measured exactly as the reported error (area-weighted on the SURFACE, so thin parts —
    legs, neck, protrusions — squeezed by the sphere map count fully). The error of a dedicated fit decreases with L
    (nested bases), so the smallest L with error <= target is found by bisection with dedicated fits (~log2(L_cap)
    fits on one frame). When even L_cap (the memory / vertex limit) does not reach the target, L_cap is returned.
    Complex shapes spread their geometry over high degrees and need a larger L.
    Returns (L, {L: error of the dedicated fit}).
    """
    tested = {}

    def err_at(L):
        if L not in tested:
            c, B, lm = fit_spharm(V, theta, phi, int(L), areas=areas, reg=reg)
            tested[L] = float(spharm_reconstruction_errors(V, B, c, lm, areas=areas)[-1][2])
            del B, c
        return tested[L]
    L_cap = int(max(L_min, L_cap))
    if err_at(L_cap) > target:
        return L_cap, tested
    lo, hi = int(L_min), L_cap                                 # invariant: err(hi) <= target
    if err_at(lo) <= target:
        return lo, tested
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if err_at(mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi, tested


def auto_mh_modes(V, F, W, M, trimesh_obj=None, target=0.02, k_cap=600, ladder=(40, 80, 120, 200, 300, 400, 600)):
    """Smallest number of manifold-harmonic modes reaching the target relative reconstruction error on this frame
    (the eigenbasis is grown along `ladder`, never beyond k_cap); returns (k, error at k)."""
    k_cap = int(min(k_cap, V.shape[0] - 2))
    last = None
    for k in [c for c in ladder if c < k_cap] + [k_cap]:
        mh = manifold_harmonics(V, F, k=k, trimesh_obj=trimesh_obj, W=W, M=M)
        errs = np.array([e[2] for e in mh["errors"]])
        last = (int(k), float(errs[-1]))
        ok = np.flatnonzero(errs <= target)
        del mh
        if len(ok):
            return int(ok[0] + 1), float(errs[ok[0]])
    return last


def max_modes_for_memory(n, budget, factor=6):
    """Largest number of eigenpairs whose basis and eigensolver workspace (≈ factor · n · k doubles) fit the budget."""
    return int(max(10, min(n - 2, budget["host"] / (8.0 * factor * n))))


def manifold_harmonics(V, F, k=100, trimesh_obj=None, W=None, M=None):
    """
    Coefficients α = Φᵀ M x of the coordinate functions in the LB eigenbasis and the RMS
    reconstruction error vs. number of modes.  x ≈ Σ_{j<k} α_j φ_j is an explicit expression in the
    intrinsic coordinates (φ_1, ..., φ_k).
    """
    if W is None or M is None:
        W, M, _ = cotan_laplacian(V, F)
    evals, Phi = laplacian_eigenbasis(W, M, k, trimesh_obj)
    alpha = Phi.T @ (M @ V)                                   # (k, 3)
    A = M.diagonal() / M.diagonal().sum()
    scale = np.sqrt(np.sum(A * np.sum((V - (A[:, None] * V).sum(0)) ** 2, axis=1)))
    errs = []
    Vg, Phig, alg, Ag = to_gpu(V), to_gpu(Phi), to_gpu(alpha), to_gpu(A)
    rec = xp.zeros_like(Vg)
    for j in range(1, Phi.shape[1] + 1):                      # running sum: O(n k) instead of O(n k²)
        rec += Phig[:, j - 1:j] @ alg[j - 1:j]
        e = float(to_cpu(xp.sqrt(xp.sum(Ag * xp.sum((Vg - rec) ** 2, axis=1)))))
        errs.append((j, e, e / max(scale, 1e-300)))
    del Vg, Phig, alg, Ag, rec
    return {"eigenvalues": evals, "basis": Phi, "coeffs": alpha, "errors": errs}


# ----------------------------------------------------------------------------- #
#  4. ARAP planar parametrization (disk topology)
# ----------------------------------------------------------------------------- #

def _boundary_loop(F):
    e_all = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    E, counts = np.unique(e_all, axis=0, return_counts=True)
    bE = E[counts == 1]
    if bE.shape[0] == 0:
        return None
    nbr = {}
    for a, b in bE:
        nbr.setdefault(a, []).append(b); nbr.setdefault(b, []).append(a)
    start = bE[0, 0]; loop = [start]; prev = -1; cur = start
    while True:
        nxt = [v for v in nbr[cur] if v != prev]
        if not nxt:
            break
        nxt = nxt[0]
        if nxt == start:
            break
        loop.append(nxt); prev, cur = cur, nxt
        if len(loop) > bE.shape[0]:
            break
    return np.asarray(loop)


def tutte_embedding(V, F, boundary):
    """Convex-combination (mean value / cotan-clipped) embedding with the boundary mapped to the unit circle by arc length."""
    n = V.shape[0]
    Wij = cotan_weights(V, F, clip_negative=True)
    Wij.data = np.maximum(Wij.data, 1e-6)
    d = np.asarray(Wij.sum(axis=1)).ravel()
    L = sparse.diags(d) - Wij
    seg = np.linalg.norm(V[boundary] - V[np.roll(boundary, -1)], axis=1)
    s = np.concatenate([[0], np.cumsum(seg)[:-1]]) / seg.sum() * 2 * np.pi
    uv = np.zeros((n, 2)); uv[boundary] = np.stack([np.cos(s), np.sin(s)], axis=1)
    interior = np.setdiff1d(np.arange(n), boundary)
    L_II = L[interior][:, interior].tocsc(); L_IB = L[interior][:, boundary]
    uv[interior] = splinalg.spsolve(L_II, -L_IB @ uv[boundary])
    return uv


def arap_planar_parametrization(V, F, n_iter=20, verbose=False):
    """
    ARAP parametrization (Liu et al. 2008): per-triangle Jacobian J_t (from the isometrically flattened
    reference triangle) → local step: R_t = closest rotation (polar decomposition, Sorkine & Alexa
    eq. 6 in 2D); global step: solve  Σ_t cot-weighted Poisson system  L u = b(R_t).
    Returns (uv, info) with the isometric distortion σ_1/σ_2 statistics.
    """
    V = np.asarray(V, dtype=np.float64); F = as_faces(F)
    boundary = _boundary_loop(F)
    if boundary is None:
        raise ValueError("ARAP planar parametrization needs a disk-topology (open) mesh.")
    uv = tutte_embedding(V, F, boundary)
    n = V.shape[0]; nf = F.shape[0]

    # local isometric frames per triangle: x_t ∈ R^{2x3}
    tri = V[F]
    e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
    ax1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-300)[:, None]
    nrm = np.cross(e1, e2); nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    ax2 = np.cross(nrm, ax1)
    X = np.zeros((nf, 3, 2))
    X[:, 1, 0] = np.einsum("ij,ij->i", e1, ax1); X[:, 1, 1] = np.einsum("ij,ij->i", e1, ax2)
    X[:, 2, 0] = np.einsum("ij,ij->i", e2, ax1); X[:, 2, 1] = np.einsum("ij,ij->i", e2, ax2)

    # cotangent weights per triangle corner (for the Poisson system)
    cot = np.zeros((nf, 3))
    for k in range(3):
        a = X[:, (k + 1) % 3] - X[:, k]; b = X[:, (k + 2) % 3] - X[:, k]
        cot[:, k] = np.einsum("ij,ij->i", a, b) / np.maximum(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), 1e-300)
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j = F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        rows += [i, j]; cols += [j, i]; vals += [-cot[:, k], -cot[:, k]]
    Wij = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    L = sparse.diags(-np.asarray(Wij.sum(axis=1)).ravel()) + Wij
    # fix one vertex (translation) via soft constraint on the mean
    Lc = (L + 1e-8 * sparse.identity(n)).tocsc()
    lu = splinalg.splu(Lc)

    def local_step(uv):
        tri_uv = uv[F]
        # J_t maps reference edges (2D) to uv edges: J = [uv edges] [X edges]^{-1}
        Ex = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]], axis=2)          # (F,2,2)
        Eu = np.stack([tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]], axis=2)
        J = Eu @ np.linalg.inv(Ex)
        U, S, Vt = np.linalg.svd(J)
        R = U @ Vt
        det = np.linalg.det(R)
        flip = det < 0
        if flip.any():
            U[flip, :, 1] *= -1
            R = U @ Vt
        return R, S

    hist = []
    for it in range(n_iter):
        R, S = local_step(uv)
        b = np.zeros((n, 2))
        for k in range(3):
            i, j = F[:, (k + 1) % 3], F[:, (k + 2) % 3]
            e_ref = X[:, (k + 1) % 3] - X[:, (k + 2) % 3]                          # (F,2)
            re = np.einsum("fij,fj->fi", R, e_ref) * cot[:, k][:, None]
            np.add.at(b, i, re); np.add.at(b, j, -re)
        uv_new = lu.solve(b)
        uv_new -= uv_new.mean(axis=0)
        hist.append(float(np.mean((S[:, 0] - 1) ** 2 + (S[:, 1] - 1) ** 2)))
        uv = uv_new
    R, S = local_step(uv)
    info = {"iterations": n_iter, "arap_energy_history": np.asarray(hist),
            "stretch_sigma1_mean": float(S[:, 0].mean()), "stretch_sigma2_mean": float(S[:, 1].mean()),
            "isometric_distortion_mean": float(np.mean(S[:, 0] / np.maximum(S[:, 1], 1e-12))),
            "flipped_triangles": int(np.sum(np.linalg.det(np.stack(
                [uv[F[:, 1]] - uv[F[:, 0]], uv[F[:, 2]] - uv[F[:, 0]]], axis=2)) <= 0))}
    if verbose:
        print(f"  ARAP planar parametrization: distortion {info['isometric_distortion_mean']:.3f}, flips {info['flipped_triangles']}")
    return uv, info


# ----------------------------------------------------------------------------- #
#  5. Sequence driver
# ----------------------------------------------------------------------------- #

def repair_to_sphere(V, F):
    """
    Makes a 'nearly closed' cell mesh usable for the spherical map without touching the vertex indexing:
      * keeps the faces of the largest connected component (dust / debris components are dropped),
      * closes every boundary loop with a fan around a virtual centroid vertex (appended at the end),
      * reports non-manifold edges (tolerated by the harmonic map).
    Returns V2, F2 (augmented), the mask of original vertices that belong to the kept component, and a
    dict describing what was done.  The sphere positions of dropped vertices are later filled by nearest
    neighbour so that every original vertex index has (θ, φ).
    """
    from scipy.sparse.csgraph import connected_components
    n = V.shape[0]
    E = mesh_edges(F)
    adj = sparse.coo_matrix((np.ones(E.shape[0]), (E[:, 0], E[:, 1])), shape=(n, n))
    n_comp, lab = connected_components(adj, directed=False)
    face_lab = lab[F[:, 0]]
    if n_comp > 1:
        counts = np.bincount(face_lab, minlength=n_comp)
        main = int(np.argmax(counts)); F2 = F[face_lab == main]
    else:
        main = 0; F2 = F.copy()
    kept = np.zeros(n, bool); kept[np.unique(F2)] = True
    # boundary loops of the kept component
    e_dir = np.concatenate([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]])
    e_key = np.sort(e_dir, axis=1)
    _, inv, cnt = np.unique(e_key, axis=0, return_inverse=True, return_counts=True)
    b_dir = e_dir[cnt[inv.ravel()] == 1]                       # boundary half-edges in face orientation
    V2 = V.copy(); n_virtual = 0; n_loops = 0
    if b_dir.shape[0]:
        badj = sparse.coo_matrix((np.ones(b_dir.shape[0]), (b_dir[:, 0], b_dir[:, 1])), shape=(n, n))
        n_l, bl = connected_components(badj, directed=False)
        loop_ids = np.unique(bl[b_dir[:, 0]])
        new_faces = []
        for L in loop_ids:
            sel = b_dir[bl[b_dir[:, 0]] == L]
            verts = np.unique(sel)
            c = V[verts].mean(axis=0)
            cid = V2.shape[0]; V2 = np.vstack([V2, c[None]])
            # face (a,b) orientation: the boundary half-edge (a,b) belongs to a face; the closing triangle is (b, a, c)
            new_faces.append(np.column_stack([sel[:, 1], sel[:, 0], np.full(sel.shape[0], cid)]))
            n_virtual += 1; n_loops += 1
        F2 = np.vstack([F2] + new_faces)
    info = {"repaired": bool(n_comp > 1 or n_loops > 0), "dropped_components": int(n_comp - 1), "closed_loops": int(n_loops),
            "virtual_vertices": int(n_virtual), "nonmanifold_edges": int(np.sum(cnt > 2)),
            "dropped_vertices": int((~kept).sum())}
    return V2, F2, kept, info


@logged_stage("Parametrization", (1, "target_folder"), "Parametrization")
def compute_parametrization(mesh_folder, target_folder, loader=None, L_max=15, reg=1e-6, n_iter=500,
                            sphere_method="conformal", area_weight=0.7, temporal_weight=0.0,
                            mh_modes=80, export_degrees=(4, 8, 15), sphere_subdiv="auto", symbolic_degree=3,
                            align_with_p2p=True, planar_if_disk=True, plots=True, verbose=True, frames=None,
                            export_on_mesh=True, high_degree=True, hd_lmax="auto", sht_backend="auto",
                            auto_target_error=0.02, memory_fraction=0.6):
    """
    Runs idea (a) on a whole scene.  `frames` (list from dyn_common.load_frames) can be passed to
    avoid reloading.  Returns a dict with paths and the stacked consistent coefficients.
    """
    """
    sphere_method : 'conformal' (stereographic harmonic map + alternating Schwarz sweeps with Möbius normalisation)
                    or 'harmonic' (projected-gradient descent of the Dirichlet energy, conformal fallback if it keeps
                    fold-overs); BOTH are followed by the spherical stretch-energy area correction (`area_weight`).
    area_weight   : 0 = conformal map (angles preserved, area concentrated on protrusions → pinched 'knots' in the
                    SPHARM reconstruction); 1 = area-preserving (uniform sampling, SPHARM-PDM-like); 0.7 (default)
                    gave the lowest SPHARM error at L = 25 on bumpy, elongated and non-star-shaped test cells.
    sphere_subdiv : icosphere level of the exported SPHARM surfaces; 'auto' = 4 for L ≤ 15, 5 for L ≤ 40, else 6
                    (a level-4 icosphere has ~4 samples per wavelength at L = 25 and looks faceted).
    export_on_mesh: also write reconstruction_T####_L##_mesh.obj, the truncated series evaluated at the sphere
                    positions of the original vertices (original connectivity, directly comparable to the input).
    temporal_weight: >0 pulls each sphere map towards the previous frame's map of the corresponding vertices
                    (through the p2p maps) for smoother coefficient trajectories c_lm(t); costs some distortion.
    quality.csv gets, per frame: fold-overs, quasi-conformal distortion K (mean, p95), log-area-ratio std,
    the same for the pure conformal stage (conformal_*), and the GCV-recommended SPHARM degree.
    """
    target_folder = Path(target_folder)
    out = ensure_dir(target_folder / "Parametrization")
    d_sph = ensure_dir(out / "Spherical"); d_spharm = ensure_dir(out / "SPHARM")
    d_mh = ensure_dir(out / "ManifoldHarmonics"); d_sym = ensure_dir(out / "symbolic")
    d_plots = ensure_dir(out / "plots")
    # per-frame results of a previous run (possibly of a different mesh set in the same scene folder) are removed,
    # otherwise stale frames would be mixed with the new ones and the viewers would read mismatching arrays
    n_old = 0
    for d in (d_sph, d_spharm, d_mh, out / "Planar"):
        if d.is_dir():
            for f in list(d.glob("*_T[0-9]*.*")):
                f.unlink(); n_old += 1
    if n_old and verbose:
        print(f"Removed {n_old} per-frame files of a previous parametrization run in {out}")
    if frames is None:
        frames = load_frames(mesh_folder, loader)
    T = len(frames)
    p2p, _ = load_p2p_chain(target_folder / "Transform_Matrices", T) if align_with_p2p else ([None] * T, None)

    quality, degree_energy, rec_err, mh_err = [], [], [], []
    coeffs = []
    U_prev = None
    # degree / number of modes: chosen from the geometry ('auto') and always kept within the available memory
    budget = memory_budget(memory_fraction)
    auto_info = {"memory_budget_bytes": budget, "target_rel_error": auto_target_error,
                 "L_max": {"requested": L_max}, "mh_modes": {"requested": mh_modes}}
    L_requested, mh_requested = L_max, mh_modes
    L_max = None if (isinstance(L_max, str) or L_max is None) else int(L_max)
    sphere_subdiv_req, export_degrees_req = sphere_subdiv, export_degrees
    template = None

    def _degree_settings(L):
        sub = sphere_subdiv_req
        if sub in (None, "auto"):
            sub = 4 if L <= 15 else (5 if L <= 40 else 6)
        ed = export_degrees_req
        if isinstance(ed, (int, np.integer)):                  # a single degree is accepted as well as a tuple
            ed = (int(ed),)
        auto_info["sphere_subdiv"] = int(sub)
        return icosphere(int(sub)), tuple(int(d) for d in ed if d <= L) or (int(L),)
    if L_max is not None and not (isinstance(L_requested, str)):
        template, export_degrees = _degree_settings(min(L_max, 4000))

    progress(total=T + 1 + (1 if high_degree else 0), advance=0)
    for i, fr in enumerate(pbar(frames, desc="Spherical parametrization")):
        V, F = fr["V"], fr["F"]
        topo = mesh_topology(V, F)
        row = {"frame": i, "name": fr["name"], **topo}
        W, M, _ = cotan_laplacian(V, F)
        A = M.diagonal()

        # --- manifold harmonics: always available ---------------------------------------------
        if i == 0:
            k_mem = max_modes_for_memory(V.shape[0], budget)
            if isinstance(mh_requested, str):              # 'auto': as many modes as the geometry needs
                k_need, e_need = auto_mh_modes(V, F, W, M, fr.get("mesh"), auto_target_error, k_cap=min(k_mem, 600))
                mh_modes = int(min(k_need, k_mem))
                auto_info["mh_modes"].update({"needed": int(k_need), "rel_error_at_needed": e_need})
            else:
                mh_modes = int(min(int(mh_requested), k_mem, V.shape[0] - 2))
            auto_info["mh_modes"].update({"memory_cap": int(k_mem), "chosen": int(mh_modes)})
            if not isinstance(mh_requested, str) and mh_modes < int(mh_requested):
                print(f"  mh_modes reduced from {mh_requested} to {mh_modes} (memory / number of vertices)")
            from PynamicMesh.core.accel import free_memory
            free_memory()
        mh = manifold_harmonics(V, F, k=mh_modes, trimesh_obj=fr.get("mesh"), W=W, M=M)
        np.save(d_mh / f"mh_coeffs_T{i:04d}.npy", mh["coeffs"])
        np.save(d_mh / f"mh_eigenvalues_T{i:04d}.npy", mh["eigenvalues"])
        for j, e, er in mh["errors"]:
            mh_err.append({"frame": i, "n_modes": j, "rms_error": e, "rel_error": er})
        row["mh_rel_error_all_modes"] = mh["errors"][-1][2]

        # --- spherical / SPHARM: genus 0 closed (repaired when holes / debris break the topology) -----
        sphere_ok = topo["topology"] == "sphere" and topo["n_unreferenced_vertices"] == 0; Vs_, Fs_, kept = V, F, None
        if not sphere_ok and topo["topology"] != "disk":
            Vs_, Fs_, kept, rep = repair_to_sphere(V, F)
            topo_rep = mesh_topology(Vs_, Fs_)
            row.update({f"repair_{k}": v for k, v in rep.items()}); row["topology_after_repair"] = topo_rep["topology"]
            sphere_ok = topo_rep["topology"] == "sphere"
            print(f"  frame {i} ({fr['name']}): topology {topo['topology']} ({topo['n_unreferenced_vertices']} unreferenced vertices) "
                  f"-> repair {rep} -> {topo_rep['topology']}")
            if not sphere_ok:
                print(f"  frame {i}: still not genus 0 after repair (chi={topo_rep['euler_characteristic']}, loops={topo_rep['n_boundary_loops']}); "
                      "spherical map attempted anyway, check fold-overs in quality.csv")
                sphere_ok = topo_rep["closed"]
        if sphere_ok:
            target = None
            if U_prev is not None and p2p[i] is not None and p2p[i].shape[0] == V.shape[0] and temporal_weight > 0:
                target = np.vstack([U_prev[p2p[i]], np.zeros((Vs_.shape[0] - V.shape[0], 3))]) if Vs_.shape[0] > V.shape[0] else U_prev[p2p[i]]
            U, info = spherical_parametrization_v2(Vs_, Fs_, method=sphere_method, area_weight=area_weight,
                                                   temporal_target=target, temporal_weight=temporal_weight,
                                                   n_iter=n_iter, verbose=False)
            row.update({k: v for k, v in info.items() if k not in ("residual_history", "qc_history", "area_energy_history")})
            if U.shape[0] != V.shape[0] or (kept is not None and not kept.all()):
                U = U[:V.shape[0]]                                   # drop the virtual hole centres
                if kept is not None and not kept.all():           # dropped debris vertices: nearest kept sphere position
                    from scipy.spatial import cKDTree
                    _, nn = cKDTree(V[kept]).query(V[~kept]); U[~kept] = U[kept][nn]
            if U_prev is not None and p2p[i] is not None and p2p[i].shape[0] == V.shape[0]:
                U, Ralign, err = align_spheres(U, U_prev, p2p[i])
                row["alignment_residual_deg"] = err
            elif U_prev is not None and coeffs:
                # fallback: align the degree-1 (ellipsoid) part of the expansion
                U, err = _align_by_degree1(U, V, coeffs[-1], L_max, A)
                row["alignment_residual_deg"] = err
            U_prev = U
            theta, phi = sphere_to_angles(U)
            np.save(d_sph / f"sphere_T{i:04d}.npy", U)
            np.save(d_sph / f"theta_phi_T{i:04d}.npy", np.stack([theta, phi], axis=1))

            if template is None or L_max is None or "chosen" not in auto_info["L_max"]:
                L_mem = max_degree_for_memory(V.shape[0], budget)
                if isinstance(L_requested, str) or L_requested is None:
                    # 'auto': the degree the geometry needs (spectral probe), within memory / samples
                    L_probe = int(min(L_mem, 100))
                    L_need, perr = needed_spharm_degree(V, theta, phi, A, auto_target_error, L_cap=L_probe, reg=reg)
                    L_max = int(max(4, min(L_need, L_mem)))
                    reached = bool(min(perr.values()) <= auto_target_error)
                    auto_info["L_max"].update({"needed": int(L_need) if reached else None, "target_reached": reached,
                                               "probe_degree": L_probe, "expected_rel_error": float(perr[L_max]),
                                               "tested_degrees": {int(k_): float(v_) for k_, v_ in sorted(perr.items())}})
                    if not reached:
                        print(f"  L_max: the target error {auto_target_error} is not reached within the memory / "
                              f"vertices (L <= {L_probe}: error {min(perr.values()):.4f}); L_max={L_max} is used")
                    from PynamicMesh.core.accel import free_memory
                    free_memory()
                else:
                    L_max = int(min(int(L_requested), L_mem))
                    if L_max < int(L_requested):
                        print(f"  L_max reduced from {L_requested} to {L_max}: the least-squares fit would need "
                              f"{spharm_memory_bytes(V.shape[0], int(L_requested))['device'] / 1e9:.3g} GB")
                need_mem = spharm_memory_bytes(V.shape[0], L_max)
                auto_info["L_max"].update({"memory_cap": int(L_mem), "chosen": int(L_max),
                                           "estimated_peak_bytes": need_mem})
                template, export_degrees = _degree_settings(L_max)
                print(f"  SPHARM degree L_max={L_max}, manifold-harmonic modes={mh_modes} "
                      f"(peak ≈ {need_mem['device'] / 1e6:.0f} MB on the {'GPU' if budget['gpu'] else 'CPU'})")
            c, B, lm = fit_spharm(V, theta, phi, L_max, areas=A, reg=reg)
            coeffs.append(c)
            np.save(d_spharm / f"coeffs_T{i:04d}.npy", c)
            E_l = spharm_degree_energy(c, lm)
            degree_energy.append({"frame": i, **{f"E_{l}": E_l[l] for l in range(L_max + 1)}})
            errs = spharm_reconstruction_errors(V, B, c, lm, areas=A)
            for l, e, er in errs:
                rec_err.append({"frame": i, "degree": l, "rms_error": e, "rel_error": er})
            row["spharm_rel_error_Lmax"] = rec_err[-1]["rel_error"]
            # generalised cross-validation over the nested truncations: GCV(L) = n·RSS_L / (n − 3(L+1)²)²
            nv = V.shape[0]
            gcv = [(nv * (e ** 2) * nv / max(nv - 3 * (l + 1) ** 2, 1) ** 2, l) for l, e, er in errs if 3 * (l + 1) ** 2 < nv]
            row["recommended_degree_gcv"] = int(min(gcv)[1]) if gcv else int(L_max)
            degs_lm = np.array([l for l, _ in lm])
            for Ld in export_degrees:
                Vs, Fs = spharm_synthesize(c, lm, L=Ld, template=template)
                write_obj(d_spharm / f"reconstruction_T{i:04d}_L{Ld:02d}.obj", Vs, Fs)
                if export_on_mesh:
                    sel = degs_lm <= Ld
                    write_obj(d_spharm / f"reconstruction_T{i:04d}_L{Ld:02d}_mesh.obj", B[:, sel] @ c[sel], F)
            # symbolic truncated series
            try:
                sym = spharm_symbolic(c, lm, L_sym=symbolic_degree)
                (d_sym / f"parametrization_T{i:04d}.tex").write_text(sym["latex"], encoding="utf-8")
                (d_sym / f"parametrization_T{i:04d}.txt").write_text(
                    "# Truncated SPHARM series (degrees <= %d); full coefficients in SPHARM/coeffs_T%04d.npy\n"
                    "# x(theta,phi) = sum_lm c_lm Y_lm(theta,phi), theta = polar angle from +z, phi = azimuth\n%s"
                    % (symbolic_degree, i, sym["text"]), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Symbolic export failed for frame {i}: {exc}")
            save_vtp(d_sph / f"sphere_T{i:04d}.vtp", U, F, {"theta": theta, "phi": phi})
        elif topo["topology"] == "disk" and planar_if_disk:
            d_pl = ensure_dir(out / "Planar")
            uv, info = arap_planar_parametrization(V, F)
            np.save(d_pl / f"uv_T{i:04d}.npy", uv)
            row.update({k: v for k, v in info.items() if k != "arap_energy_history"})
        quality.append(row)
        progress(f"frame {i + 1}/{T} {fr['name']}")
        # release the memory of this frame (basis, eigenvectors, GPU blocks cached by CuPy): every frame starts with
        # the same free memory as the first one
        mh = B = W = M = None
        from PynamicMesh.core.accel import free_memory
        free_memory()

    from PynamicMesh.core.dyn_common import add_time_columns      # known frame times: 'Time' column
    add_time_columns(pd.DataFrame(quality), target_folder, frame_col="frame").to_csv(out / "quality.csv", index=False)
    pd.DataFrame(mh_err).to_csv(d_mh / "mh_reconstruction_error.csv", index=False)
    with open(out / "auto_parameters.json", "w", encoding="utf-8") as fh:
        json.dump(auto_info, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if L_max is None:
        L_max = int(L_requested) if not isinstance(L_requested, str) and L_requested is not None else 15
    result = {"folder": str(out), "quality": quality}
    if coeffs:
        C = np.stack(coeffs)                                       # (T, (L+1)^2, 3)
        np.save(d_spharm / "coeffs_consistent.npy", C)
        add_time_columns(pd.DataFrame(degree_energy), target_folder, frame_col="frame").to_csv(
            d_spharm / "spharm_degree_energy.csv", index=False)
        pd.DataFrame(rec_err).to_csv(d_spharm / "reconstruction_error.csv", index=False)
        with open(d_spharm / "basis_info.json", "w", encoding="utf-8") as fh:
            json.dump({"L_max": L_max, "ordering": "(l, m) for l=0..L, m=-l..l", "real_basis":
                       "Y_l0 = Re Y_l^0; m>0: sqrt2 Re Y_l^m; m<0: sqrt2 Im Y_l^|m|",
                       "theta": "polar angle from +z", "phi": "azimuth atan2(y,x)", "n_frames": T,
                       "regularisation": reg, "sphere_method": sphere_method, "area_weight": area_weight,
                       "icosphere_subdiv": int(auto_info.get("sphere_subdiv", 4))}, fh, indent=2)
        result["coeffs_consistent"] = C
    if plots:
        _plots(out, quality, rec_err, degree_energy, mh_err)
    progress("plots and summary")
    # high-degree SPHARM with a fast spherical harmonic transform (torch-harmonics on the GPU / NumPy): the least-
    # squares fit above is limited to moderate degrees; shapes with long thin parts need L ~ 64-160
    if high_degree and coeffs:
        try:
            from PynamicMesh.core.spherical_spectral import high_degree_spharm
            hd = high_degree_spharm(target_folder, frames, lmax=hd_lmax, backend=sht_backend, L_ls=L_max,
                                    plots=plots, verbose=verbose)
            if hd is not None:
                result["high_degree"] = hd
                ls_err = [r["rel_error"] for r in rec_err if r["degree"] == L_max]
                if verbose and ls_err:
                    print(f"  SPHARM relative error: least squares L={L_max}: {np.median(ls_err):.4f} | "
                          f"fast SHT L={hd['lmax']}: {hd['median_rel_error_at_lmax']:.4f}")
        except Exception as exc:  # noqa: BLE001 - the high-degree stage must never break the parametrization
            print(f"[Warning] high-degree SPHARM skipped: {exc}")
        progress("high-degree SPHARM")
    if verbose:
        n_sph = sum(1 for q in quality if q["topology"] == "sphere")
        print(f"Parametrization: {T} frames ({n_sph} genus-0 → SPHARM L={L_max}); results in {out}")
    return result


def _align_by_degree1(U, V, c_prev, L, A):
    """Fallback alignment (no p2p): rotate the sphere so that the degree-1 coefficients match the previous frame."""
    theta, phi = sphere_to_angles(U)
    c, _, lm = fit_spharm(V, theta, phi, 1, areas=A)
    P = c[1:4]; Q = c_prev[1:4]                              # 3 vectors (m=-1,0,1) in R^3 — rotate the *domain*
    # Rotating the domain by R permutes/mixes degree-1 harmonics like a vector: c1 ← D(R) c1.
    R = kabsch_rotation(P.T, Q.T)                            # treat the 3 coefficient columns as points
    Ua = U @ R.T
    return Ua, float(np.linalg.norm(P.T @ R.T - Q.T))


def _plots(out, quality, rec_err, degree_energy, mh_err):
    d = Path(out) / "plots"
    q = pd.DataFrame(quality)
    if rec_err:
        df = pd.DataFrame(rec_err)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for fr, g in df.groupby("frame"):
            ax.semilogy(g["degree"], g["rel_error"], alpha=0.35, color="tab:blue")
        med = df.groupby("degree")["rel_error"].median()
        ax.semilogy(med.index, med.values, color="k", lw=2, label="median over frames")
        ax.set_xlabel("SPHARM degree L"); ax.set_ylabel("relative RMS error"); ax.legend()
        ax.set_title("SPHARM reconstruction error vs. degree"); fig.tight_layout()
        fig.savefig(d / "1_spharm_reconstruction_error.png", dpi=200); plt.close(fig)
        de = pd.DataFrame(degree_energy).set_index("frame")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(np.log10(de.values.T + 1e-16), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("frame"); ax.set_ylabel("degree l"); ax.set_title("log10 rotation-invariant energy E_l(t)")
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(d / "2_spharm_degree_energy.png", dpi=200); plt.close(fig)
    if mh_err:
        df = pd.DataFrame(mh_err)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for fr, g in df.groupby("frame"):
            ax.semilogy(g["n_modes"], g["rel_error"], alpha=0.35, color="tab:green")
        ax.set_xlabel("number of LB modes k"); ax.set_ylabel("relative RMS error")
        ax.set_title("Manifold-harmonics reconstruction error"); fig.tight_layout()
        fig.savefig(d / "3_manifold_harmonics_error.png", dpi=200); plt.close(fig)
    if "foldovers" in q.columns:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].plot(q["frame"], q["foldovers"], marker="o"); axes[0].set_title("fold-overs of the sphere map")
        axes[1].plot(q["frame"], q["area_distortion_L2"], marker="s", color="tab:orange"); axes[1].set_title("area distortion (L2 of log ratio)")
        axes[2].plot(q["frame"], q["angle_distortion_mean_deg"], marker="^", color="tab:red"); axes[2].set_title("mean angle distortion [deg]")
        for a in axes:
            a.set_xlabel("frame")
        fig.tight_layout(); fig.savefig(d / "4_sphere_map_quality.png", dpi=200); plt.close(fig)