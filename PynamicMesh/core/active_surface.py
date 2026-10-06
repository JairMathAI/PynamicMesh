"""
A virtual laboratory for the shape dynamics of *active surfaces* (cell cortex,
epithelia, active membranes) on arbitrary triangulated meshes.

Physics
-------
The model follows the covariant active-surface theory of
Salbreux & Jülicher, *Mechanics of active surfaces*, Phys. Rev. E 96, 032404 (2017),
restricted to the experimentally most relevant class: an **isotropic,
non-chiral fluid surface with broken up-down symmetry** (their Sec. III F), at
low Reynolds number, embedded in a viscous medium, and driven by a
mechano-chemical regulator field c(x, t) (e.g. active myosin density) that
sets the local chemical drive  Δμ → Δμ·c.

Constitutive relations implemented (paper Eqs. 52–55, 45–46):

    in-plane tension   t̄ᶦʲ = [γ_H + ζc + (−κC₀ + ζ'c) C_kᵏ] gᶦʲ + 2 ζ̃ c C̃ᶦʲ
    bending moment     m̄ᶦʲ = [(κ_eff + κ_g) C_kᵏ − b] gᶦʲ − κ_g,eff Cᶦʲ
    with               κ_eff   = κ + (ζ̃_c + ζ'_c) c        (active renormalisation of κ)
                       b       = κ C₀ − ζ_c c               (active torque ⇒ spontaneous curvature)

Force balance (paper Eqs. 9–10, overdamped, external medium as local drag ξ,
surface shear viscosity η as a Laplacian damping of the velocity) is solved
every step as a sparse linear system; the fourth-order bending operator is
treated semi-implicitly so that large stable time steps are possible on fine
meshes of real cells.

Normal force density (derived variationally from the effective energy
E = ∫ [ κ_eff/2 (2H)² − b·2H + s ] dA plus the non-variational active terms):

    f_n = Δ(2κ_eff H − b) + 4κ_eff H(H² − K) + 2bK − 2sH
          − 4ζ'c H² − 4ζ̃c (H² − K) + p

    s = γ_H + ζ c + κC₀²/2 + k_A (A−A₀)/A₀        (total isotropic tension)
    p = pressure (Lagrange multiplier for volume, or penalty)

Tangential force density (drives cortical flows toward high tension):

    f_t = ∇s + 2H²∇κ_eff − 2H∇b + ∇(2ζ'cH) + 2ζ̃ c ∇H

Chemistry: every regulator species is transported with the material (dilution by
area change for surface densities), diffuses (implicit), and reacts according to a
pluggable :class:`ChemistryModel`:  ∂ₜc = DΔc − c(∇·v) + R(c, …; H, s, ∇·v, σ).
The reaction terms may depend on the local mechanics (curvature H, tension s, area
strain rate ∇·v) and on lab-frame stimuli σ(x,t) (optogenetics, drugs), which closes
the mechano-chemical feedback loop. Built-in models: LinearTurnover (legacy k_turn),
MechanosensitiveTurnover, ExcitableRho (pulsatile / excitable Rho–actomyosin cortex),
TuringPolarity (mass-conserved wave-pinning polarity). Events (LaserAblation,
ParameterStep) reproduce standard perturbation experiments.

Approximations w.r.t. the full theory (documented on purpose):
  * no chiral / planar-chiral couplings (η_C, ζ_C, ζ_PC …),
  * the up-down asymmetric viscosity η̄ and the normal moment mⁿ are neglected,
  * the tangential divergence term 2ζ̃ C̃ᶦʲ∂ᵢc is dropped (needs the full shape
    operator); the 2ζ̃ c ∇H part is kept,
  * gradients of κ_g,eff give no bulk force (Gauss–Bonnet) and are ignored,
  * the medium is a local drag (no bulk Stokes flow), the surface is
    compressible with viscosity acting as a vector Laplacian on the velocity.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyvista as pv

__all__ = [
    "DiscreteGeometryEngine",
    "ActiveSurfaceConstitutiveModel",
    "SimulationConfig",
    "ActiveSurfaceSimulator",
    "VirtualLaboratory",
    "ExperimentResult",
    "AnchorSpring",
    "LocalNormalForce",
    "UniformBodyForce",
    "MechanicalState", "ChemistryModel", "LinearTurnover", "MechanosensitiveTurnover",
    "ExcitableRho", "TuringPolarity",
    "Stimulus", "GaussianPulse", "UniformStimulus", "PatternStimulus",
    "Event", "LaserAblation", "ParameterStep",
    "Protocol", "ExperimentRun", "ComputeBackend", "InterruptWatcher",
    "prepare_mesh", "to_physical_units",
    "uniform_field",
    "gaussian_cap",
    "equatorial_ring",
    "noisy_field",
    "from_function",
    "PRESETS", "CELL_PRESETS", "CELL_SHELL",
]

class ComputeBackend:
    """Selects where the sparse linear systems (velocity operator, species diffusion) are solved.

    CPU: SuperLU direct factorisation (exact, one factorisation + a few solves per step).
    GPU: CuPy — the operator and right-hand sides are moved to the device and solved with a
    Jacobi-preconditioned block conjugate gradient (all right-hand sides at once). The
    operators are symmetric positive definite, so CG converges; if it does not reach the
    tolerance the solve silently falls back to the CPU factorisation for that step.

    ``mode``: 'auto' (use the GPU when CuPy + a CUDA device are found *and* the mesh is
    large enough), 'on' (use the GPU whenever available), 'off'.  ``min_vertices``: below
    this size the direct CPU solver is faster than launching GPU kernels (the systems are
    small and sparse), so the GPU is only engaged for larger meshes in 'auto' mode.
    """

    def __init__(self, mode: str = "auto", min_vertices: int = 8000, cg_tol: float = 1e-9,
                 cg_maxiter: int = 3000):
        if mode not in ("auto", "on", "off"):
            raise ValueError("gpu mode must be 'auto', 'on' or 'off'")
        self.mode, self.min_vertices = mode, int(min_vertices)
        self.cg_tol, self.cg_maxiter = cg_tol, cg_maxiter
        self.xp = None            # array module on the device (cupy)
        self.xsparse = None       # cupyx.scipy.sparse
        self.device_name = None
        self.available = False
        self.n_fallbacks = 0
        if mode != "off":
            try:
                import cupy
                import cupyx.scipy.sparse as cusparse
                if cupy.cuda.runtime.getDeviceCount() > 0:
                    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
                    name = props.get("name", b"GPU")
                    self.device_name = name.decode() if isinstance(name, bytes) else str(name)
                    self.xp, self.xsparse, self.available = cupy, cusparse, True
            except Exception:                    # no cupy / no CUDA driver / no device
                self.available = False

 
    def use_gpu(self, n: int) -> bool:
        return self.available and (self.mode == "on" or n >= self.min_vertices)

    def describe(self) -> str:
        if self.mode == "off":
            return "GPU: disabled (gpu='off'), CPU direct solver"
        if not self.available:
            return "GPU: no CuPy/CUDA device detected, CPU direct solver"
        return (f"GPU: CuPy on '{self.device_name}' — used for the sparse solves "
                + ("always" if self.mode == "on" else f"for meshes with ≥ {self.min_vertices} vertices"))


    def factorize(self, A: sp.spmatrix, n: int):
        """Return an object with ``solve(B)`` for a sparse SPD matrix A (B: (n,) or (n,k) numpy)."""
        if self.use_gpu(n):
            return _GPUSolve(A, self)
        return _CPUSolve(A)


class _CPUSolve:
    def __init__(self, A):
        self.lu = spla.splu(sp.csc_matrix(A))

    def solve(self, B):
        return self.lu.solve(np.ascontiguousarray(B))


class _GPUSolve:
    """Jacobi-preconditioned block CG on the device (identical algorithm for any array module)."""

    def __init__(self, A, backend: ComputeBackend, xp=None, xsparse=None):
        self.backend = backend
        self.xp = xp or backend.xp
        self.xsparse = xsparse or backend.xsparse
        A = sp.csr_matrix(A)
        self.A_host = A
        self.A = self.xsparse.csr_matrix(A)
        d = A.diagonal()
        self.Minv = self.xp.asarray(1.0 / np.where(np.abs(d) > 1e-300, d, 1.0))

    def _asnumpy(self, x):
        return x.get() if hasattr(x, "get") else np.asarray(x)

    def solve(self, B):
        xp = self.xp
        B = np.asarray(B, float)
        vec = B.ndim == 1
        Bd = xp.asarray(B.reshape(-1, 1) if vec else B)
        X, ok = self._block_cg(Bd)
        if not ok:
            self.backend.n_fallbacks += 1
            out = spla.splu(sp.csc_matrix(self.A_host)).solve(np.ascontiguousarray(B))
            return out
        out = self._asnumpy(X)
        return out[:, 0] if vec else out

    def _block_cg(self, B):
        xp, A, Minv = self.xp, self.A, self.Minv
        tol, maxiter = self.backend.cg_tol, self.backend.cg_maxiter
        X = xp.zeros_like(B)
        R = B.copy()
        Z = Minv[:, None] * R
        P = Z.copy()
        rz = xp.sum(R * Z, axis=0)
        bnorm = xp.sqrt(xp.sum(B * B, axis=0))
        bnorm = xp.where(bnorm > 0, bnorm, 1.0)
        for it in range(maxiter):
            AP = A @ P
            pAp = xp.sum(P * AP, axis=0)
            alpha = xp.where(pAp > 0, rz / xp.where(pAp > 0, pAp, 1.0), 0.0)
            X = X + alpha[None, :] * P
            R = R - alpha[None, :] * AP
            res = xp.sqrt(xp.sum(R * R, axis=0)) / bnorm
            if it % 10 == 0 or it == maxiter - 1:
                if float(self._asnumpy(res.max())) < tol:
                    return X, True
            Z = Minv[:, None] * R
            rz_new = xp.sum(R * Z, axis=0)
            beta = rz_new / xp.where(rz > 0, rz, 1.0)
            P = Z + beta[None, :] * P
            rz = rz_new
        res = xp.sqrt(xp.sum(R * R, axis=0)) / bnorm
        return X, bool(float(self._asnumpy(res.max())) < tol * 100)   # accept a slightly looser solution


_BACKENDS: Dict[Tuple[str, int], ComputeBackend] = {}


def _get_backend(mode: str = "auto", min_vertices: int = 8000) -> ComputeBackend:
    key = (mode, int(min_vertices))
    if key not in _BACKENDS:
        _BACKENDS[key] = ComputeBackend(mode, min_vertices)
    return _BACKENDS[key]



#  Keyboard interruption of the running experiment (Enter = stop, go to next)  #
class InterruptWatcher:
    """Watches stdin from a daemon thread; pressing Enter in the console sets a flag that the
    time loop checks after every completed step.  The running experiment then stops cleanly
    (frames, metrics and logs are saved) and the driver continues with the next one.

    Only active when stdin is an interactive console (``isatty``); in IDE consoles, notebooks
    or piped runs it is inert unless forced (``force=True`` or env
    ``ACTIVE_SURFACE_FORCE_INTERRUPT=1``).  Set env ``ACTIVE_SURFACE_NO_INTERRUPT=1`` to
    disable it entirely (e.g. if your driver itself reads from stdin).
    """
    _instance: Optional["InterruptWatcher"] = None

    def __init__(self, stream=None, force: bool = False):
        self.stream = stream if stream is not None else sys.stdin
        self._event = threading.Event()
        self.active = False
        if os.environ.get("ACTIVE_SURFACE_NO_INTERRUPT", "") not in ("", "0"):
            return
        force = force or os.environ.get("ACTIVE_SURFACE_FORCE_INTERRUPT", "") not in ("", "0")
        try:
            interactive = force or (self.stream is not None and self.stream.isatty())
        except Exception:
            interactive = False
        if not interactive:
            return
        self._thread = threading.Thread(target=self._loop, name="active-surface-interrupt", daemon=True)
        self._thread.start()
        self.active = True

    def _loop(self):
        while True:
            try:
                line = self.stream.readline()
            except Exception:
                break
            if line == "":          # EOF: console closed
                break
            self._event.set()

    def clear(self):
        self._event.clear()

    def triggered(self) -> bool:
        return self.active and self._event.is_set()

    def trigger(self):
        """Programmatic interruption (e.g. from a callback)."""
        self._event.set()

    @classmethod
    def get(cls, force: bool = False) -> "InterruptWatcher":
        if cls._instance is None:
            cls._instance = cls(force=force)
        return cls._instance


def _faces_from_polydata(mesh: pv.PolyData) -> np.ndarray:
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    return mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def _polydata_from_arrays(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    pv_faces = np.c_[np.full(len(faces), 3, dtype=np.int64), faces].ravel()
    return pv.PolyData(np.ascontiguousarray(points, dtype=float), pv_faces)


def _repair_topology(points: np.ndarray, faces: np.ndarray, rel_tol: float = 1e-7
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove degenerate / duplicate / non-manifold triangles and unreferenced points.

    Segmentation meshes (marching cubes, MATLAB isosurface, decimation output) typically
    contain zero-area triangles, duplicated vertices and edges shared by >2 faces; any
    of these makes the cotangent operators singular.
    """
    P = np.asarray(points, float)
    F = np.asarray(faces, np.int64)
    # merge coincident vertices
    diag = np.linalg.norm(P.max(0) - P.min(0)) + 1e-300
    key = np.round((P - P.min(0)) / (diag * rel_tol)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    P = P[first]
    F = inv.ravel()[F]
    for _ in range(10):
        # faces with repeated indices
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        # zero-area faces
        a = 0.5 * np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1)
        F = F[a > 1e-9 * a.mean()]
        # duplicate faces (same vertex set)
        _, idx = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
        F = F[np.sort(idx)]
        # non-manifold edges: drop every face touching an edge shared by > 2 faces
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        uE, inv_e, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
        bad_edge = cnt > 2
        if not bad_edge.any():
            break
        bad_face = bad_edge[inv_e.ravel()].reshape(3, -1).any(axis=0)
        F = F[~bad_face]
    # drop unreferenced points
    used = np.unique(F)
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    return P[used], remap[F]


def _implicit_remesh(mesh: pv.PolyData, target_vertices: int) -> pv.PolyData:
    """Isotropic remesh without external dependencies: signed distance of the input
    sampled on a regular grid, marching cubes at level 0, quadric decimation.  Produces a
    closed, manifold triangulation independent of the input's topological defects."""
    surf = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True,
                                point_normals=True, cell_normals=True)
    area = surf.area
    h = np.sqrt(1.6 * area / (3.0 * target_vertices))      # ~1.5x target vertices before decimation
    bounds = np.array(surf.bounds).reshape(3, 2)
    size = bounds[:, 1] - bounds[:, 0]
    dims = np.ceil(size / h).astype(int) + 8
    scale = max(1.0, (dims.prod() / 4.0e6) ** (1.0 / 3.0))   # memory cap ~4e6 grid points
    h *= scale
    dims = np.ceil(size / h).astype(int) + 8
    grid = pv.ImageData(dimensions=dims + 1, spacing=(h, h, h),
                        origin=bounds[:, 0] - 4 * h)
    grid = grid.compute_implicit_distance(surf)
    iso = grid.contour([0.0], scalars="implicit_distance").triangulate().clean()
    iso = iso.connectivity("largest")
    iso = pv.PolyData(iso.points, iso.faces).triangulate().clean()
    if iso.n_points > target_vertices:
        iso = iso.decimate(1.0 - target_vertices / iso.n_points).triangulate().clean()
    return iso


def _relax_tangential(X: np.ndarray, F: np.ndarray, n_iter: int = 30, step: float = 0.5,
                      target_quality: float = 0.4) -> np.ndarray:
    """Improve triangle quality by tangential Laplacian relaxation (shape-preserving
    to first order: displacements are projected onto the tangent plane)."""
    X = X.copy()
    for _ in range(n_iter):
        geo = DiscreteGeometryEngine(X, F)
        if geo.triangle_quality().min() >= target_quality:
            break
        centroid = (geo.adjacency @ X) / geo.degree[:, None]
        u = centroid - X
        u -= np.einsum("ij,ij->i", u, geo.N)[:, None] * geo.N
        X += step * u
    return X


def _close_and_repair(m: pv.PolyData) -> pv.PolyData:
    P, F = _repair_topology(np.asarray(m.points, float), _faces_from_polydata(m))
    m = _polydata_from_arrays(P, F).connectivity("largest")
    m = pv.PolyData(m.points, m.faces).triangulate().clean()
    for _ in range(3):
        if not m.n_open_edges:
            break
        m = m.fill_holes(hole_size=1e30).triangulate().clean()
        P, F = _repair_topology(np.asarray(m.points, float), _faces_from_polydata(m))
        m = _polydata_from_arrays(P, F)
    return m


def prepare_mesh(mesh: pv.PolyData,
                 target_vertices: Optional[int] = None,
                 normalize_size: bool = True,
                 remesh: str = "auto",
                 relax_iterations: int = 60,
                 min_quality: float = 0.3,
                 verbose: bool = True) -> pv.PolyData:
    """Make a real-world surface mesh simulation-ready.

    Pipeline: topology repair (degenerate / duplicate / non-manifold faces, holes) →
    optional remesh to ``target_vertices`` → tangential relaxation of triangle quality →
    consistent outward orientation → size normalisation.

    remesh : 'auto'     use pyacvd if installed, otherwise the built-in implicit remesher
             'acvd'     force pyacvd (raises if missing)
             'implicit' force the built-in signed-distance / marching-cubes remesher
             'none'     keep the input connectivity (only repair + relax)
    normalize_size : rescale so that the equivalent-sphere radius R_eq = (3V/4π)^(1/3) = 1
             and centre the cell; factors are kept in ``field_data`` (see
             :func:`to_physical_units`).  Presets and default time steps assume this.
    """
    m = _close_and_repair(mesh.copy().triangulate().clean())
    method = "none"
    if target_vertices is not None and remesh != "none":
        if remesh in ("auto", "acvd"):
            try:
                import pyacvd  # type: ignore
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clus = pyacvd.Clustering(m)
                    if m.n_points < 3 * target_vertices:
                        clus.subdivide(2)
                    clus.cluster(target_vertices)
                    m = clus.create_mesh().triangulate().clean()
                method = "acvd"
            except ImportError:
                if remesh == "acvd":
                    raise
        if method == "none":
            m = _implicit_remesh(m, target_vertices)
            method = "implicit"
        m = _close_and_repair(m)

    m = m.compute_normals(auto_orient_normals=True, consistent_normals=True,
                          point_normals=False, cell_normals=True)
    faces = _faces_from_polydata(m)
    pts = np.asarray(m.points, dtype=float)
    if _signed_volume(pts, faces) < 0:
        faces = faces[:, ::-1]

    if relax_iterations > 0:
        pts = _relax_tangential(pts, faces, n_iter=relax_iterations, target_quality=0.45)

    centroid, scale = np.zeros(3), 1.0
    if normalize_size:
        geo = DiscreteGeometryEngine(pts, faces)
        centroid = np.average(pts, axis=0, weights=geo.A)
        scale = (3.0 * abs(geo.volume) / (4.0 * np.pi)) ** (1.0 / 3.0)
        pts = (pts - centroid) / scale

    out = _polydata_from_arrays(pts, faces)
    out.field_data["length_scale"] = np.array([scale])
    out.field_data["centroid"] = centroid
    q = DiscreteGeometryEngine(out).triangle_quality()
    if verbose:
        print(f"[prepare_mesh] {out.n_points} vertices, {out.n_cells} triangles, remesh={method}, "
              f"R_eq scale={scale:.4g}, open edges={out.n_open_edges}, "
              f"triangle quality min={q.min():.2f} (median {np.median(q):.2f})")
    if out.n_open_edges:
        warnings.warn("Mesh is still not closed; volume constraints are ill-defined. "
                      "Try prepare_mesh(..., remesh='implicit').")
    if q.min() < min_quality:
        warnings.warn(f"Min triangle quality {q.min():.2f} < {min_quality}; consider "
                      "target_vertices with remesh='implicit' or installing pyacvd.")
    return out


def to_physical_units(mesh: pv.PolyData, reference: pv.PolyData) -> pv.PolyData:
    """Map a simulated mesh back to the physical units of the input mesh."""
    scale = float(reference.field_data["length_scale"][0])
    centroid = np.asarray(reference.field_data["centroid"], float)
    out = mesh.copy()
    out.points = mesh.points * scale + centroid
    return out


def _signed_volume(X: np.ndarray, F: np.ndarray) -> float:
    p0, p1, p2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)



#  Discrete differential geometry                                             #
class DiscreteGeometryEngine:
    """Discrete differential-geometry operators on a triangle mesh.

    Conventions: outward unit normal **n**, mean curvature H = (c₁+c₂)/2 so that
    a sphere of radius R has H = 1/R > 0, K = 1/R². The paper's trace of the
    curvature tensor is C_kᵏ = 2H.

    Operators (all sparse / vectorised):
      * ``W``  cotangent stiffness matrix, ``M`` barycentric vertex areas so
        that the Laplace–Beltrami operator is Δf = M⁻¹ W f (negative semidefinite)
      * ``H`` from the Laplacian of the embedding  ΔX = −2H n
      * ``K`` from the angle defect (discrete Gauss–Bonnet)
      * ``gradient(f)`` tangential vertex gradient (face gradients, area averaged)
    """

    def __init__(self, mesh_or_points: Union[pv.PolyData, np.ndarray],
                 faces: Optional[np.ndarray] = None, edges: Optional[np.ndarray] = None):
        """``edges``: the unique sorted edge list of a previous engine with the *same* faces
        (topology-only quantity); passing it skips an O(E log E) sort every rebuild."""
        if isinstance(mesh_or_points, pv.PolyData):
            self.F = _faces_from_polydata(mesh_or_points)
            self.X = np.asarray(mesh_or_points.points, dtype=float)
        else:
            self.X = np.asarray(mesh_or_points, dtype=float)
            self.F = np.asarray(faces, dtype=np.int64)
        self._edges_in = edges
        self.n_vertices = self.X.shape[0]
        self.n_faces = self.F.shape[0]
        self._compute()

    # ------------------------------------------------------------------ #
    def _compute(self) -> None:
        X, F, n = self.X, self.F, self.n_vertices
        p0, p1, p2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
        # edge opposite to corner k, oriented counter-clockwise
        e0, e1, e2 = p2 - p1, p0 - p2, p1 - p0
        fn = np.cross(e2, -e1)                       # (p1-p0) x (p2-p0)
        dblA = np.linalg.norm(fn, axis=1)
        tiny = 1e-14 * (dblA.mean() + 1e-300)
        dblA = np.maximum(dblA, tiny)
        self.face_area = 0.5 * dblA
        self.face_normals = fn / dblA[:, None]
        self._e = (e0, e1, e2)

        # corner cotangents / angles
        d0 = np.einsum("ij,ij->i", e2, -e1)
        d1 = np.einsum("ij,ij->i", -e2, e0)
        d2 = np.einsum("ij,ij->i", -e0, e1)
        # clip cotangents of near-degenerate corners (angles < 1 deg) for robustness
        cmax = 1.0 / np.tan(np.radians(1.0))
        cot0, cot1, cot2 = (np.clip(d / dblA, -cmax, cmax) for d in (d0, d1, d2))
        ang0, ang1, ang2 = (np.arctan2(dblA, d0), np.arctan2(dblA, d1),
                            np.arctan2(dblA, d2))

        # cotangent stiffness  (Δf)_i = (1/A_i) Σ_j ½(cotα+cotβ)(f_j − f_i)
        rows = np.concatenate([F[:, 1], F[:, 2], F[:, 2], F[:, 0], F[:, 0], F[:, 1]])
        cols = np.concatenate([F[:, 2], F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0]])
        vals = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
        W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        W.sum_duplicates()
        W = W - sp.diags(np.asarray(W.sum(axis=1)).ravel())
        self.W = W.tocsr()

        # barycentric vertex areas (robust for obtuse triangles)
        A = np.bincount(F.ravel(), weights=np.repeat(self.face_area / 3.0, 3),
                        minlength=n)
        if np.any(A <= 0):
            warnings.warn("Isolated vertices detected; run prepare_mesh().")
            A = np.where(A <= 0, 1e-12 * A[A > 0].mean(), A)
        self.A = A
        self.M = sp.diags(A)
        self.Minv = sp.diags(1.0 / A)

        # area-weighted vertex normals
        N = np.zeros((n, 3))
        for k in range(3):
            np.add.at(N, F[:, k], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
        self.N = N

        # Gaussian curvature by angle defect
        angle_sum = np.bincount(F.T.ravel(), weights=np.concatenate([ang0, ang1, ang2]),
                                minlength=n)
        self.K = (2.0 * np.pi - angle_sum) / A

        # mean curvature from the Laplacian of the position: ΔX = −2 H n
        LX = (self.W @ X) / A[:, None]
        self.H = -0.5 * np.einsum("ij,ij->i", LX, N)

        # adjacency (for mesh regularisation) and edge lengths
        if self._edges_in is not None:
            self.edges = self._edges_in
        else:
            E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
            self.edges = np.unique(E, axis=0)
        adj = self.W.copy()
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj.data[:] = 1.0
        self.adjacency = adj
        self.degree = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1.0)
        el = np.concatenate([np.linalg.norm(e, axis=1) for e in self._e])
        self.min_edge = float(el.min())
        self.mean_edge = float(el.mean())
        # robust length scale for CFL (ignores a few sliver edges)
        self.cfl_edge = float(max(np.percentile(el, 5), 0.25 * self.mean_edge))

        self.area = float(self.face_area.sum())
        self.volume = float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)

    # ------------------------------------------------------------------ #
    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """Laplace–Beltrami of a vertex scalar (or vector) field."""
        if f.ndim == 1:
            return (self.W @ f) / self.A
        return (self.W @ f) / self.A[:, None]

    def gradient(self, f: np.ndarray) -> np.ndarray:
        """Tangential vertex gradient of a scalar field (n×3)."""
        F = self.F
        e0, e1, e2 = self._e
        nf = self.face_normals
        f0, f1, f2 = f[F[:, 0]], f[F[:, 1]], f[F[:, 2]]
        g = (f0[:, None] * np.cross(nf, e0) + f1[:, None] * np.cross(nf, e1)
             + f2[:, None] * np.cross(nf, e2)) / (2.0 * self.face_area)[:, None]
        G = np.zeros((self.n_vertices, 3))
        w = (g * (self.face_area / 3.0)[:, None])
        for k in range(3):
            np.add.at(G, F[:, k], w)
        G /= self.A[:, None]
        G -= np.einsum("ij,ij->i", G, self.N)[:, None] * self.N   # project to tangent plane
        return G

    def get_enclosed_volume_and_area(self) -> Tuple[float, float]:
        return self.volume, self.area

    def get_mean_curvature(self) -> np.ndarray:
        return self.H

    def get_gaussian_curvature(self) -> np.ndarray:
        return self.K

    def compute_laplace_beltrami(self) -> sp.csr_matrix:
        return (self.Minv @ self.W).tocsr()

    def local_min_max(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Min / max of f over each vertex's one-ring (including itself)."""
        E = self.edges
        lo, hi = f.copy(), f.copy()
        np.minimum.at(lo, E[:, 0], f[E[:, 1]]); np.minimum.at(lo, E[:, 1], f[E[:, 0]])
        np.maximum.at(hi, E[:, 0], f[E[:, 1]]); np.maximum.at(hi, E[:, 1], f[E[:, 0]])
        return lo, hi

    def triangle_quality(self) -> np.ndarray:
        """Per-face quality 4√3·A / Σℓ² (1 = equilateral, → 0 degenerate)."""
        e0, e1, e2 = self._e
        s = sum(np.einsum("ij,ij->i", e, e) for e in (e0, e1, e2))
        return 4.0 * np.sqrt(3.0) * self.face_area / np.maximum(s, 1e-300)


#  External (non-surface) forces                                              #
class ExternalForce:
    """Base class: return a force *density* (n×3) given the geometry."""

    def __call__(self, geo: DiscreteGeometryEngine, t: float) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class AnchorSpring(ExternalForce):
    """Focal-adhesion-like harmonic tether of all vertices within ``radius`` of
    ``point`` toward it (Hooke's law, stiffness per unit area)."""
    point: Sequence[float]
    stiffness: float = 1.0
    radius: float = 1.0
    ramp_time: float = 0.0

    def __call__(self, geo, t):
        P = np.asarray(self.point, float)
        d = P - geo.X
        mask = np.linalg.norm(d, axis=1) < self.radius
        F = np.zeros_like(geo.X)
        amp = 1.0 if self.ramp_time <= 0 else min(1.0, t / self.ramp_time)
        F[mask] = amp * self.stiffness * d[mask]
        return F


@dataclass
class LocalNormalForce(ExternalForce):
    """Gaussian patch of normal force (e.g. actin-polymerisation pressure for
    a protrusion, > 0 pushes outward; < 0 indents)."""
    point: Sequence[float]
    strength: float = 1.0
    radius: float = 1.0
    t_on: float = 0.0
    t_off: float = np.inf

    def __call__(self, geo, t):
        if not (self.t_on <= t <= self.t_off):
            return np.zeros_like(geo.X)
        P = np.asarray(self.point, float)
        d2 = np.sum((geo.X - P) ** 2, axis=1)
        mag = self.strength * np.exp(-d2 / (2.0 * self.radius ** 2))
        return mag[:, None] * geo.N


@dataclass
class UniformBodyForce(ExternalForce):
    """Uniform force density (e.g. gravity / flow shear)."""
    vector: Sequence[float] = (0.0, 0.0, 0.0)

    def __call__(self, geo, t):
        return np.tile(np.asarray(self.vector, float), (geo.n_vertices, 1))


#  Regulator chemistry: reaction models, stimuli and events                    #
@dataclass
class MechanicalState:
    """Snapshot of the mechanics handed to :meth:`ChemistryModel.rates` every step.

    All arrays are per vertex on the *updated* geometry.
      t, dt        : current time and step
      geo          : DiscreteGeometryEngine (areas, normals, curvatures, operators)
      H, K         : mean / Gaussian curvature (sphere of radius R: H = 1/R)
      tension      : total isotropic tension s = γ_H + ζφ(c) + ... used in the last force evaluation
      kappa_eff    : effective bending rigidity field
      strain_rate  : material area strain rate  (1/A) dA/dt = ∇·v  (< 0: local compression)
      normal_velocity, speed : components of the last velocity field
      c_eq         : material-attached target pattern of the mechanical regulator
      stimulus     : external signal σ(x,t) ≥ 0 summed over all active Stimulus objects
                     (optogenetic light, morphogen, ...)
    """
    t: float
    dt: float
    geo: "DiscreteGeometryEngine"
    H: np.ndarray
    K: np.ndarray
    tension: np.ndarray
    kappa_eff: np.ndarray
    strain_rate: np.ndarray
    normal_velocity: np.ndarray
    speed: np.ndarray
    c_eq: np.ndarray
    stimulus: np.ndarray


class ChemistryModel:
    """Base class for the reaction kinetics of the regulator field(s).

    The simulator handles, for every species, transport on the moving surface
    (Lagrangian advection + dilution by area change for surface densities, ALE
    correction), implicit surface diffusion, and then calls :meth:`rates` for the
    reaction terms, which are integrated with Heun's method in ``substeps`` sub-steps.
    Subclasses define::

        species   : names of the fields; the FIRST one is the mechanical drive ``c``
                    that multiplies all active coefficients of the constitutive model.
        diffusion : {name: D}. A missing entry for ``c`` falls back to model.D_chem,
                    other missing species do not diffuse.
        diluted   : names of species that are surface densities (diluted when the
                    surface stretches). Default: all species. Volume-borne or
                    cytoplasm-buffered quantities can be excluded.
        rates(fields, mech) -> {name: d field / dt}

    ``noise`` adds multiplicative Langevin noise  σ √(f) √dt ξ  to each species
    (stochastic binding/unbinding), ``seed`` makes it reproducible.
    """
    species: Tuple[str, ...] = ("c",)
    diffusion: Dict[str, float] = {}
    diluted: Optional[Sequence[str]] = None
    noise: float = 0.0
    substeps: int = 1
    seed: Optional[int] = 0

    def __post_init__(self):
        if not self.species or self.species[0] != "c":
            raise ValueError("The first species must be named 'c' (the mechanical drive)")
        if self.diluted is None:
            self.diluted = tuple(self.species)
        self._rng = np.random.default_rng(self.seed)

    # API ---------------------------------------------------------------
    def initial_fields(self, geo: "DiscreteGeometryEngine", c0: np.ndarray) -> Dict[str, np.ndarray]:
        """Default initial values of the *extra* species (all zero)."""
        return {s: np.zeros(geo.n_vertices) for s in self.species[1:]}

    def rates(self, fields: Dict[str, np.ndarray], mech: MechanicalState) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        d = {k: v for k, v in asdict(self).items()} if hasattr(self, "__dataclass_fields__") else {}
        d["type"] = type(self).__name__
        d["species"] = list(self.species)
        return d

    # used by the simulator --------------------------------------------
    def D_of(self, name: str, model: "ActiveSurfaceConstitutiveModel") -> float:
        if name in self.diffusion:
            return float(self.diffusion[name])
        return float(model.D_chem) if name == "c" else 0.0

    def integrate(self, fields: Dict[str, np.ndarray], mech: MechanicalState) -> Dict[str, np.ndarray]:
        """Heun (RK2) integration of the reaction terms over mech.dt, positivity preserving."""
        n = max(1, int(self.substeps))
        h = mech.dt / n
        f = {k: v.copy() for k, v in fields.items()}
        for _ in range(n):
            r1 = self.rates(f, mech)
            f1 = {k: np.maximum(f[k] + h * r1[k], 0.0) if k in r1 else f[k] for k in f}
            r2 = self.rates(f1, mech)
            for k in f:
                if k in r1:
                    f[k] = np.maximum(f[k] + 0.5 * h * (r1[k] + r2[k]), 0.0)
        if self.noise > 0:
            for k in f:
                xi = self._rng.standard_normal(f[k].shape)
                f[k] = np.maximum(f[k] + self.noise * np.sqrt(np.maximum(f[k], 0.0) * mech.dt) * xi, 0.0)
        return f


@dataclass
class LinearTurnover(ChemistryModel):
    """Default kinetics (identical to the legacy ``k_turn`` behaviour):

        ∂ₜc = k_turn (c_eq + σ − c)

    c_eq is the material-attached pattern set by :meth:`set_initial_chemical_field`,
    σ the external stimulus (optogenetic activation adds to the target level).
    """
    k_turn: float = 0.0
    stimulus_gain: float = 1.0
    diffusion: Dict[str, float] = field(default_factory=dict)
    noise: float = 0.0
    substeps: int = 1
    seed: Optional[int] = 0
    species: Tuple[str, ...] = ("c",)
    diluted: Optional[Sequence[str]] = None

    def rates(self, f, mech):
        if self.k_turn <= 0:
            return {}
        return {"c": self.k_turn * (mech.c_eq + self.stimulus_gain * mech.stimulus - f["c"])}


@dataclass
class MechanosensitiveTurnover(ChemistryModel):
    """Turnover toward a target that is set by the local mechanics (all couplings default to 0):

        ∂ₜc = k_turn [ c_eq (1 + α_H (H − H_ref)) (1 + α_s (s/s_ref − 1)) + σ − c ]
              + k_turn α_comp c_eq max(−∇·v, 0)

    alpha_curvature   : curvature sensing (BAR-domain-like recruitment to high H; with a
                        negative zeta_c this closes a positive feedback loop → folds/tubes).
                        H_ref defaults to the area-weighted mean curvature at t=0.
    alpha_tension     : tension-dependent binding (catch-bond-like: > 0 recruits myosin where
                        the cortex is under tension; s_ref = tension at t=0).
    alpha_compression : recruitment by compressive strain rate (Munjal et al. 2015: myosin
                        accumulates where the surface is compressed → pulsed contractions).
    """
    k_turn: float = 1.0
    alpha_curvature: float = 0.0
    alpha_tension: float = 0.0
    alpha_compression: float = 0.0
    H_ref: Optional[float] = None
    s_ref: Optional[float] = None
    stimulus_gain: float = 1.0
    diffusion: Dict[str, float] = field(default_factory=dict)
    noise: float = 0.0
    substeps: int = 1
    seed: Optional[int] = 0
    species: Tuple[str, ...] = ("c",)
    diluted: Optional[Sequence[str]] = None

    def rates(self, f, mech):
        if self.H_ref is None:
            self.H_ref = float(np.average(mech.H, weights=mech.geo.A))
        if self.s_ref is None:
            self.s_ref = float(max(np.average(mech.tension, weights=mech.geo.A), 1e-9))
        target = mech.c_eq * (1.0 + self.alpha_curvature * (mech.H - self.H_ref))
        target = target * (1.0 + self.alpha_tension * (mech.tension / self.s_ref - 1.0))
        target = np.maximum(target, 0.0) + self.stimulus_gain * mech.stimulus
        dc = self.k_turn * (target - f["c"])
        if self.alpha_compression != 0.0:
            dc = dc + self.k_turn * self.alpha_compression * mech.c_eq * np.maximum(-mech.strain_rate, 0.0)
        return {"c": dc}


@dataclass
class ExcitableRho(ChemistryModel):
    """Two-species excitable / oscillatory Rho–actomyosin cortex (Bement et al. 2015,
    Michaux et al. 2018 type activator–inhibitor kinetics):

        ∂ₜρ = k_b (1 + σ) + k_a ρⁿ/(Kⁿ + ρⁿ) − k_d ρ − k_i c ρ   + α_comp k_b max(−∇·v,0)
        ∂ₜc = k_r ρ − k_c c

    ρ = active RhoA (fast activator with GEF-mediated autocatalysis), c = actomyosin
    (slow inhibitor via F-actin-recruited GAPs) which is also the mechanical drive.
    Two calibrated regimes (time unit ξR²/γ):

        oscillatory : k_b=2,  k_a=80, K=1, k_d=4, k_i=40, k_r=10, k_c=8   (period ≈ 0.53, c ∈ [0.4, 1.2])
        excitable   : k_b=1,  k_a=60, K=1, k_d=8, k_i=40, k_r=10, k_c=8   (rest c ≈ 0.19, a pulse
                      σ ≈ 6 for 0.1 time units fires a single wave to c ≈ 0.8)

    With diffusion of ρ faster than c (D_rho > D_c) the oscillations become traveling
    waves and target patterns; ``noise`` nucleates them at random sites. alpha_compression
    couples Rho activation to compression (mechanochemical positive feedback).
    """
    k_b: float = 2.0
    k_a: float = 80.0
    K: float = 1.0
    hill_n: float = 2.0
    k_d: float = 4.0
    k_i: float = 40.0
    k_r: float = 10.0
    k_c: float = 8.0
    alpha_compression: float = 0.0
    rho0: float = 0.05
    diffusion: Dict[str, float] = field(default_factory=lambda: {"c": 0.002, "rho": 0.01})
    noise: float = 0.0
    substeps: int = 4
    seed: Optional[int] = 0
    species: Tuple[str, ...] = ("c", "rho")
    diluted: Optional[Sequence[str]] = None

    @classmethod
    def excitable(cls, **kw) -> "ExcitableRho":
        p = dict(k_b=1.0, k_a=60.0, K=1.0, k_d=8.0, k_i=40.0, k_r=10.0, k_c=8.0)
        p.update(kw)
        return cls(**p)

    def initial_fields(self, geo, c0):
        return {"rho": np.full(geo.n_vertices, self.rho0)}

    def rates(self, f, mech):
        rho, c = f["rho"], f["c"]
        rn = rho ** self.hill_n
        act = self.k_b * (1.0 + mech.stimulus)
        if self.alpha_compression != 0.0:
            act = act + self.alpha_compression * self.k_b * np.maximum(-mech.strain_rate, 0.0)
        d_rho = act + self.k_a * rn / (self.K ** self.hill_n + rn) - self.k_d * rho - self.k_i * c * rho
        d_c = self.k_r * rho - self.k_c * c
        return {"rho": d_rho, "c": d_c}


@dataclass
class TuringPolarity(ChemistryModel):
    """Mass-conserved activator–substrate polarity module (wave-pinning / Cdc42–PAR type,
    Mori, Jilkine & Edelstein-Keshet 2008), producing a single stable cap from noise:

        ∂ₜc = (k_0 + k_a c²/(K² + c²)) u − k_d c
        ∂ₜu = −(k_0 + k_a c²/(K² + c²)) u + k_d c

    c = active form (slow diffusion, mechanical drive), u = fast-diffusing inactive form.
    Both are surface densities, so the total ∫(c+u)dA is exactly conserved (also under
    cortical flow); the cap size is set by ``total`` (mean of c+u per unit area).
    Cortical flow toward the contractile cap advects c and reinforces the polarity
    (mechanochemical polarisation); with zeta<0 the cap can instead protrude.
    """
    k_0: float = 0.5
    k_a: float = 8.0
    K: float = 1.0
    k_d: float = 4.0
    total: float = 1.5
    diffusion: Dict[str, float] = field(default_factory=lambda: {"c": 0.002, "u": 1.0})
    noise: float = 0.0
    substeps: int = 2
    seed: Optional[int] = 0
    species: Tuple[str, ...] = ("c", "u")
    diluted: Optional[Sequence[str]] = None

    def initial_fields(self, geo, c0):
        return {"u": np.maximum(self.total - c0, 0.0)}

    def rates(self, f, mech):
        c, u = f["c"], f["u"]
        kon = self.k_0 + self.k_a * c ** 2 / (self.K ** 2 + c ** 2)
        d_c = kon * u - self.k_d * c
        return {"c": d_c, "u": -d_c}


# ---- external signals (lab frame, time dependent) --------------------------- #
class Stimulus:
    """Lab-frame signal σ(x, t) ≥ 0 (optogenetic light pattern, morphogen, drug wash-in).
    Return a per-vertex array; the simulator sums all stimuli into ``mech.stimulus``."""

    def __call__(self, geo: "DiscreteGeometryEngine", t: float) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class GaussianPulse(Stimulus):
    """Optogenetic-like activation: Gaussian spot of ``amplitude`` (width ``radius``) on
    between t_on and t_off, optionally moving with ``velocity``, repeating every ``period``
    (duty = on-time fraction) for pulsed illumination protocols."""
    center: Sequence[float]
    radius: float = 0.3
    amplitude: float = 1.0
    t_on: float = 0.0
    t_off: float = np.inf
    velocity: Sequence[float] = (0.0, 0.0, 0.0)
    period: float = np.inf
    duty: float = 1.0

    def __call__(self, geo, t):
        on = self.t_on <= t <= self.t_off
        if on and np.isfinite(self.period):
            on = ((t - self.t_on) % self.period) < self.duty * self.period
        if not on:
            return np.zeros(geo.n_vertices)
        P = np.asarray(self.center, float) + np.asarray(self.velocity, float) * (t - self.t_on)
        d2 = np.sum((geo.X - P) ** 2, axis=1)
        return self.amplitude * np.exp(-d2 / (2.0 * self.radius ** 2))


@dataclass
class UniformStimulus(Stimulus):
    """Global drug / temperature step: σ = amplitude for t_on ≤ t ≤ t_off (e.g. calyculin
    wash-in raises the target myosin level everywhere)."""
    amplitude: float = 1.0
    t_on: float = 0.0
    t_off: float = np.inf

    def __call__(self, geo, t):
        v = self.amplitude if self.t_on <= t <= self.t_off else 0.0
        return np.full(geo.n_vertices, v)


@dataclass
class PatternStimulus(Stimulus):
    """Arbitrary σ = fn(points, t) (n×3 coordinates → n values)."""
    fn: Callable[[np.ndarray, float], np.ndarray]

    def __call__(self, geo, t):
        return np.maximum(np.asarray(self.fn(geo.X, t), float), 0.0)


# ---- discrete events ----------------------------------------------------------- #
class Event:
    """One-shot perturbation of the simulator state at ``t_event`` (``apply``), with an
    optional per-step ``update`` afterwards (recovery)."""
    t_event: float = 0.0

    def __init__(self):
        self.done = False

    def apply(self, sim: "ActiveSurfaceSimulator"):  # pragma: no cover
        raise NotImplementedError

    def update(self, sim: "ActiveSurfaceSimulator"):
        pass

    def on_refine(self, sim: "ActiveSurfaceSimulator", interp: Callable[[np.ndarray], np.ndarray]):
        """Called after adaptive refinement; ``interp`` extends any per-vertex array."""
        pass


class LaserAblation(Event):
    """Laser ablation of the cortex in a spot: at ``t_cut`` all regulator species and the
    target pattern c_eq are set to zero within ``radius`` of ``center`` (the active tension
    drops, the surrounding cortex recoils — the standard cortical tension assay). The
    target pattern recovers as 1 − exp(−(t − t_cut)/recovery_time) (cortex re-assembly).
    ``recoil_speed(sim)`` gives the mean outward speed of the wound margin for read-out;
    ``mask`` holds the ablated vertices."""

    def __init__(self, center, radius=0.2, t_cut=0.5, recovery_time=1.0):
        super().__init__()
        self.center = np.asarray(center, float)
        self.radius = float(radius)
        self.t_event = float(t_cut)
        self.recovery_time = float(recovery_time)
        self.mask = None
        self._c_eq_saved = None

    def apply(self, sim):
        d = np.linalg.norm(sim.geo.X - self.center, axis=1)
        self.mask = d < self.radius
        self._margin = (d >= self.radius) & (d < 1.5 * self.radius)
        self._c_eq_saved = sim.c_eq.copy()
        sim.c[self.mask] = 0.0
        for k in sim.fields:
            sim.fields[k][self.mask] = 0.0
        sim.c_eq[self.mask] = 0.0

    def update(self, sim):
        if self.mask is None or self.recovery_time <= 0:
            return
        frac = 1.0 - np.exp(-(sim.t - self.t_event) / self.recovery_time)
        sim.c_eq[self.mask] = frac * self._c_eq_saved[self.mask]

    def on_refine(self, sim, interp):
        if self.mask is not None:
            self.mask, self._margin = interp(self.mask), interp(self._margin)
            self._c_eq_saved = interp(self._c_eq_saved)

    def recoil_speed(self, sim) -> float:
        """Mean velocity of the wound margin away from the cut centre (> 0: recoil)."""
        if self.mask is None or not np.any(self._margin):
            return 0.0
        r = sim.geo.X[self._margin] - self.center
        r /= np.maximum(np.linalg.norm(r, axis=1), 1e-300)[:, None]
        return float(np.mean(np.einsum("ij,ij->i", sim.velocity[self._margin], r)))


class ParameterStep(Event):
    """Change constitutive parameters at ``t_event`` (drug: blebbistatin ⇒ zeta → 0,
    cytochalasin ⇒ E_shear → 0, ...). ``changes`` = {param: new_value}."""

    def __init__(self, t_event: float, **changes):
        super().__init__()
        self.t_event = float(t_event)
        self.changes = changes

    def apply(self, sim):
        for k, v in self.changes.items():
            if k not in sim.model.__dataclass_fields__:
                raise KeyError(f"Unknown model parameter '{k}'")
            setattr(sim.model, k, v)
        sim.elastic = (sim.model.E_shear > 0) or (sim.model.E_area > 0) or sim.model.curvature_memory



#  Constitutive model                                                          #
@dataclass
class ActiveSurfaceConstitutiveModel:
    """Material parameters of a non-chiral, up-down asymmetric active fluid surface.

    Every active coefficient multiplies the regulator field c(x,t) (Δμ ≡ 1 is
    absorbed into the coefficient), so setting c=0 recovers a passive Helfrich
    membrane and a uniform c=1 gives the spatially homogeneous active surface of
    the paper's Sec. III F.

    Passive (Helfrich) parameters
    -----------------------------
    kappa_b : bending rigidity κ
    kappa_g : Gaussian bending modulus κ_g (energy bookkeeping only for closed surfaces)
    gamma_0 : passive surface tension γ_H (> 0)
    C0      : spontaneous curvature of the trace 2H (sphere of radius R has 2H = 2/R)

    Active couplings (paper Eqs. 45–46, 52–55)
    -----------------------------------------
    zeta          : active isotropic tension ζ (>0 contractile, e.g. myosin).
                    Buckling instability when ζ c < −γ_H (Eq. 57).
    zeta_c        : active torque ζ_c; shifts the spontaneous curvature
                    (κ̄C̄₀ = κC₀ − ζ_c c). >0 favours bending *inward* (negative
                    curvature) where c is high; gradients of ζ_c c fold the surface.
    zeta_prime    : tension–curvature coupling ζ' (t̄ᶦʲ ∋ ζ' c C_kᵏ gᶦʲ). Non-variational.
    zeta_tilde    : anisotropic active tension ζ̃ (t̄ᶦʲ ∋ 2ζ̃ c C̃ᶦʲ): acts only where
                    the two principal curvatures differ (tubes, saddles, necks).
    zeta_c_tilde,
    zeta_c_prime  : renormalise the bending rigidity κ_eff = κ + (ζ̃_c + ζ'_c) c.
                    Negative values soften the surface (Eq. 58 instability).

    Dissipation
    -----------
    eta_drag : friction ξ with the surrounding medium (per unit area), sets the time scale.
    eta_s    : surface shear viscosity η, implemented as −η Δv (implicit).

    Constraints
    -----------
    volume_constraint : 'lagrange' (exact, osmotic pressure computed each step),
                        'penalty' (p = −kV (V−V₀)/V₀) or 'none'.
    kV, kA            : penalty stiffness of volume and of area (area penalty acts
                        as an extra isotropic tension; use kA=0 for a cortex whose
                        area is not conserved, kA>0 for lipid-membrane-like surfaces).

    Regulator chemistry
    -------------------
    D_chem       : surface diffusion coefficient of c.
    k_turn       : turnover rate toward the equilibrium pattern c_eq (set from the
                   initial field unless overridden in the simulator).
    c_saturation : the mechanical drive is φ(c) = c / (1 + c/c_sat) (binding-site
                   saturation of the motor/regulator); ∞ = linear. A finite value
                   bounds the contractile instability at a physical density.

    Reference shape (elastic / viscoelastic shell, paper Sec. IV)
    ------------------------------------------------------------
    A purely fluid surface (defaults) relaxes any initial shape to a sphere on the
    time scale ξ/(κq⁴), i.e. instantly for the fine features of a real cell.  To
    simulate deformations *of a given cell shape*, give the surface a reference
    configuration that it remembers:
    E_shear          : 2D stretching/shear modulus (edge springs, rest lengths = input mesh)
    E_area           : 2D local area modulus (per-triangle rest area = input mesh)
    curvature_memory : use the input curvature as spontaneous curvature field C₀(x) = 2H₀(x)
    tau_remodel      : cortex turnover time; the reference configuration relaxes toward the
                       current shape with this time constant (∞ = permanently elastic,
                       small = fluid). Deformations that last longer than τ become permanent.
    """
    # passive
    kappa_b: float = 1.0
    kappa_g: float = 0.0
    gamma_0: float = 0.5
    C0: float = 0.0
    # active
    zeta: float = 0.0
    zeta_c: float = 0.0
    zeta_prime: float = 0.0
    zeta_tilde: float = 0.0
    zeta_c_tilde: float = 0.0
    zeta_c_prime: float = 0.0
    # dissipation
    eta_drag: float = 1.0
    eta_s: float = 0.0
    # constraints
    volume_constraint: str = "lagrange"
    kV: float = 10.0
    kA: float = 0.0
    # chemistry
    D_chem: float = 0.0
    k_turn: float = 0.0
    c_saturation: float = np.inf
    # viscoelastic reference shape (paper Sec. IV, active elastic thin shell)
    E_shear: float = 0.0
    E_area: float = 0.0
    curvature_memory: bool = False
    tau_remodel: float = np.inf

    def __post_init__(self):
        if self.volume_constraint not in ("lagrange", "penalty", "none"):
            raise ValueError("volume_constraint must be 'lagrange', 'penalty' or 'none'")
        if self.eta_drag <= 0:
            raise ValueError("eta_drag must be > 0 (sets the overdamped time scale)")
        if self.kappa_b < 0 or self.eta_s < 0 or self.D_chem < 0:
            raise ValueError("kappa_b, eta_s and D_chem must be non-negative")

    # convenience -------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict) -> "ActiveSurfaceConstitutiveModel":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        unknown = set(d) - set(known)
        if unknown:
            warnings.warn(f"Ignoring unknown model parameters: {sorted(unknown)}")
        return cls(**known)

    def to_dict(self) -> Dict:
        return asdict(self)

    def drive(self, c: np.ndarray) -> np.ndarray:
        """Mechanical drive φ(c) that multiplies all active coefficients."""
        if not np.isfinite(self.c_saturation):
            return c
        return c / (1.0 + c / self.c_saturation)

    def effective_coefficients(self, c: np.ndarray, area: float, A0: float, C0=None):
        """Return (κ_eff, b, s) vertex fields for a regulator drive φ(c); C0 may be a field."""
        C0 = self.C0 if C0 is None else C0
        kappa_eff = self.kappa_b + (self.zeta_c_tilde + self.zeta_c_prime) * c
        kappa_eff = np.maximum(kappa_eff, 1e-3 * self.kappa_b)  # keep the surface stable
        b = self.kappa_b * C0 - self.zeta_c * c
        s = (self.gamma_0 + self.zeta * c + 0.5 * self.kappa_b * C0 ** 2
             + self.kA * (area - A0) / A0)
        return kappa_eff, np.broadcast_to(b, c.shape).copy(), np.broadcast_to(s, c.shape).copy()



#  Simulator                                                                   #
@dataclass
class SimulationConfig:
    """Numerical settings (independent of the physics)."""
    cfl: float = 0.2                 # max vertex displacement per step, in units of min edge length
    stabilization: float = 1.0       # weight of the semi-implicit bending operator (≥ 1 recommended)
    regularize_mesh: float = 0.1     # tangential Laplacian mesh relaxation per step (0 disables; ALE)
    quality_threshold: float = 0.3   # below this min triangle quality, extra relaxation passes are applied
    max_regularize_passes: int = 10
    tangential_flow: bool = True     # False: vertices move only along the normal (no cortical flow)
    max_force: Optional[float] = None  # optional clipping of force density magnitude
    volume_relaxation: float = 1.0   # fraction of the volume error removed per step ('lagrange' mode)
    min_dt: float = 1e-8
    store_fields: bool = True
    curvature_memory_smoothing: float = 1.0  # smoothing length (mean edges) of the reference curvature
    # compute backend for the sparse solves (see ComputeBackend)
    gpu: str = "auto"                # 'auto' | 'on' | 'off'
    gpu_min_vertices: int = 8000     # 'auto': use the GPU only for meshes at least this large
    # adaptive refinement of over-stretched regions (edge bisection, fields & rest state carried along)
    refine_ratio: float = 0.0        # split edges longer than refine_ratio × initial mean edge (0 disables; 1.6 typical)
    refine_every: int = 1            # check every N steps
    max_refine_passes: int = 3       # bisection passes per check (each face is split at most once per pass)
    max_vertex_factor: float = 3.0   # stop refining beyond this multiple of the initial vertex count


class ActiveSurfaceSimulator:
    """Time integrator for :class:`ActiveSurfaceConstitutiveModel` on a mesh.

    Usage
    -----
    >>> sim = ActiveSurfaceSimulator(mesh, model)
    >>> sim.set_initial_chemical_field(c0)
    >>> dt, metrics = sim.step_forward(dt_max=0.01)
    >>> traj, hist = sim.run(total_time=1.0, save_every=5)
    """

    def __init__(self, mesh: pv.PolyData, model: ActiveSurfaceConstitutiveModel,
                 config: Optional[SimulationConfig] = None,
                 external_forces: Optional[Sequence[ExternalForce]] = None,
                 prepare: bool = False,
                 chemistry: Optional[ChemistryModel] = None,
                 stimuli: Optional[Sequence[Stimulus]] = None,
                 events: Optional[Sequence[Event]] = None):
        if prepare:
            mesh = prepare_mesh(mesh, verbose=False)
        self.model = model
        self.config = config or SimulationConfig()
        self.external_forces: List[ExternalForce] = list(external_forces or [])
        # reaction kinetics of the regulator(s); default reproduces the legacy k_turn relaxation
        self.chemistry: ChemistryModel = chemistry if chemistry is not None else LinearTurnover(k_turn=model.k_turn)
        if chemistry is not None and model.k_turn > 0 and not isinstance(chemistry, LinearTurnover):
            warnings.warn("model.k_turn is ignored because an explicit ChemistryModel was given")
        self.stimuli: List[Stimulus] = list(stimuli or [])
        self.events: List[Event] = sorted(events or [], key=lambda e: e.t_event)
        self.F = _faces_from_polydata(mesh)
        self.X = np.array(mesh.points, dtype=float)
        self.geo = DiscreteGeometryEngine(self.X, self.F)
        if self.geo.volume < 0:
            self.F = self.F[:, ::-1]
            self.geo = DiscreteGeometryEngine(self.X, self.F)
        self.V0, self.A0 = self.geo.volume, self.geo.area
        self._init_reference()
        q = float(self.geo.triangle_quality().min())
        if q < 0.05:
            raise ValueError(
                f"Min triangle quality {q:.2e}: the mesh has (near-)degenerate triangles and "
                "cannot be simulated. Run prepare_mesh(mesh, target_vertices=...) first.")
        if q < 0.15:
            warnings.warn(f"Min triangle quality {q:.3f} is poor; results may be noisy. "
                          "Consider prepare_mesh(mesh, target_vertices=...).")
        R_eq = (3.0 * abs(self.V0) / (4.0 * np.pi)) ** (1.0 / 3.0)
        if not (0.2 < R_eq < 5.0):
            warnings.warn(f"Equivalent radius is {R_eq:.3g}; presets and default dt assume "
                          "R_eq ~ 1. Use prepare_mesh(normalize_size=True).")
        self.t = 0.0
        self.step_count = 0
        self.c = np.zeros(self.geo.n_vertices)
        self.c_eq = np.zeros(self.geo.n_vertices)
        self.fields: Dict[str, np.ndarray] = self.chemistry.initial_fields(self.geo, self.c)
        self.strain_rate = np.zeros(self.geo.n_vertices)
        self.stimulus = np.zeros(self.geo.n_vertices)
        self.velocity = np.zeros_like(self.X)
        self.pressure = 0.0
        self.X_centroid0 = np.average(self.X, axis=0, weights=self.geo.A)
        self.n0, self.h0 = self.geo.n_vertices, self.geo.mean_edge   # initial resolution (refinement)
        self.n_refined = 0
        self.backend = _get_backend(self.config.gpu, self.config.gpu_min_vertices)
        self.interrupted = False
        self.regularize_passes = 0
        self.terminated_early = False
        self.last_forces: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def _init_reference(self):
        """Store the reference (rest) configuration for the elastic terms."""
        geo, m = self.geo, self.model
        self.elastic = (m.E_shear > 0) or (m.E_area > 0) or m.curvature_memory
        self.ref_edges = geo.edges
        d = geo.X[self.ref_edges[:, 1]] - geo.X[self.ref_edges[:, 0]]
        self.l0 = np.linalg.norm(d, axis=1)
        self.Af0 = geo.face_area.copy()
        if m.curvature_memory:
            # reference spontaneous curvature = initial 2H, lightly smoothed (2 edge lengths)
            ell2 = (self.config.curvature_memory_smoothing * geo.mean_edge) ** 2
            if ell2 > 0:
                C0 = spla.spsolve(sp.csc_matrix(geo.M - ell2 * geo.W), geo.A * (2.0 * geo.H))
            else:
                C0 = 2.0 * geo.H
            self.C0_field = np.asarray(C0, float)
        else:
            self.C0_field = None

    def _elastic_forces(self, geo: DiscreteGeometryEngine) -> np.ndarray:
        """Force density from in-plane elasticity w.r.t. the reference configuration."""
        m = self.model
        Fp = np.zeros_like(geo.X)                               # point forces
        if m.E_shear > 0:
            E = self.ref_edges
            d = geo.X[E[:, 1]] - geo.X[E[:, 0]]
            l = np.maximum(np.linalg.norm(d, axis=1), 1e-300)
            f = (m.E_shear * (l - self.l0))[:, None] * (d / l[:, None])   # k_e = E·l0
            np.add.at(Fp, E[:, 0], f)
            np.add.at(Fp, E[:, 1], -f)
        if m.E_area > 0:
            F = geo.F
            coef = -m.E_area * (geo.face_area - self.Af0) / self.Af0
            nf = geo.face_normals
            for k, e in enumerate(geo._e):                      # ∇_{p_k} A_f = ½ n × e_k
                np.add.at(Fp, F[:, k], (0.5 * coef)[:, None] * np.cross(nf, e))
        return Fp / geo.A[:, None]

    def _remodel_reference(self, geo: DiscreteGeometryEngine, dt: float):
        m = self.model
        if not self.elastic or not np.isfinite(m.tau_remodel):
            return
        a = min(1.0, dt / max(m.tau_remodel, 1e-300))
        d = geo.X[self.ref_edges[:, 1]] - geo.X[self.ref_edges[:, 0]]
        self.l0 += a * (np.linalg.norm(d, axis=1) - self.l0)
        self.Af0 += a * (geo.face_area - self.Af0)
        if self.C0_field is not None:
            self.C0_field += a * (2.0 * geo.H - self.C0_field)

    def set_initial_chemical_field(self, c: np.ndarray, c_eq: Optional[np.ndarray] = None):
        c = np.asarray(c, dtype=float)
        if c.shape != (self.geo.n_vertices,):
            raise ValueError(f"c must have shape ({self.geo.n_vertices},), got {c.shape}")
        self.c = np.maximum(c, 0.0).copy()
        self.c_eq = self.c.copy() if c_eq is None else np.asarray(c_eq, float).copy()
        self.fields = self.chemistry.initial_fields(self.geo, self.c)

    def set_initial_fields(self, **fields: np.ndarray):
        """Override initial values of extra species, e.g. ``set_initial_fields(rho=...)``."""
        for k, v in fields.items():
            if k not in self.chemistry.species[1:]:
                raise KeyError(f"'{k}' is not a species of {type(self.chemistry).__name__}: "
                               f"{self.chemistry.species[1:]}")
            v = np.asarray(v, float)
            if v.shape != (self.geo.n_vertices,):
                raise ValueError(f"{k} must have shape ({self.geo.n_vertices},), got {v.shape}")
            self.fields[k] = np.maximum(v, 0.0).copy()

    def add_external_force(self, f: ExternalForce):
        self.external_forces.append(f)

    def add_stimulus(self, s: Stimulus):
        self.stimuli.append(s)

    def add_event(self, e: Event):
        self.events.append(e)
        self.events.sort(key=lambda ev: ev.t_event)

    # ------------------------------------------------------------------ #
    def default_dt(self) -> float:
        """Explicit stability estimate for the tension-driven part (ξh²/γ)."""
        m = self.model
        gam = m.gamma_0 + abs(m.zeta) * (self.c.max() if self.c.size else 0.0) + m.kA
        h = self.geo.mean_edge
        dt_explicit = 0.25 * m.eta_drag * h * h / max(gam, 1e-9)
        return dt_explicit if self.config.stabilization <= 0 else max(dt_explicit, 0.01)

    def compute_forces(self, geo: Optional[DiscreteGeometryEngine] = None
                       ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Surface force density (n×3) without pressure, plus diagnostics."""
        geo = geo or self.geo
        m = self.model
        c = m.drive(self.c)                 # saturating mechanical drive φ(c)
        H, K, N = geo.H, geo.K, geo.N
        kappa_eff, b, s = m.effective_coefficients(c, geo.area, self.A0, C0=self.C0_field)

        # normal component (variational + non-variational active terms)
        f_n = (geo.laplacian(2.0 * kappa_eff * H - b)
               + 4.0 * kappa_eff * H * (H ** 2 - K)
               + 2.0 * b * K
               - 2.0 * s * H
               - 4.0 * m.zeta_prime * c * H ** 2
               - 4.0 * m.zeta_tilde * c * (H ** 2 - K))

        # tangential component (gradients of material coefficients / curvature couplings)
        f_t = geo.gradient(s)
        if np.any(kappa_eff != kappa_eff[0]):
            f_t += 2.0 * H[:, None] ** 2 * geo.gradient(kappa_eff)
        if np.any(b != b[0]):
            f_t -= 2.0 * H[:, None] * geo.gradient(b)
        if m.zeta_prime != 0.0:
            f_t += geo.gradient(2.0 * m.zeta_prime * c * H)
        if m.zeta_tilde != 0.0:
            f_t += 2.0 * m.zeta_tilde * c[:, None] * geo.gradient(H)

        F = f_n[:, None] * N + f_t
        if self.elastic and (m.E_shear > 0 or m.E_area > 0):
            F += self._elastic_forces(geo)
        F_ext = np.zeros_like(F)
        for ext in self.external_forces:
            F_ext += ext(geo, self.t)
        F += F_ext

        if self.config.max_force is not None:
            mag = np.linalg.norm(F, axis=1)
            over = mag > self.config.max_force
            F[over] *= (self.config.max_force / mag[over])[:, None]

        diag = {"normal_force": f_n, "tangential_force": f_t, "external_force": F_ext,
                "active_tension": m.zeta * c, "active_torque": m.zeta_c * c,
                "kappa_eff": kappa_eff, "tension": s}
        return F, diag

    # ------------------------------------------------------------------ #
    def _build_operator(self, geo: DiscreteGeometryEngine, dt: float, kappa_max: float,
                        s_max: float = 0.0):
        """ξM − ηW + dt·(s_max + E·h)(−W) + dt·κ·W M⁻¹ W  (SPD): drag, in-plane viscosity,
        and the semi-implicit (linearised) tension, elastic and bending operators."""
        m, cfg = self.model, self.config
        Aop = m.eta_drag * geo.M
        if m.eta_s > 0:
            Aop = Aop - m.eta_s * geo.W
        if s_max > 0 and cfg.stabilization > 0:
            Aop = Aop - (dt * cfg.stabilization * s_max) * geo.W
        if kappa_max > 0 and cfg.stabilization > 0:
            Aop = Aop + (dt * cfg.stabilization * kappa_max) * (geo.W @ geo.Minv @ geo.W)
        E_tot = m.E_shear + m.E_area
        if E_tot > 0 and cfg.stabilization > 0:
            Aop = Aop - (dt * cfg.stabilization * E_tot * geo.mean_edge) * geo.W
        return self.backend.factorize(Aop, geo.n_vertices)      # CPU SuperLU or GPU block-CG

    def _solve_velocity(self, geo, F, dt, kappa_max, s_max=0.0):
        m, cfg = self.model, self.config
        lu = self._build_operator(geo, dt, kappa_max, s_max)
        rhs = geo.A[:, None] * F
        V = lu.solve(rhs)
        p = 0.0
        if m.volume_constraint == "penalty":
            p = -m.kV * (geo.volume - self.V0) / self.V0
            V += lu.solve(geo.A[:, None] * (p * geo.N))
        elif m.volume_constraint == "lagrange":
            Vp = lu.solve(geo.A[:, None] * geo.N)
            flux0 = float(np.sum(geo.A * np.einsum("ij,ij->i", V, geo.N)))
            fluxp = float(np.sum(geo.A * np.einsum("ij,ij->i", Vp, geo.N)))
            target = -cfg.volume_relaxation * (geo.volume - self.V0) / dt
            p = (target - flux0) / fluxp if abs(fluxp) > 1e-300 else 0.0
            V += p * Vp
        return V, p

    # ------------------------------------------------------------------ #
    def step_forward(self, dt_max: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """Advance one adaptive step (≤ dt_max). Returns (dt, metrics)."""
        m, cfg, geo = self.model, self.config, self.geo
        dt = dt_max if dt_max is not None else self.default_dt()

        F, diag = self.compute_forces(geo)
        kappa_max = float(diag["kappa_eff"].max())
        s_max = float(max(diag["tension"].max(), 0.0))
        V, p = self._solve_velocity(geo, F, dt, kappa_max, s_max)
        if not cfg.tangential_flow:
            V = np.einsum("ij,ij->i", V, geo.N)[:, None] * geo.N

        # CFL on the displacement; re-solve if the step must shrink (operator depends on dt)
        vmax = float(np.linalg.norm(V, axis=1).max())
        h = geo.cfl_edge
        if vmax * dt > cfg.cfl * h:
            dt = max(cfg.cfl * h / vmax, cfg.min_dt)
            V, p = self._solve_velocity(geo, F, dt, kappa_max, s_max)
            if not cfg.tangential_flow:
                V = np.einsum("ij,ij->i", V, geo.N)[:, None] * geo.N
            vmax = float(np.linalg.norm(V, axis=1).max())
        if not np.isfinite(vmax) or dt <= cfg.min_dt:
            q = float(geo.triangle_quality().min())
            hint = (" Min triangle quality is %.3f: remesh the input with prepare_mesh"
                    "(target_vertices=...) and install pyacvd." % q) if q < 0.15 else ""
            raise RuntimeError(
                "Simulation became unstable (non-finite velocity or dt hit min_dt)." + hint +
                " Other remedies: increase eta_s, D_chem or k_turn, set a finite "
                "c_saturation, reduce |zeta|, or lower dt_max.")

        # --- material update (Lagrangian) of all regulator species ---
        chem = self.chemistry
        X_new = self.X + dt * V
        geo_new = DiscreteGeometryEngine(X_new, self.F, edges=geo.edges)
        ratio = geo.A / geo_new.A                        # A_old / A_new  (< 1: stretching)
        strain_rate = (1.0 / ratio - 1.0) / dt           # material area strain rate ∇·v
        fields = {"c": self.c, **self.fields}
        new = {}
        for name, f in fields.items():
            if name in chem.diluted:                     # surface density: conserved amount f·A
                fn = f * ratio
                lo, hi = geo.local_min_max(f)
                fn = np.clip(fn, lo * ratio.min(), hi * max(ratio.max(), 1.0))
            else:                                        # buffered / volume-borne: value advected
                fn = f.copy()
            new[name] = fn
        c_eq_new = self.c_eq.copy()                      # material-attached target pattern

        # --- ALE tangential mesh regularisation (fields are corrected semi-Lagrangian) ---
        if cfg.regularize_mesh > 0:
            passes = 0
            while True:
                if self.elastic:   # reference state before the (non-material) mesh move
                    l_b = np.linalg.norm(X_new[self.ref_edges[:, 1]] - X_new[self.ref_edges[:, 0]], axis=1)
                    Af_b = geo_new.face_area.copy()
                centroid = (geo_new.adjacency @ X_new) / geo_new.degree[:, None]
                u = centroid - X_new
                u -= np.einsum("ij,ij->i", u, geo_new.N)[:, None] * geo_new.N
                u *= cfg.regularize_mesh if passes == 0 else 0.5

                def _advect(fld):
                    lo, hi = geo_new.local_min_max(fld)
                    return np.clip(fld - np.einsum("ij,ij->i", u, geo_new.gradient(fld)), lo, hi)
                mass_before = {k: float(np.sum(new[k] * geo_new.A)) for k in new if k in chem.diluted}
                for name in new:
                    new[name] = _advect(new[name])
                c_eq_new = _advect(c_eq_new)
                strain_rate = _advect(strain_rate)
                if self.C0_field is not None:
                    self.C0_field = _advect(self.C0_field)
                X_new = X_new + u
                geo_new = DiscreteGeometryEngine(X_new, self.F, edges=geo.edges)
                # the semi-Lagrangian correction is not conservative: restore the total amount of
                # every surface density (the mesh move is not a material process)
                for name, m_b in mass_before.items():
                    m_a = float(np.sum(new[name] * geo_new.A))
                    if m_a > 1e-300:
                        new[name] *= m_b / m_a
                if self.elastic:   # the mesh move is not a material deformation: carry the rest state along
                    l_a = np.linalg.norm(X_new[self.ref_edges[:, 1]] - X_new[self.ref_edges[:, 0]], axis=1)
                    self.l0 = np.maximum(self.l0 + (l_a - l_b), 0.1 * l_a)
                    self.Af0 = np.maximum(self.Af0 + (geo_new.face_area - Af_b), 0.1 * geo_new.face_area)
                passes += 1
                if (geo_new.triangle_quality().min() >= cfg.quality_threshold
                        or passes >= cfg.max_regularize_passes):
                    break
            self.regularize_passes = passes

        # --- chemistry: implicit surface diffusion per species ---
        for name in new:
            D = chem.D_of(name, m)
            if D > 0:
                Dop = sp.csc_matrix(geo_new.M - dt * D * geo_new.W)
                new[name] = self.backend.factorize(Dop, geo_new.n_vertices).solve(geo_new.A * new[name])

        # --- discrete events (ablation, drug steps) act on the state *before* the reactions ---
        t_new = self.t + dt
        self.X, self.geo, self.c, self.fields, self.c_eq = X_new, geo_new, new["c"], \
            {k: v for k, v in new.items() if k != "c"}, c_eq_new
        for ev in self.events:
            if not ev.done and t_new >= ev.t_event:
                ev.apply(self)
                ev.done = True
            elif ev.done:
                ev.update(self)
        new = {"c": self.c, **self.fields}

        # --- external lab-frame stimulus σ(x,t) ---
        stim = np.zeros(geo_new.n_vertices)
        for s in self.stimuli:
            stim += s(geo_new, t_new)

        # --- reactions (Heun, positivity preserving, optional Langevin noise) ---
        mech = MechanicalState(t=t_new, dt=dt, geo=geo_new, H=geo_new.H, K=geo_new.K,
                               tension=diag["tension"], kappa_eff=diag["kappa_eff"],
                               strain_rate=strain_rate,
                               normal_velocity=np.einsum("ij,ij->i", V, geo_new.N),
                               speed=np.linalg.norm(V, axis=1), c_eq=self.c_eq, stimulus=stim)
        new = chem.integrate(new, mech)
        c_new = np.maximum(new["c"], 0.0)
        if np.isfinite(m.c_saturation):        # excluded volume: no meaningful drive beyond ~10 c_sat
            c_new = np.minimum(c_new, 10.0 * m.c_saturation)

        self._remodel_reference(geo_new, dt)
        self.c = c_new
        self.fields = {k: v for k, v in new.items() if k != "c"}
        self.strain_rate, self.stimulus = strain_rate, stim
        self.velocity, self.pressure = V, p
        self.last_forces = diag
        self.t = t_new
        self.step_count += 1
        if cfg.refine_ratio > 0 and self.step_count % max(1, cfg.refine_every) == 0:
            self._refine_mesh()
        return dt, self.metrics(vmax=vmax)

  
    #  Adaptive refinement of over-stretched regions                      #
    @staticmethod
    def _edge_keys(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        return lo.astype(np.int64) * n + hi

    def _refine_mesh(self) -> int:
        """Bisect edges longer than ``refine_ratio·h0`` (h0 = initial mean edge).

        The cortex is a fluid-like material: when a region swells (bleb, protrusion,
        buckle) the triangles there stretch and the discretisation coarsens, which makes
        the surface look faceted and degrades the curvature operators. Here over-long
        edges are split at their midpoint; both adjacent faces are bisected (no
        T-junctions), each face is split at most once per pass, and every vertex field
        (regulators, c_eq, reference curvature, velocity, diagnostics) is interpolated
        linearly. The elastic rest state is carried along consistently: the halves of a
        split edge inherit l0/2, the new edge to the opposite vertex gets its current
        length divided by the mean stretch of the parent face, the two child faces get
        Af0/2. Events are notified through ``Event.on_refine``. Returns #new vertices.
        """
        cfg = self.config
        n_max = int(cfg.max_vertex_factor * self.n0)
        added_total = 0
        for _ in range(max(1, cfg.max_refine_passes)):
            geo, X, F = self.geo, self.X, self.F
            n = geo.n_vertices
            if n >= n_max:
                break
            E = geo.edges
            L = np.linalg.norm(X[E[:, 1]] - X[E[:, 0]], axis=1)
            marked = L > cfg.refine_ratio * self.h0
            if not marked.any():
                break
            # edge index of the 3 edges of every face (edge k is opposite corner k)
            ekeys = self._edge_keys(E[:, 0], E[:, 1], n)          # sorted (E is lexicographically sorted)
            fe = np.stack([np.searchsorted(ekeys, self._edge_keys(F[:, (k + 1) % 3], F[:, (k + 2) % 3], n))
                           for k in range(3)], axis=1)            # (n_faces, 3)
            # every face votes for its longest marked edge; an edge is split only if both of
            # its faces voted for it (=> each face is split at most once, no T-junctions)
            Lf = np.where(marked[fe], L[fe], -1.0)
            choice = np.argmax(Lf, axis=1)
            has = Lf[np.arange(len(F)), choice] > 0
            votes = np.zeros(len(E), dtype=np.int64)
            np.add.at(votes, fe[has, choice[has]], 1)
            split = np.where(votes == 2)[0]
            if len(split) == 0:
                break
            if n + len(split) > n_max:
                split = split[np.argsort(-L[split])[: n_max - n]]
            n_new = len(split)
            new_index = np.full(len(E), -1, dtype=np.int64)
            new_index[split] = n + np.arange(n_new)
            a, b = E[split, 0], E[split, 1]

            # ---- vertex fields: linear interpolation at the midpoints -------------------
            def interp(f):
                f = np.asarray(f)
                if f.shape[0] != n:
                    return f
                if f.dtype == bool:
                    return np.concatenate([f, f[a] | f[b]])
                return np.concatenate([f, 0.5 * (f[a] + f[b])])
            X_new = interp(X)

            # ---- faces: bisect every face that contains a split edge --------------------
            fsplit = new_index[fe]                                    # (n_faces, 3), -1 if not split
            which = np.argmax(fsplit, axis=1)                         # the (unique) split edge, if any
            is_split = fsplit[np.arange(len(F)), which] >= 0
            keep = F[~is_split]
            Fs, j = F[is_split], which[is_split]
            m = fsplit[is_split, j]
            vj = Fs[np.arange(len(Fs)), j]
            v1 = Fs[np.arange(len(Fs)), (j + 1) % 3]
            v2 = Fs[np.arange(len(Fs)), (j + 2) % 3]
            child1 = np.stack([vj, v1, m], axis=1)                    # orientation preserved
            child2 = np.stack([vj, m, v2], axis=1)
            F_new = np.concatenate([keep, child1, child2])

            # ---- elastic rest state ------------------------------------------------------
            rkeys = self._edge_keys(self.ref_edges[:, 0], self.ref_edges[:, 1], n)
            order = np.argsort(rkeys)
            rkeys_s, l0_s = rkeys[order], self.l0[order]
            def l0_of(u, v):
                idx = np.searchsorted(rkeys_s, self._edge_keys(u, v, n))
                idx = np.minimum(idx, len(rkeys_s) - 1)
                found = rkeys_s[idx] == self._edge_keys(u, v, n)
                out = np.linalg.norm(X[u] - X[v], axis=1)
                out[found] = l0_s[idx[found]]
                return out
            N2 = n + n_new
            l_split = L[split]
            l0_split = l0_of(a, b)
            # halves of the split edges
            k_half = np.concatenate([self._edge_keys(a, new_index[split], N2),
                                     self._edge_keys(new_index[split], b, N2)])
            v_half = np.concatenate([0.5 * l0_split, 0.5 * l0_split])
            # new edge (v_j, m): current length / mean stretch of the parent face
            stretch = np.zeros(len(Fs))
            for k in range(3):
                u_, w_ = Fs[:, (k + 1) % 3], Fs[:, (k + 2) % 3]
                stretch += np.linalg.norm(X[u_] - X[w_], axis=1) / np.maximum(l0_of(u_, w_), 1e-300)
            stretch /= 3.0
            k_diag = self._edge_keys(vj, m, N2)
            v_diag = np.linalg.norm(X_new[vj] - X_new[m], axis=1) / np.maximum(stretch, 1e-300)
            # all old edges re-keyed for the new vertex count
            k_old = self._edge_keys(self.ref_edges[:, 0], self.ref_edges[:, 1], N2)
            keys_all = np.concatenate([k_old, k_half, k_diag])
            vals_all = np.concatenate([self.l0, v_half, v_diag])
            o = np.argsort(keys_all, kind="stable")
            keys_all, vals_all = keys_all[o], vals_all[o]
            Af0_new = np.concatenate([self.Af0[~is_split], 0.5 * self.Af0[is_split], 0.5 * self.Af0[is_split]])

            # ---- commit -----------------------------------------------------------------
            geo_new = DiscreteGeometryEngine(X_new, F_new)
            En = geo_new.edges
            kq = self._edge_keys(En[:, 0], En[:, 1], N2)
            idx = np.minimum(np.searchsorted(keys_all, kq), len(keys_all) - 1)
            found = keys_all[idx] == kq
            l0_new = np.linalg.norm(X_new[En[:, 1]] - X_new[En[:, 0]], axis=1)
            l0_new[found] = vals_all[idx[found]]
            self.X, self.F, self.geo = X_new, F_new, geo_new
            self.ref_edges, self.l0, self.Af0 = En, l0_new, Af0_new
            self.c, self.c_eq = interp(self.c), interp(self.c_eq)
            self.fields = {k: interp(v) for k, v in self.fields.items()}
            if self.C0_field is not None:
                self.C0_field = interp(self.C0_field)
            self.strain_rate, self.stimulus = interp(self.strain_rate), interp(self.stimulus)
            self.velocity = interp(self.velocity)
            self.last_forces = {k: interp(v) for k, v in self.last_forces.items()}
            for ev in self.events:
                ev.on_refine(self, interp)
            self.n_refined += n_new
            added_total += n_new
        return added_total

    # ------------------------------------------------------------------ #
    def run(self, total_time: float, dt_max: Optional[float] = None, save_every: int = 1,
            max_steps: int = 10 ** 6, callback: Optional[Callable] = None,
            verbose: bool = False, log: Optional[Callable[[str], None]] = None,
            progress: bool = False, desc: str = "",
            trajectory: Optional[List[pv.PolyData]] = None, history: Optional[List[Dict]] = None,
            interrupt: Optional[InterruptWatcher] = None
            ) -> Tuple[List[pv.PolyData], List[Dict[str, float]]]:
        """Integrate for ``total_time``.  ``log`` receives the periodic status lines (default:
        print when verbose); ``progress`` shows a transient tqdm bar (removed when finished).
        Pass existing ``trajectory``/``history`` lists to append (used by sequences).
        ``interrupt``: an InterruptWatcher; when it fires (Enter in the console) the loop stops
        after the current step completes and ``self.interrupted`` is set."""
        emit = log if log is not None else (print if verbose else None)
        self.interrupted = False
        if interrupt is not None:
            interrupt.clear()                    # ignore key presses from before this run
        if trajectory is None:
            trajectory = [self.to_polydata()]
            history = [self.metrics()]
        t_start = self.t
        t_end = self.t + total_time
        k = 0
        t0 = time.time()
        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=float(total_time), desc=desc[:40], leave=False, dynamic_ncols=True,
                           unit="τ", bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f}{unit} [{elapsed}<{remaining}{postfix}]")
            except ImportError:            # tqdm not installed: silently run without a bar
                bar = None
        try:
            while self.t < t_end - 1e-12 and k < max_steps:
                cap = t_end - self.t
                dt_req = min(dt_max, cap) if dt_max is not None else min(self.default_dt(), cap)
                try:
                    dt, met = self.step_forward(dt_max=dt_req)
                except RuntimeError as exc:
                    msg = f"Stopped at t={self.t:.4g}: {exc}"
                    warnings.warn(msg)
                    if emit is not None:
                        emit("  " + msg)
                    self.terminated_early = True
                    break
                k += 1
                if k % save_every == 0:
                    trajectory.append(self.to_polydata())
                    history.append(met)
                if callback is not None:
                    callback(self, met)
                if bar is not None:
                    bar.update(min(dt, t_end - (self.t - dt)))
                    if k % 10 == 0:
                        bar.set_postfix_str(f"dt={dt:.1e} vmax={met['Max_Velocity']:.2e} n={met['N_vertices']}", refresh=False)
                if emit is not None and k % (10 * save_every) == 0:
                    emit(f"  t={self.t:8.4f} dt={dt:.2e} vmax={met['Max_Velocity']:.3e} "
                         f"E={met['Helfrich_Energy']:.4f} V/V0={met['Volume']/self.V0:.4f} "
                         f"n={met['N_vertices']} minQ={met['Min_Triangle_Quality']:.2f}")
                if interrupt is not None and interrupt.triggered():
                    self.interrupted = True
                    interrupt.clear()
                    if k % save_every != 0:          # keep the last computed state as a frame
                        trajectory.append(self.to_polydata())
                        history.append(met)
                    if emit is not None:
                        emit(f"  interrupted by user (Enter) at t={self.t:.4g} after {k} steps; "
                             "saving what was computed and continuing with the next experiment")
                    break
        finally:
            if bar is not None:
                bar.close()
        if emit is not None:
            emit(f"  finished {k} steps ({self.t - t_start:.4g} time units) in {time.time()-t0:.1f}s"
                 + (f", {self.n_refined} vertices added by refinement" if self.n_refined else "")
                 + (f", {self.backend.n_fallbacks} GPU->CPU solver fallbacks" if self.backend.n_fallbacks else ""))
        return trajectory, history

    # ------------------------------------------------------------------ #
    def apply_protocol_stage(self, stage: "Protocol", t0: Optional[float] = None):
        """Switch the running simulation to a new stage (mechanics, regulator, forces,
        stimuli, events, chemistry) *without* resetting geometry or the elastic reference.
        Times inside the stage (t_on, t_event, ...) are relative to the stage start."""
        t0 = self.t if t0 is None else t0
        if stage.params:
            new_model = ActiveSurfaceConstitutiveModel.from_dict(stage.params)
            self.model = new_model
            self.elastic = (new_model.E_shear > 0) or (new_model.E_area > 0) or new_model.curvature_memory
            if new_model.curvature_memory and self.C0_field is None:
                geo = self.geo
                ell2 = (self.config.curvature_memory_smoothing * geo.mean_edge) ** 2
                C0 = spla.spsolve(sp.csc_matrix(geo.M - ell2 * geo.W), geo.A * (2.0 * geo.H)) if ell2 > 0 else 2.0 * geo.H
                self.C0_field = np.asarray(C0, float)
        mesh = self.to_polydata()
        c0 = stage.field_c0(mesh)
        if c0 is not None and stage.c_mode != "keep":
            if c0.shape != (self.geo.n_vertices,):
                raise ValueError("stage c0 has the wrong size; use a callable c0 (mesh -> array) "
                                 "so it can be evaluated on the current mesh")
            if stage.c_mode == "set":
                self.c = np.maximum(c0, 0.0).copy()
                self.c_eq = self.c.copy()
            elif stage.c_mode == "add":
                self.c = np.maximum(self.c + c0, 0.0)
                self.c_eq = np.maximum(self.c_eq + c0, 0.0)
            else:
                raise ValueError("c_mode must be 'set', 'add' or 'keep'")
        ceq = stage.field_c_eq(mesh)
        if ceq is not None:
            self.c_eq = np.maximum(ceq, 0.0).copy()
        if stage.chemistry is not None:
            self.set_chemistry(stage.chemistry)
        f0 = stage.eval_fields0(mesh)
        if f0:
            self.set_initial_fields(**f0)
        self.external_forces = [_TimeShift(f, t0) for f in stage.external_forces]
        self.stimuli = [_TimeShift(s, t0) for s in stage.stimuli]
        self.events = sorted(stage.fresh_events(t0), key=lambda e: e.t_event)

    def set_chemistry(self, chem: ChemistryModel):
        """Replace the reaction model; species present in both are kept, new ones initialised."""
        old_fields = self.fields
        self.chemistry = chem
        self.fields = chem.initial_fields(self.geo, self.c)
        for k in self.fields:
            if k in old_fields and old_fields[k].shape == self.fields[k].shape:
                self.fields[k] = old_fields[k]

    # ------------------------------------------------------------------ #
    def energies(self, geo: Optional[DiscreteGeometryEngine] = None) -> Dict[str, float]:
        geo = geo or self.geo
        m = self.model
        H, K, A = geo.H, geo.K, geo.A
        C0 = self.C0_field if self.C0_field is not None else m.C0   # reference curvature field if any
        e_bend = float(np.sum(A * 0.5 * m.kappa_b * (2 * H - C0) ** 2))
        e_gauss = float(m.kappa_g * np.sum(A * K))
        e_tens = float(m.gamma_0 * geo.area)
        e_active = float(np.sum(A * (m.zeta * m.drive(self.c))))  # work-like bookkeeping of active tension
        return {"Bending_Energy": e_bend, "Gaussian_Energy": e_gauss,
                "Tension_Energy": e_tens, "Helfrich_Energy": e_bend + e_gauss + e_tens,
                "Active_Tension_Integral": e_active}

    def metrics(self, vmax: Optional[float] = None) -> Dict[str, float]:
        geo = self.geo
        speed = np.linalg.norm(self.velocity, axis=1)
        centroid = np.average(self.X, axis=0, weights=geo.A)
        # shape descriptors: sphericity (1 = sphere) and aspect ratio of the inertia ellipsoid
        sphericity = (np.pi ** (1 / 3) * (6.0 * abs(geo.volume)) ** (2 / 3)) / max(geo.area, 1e-300)
        Xc = self.X - centroid
        cov = (Xc * geo.A[:, None]).T @ Xc / geo.A.sum()
        ev = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 1e-300))
        met = {"time": self.t, "step": self.step_count,
               "Area": geo.area, "Volume": geo.volume,
               "Area_ratio": geo.area / self.A0, "Volume_ratio": geo.volume / self.V0,
               "Pressure": self.pressure,
               "Max_Velocity": float(speed.max()) if vmax is None else vmax,
               "Mean_Speed": float(np.average(speed, weights=geo.A)),
               "Retrograde_Flow": float(np.average(
                   np.linalg.norm(self.velocity - np.einsum("ij,ij->i", self.velocity, geo.N)[:, None] * geo.N,
                                  axis=1), weights=geo.A)),          # mean tangential (cortical) flow speed
               "Sphericity": float(sphericity),
               "Aspect_Ratio": float(ev[-1] / ev[0]),
               "Centroid_Displacement": float(np.linalg.norm(centroid - self.X_centroid0)),
               "H_mean": float(np.average(geo.H, weights=geo.A)),
               "H_std": float(np.sqrt(np.average((geo.H - np.average(geo.H, weights=geo.A)) ** 2,
                                                 weights=geo.A))),
               "c_mean": float(np.average(self.c, weights=geo.A)),
               "c_max": float(self.c.max()) if self.c.size else 0.0,
               "c_std": float(np.sqrt(np.average((self.c - np.average(self.c, weights=geo.A)) ** 2,
                                                 weights=geo.A))),  # patterning / pulsatility read-out
               "Stimulus_max": float(self.stimulus.max()) if self.stimulus.size else 0.0,
               "N_vertices": int(geo.n_vertices), "Max_Edge_Ratio": float(
                   np.linalg.norm(self.X[geo.edges[:, 1]] - self.X[geo.edges[:, 0]], axis=1).max() / self.h0),
               "Min_Triangle_Quality": float(geo.triangle_quality().min())}
        for k, v in self.fields.items():
            met[f"{k}_mean"] = float(np.average(v, weights=geo.A))
            met[f"{k}_max"] = float(v.max())
        met.update(self.energies(geo))
        return met

    def furrow_radius(self, axis=(1, 0, 0), width: float = 0.05) -> float:
        """Mean distance from ``axis`` (through the centroid) of the vertices within ``width`` of
        the equatorial plane — the cytokinesis furrow read-out (R_eq units)."""
        a = np.asarray(axis, float); a /= np.linalg.norm(a)
        X = self.X - np.average(self.X, axis=0, weights=self.geo.A)
        h = X @ a
        band = np.abs(h) < width
        if not np.any(band):
            band = np.abs(h) < 3 * self.geo.mean_edge
        r = np.linalg.norm(X[band] - np.outer(h[band], a), axis=1)
        return float(r.mean())

    def to_polydata(self) -> pv.PolyData:
        pd = _polydata_from_arrays(self.X, self.F)
        if self.config.store_fields:
            geo = self.geo
            pd.point_data["concentration"] = self.c
            pd.point_data["mean_curvature"] = geo.H
            pd.point_data["gaussian_curvature"] = geo.K
            pd.point_data["velocity"] = self.velocity
            pd.point_data["speed"] = np.linalg.norm(self.velocity, axis=1)
            pd.point_data["normal_velocity"] = np.einsum("ij,ij->i", self.velocity, geo.N)
            phi = self.model.drive(self.c)
            pd.point_data["active_tension"] = self.model.zeta * phi
            pd.point_data["active_torque"] = self.model.zeta_c * phi
            pd.point_data["strain_rate"] = self.strain_rate
            if self.stimuli:
                pd.point_data["stimulus"] = self.stimulus
            for k, v in self.fields.items():
                pd.point_data[k] = v
            for key in ("normal_force", "tension"):
                if key in self.last_forces:
                    pd.point_data[key] = self.last_forces[key]
            if self.C0_field is not None:
                pd.point_data["reference_curvature"] = 0.5 * self.C0_field
            pd.field_data["time"] = np.array([self.t])
        return pd

    @property
    def mesh(self) -> pv.PolyData:
        return self.to_polydata()



#  Regulator field helpers                                                    #
def _pts(mesh) -> np.ndarray:
    return np.asarray(mesh.points if isinstance(mesh, pv.PolyData) else mesh, float)


def uniform_field(mesh, value: float = 1.0) -> np.ndarray:
    return np.full(len(_pts(mesh)), float(value))


def gaussian_cap(mesh, center, width: float, amplitude: float = 1.0,
                 background: float = 0.0) -> np.ndarray:
    """Polarised cap: c = background + amplitude·exp(−|x−center|²/(2 width²))."""
    P = _pts(mesh)
    d2 = np.sum((P - np.asarray(center, float)) ** 2, axis=1)
    return background + amplitude * np.exp(-d2 / (2.0 * width ** 2))


def equatorial_ring(mesh, axis=(1, 0, 0), width: float = 0.2, amplitude: float = 1.0,
                    background: float = 0.0, offset: float = 0.0) -> np.ndarray:
    """Contractile ring (cytokinesis-like) around ``axis`` through the centroid."""
    P = _pts(mesh)
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    h = (P - P.mean(axis=0)) @ a - offset
    return background + amplitude * np.exp(-h ** 2 / (2.0 * width ** 2))


def noisy_field(mesh, mean: float = 1.0, std: float = 0.1, seed: Optional[int] = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.maximum(rng.normal(mean, std, len(_pts(mesh))), 0.0)


def from_function(mesh, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """c_i = fn(points) for an arbitrary user function of the n×3 coordinates."""
    return np.asarray(fn(_pts(mesh)), float)



#  Presets (starting points for hypothesis testing)                           #

# Units: lengths in R_eq (equivalent-sphere radius), tensions in units of the reference
# tension scale, time in ξR²/γ.  Real cells have κ/(γR²) ~ 1e-5..1e-2 (κ~1e-18 J, γ~3e-4 N/m,
# R~10 µm), i.e. bending is a *small* correction: kappa_b ~ 0.01.  A kappa_b of order 1 turns
# the cell into a stiff vesicle whose bending length √(κ/γ) ~ R erases every feature.
PRESETS: Dict[str, Dict] = {
    "passive_relaxation": dict(kappa_b=0.02, gamma_0=0.5, eta_drag=1.0, eta_s=0.05),
    "cortical_contraction": dict(kappa_b=0.02, gamma_0=0.2, zeta=2.0, eta_drag=1.0, eta_s=0.3,
                                 D_chem=0.05, k_turn=1.0, c_saturation=3.0),
    # sphere unstable to mode l when |gamma_0 + zeta c|(l-1)(l+2) > kappa l(l+1)(l-1)(l+2)/R^2
    "active_buckling": dict(kappa_b=0.05, gamma_0=0.2, zeta=-1.5, eta_drag=1.0, eta_s=0.05,
                            k_turn=5.0, volume_constraint="lagrange"),
    "polar_protrusion": dict(kappa_b=0.02, gamma_0=0.5, zeta=-1.0, zeta_c=-0.2, eta_drag=1.0,
                             eta_s=0.3, D_chem=0.05, k_turn=1.0, c_saturation=3.0),
    "cytokinesis_ring": dict(kappa_b=0.01, gamma_0=0.3, zeta=3.0, eta_drag=1.0, eta_s=0.3,
                             D_chem=0.05, k_turn=1.0, c_saturation=3.0),
    "active_torque_folding": dict(kappa_b=0.05, gamma_0=0.3, zeta_c=0.3, eta_drag=1.0, eta_s=0.3,
                                  D_chem=0.05, k_turn=1.0),
    "curvature_tension_instability": dict(kappa_b=0.05, gamma_0=0.2, zeta_prime=1.5, zeta_tilde=0.5,
                                          eta_drag=1.0, eta_s=0.05),
}

# Viscoelastic cortex that remembers the *input cell shape* (use these for real cell meshes).
# E_* ~ 2D elastic moduli in units of the tension scale; tau_remodel = cortex turnover time:
# deformations shorter than tau are elastic and reversible, longer ones become permanent.
CELL_SHELL = dict(E_shear=3.0, E_area=3.0, curvature_memory=True, tau_remodel=5.0)
CELL_PRESETS: Dict[str, Dict] = {
    "cell_passive":        {**PRESETS["passive_relaxation"], **CELL_SHELL},
    "cell_contraction":    {**PRESETS["cortical_contraction"], **CELL_SHELL},
    "cell_ring":           {**PRESETS["cytokinesis_ring"], **CELL_SHELL},
    "cell_protrusion":     {**PRESETS["polar_protrusion"], **CELL_SHELL},
    "cell_torque_folding": {**PRESETS["active_torque_folding"], **CELL_SHELL},
    "cell_buckling":       {**PRESETS["active_buckling"], **CELL_SHELL, "E_shear": 0.5, "E_area": 0.5},
}



#  Protocols: composable / chainable experiment building blocks               #

class _TimeShift:
    """Wrap a Stimulus / ExternalForce so that its clock starts at ``t0`` (used by sequences)."""

    def __init__(self, inner, t0: float):
        self.inner, self.t0 = inner, float(t0)

    def __call__(self, geo, t):
        return self.inner(geo, t - self.t0)

    def __repr__(self):
        return f"{self.inner!r}@t0={self.t0:g}"


@dataclass
class Protocol:
    """A reusable experiment definition (building block).

    Fields mirror :meth:`VirtualLaboratory.run_experiment`.  ``c0``/``c_eq`` may be arrays
    or callables ``mesh -> array`` (preferred: they can be re-evaluated on any mesh, e.g.
    after refinement or on a different cell).  ``c_mode`` says how the regulator is applied
    when the protocol starts as a *stage* of a sequence: 'set' (c := c0, c_eq := c0),
    'add' (c += c0, c_eq += c0) or 'keep' (only mechanics/forces/stimuli change).

    Composition (parallel):  ``A + B``  or  ``Protocol.compose(A, B, ..., name=..., params={...})``
        * constitutive params merged left→right (later wins; overrides are recorded in ``notes``,
          explicit ``params`` win over everything),
        * regulator fields combined (``c0_mode`` 'sum' | 'max'),
        * external forces / stimuli / events concatenated,
        * at most one ChemistryModel (pass ``chemistry=`` to resolve a conflict),
        * total_time = max of the parts unless given.
    Composition (sequential): ``A >> B``  gives a list of stages for
    :meth:`VirtualLaboratory.run_sequence`; stage-relative times of forces, stimuli and
    events are shifted automatically.
    """
    name: str
    params: Dict = field(default_factory=dict)
    c0: Optional[Union[np.ndarray, Callable]] = None
    c_eq: Optional[Union[np.ndarray, Callable]] = None
    total_time: float = 1.0
    external_forces: List[ExternalForce] = field(default_factory=list)
    chemistry: Optional[ChemistryModel] = None
    stimuli: List[Stimulus] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    fields0: Optional[Dict[str, Union[np.ndarray, Callable]]] = None
    c_mode: str = "set"
    run_kwargs: Dict = field(default_factory=dict)      # dt_max, save_every, ...
    notes: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)

    # evaluation on a mesh ------------------------------------------------
    @staticmethod
    def _eval(f, mesh):
        if f is None:
            return None
        return np.asarray(f(mesh) if callable(f) else f, float)

    def field_c0(self, mesh) -> Optional[np.ndarray]:
        return self._eval(self.c0, mesh)

    def field_c_eq(self, mesh) -> Optional[np.ndarray]:
        return self._eval(self.c_eq, mesh)

    def eval_fields0(self, mesh) -> Optional[Dict[str, np.ndarray]]:
        if not self.fields0:
            return None
        return {k: self._eval(v, mesh) for k, v in self.fields0.items()}

    def fresh_events(self, t0: float = 0.0) -> List[Event]:
        """Independent copies (events carry state) with their clock shifted by t0."""
        import copy
        out = []
        for e in self.events:
            e2 = copy.deepcopy(e)
            e2.done = False
            e2.t_event = e.t_event + t0
            out.append(e2)
        return out

    def describe(self) -> Dict:
        return {"name": self.name, "params": self.params, "total_time": self.total_time,
                "c_mode": self.c_mode, "chemistry": self.chemistry.to_dict() if self.chemistry else None,
                "external_forces": [repr(f) for f in self.external_forces],
                "stimuli": [repr(s) for s in self.stimuli], "events": [type(e).__name__ for e in self.events],
                "parents": self.parents, "notes": self.notes}

    # composition ---------------------------------------------------------
    @staticmethod
    def compose(*protocols: "Protocol", name: Optional[str] = None, params: Optional[Dict] = None,
                c0_mode: str = "sum", total_time: Optional[float] = None,
                chemistry: Optional[ChemistryModel] = None, c_mode: Optional[str] = None) -> "Protocol":
        if not protocols:
            raise ValueError("compose needs at least one protocol")
        notes: List[str] = []
        merged: Dict = {}
        for p in protocols:
            for k, v in p.params.items():
                if k in merged and merged[k] != v:
                    notes.append(f"{k}: {merged[k]!r} -> {v!r} (from {p.name})")
                merged[k] = v
        if params:
            for k, v in params.items():
                if k in merged and merged[k] != v:
                    notes.append(f"{k}: {merged[k]!r} -> {v!r} (explicit)")
            merged.update(params)
        chems = [p.chemistry for p in protocols if p.chemistry is not None]
        if chemistry is None:
            if len({id(c) for c in chems}) > 1:
                raise ValueError("Protocols define different ChemistryModels; pass chemistry= to compose()")
            chemistry = chems[0] if chems else None
        c0_parts = [p.c0 for p in protocols if p.c0 is not None]
        ceq_parts = [p.c_eq for p in protocols if p.c_eq is not None]

        def combine(parts):
            if not parts:
                return None
            def f(mesh):
                vals = [Protocol._eval(q, mesh) for q in parts]
                return np.sum(vals, axis=0) if c0_mode == "sum" else np.max(vals, axis=0)
            return f
        f0 = {}
        for p in protocols:
            if p.fields0:
                f0.update(p.fields0)
        rk = {}
        for p in protocols:
            rk.update(p.run_kwargs)
        return Protocol(name=name or "+".join(p.name for p in protocols), params=merged,
                        c0=combine(c0_parts), c_eq=combine(ceq_parts),
                        total_time=total_time if total_time is not None else max(p.total_time for p in protocols),
                        external_forces=[f for p in protocols for f in p.external_forces],
                        chemistry=chemistry,
                        stimuli=[s for p in protocols for s in p.stimuli],
                        events=[e for p in protocols for e in p.events],
                        fields0=f0 or None, c_mode=c_mode or protocols[0].c_mode, run_kwargs=rk,
                        notes=notes, parents=[p.name for p in protocols])

    def with_(self, **changes) -> "Protocol":
        """Copy with modified attributes; constitutive params can be given as params={...} (merged)."""
        import copy
        p = copy.copy(self)
        if "params" in changes:
            p.params = {**self.params, **changes.pop("params")}
        for k, v in changes.items():
            setattr(p, k, v)
        p.parents = [self.name]
        return p

    def __add__(self, other: "Protocol") -> "Protocol":
        return Protocol.compose(self, other)

    def __rshift__(self, other) -> List["Protocol"]:
        return [self] + (list(other) if isinstance(other, (list, tuple)) else [other])

    def __rrshift__(self, other) -> List["Protocol"]:      # [A, B] >> C
        return list(other) + [self]



#  Virtual laboratory                                                          #

@dataclass
class ExperimentResult:
    name: str
    params: Dict
    config: Dict
    trajectory: List[pv.PolyData]
    history: List[Dict[str, float]]
    directory: Optional[Path] = None
    wall_time: float = 0.0
    stages: List[Dict] = field(default_factory=list)      # for sequences: [{"name", "t_start", "t_end"}]
    protocol: Optional[Dict] = None                        # description of the Protocol that produced it

    def to_dataframe(self):
        import pandas as pd  # optional dependency
        return pd.DataFrame(self.history)

    def final_metrics(self) -> Dict[str, float]:
        return self.history[-1]

    def save_metrics_csv(self, path: Union[str, Path]):
        keys = list(self.history[0].keys())
        with open(path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in self.history:
                f.write(",".join(f"{row.get(k, float('nan')):.10g}" for k in keys) + "\n")


class ExperimentRun:
    """One execution of the driver = one output folder  ``<root>/Experiment_<id>/``.

    ``root`` defaults to ``Results/Virtual_lab`` (created if missing, relative to the current
    working directory); ``id`` = zero-padded run counter + timestamp, unique per execution.
    Inside: one sub-folder per VirtualLaboratory (with one folder per experiment), the GIFs
    in ``simulation_gifs/``, ``log_output.txt`` with everything that used to be printed, and
    ``run_info.json``.  ``echo=True`` additionally prints log lines (above the progress bar).

    >>> run = ExperimentRun()
    >>> lab = run.lab("active_surface_lab")          # VirtualLaboratory writing into the run folder
    >>> run.log("length scale", scale)
    >>> run.export_animations(lab.results, names)     # -> simulation_gifs/<name>.gif
    """

    def __init__(self, root: Union[str, Path] = Path("Results") / "Virtual_lab",
                 prefix: str = "Experiment", echo: bool = False, run_id: Optional[str] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            existing = []
            for d in self.root.glob(f"{prefix}_*"):
                head = d.name[len(prefix) + 1:].split("_")[0]
                if head.isdigit():
                    existing.append(int(head))
            counter = (max(existing) + 1) if existing else 1
            run_id = f"{counter:03d}_{time.strftime('%Y%m%d-%H%M%S')}"
            while (self.root / f"{prefix}_{run_id}").exists():
                counter += 1
                run_id = f"{counter:03d}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.id = run_id
        self.dir = self.root / f"{prefix}_{run_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.gif_dir = self.dir / "simulation_gifs"
        self.gif_dir.mkdir(exist_ok=True)
        self.log_path = self.dir / "log_output.txt"
        self.echo = echo
        self.labs: List["VirtualLaboratory"] = []
        self._t0 = time.time()
        import platform, sys, os
        info = {"id": self.id, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cwd": os.getcwd(),
                "python": sys.version.split()[0], "platform": platform.platform(),
                "numpy": np.__version__, "pyvista": pv.__version__}
        with open(self.dir / "run_info.json", "w") as f:
            json.dump(info, f, indent=2)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# Experiment run {self.id}   {info['started']}\n# output: {self.dir.resolve()}\n\n")
        self.backend = _get_backend()
        self.interrupt = InterruptWatcher.get()
        self.log(self.backend.describe(), echo=True)
        self.log("Interrupt: press Enter in the console to stop the running experiment and continue with the next"
                 if self.interrupt.active else
                 "Interrupt: stdin is not an interactive console; Enter-to-skip is disabled", echo=True)

    # logging --------------------------------------------------------------
    def log(self, *parts, echo: Optional[bool] = None):
        """Append a line (timestamped) to log_output.txt; also print if echo."""
        msg = " ".join(str(p) for p in parts)
        stamp = time.strftime("%H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in msg.split("\n"):
                f.write(f"[{stamp}] {line}\n")
        if self.echo if echo is None else echo:
            try:
                from tqdm.auto import tqdm
                tqdm.write(msg)          # does not break an active progress bar
            except ImportError:
                print(msg)

    def path(self, *parts: Union[str, Path]) -> Path:
        """A path inside the run folder (parent directories are created)."""
        p = self.dir.joinpath(*[str(x) for x in parts])
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def lab(self, name: str = "active_surface_lab", **kwargs) -> "VirtualLaboratory":
        lab = VirtualLaboratory(self.dir / name, run=self, **kwargs)
        self.labs.append(lab)
        return lab

    def export_animations(self, results: Dict[str, "ExperimentResult"], names: Optional[Sequence[str]] = None,
                          **kwargs) -> List[Path]:
        """GIFs of the given results into ``simulation_gifs/`` (all results if names is None)."""
        out = []
        for name in (names if names is not None else list(results)):
            if name not in results:
                self.log(f"[gif] '{name}' not found in results, skipped")
                continue
            fn = self.gif_dir / (str(name).replace(" ", "_") + ".gif")
            VirtualLaboratory.export_animation(results[name], fn, log=self.log, **kwargs)
            out.append(fn)
        return out

    def finish(self):
        self.log(f"run finished in {time.time() - self._t0:.1f}s; "
                 f"{sum(len(l.results) for l in self.labs)} experiments in {self.dir}")


class VirtualLaboratory:
    """Runs, stores and compares active-surface experiments.

    Each experiment gets its own folder with ``params.json``, ``metrics.csv``,
    ``log_output.txt`` and the saved frames (``.vtp`` keeps all point fields; ``.obj`` is
    geometry only).  Attach the lab to an :class:`ExperimentRun` (``run.lab(...)``) to collect
    everything under ``Results/Virtual_lab/Experiment_<id>/``; status lines then go to the
    logs instead of the console and a transient tqdm bar shows the progress of the running
    experiment.
    """

    def __init__(self, output_dir: Union[str, Path] = "active_surface_lab",
                 save_formats: Sequence[str] = ("obj",), save_fields: bool = True,
                 verbose: bool = True, config: Optional[SimulationConfig] = None,
                 run: Optional[ExperimentRun] = None, progress: bool = True,
                 interruptible: bool = True):
        """save_formats: any of 'obj', 'ply', 'stl', 'vtp'.  OBJ/STL carry geometry only, so
        with ``save_fields=True`` every frame also gets ``frame_XXXX_fields.npz`` holding the
        point arrays (concentration, curvature, velocity, ...) and the time; reload with
        :meth:`load_trajectory`.  ``verbose`` controls the status lines (console if no run is
        attached, log files otherwise); ``progress`` the tqdm bar."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_formats = tuple(save_formats)
        self.save_fields = save_fields
        self.verbose = verbose
        self.progress = progress
        self.run = run
        self.config = config or SimulationConfig()
        self.interrupt: Optional[InterruptWatcher] = InterruptWatcher.get() if interruptible else None
        if self.interrupt is not None and not self.interrupt.active:
            self.interrupt = None
        self.backend = _get_backend(self.config.gpu, self.config.gpu_min_vertices)
        self.results: Dict[str, ExperimentResult] = {}
        self.protocols: Dict[str, Protocol] = {}
        self.last_simulator: Optional[ActiveSurfaceSimulator] = None
        self._exp_log: Optional[Path] = None

    # logging ----------------------------------------------------------------
    def _log(self, msg: str):
        if self._exp_log is not None:
            with open(self._exp_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        if self.run is not None:
            self.run.log(msg)
        elif self.verbose:
            print(msg)

    # ------------------------------------------------------------------ #
    def _make_simulator(self, mesh, params, external_forces, prepare, chemistry, stimuli, events):
        model = (params if isinstance(params, ActiveSurfaceConstitutiveModel)
                 else ActiveSurfaceConstitutiveModel.from_dict(params))
        return ActiveSurfaceSimulator(mesh, model, config=self.config, external_forces=external_forces,
                                      prepare=prepare, chemistry=chemistry, stimuli=stimuli, events=events)

    def _finish(self, sim, exp_name, exp_dir, traj, hist, total_time, wall, save_frames,
                stages=None, protocol=None) -> ExperimentResult:
        res = ExperimentResult(exp_name, sim.model.to_dict(), asdict(self.config), traj, hist, exp_dir, wall,
                               stages=stages or [], protocol=protocol)
        with open(exp_dir / "params.json", "w") as f:
            json.dump({"model": sim.model.to_dict(), "numerics": asdict(self.config),
                       "chemistry": sim.chemistry.to_dict(),
                       "stimuli": [repr(s) for s in sim.stimuli],
                       "events": [type(e).__name__ for e in sim.events],
                       "total_time": total_time, "n_frames": len(traj), "t_end": sim.t,
                       "interrupted_by_user": sim.interrupted, "terminated_early": sim.terminated_early,
                       "compute": sim.backend.describe(),
                       "stages": stages or [], "protocol": protocol}, f, indent=2, default=str)
        res.save_metrics_csv(exp_dir / "metrics.csv")
        if save_frames:
            self.save_trajectory(traj, exp_dir)
        fm = hist[-1]
        self._log(f"[{exp_name}] done in {wall:.1f}s | E={fm['Helfrich_Energy']:.4f} "
                  f"A/A0={fm['Area_ratio']:.3f} V/V0={fm['Volume_ratio']:.4f} "
                  f"vmax={fm['Max_Velocity']:.2e} minQ={fm['Min_Triangle_Quality']:.2f} "
                  f"n={fm['N_vertices']}" + (" (terminated early)" if sim.terminated_early else "")
                  + (f" (STOPPED BY USER at t={sim.t:.4g} of {total_time})" if sim.interrupted else ""))
        self.results[exp_name] = res
        self._exp_log = None
        return res

    def run_experiment(self, mesh: pv.PolyData, params: Union[Dict, ActiveSurfaceConstitutiveModel],
                       c0: Optional[np.ndarray] = None, total_time: float = 1.0,
                       exp_name: str = "experiment", dt_max: Optional[float] = None,
                       save_every: int = 1, external_forces: Optional[Sequence[ExternalForce]] = None,
                       c_eq: Optional[np.ndarray] = None, prepare: bool = False,
                       save_frames: bool = True, callback: Optional[Callable] = None,
                       chemistry: Optional[ChemistryModel] = None,
                       stimuli: Optional[Sequence[Stimulus]] = None,
                       events: Optional[Sequence[Event]] = None,
                       fields0: Optional[Dict[str, np.ndarray]] = None,
                       _protocol: Optional[Dict] = None) -> ExperimentResult:
        """Run one experiment.  Biology hooks:
          chemistry : ChemistryModel (LinearTurnover, MechanosensitiveTurnover, ExcitableRho,
                      TuringPolarity, or your own) — reaction kinetics of the regulator(s)
          stimuli   : lab-frame signals σ(x,t) (GaussianPulse = optogenetics, UniformStimulus = drug)
          events    : LaserAblation, ParameterStep (drug wash-in), ...
          fields0   : initial values of extra species, e.g. {"rho": ...}
        The simulator is available in ``callback(sim, metrics)`` for custom read-outs
        (e.g. sim.furrow_radius(), ablation.recoil_speed(sim))."""
        exp_dir = self.output_dir / exp_name.replace(" ", "_")
        exp_dir.mkdir(parents=True, exist_ok=True)
        self._exp_log = exp_dir / "log_output.txt"
        open(self._exp_log, "w").close()
        sim = self._make_simulator(mesh, params, external_forces, prepare, chemistry, stimuli, events)
        if c0 is None:
            c0 = np.zeros(sim.geo.n_vertices)
        sim.set_initial_chemical_field(c0, c_eq=c_eq)
        if fields0:
            sim.set_initial_fields(**fields0)
        self.last_simulator = sim
        self._log(f"[{exp_name}] {sim.geo.n_vertices} vertices, T={total_time}, "
                  f"dt_max={dt_max if dt_max is not None else sim.default_dt():.3e}, "
                  f"chemistry={type(sim.chemistry).__name__}"
                  + (f", {len(sim.stimuli)} stimuli" if sim.stimuli else "")
                  + (f", {len(sim.events)} events" if sim.events else "")
                  + (f", {len(sim.external_forces)} external forces" if sim.external_forces else ""))
        t0 = time.time()
        traj, hist = sim.run(total_time, dt_max=dt_max, save_every=save_every, callback=callback,
                             log=self._log if self.verbose else None, progress=self.progress, desc=exp_name,
                             interrupt=self.interrupt)
        return self._finish(sim, exp_name, exp_dir, traj, hist, total_time, time.time() - t0,
                            save_frames, protocol=_protocol)

    # ---- protocols -------------------------------------------------------- #
    def run_protocol(self, mesh: pv.PolyData, protocol: Protocol, exp_name: Optional[str] = None,
                     callback: Optional[Callable] = None, **run_kwargs) -> ExperimentResult:
        """Run a :class:`Protocol` (single or composed) and register it in ``self.protocols``
        so it can be used as a building block later (``lab.protocols[name]``)."""
        exp_name = exp_name or protocol.name
        self.protocols[exp_name] = protocol
        kw = {**protocol.run_kwargs, **run_kwargs}
        if protocol.notes:
            self._log(f"[{exp_name}] composed from {protocol.parents}; parameter overrides: "
                      + "; ".join(protocol.notes))
        return self.run_experiment(mesh, protocol.params, protocol.field_c0(mesh), protocol.total_time,
                                   exp_name=exp_name, external_forces=list(protocol.external_forces),
                                   c_eq=protocol.field_c_eq(mesh), chemistry=protocol.chemistry,
                                   stimuli=list(protocol.stimuli), events=protocol.fresh_events(0.0),
                                   fields0=protocol.eval_fields0(mesh), callback=callback,
                                   _protocol=protocol.describe(), **kw)

    def run_sequence(self, mesh: pv.PolyData, stages: Sequence[Protocol], exp_name: str,
                     dt_max: Optional[float] = None, save_every: int = 1, save_frames: bool = True,
                     callback: Optional[Callable] = None) -> ExperimentResult:
        """Chain protocols in time on ONE simulation: stage k+1 starts from the final geometry,
        regulator fields and elastic reference of stage k (mechanics, forces, stimuli, events and
        chemistry are switched; see :class:`Protocol` ``c_mode``).  One trajectory / metrics file;
        stage boundaries in ``result.stages`` and ``stages.json``."""
        stages = list(stages)
        exp_dir = self.output_dir / exp_name.replace(" ", "_")
        exp_dir.mkdir(parents=True, exist_ok=True)
        self._exp_log = exp_dir / "log_output.txt"
        open(self._exp_log, "w").close()
        first = stages[0]
        sim = self._make_simulator(mesh, first.params, list(first.external_forces), False,
                                   first.chemistry, list(first.stimuli), first.fresh_events(0.0))
        c0 = first.field_c0(mesh)
        sim.set_initial_chemical_field(np.zeros(sim.geo.n_vertices) if c0 is None else c0,
                                       c_eq=first.field_c_eq(mesh))
        f0 = first.eval_fields0(mesh)
        if f0:
            sim.set_initial_fields(**f0)
        self.last_simulator = sim
        total = sum(s.total_time for s in stages)
        self._log(f"[{exp_name}] sequence of {len(stages)} stages, T={total}: "
                  + " -> ".join(f"{s.name} ({s.total_time})" for s in stages))
        traj, hist, info = None, None, []
        t0 = time.time()
        for k, stage in enumerate(stages):
            if k > 0:
                sim.apply_protocol_stage(stage)
            t_start = sim.t
            self._log(f"[{exp_name}] stage {k + 1}/{len(stages)} '{stage.name}' from t={t_start:.4g} "
                      f"(c_mode={stage.c_mode}, chemistry={type(sim.chemistry).__name__})")
            traj, hist = sim.run(stage.total_time, dt_max=dt_max, save_every=save_every, callback=callback,
                                 log=self._log if self.verbose else None, progress=self.progress,
                                 desc=f"{exp_name} [{k + 1}/{len(stages)} {stage.name}]",
                                 trajectory=traj, history=hist, interrupt=self.interrupt)
            info.append({"name": stage.name, "t_start": t_start, "t_end": sim.t, "frame_end": len(traj) - 1,
                         "protocol": stage.describe()})
            if sim.terminated_early:
                break
        with open(exp_dir / "stages.json", "w") as f:
            json.dump(info, f, indent=2, default=str)
        self.protocols[exp_name] = Protocol(exp_name, params=dict(stages[-1].params), total_time=total,
                                            parents=[s.name for s in stages],
                                            notes=["sequence; re-run with run_sequence"])
        return self._finish(sim, exp_name, exp_dir, traj, hist, total, time.time() - t0, save_frames,
                            stages=info, protocol={"sequence": [s.describe() for s in stages]})

    # ------------------------------------------------------------------ #
    def save_trajectory(self, traj: Sequence[pv.PolyData], exp_dir: Union[str, Path]):
        exp_dir = Path(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(traj):
            for fmt in self.save_formats:
                m.save(str(exp_dir / f"frame_{i:04d}.{fmt}"))
            if self.save_fields and (m.point_data or m.field_data):
                arrays = {k: np.asarray(m.point_data[k]) for k in m.point_data.keys()}
                if "time" in m.field_data:
                    arrays["time"] = np.asarray(m.field_data["time"])
                np.savez_compressed(exp_dir / f"frame_{i:04d}_fields.npz", **arrays)
        # acquisition (simulation) times next to the meshes: the analysis pipeline detects frame_times.csv in a
        # scene folder and then uses the real time between the frames (velocities, rates, lifetimes, GIF timing)
        times = [float(np.asarray(m.field_data["time"]).ravel()[0]) if "time" in m.field_data else None for m in traj]
        if traj and all(t is not None for t in times):
            fmt0 = self.save_formats[0] if self.save_formats else "obj"
            with open(exp_dir / "frame_times.csv", "w", encoding="utf-8") as fh:
                fh.write("frame,file,time,unit\n")
                for i, t in enumerate(times):
                    fh.write(f"{i},frame_{i:04d}.{fmt0},{t:.10g},s\n")

    @staticmethod
    def load_trajectory(exp_dir: Union[str, Path], fmt: str = "obj") -> List[pv.PolyData]:
        """Reload saved frames (geometry + field sidecars) as PolyData with point arrays."""
        exp_dir = Path(exp_dir)
        traj = []
        for f in sorted(exp_dir.glob(f"frame_????.{fmt}")):
            m = pv.read(str(f))
            if not isinstance(m, pv.PolyData):
                m = m.extract_surface()
            m = m.triangulate()
            side = exp_dir / (f.stem + "_fields.npz")
            if side.exists():
                with np.load(side) as z:
                    for k in z.files:
                        arr = z[k]
                        if k == "time":
                            m.field_data["time"] = arr
                        elif len(arr) == m.n_points:
                            m.point_data[k] = arr
            traj.append(m)
        return traj

    def sweep(self, mesh: pv.PolyData, base_params: Dict, param_name: str, values: Sequence[float],
              c0: Optional[np.ndarray] = None, total_time: float = 1.0, prefix: str = "sweep",
              **kwargs) -> Dict[float, ExperimentResult]:
        """One-parameter sweep; returns {value: result}."""
        out = {}
        for v in values:
            p = dict(base_params)
            p[param_name] = v
            out[v] = self.run_experiment(mesh, p, c0, total_time,
                                         exp_name=f"{prefix}_{param_name}={v:g}", **kwargs)
        return out

    @staticmethod
    def compare(results: Dict[str, ExperimentResult], keys=("Helfrich_Energy", "Area_ratio",
                                                            "Sphericity", "Aspect_Ratio",
                                                            "Centroid_Displacement", "c_std", "c_max"),
                log: Optional[Callable[[str], None]] = None, title: str = "comparison"):
        """Table of final metrics across experiments (printed, or sent to ``log``; returned)."""
        rows = {name: {k: r.final_metrics().get(k, float("nan")) for k in keys} for name, r in results.items()}
        w = max(len(n) for n in rows) if rows else 4
        lines = [f"== {title} ==", "name".ljust(w) + " | " + " | ".join(k.rjust(16) for k in keys)]
        for n, r in rows.items():
            lines.append(n.ljust(w) + " | " + " | ".join(f"{r[k]:16.5g}" for k in keys))
        text = "\n".join(lines)
        (log or print)(text)
        return rows

    # ------------------------------------------------------------------ #
    @staticmethod
    def export_animation(result: ExperimentResult, filename: Union[str, Path] = "cell_dynamics.gif",
                         scalars: str = "concentration", fps: int = 20, cmap: str = "plasma",
                         clim: Optional[Tuple[float, float]] = None, show_edges: bool = False,
                         log: Optional[Callable[[str], None]] = None):
        """Off-screen GIF of a trajectory (requires a working VTK render backend).
        Frames may have different vertex counts (adaptive refinement)."""
        emit = log or print
        traj = result.trajectory
        filename = str(filename)
        try:
            plotter = pv.Plotter(off_screen=True)
            plotter.open_gif(filename, fps=fps)
            plotter.set_background("#111111")
            has_scalars = all(scalars in m.point_data for m in traj)
            if clim is None and has_scalars:
                vals = np.concatenate([np.asarray(m.point_data[scalars]) for m in traj])
                clim = (float(vals.min()), float(vals.max()))
            plotter.add_mesh(traj[0], scalars=scalars if has_scalars else None, cmap=cmap, clim=clim,
                             smooth_shading=True, show_edges=show_edges, name="cell",
                             scalar_bar_args={"title": scalars})
            plotter.camera_position = "iso"
            cam = plotter.camera.copy()
            for m in traj:
                plotter.add_mesh(m, scalars=scalars if has_scalars else None, cmap=cmap, clim=clim,
                                 smooth_shading=True, show_edges=show_edges, name="cell",
                                 reset_camera=False, scalar_bar_args={"title": scalars})
                plotter.camera = cam
                plotter.write_frame()
            plotter.close()
            emit(f"Animation saved to {filename}")
        except Exception as exc:  # headless environments
            emit(f"Animation export skipped for {filename} ({exc}).")

    @staticmethod
    def render_side_by_side(results: Dict[str, ExperimentResult], frame_idx: int = -1,
                            scalars: str = "concentration", show_velocity: bool = True):
        plotter = pv.Plotter(shape=(1, len(results)), border=False)
        for i, (name, res) in enumerate(results.items()):
            plotter.subplot(0, i)
            m = res.trajectory[frame_idx]
            plotter.add_text(f"{name}", font_size=10)
            plotter.add_mesh(m, scalars=scalars if scalars in m.point_data else None,
                             cmap="magma", smooth_shading=True,
                             scalar_bar_args={"title": scalars})
            if show_velocity and "velocity" in m.point_data:
                arrows = m.glyph(orient="velocity", scale="speed", factor=0.5)
                plotter.add_mesh(arrows, color="cyan", opacity=0.7)
            plotter.set_background("#1E1E1E")
        plotter.link_views()
        plotter.show()