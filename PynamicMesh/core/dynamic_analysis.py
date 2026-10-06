"""
dynamic_analysis.py  —  : descriptive analyses of the deformation
==================================================================================================
No dynamical model (differential equation) is fitted here: every analysis describes the observed sequence.

1. **Helmholtz–Hodge decomposition of the surface velocity field** (Polthier & Preuß 2003; Bhatia et al.
   2013).  The tangential velocity v_T on the reference surface is split into
        v_T = ∇α  +  J∇β  +  h       (curl-free / divergence-free / harmonic)
   with the potential α (sources/sinks = local area production) and the stream function β (vortices =
   rotational cortical flows), computed on faces with the cotangent Laplacian.  The energy fractions of
   the three components characterise the *type* of motion; the normal component v_n is the growth field.
2. **Finite-time Lyapunov exponents** of the surface flow map Φ_0→t (Haller 2015): per-face deformation
   gradient of the map between the reference and frame t, Cauchy–Green tensor, FTLE = ln√λ_max / t.
   Ridges of the FTLE field are the Lagrangian coherent structures separating regions of the surface with
   different fates.
3. **Shape space**: PCA of the SPHARM coefficients (Kendall-type shape space after the alignment) or of
   the trajectories → principal deformation modes exported as meshes (mean ± 2σ), mean shape, shape-distance
   matrix, classical MDS embedding of the trajectory, recurrence matrix, Shape-DNA (Reuter et al. 2006).
4. **Reduced coordinates**: POD (SVD) of the SPHARM coefficients and of the vertex trajectories — the
   low-dimensional description of the sequence used by 5 and 6.
5. **Change-point detection**: binary segmentation with a BIC penalty on the reduced coordinates
   (multivariate mean-shift model), on their speed and on the ARAP energy series → regimes of the deformation.
6. **Persistent homology** (optional, ripser/gudhi if installed) of the reduced trajectory: cycles in the
   shape trajectory indicate periodic behaviour.

Results/<scene>/DynamicAnalysis/
    Hodge/hodge_T####.vtp, hodge_energy.csv, potentials_T####.npy, components_T####.npy
    FTLE/ftle_T####.npy, ftle.vtp, ftle_summary.csv
    ShapeSpace/modes.npy, mode_meshes/, distance_matrix.npy, mds.csv, recurrence.npy, shape_dna_distance.npy
    ReducedCoordinates/<source>_coords.npy, <source>_energy.csv
    ChangePoints/change_points.json
    Topology/persistence.csv (optional)
    plots/
    report.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.spatial.distance import pdist, squareform

from PynamicMesh.core.dyn_common import pbar
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots
from PynamicMesh.core.dyn_common import (find_frame_times, 
    logged_stage, progress,
    xp, GPU_AVAILABLE, to_gpu, to_cpu, ensure_dir, cotan_laplacian, face_gradient_operator, face_normals_areas,
    vertex_normals, vertex_areas, mean_curvature, gaussian_curvature, laplacian_eigenbasis, write_obj, save_vtp,
    icosphere, smooth_derivative, kabsch_rotation
)


# ----------------------------------------------------------------------------- #
#  1. Helmholtz–Hodge decomposition
# ----------------------------------------------------------------------------- #

class SurfaceHodge:
    """Discrete HHD of face-based tangential vector fields on a fixed triangle mesh."""

    def __init__(self, V, F):
        self.V, self.F = V, F
        self.G, self.n_f, self.area = face_gradient_operator(V, F)            # (3F, n)
        nf = F.shape[0]
        self.Af = sparse.diags(np.repeat(self.area, 3))
        self.W = (self.G.T @ self.Af @ self.G).tocsc()                        # cotangent stiffness (= Σ A ∇φ·∇φ)
        n = V.shape[0]
        self.Wreg = (self.W + 1e-9 * abs(self.W.diagonal()).mean() * sparse.identity(n, format="csc"))
        self.lu = splinalg.splu(self.Wreg)
        # J: rotation by 90° in each face (n_f × g)
        self.J = self._J_operator()

    def _J_operator(self):
        nf = self.F.shape[0]
        n = self.n_f
        rows, cols, vals = [], [], []
        # (n × g)_a = ε_abc n_b g_c
        eps = np.zeros((3, 3, 3)); eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1; eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if eps[a, b, c] != 0:
                        rows.append(3 * np.arange(nf) + a); cols.append(3 * np.arange(nf) + c); vals.append(eps[a, b, c] * n[:, b])
        return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * nf, 3 * nf)).tocsr()

    def face_field_from_vertices(self, vel):
        """Vertex velocities → face-averaged tangential (and normal) components."""
        vf = vel[self.F].mean(axis=1)
        vn = np.einsum("ij,ij->i", vf, self.n_f)
        vt = vf - vn[:, None] * self.n_f
        return vt, vn

    def decompose(self, vt):
        """vt (F,3) tangential → dict with α, β (vertex), the three face components and energy fractions."""
        v = vt.ravel()
        rhs_a = self.G.T @ (self.Af @ v)                     # ∫ ∇φ·v  = −∫ φ div v  (weak divergence)
        alpha = self.lu.solve(rhs_a); alpha -= alpha.mean()
        JG = self.J @ self.G
        rhs_b = JG.T @ (self.Af @ v)                         # weak curl
        beta = self.lu.solve(rhs_b); beta -= beta.mean()
        grad = (self.G @ alpha).reshape(-1, 3)
        cograd = (JG @ beta).reshape(-1, 3)
        harm = vt - grad - cograd

        def energy(f):
            return float(np.sum(self.area * np.sum(f ** 2, axis=1)))
        Etot = max(energy(vt), 1e-300)
        return {"alpha": alpha, "beta": beta, "curl_free": grad, "div_free": cograd, "harmonic": harm,
                "E_total": Etot, "frac_curl_free": energy(grad) / Etot, "frac_div_free": energy(cograd) / Etot,
                "frac_harmonic": energy(harm) / Etot, "divergence": self._vertex_div(vt)}

    def _vertex_div(self, vt):
        """Vertex divergence  div v_i = −(1/A_i) Σ_f A_f ∇φ_i · v_f."""
        d = -(self.G.T @ (self.Af @ vt.ravel()))
        return d / np.maximum(vertex_areas(self.V, self.F), 1e-300)


def hodge_analysis(X, F, vel, out_dir, plots=True, verbose=True):
    """HHD of the velocity field of every frame, on the frame's own geometry (moving surface)."""
    out = ensure_dir(out_dir)
    T = X.shape[0]
    rows = []
    alpha_all, beta_all = np.zeros((T, X.shape[1])), np.zeros((T, X.shape[1]))
    for k in pbar(range(T), desc="Hodge decomposition"):
        hh = SurfaceHodge(X[k], F)
        vt, vn = hh.face_field_from_vertices(vel[k])
        dec = hh.decompose(vt)
        alpha_all[k], beta_all[k] = dec["alpha"], dec["beta"]
        En = float(np.sum(hh.area * vn ** 2)); Et = dec["E_total"]
        rows.append({"frame": k, "E_tangential": Et, "E_normal": En, "normal_fraction": En / max(En + Et, 1e-300),
                     "frac_curl_free": dec["frac_curl_free"], "frac_div_free": dec["frac_div_free"], "frac_harmonic": dec["frac_harmonic"],
                     "max_divergence": float(np.abs(dec["divergence"]).max()), "mean_divergence": float(dec["divergence"].mean())})
        np.save(out / f"potentials_T{k:04d}.npy", np.stack([dec["alpha"], dec["beta"], dec["divergence"]], axis=1))
        np.save(out / f"components_T{k:04d}.npy", np.stack([vt, dec["curl_free"], dec["div_free"], dec["harmonic"]]))
        save_vtp(out / f"hodge_T{k:04d}.vtp", X[k], F, {"alpha_potential": dec["alpha"], "beta_stream": dec["beta"],
                                                        "divergence": dec["divergence"], "velocity": vel[k],
                                                        "normal_speed": np.einsum("ij,ij->i", vel[k], vertex_normals(X[k], F))})
    df = pd.DataFrame(rows); df.to_csv(out / "hodge_energy.csv", index=False)
    if plots:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.stackplot(df["frame"], df["frac_curl_free"], df["frac_div_free"], df["frac_harmonic"],
                     labels=["curl-free grad(α) (sources/sinks)", "div-free J grad(β) (vortices)", "harmonic"], alpha=0.85)
        ax.plot(df["frame"], df["normal_fraction"], "k--", label="normal energy fraction (growth)")
        ax.set_ylim(0, 1); ax.set_xlabel("frame"); ax.set_ylabel("fraction of kinetic energy"); ax.legend(fontsize=8, loc="upper right")
        ax.set_title("Helmholtz–Hodge decomposition of the surface velocity"); fig.tight_layout()
        fig.savefig(Path(out).parent / "plots" / "1_hodge_energy_fractions.png", dpi=200); plt.close(fig)
    if verbose:
        print(f"Hodge: mean fractions curl-free {df['frac_curl_free'].mean():.2f}, div-free {df['frac_div_free'].mean():.2f}, "
              f"harmonic {df['frac_harmonic'].mean():.2f}; normal (growth) share {df['normal_fraction'].mean():.2f}")
    return df, alpha_all, beta_all


# ----------------------------------------------------------------------------- #
#  2. Finite-time Lyapunov exponents
# ----------------------------------------------------------------------------- #

def ftle_analysis(X, F, t, out_dir, plots=True, verbose=True):
    """Per-face deformation gradient of Φ_{0→t}, Cauchy–Green eigenvalues → FTLE fields (vertex averaged)."""
    out = ensure_dir(out_dir)
    T, n, _ = X.shape
    V0 = X[0]
    tri0 = V0[F]; e1 = tri0[:, 1] - tri0[:, 0]; e2 = tri0[:, 2] - tri0[:, 0]
    # orthonormal tangent basis of each reference face
    b1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-300)[:, None]
    nrm = np.cross(e1, e2); nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    b2 = np.cross(nrm, b1)
    E0 = np.stack([np.stack([np.einsum("ij,ij->i", e1, b1), np.einsum("ij,ij->i", e1, b2)], 1),
                   np.stack([np.einsum("ij,ij->i", e2, b1), np.einsum("ij,ij->i", e2, b2)], 1)], axis=2)  # (F,2,2) columns = edges in 2D
    E0inv = np.linalg.inv(E0 + 1e-14 * np.eye(2)[None])
    _, a0 = face_normals_areas(V0, F)
    ftle_v = np.zeros((T, n)); rows = []
    for k in pbar(range(1, T), desc="FTLE"):
        trik = X[k][F]
        Ek = np.stack([trik[:, 1] - trik[:, 0], trik[:, 2] - trik[:, 0]], axis=2)     # (F,3,2)
        Fg = Ek @ E0inv                                                                # (F,3,2) deformation gradient
        C = np.transpose(Fg, (0, 2, 1)) @ Fg                                           # (F,2,2) Cauchy–Green
        lam = np.linalg.eigvalsh(C)
        tau = max(t[k] - t[0], 1e-12)
        ftle_f = np.log(np.sqrt(np.maximum(lam[:, -1], 1e-300))) / tau
        strain_f = np.log(np.sqrt(np.maximum(lam[:, -1], 1e-300)))
        acc = np.zeros(n); wsum = np.zeros(n)
        for c in range(3):
            np.add.at(acc, F[:, c], ftle_f * a0); np.add.at(wsum, F[:, c], a0)
        ftle_v[k] = acc / np.maximum(wsum, 1e-300)
        np.save(out / f"ftle_T{k:04d}.npy", ftle_v[k])
        rows.append({"frame": k, "ftle_mean": float(np.average(ftle_f, weights=a0)), "ftle_max": float(ftle_f.max()),
                     "ftle_p95": float(np.percentile(ftle_f, 95)), "max_stretch_ratio": float(np.exp(strain_f.max())),
                     "area_ratio_mean": float(np.average(np.sqrt(np.prod(np.maximum(lam, 1e-300), axis=1)), weights=a0))})
    df = pd.DataFrame(rows); df.to_csv(out / "ftle_summary.csv", index=False)
    save_vtp(out / "ftle.vtp", V0, F, {f"ftle_T{k:04d}": ftle_v[k] for k in range(1, T)} | {"ftle_final": ftle_v[-1]})
    if plots and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(df["frame"], df["ftle_mean"], "o-", label="mean FTLE"); ax.plot(df["frame"], df["ftle_p95"], "s--", label="95th percentile")
        ax.set_xlabel("frame"); ax.set_ylabel("FTLE [1/time]"); ax.legend(); ax.set_title("finite-time Lyapunov exponents of the surface flow map")
        fig.tight_layout(); fig.savefig(Path(out).parent / "plots" / "3_ftle.png", dpi=200); plt.close(fig)
    if verbose and not df.empty:
        print(f"FTLE: final mean {df['ftle_mean'].iloc[-1]:.3g}, max stretch ratio {df['max_stretch_ratio'].iloc[-1]:.3g}")
    return ftle_v, df


# ----------------------------------------------------------------------------- #
#  3. Shape space
# ----------------------------------------------------------------------------- #

def _median_error_at(csv, degree=None):
    """Median relative reconstruction error of a reconstruction_error.csv at `degree` (default: its highest degree)."""
    if not Path(csv).exists():
        return None
    e = pd.read_csv(csv)
    d = e.degree.max() if degree is None else degree
    sub = e[e.degree == d]
    return float(sub.rel_error.median()) if len(sub) else None


def manifold_harmonic_coordinates(X, F0, k=60, align=True, max_error=0.05, max_worst_error=0.15,
                                  candidates=(60, 120, 200, 300)):
    """
    Intrinsic shape coordinates of the tracked mesh: every frame (rigidly aligned to the reference, so the
    coordinates describe the SHAPE and not the pose) projected on the first k Laplace–Beltrami eigenfunctions of the
    reference mesh, a_t = Φᵀ M Y_t. No sphere map is needed (any topology, any complexity) and the coordinates are
    consistent in time by construction (one basis, one triangulation). Returns (A (T, k, 3), Φ, centroid,
    relative reconstruction error per frame, 95th-percentile per-vertex error per frame, chosen k).
    k='auto': the eigenbasis is computed once for the largest candidate and the SMALLEST number of modes whose median
    error ≤ max_error / 2 and worst-region (95th percentile) error ≤ max_worst_error is kept (thin parts such as legs
    need many modes, cells few).
    """
    X = np.asarray(X, dtype=np.float64); F0 = np.asarray(F0, dtype=np.int64)
    W, M, _ = cotan_laplacian(X[0], F0)
    auto = (k == "auto")
    kmax = int(min(max(candidates) if auto else k, X.shape[1] - 2))
    _, Phi_all = laplacian_eigenbasis(W, M, kmax)
    Md = np.asarray(M.diagonal()).ravel()
    c0 = (X[0] * Md[:, None]).sum(0) / Md.sum()
    P0 = X[0] - c0
    Ys = []
    for Xt in X:
        Y = Xt - (Xt * Md[:, None]).sum(0) / Md.sum()
        if align:
            Y = Y @ kabsch_rotation(Y, P0, Md).T
        Ys.append(Y)

    def evaluate(kk):
        Phi = Phi_all[:, :kk]; A, med, p95 = [], [], []
        for Y in Ys:
            a = Phi.T @ (Md[:, None] * Y); e = np.sqrt(np.sum((Y - Phi @ a) ** 2, 1))
            sp = max(np.sqrt(np.mean(np.sum(Y ** 2, 1))), 1e-300)
            A.append(a); med.append(float(np.sqrt(np.mean(e ** 2)) / sp)); p95.append(float(np.percentile(e, 95) / sp))
        return np.array(A), Phi, np.array(med), np.array(p95)
    ks = sorted({min(c, kmax) for c in candidates}) if auto else [kmax]
    for kk in ks:
        A, Phi, med, p95 = evaluate(kk)
        if np.median(med) <= max_error / 2 and np.median(p95) <= max_worst_error:
            break
    return A, Phi, c0, med, p95, kk


def shape_coordinates(target_folder, X=None, F0=None, frames=None, source="auto", max_error=0.05, mh_modes="auto",
                      verbose=True, max_worst_error=0.15):
    """
    Shape coordinates used by the shape space, the reduced coordinates, the change points and the homology, chosen by
    the QUALITY of the representation (the median relative reconstruction error of the shapes):
      'spharm'     least-squares SPHARM of the parametrization (aligned coefficients)
      'spharm_hd'  high-degree SPHARM (fast spherical harmonic transform, Parametrization/SPHARM_HD)
      'manifold'   manifold harmonics of the tracked mesh (manifold_harmonic_coordinates)
      'trajectories' the aligned vertex positions themselves
    source='auto': the SPHARM representations are used only when their error is at most max_error (high-degree
    preferred), otherwise the manifold harmonics of the trajectories (complex shapes: the sphere map cannot represent
    them well), otherwise the trajectories. Returns a dict with label, Z (T, D), errors, reason and a mesh synthesiser.
    """
    par = Path(target_folder) / "Parametrization"
    cand, errors, worst = {}, {}, {}
    f_ls = par / "SPHARM" / "coeffs_consistent.npy"
    if f_ls.exists():
        info = json.load(open(par / "SPHARM" / "basis_info.json")) if (par / "SPHARM" / "basis_info.json").exists() else {}
        e = _median_error_at(par / "SPHARM" / "reconstruction_error.csv", info.get("L_max"))
        cand["spharm"] = lambda: np.load(f_ls).reshape(np.load(f_ls).shape[0], -1); errors["spharm"] = e
    f_hd = par / "SPHARM_HD" / "coeffs_hd_consistent.npy"
    if f_hd.exists():
        e = _median_error_at(par / "SPHARM_HD" / "reconstruction_error.csv")
        eh = pd.read_csv(par / "SPHARM_HD" / "reconstruction_error.csv") if (par / "SPHARM_HD" / "reconstruction_error.csv").exists() else None
        if eh is not None and "p95_rel_error" in eh:
            worst["spharm_hd"] = float(eh[eh.degree == eh.degree.max()].p95_rel_error.median())

        def _hd():
            C = np.load(f_hd)                                         # (T, 3, L+1, L+1) complex
            L = C.shape[-1] - 1
            mask = np.tril(np.ones((L + 1, L + 1), dtype=bool))       # m <= l
            Cm = C[:, :, mask]
            return np.concatenate([Cm.real, Cm.imag], axis=-1).reshape(C.shape[0], -1)
        cand["spharm_hd"] = _hd; errors["spharm_hd"] = e
    mh = None
    if X is not None and F0 is not None and source in ("auto", "manifold"):
        mh = manifold_harmonic_coordinates(X, F0, mh_modes, max_error=max_error, max_worst_error=max_worst_error)
        errors["manifold"] = float(np.median(mh[3])); worst["manifold"] = float(np.median(mh[4]))
    if X is not None:
        errors.setdefault("trajectories", 0.0)
    if source == "auto":
        ok = [s_ for s_ in ("spharm_hd", "spharm") if errors.get(s_) is not None and errors[s_] <= max_error
              and (worst.get(s_) is None or worst[s_] <= max_worst_error)]
        if ok:
            chosen = min(ok, key=lambda s_: errors[s_])
            reason = (f"{chosen}: reconstruction error {errors[chosen]:.4f} <= {max_error}" +
                      (f", worst regions {worst[chosen]:.4f} <= {max_worst_error}" if chosen in worst else
                       " (no per-vertex error available: median only)"))
        elif mh is not None:
            chosen = "manifold"
            bad = {k_: round(v, 4) for k_, v in errors.items() if k_ in ("spharm", "spharm_hd") and v is not None}
            reason = ("manifold harmonics of the trajectories" + f" ({mh[5]} modes): " +
                      (f"SPHARM errors {bad} (median) / {({k_: round(v, 4) for k_, v in worst.items() if k_ != 'manifold'})} "
                       f"(worst regions) above {max_error} / {max_worst_error}" if bad else "no SPHARM coefficients"))
        elif "spharm_hd" in cand or "spharm" in cand:
            chosen = "spharm_hd" if "spharm_hd" in cand else "spharm"; reason = f"{chosen}: only representation available"
        elif X is not None:
            chosen = "trajectories"; reason = "trajectories: no other representation"
        else:
            return None
    else:
        chosen = source; reason = f"{source}: requested"
    synth = None
    if chosen == "manifold":
        A, Phi, c0, err, _, _ = mh
        Z = A.reshape(A.shape[0], -1)
        synth = lambda z: (Phi @ z.reshape(-1, 3) + c0, F0)
    elif chosen == "trajectories":
        Z = np.asarray(X).reshape(X.shape[0], -1)
        synth = lambda z: (z.reshape(-1, 3), F0)
    elif chosen in cand:
        Z = cand[chosen]()
    else:
        raise ValueError(f"shape source '{chosen}' not available (available: {sorted(cand) + (['manifold'] if mh else [])})")
    label = {"spharm": "SPHARM_coefficients", "spharm_hd": "SPHARM_HD_coefficients", "manifold": "manifold_harmonics",
             "trajectories": "vertex_trajectories"}[chosen]
    if verbose:
        print(f"Shape coordinates: {reason}")
    return {"source": chosen, "label": label, "Z": Z, "errors": errors, "worst_region_errors": worst, "reason": reason,
            "synth": synth, "max_error": max_error, "max_worst_error": max_worst_error,
            "mh_modes_used": None if mh is None else int(mh[5])}


def shape_space_analysis(target_folder, out_dir, X=None, F0=None, frames=None, n_modes=4, sphere_subdiv=4, plots=True, verbose=True,
                         selection=None):
    out = ensure_dir(out_dir); d_modes = ensure_dir(out / "mode_meshes")
    res = {}
    cfile = Path(target_folder) / "Parametrization" / "SPHARM" / "coeffs_consistent.npy"
    source = None
    if selection is not None:                          # quality-aware choice (shape_coordinates)
        Z = selection["Z"]; source = selection["source"]
        with open(out / "source.json", "w", encoding="utf-8") as fh:
            json.dump({k_: v for k_, v in selection.items() if k_ in ("source", "label", "errors", "worst_region_errors", "reason",
                                                                       "max_error", "max_worst_error", "mh_modes_used")},
                      fh, indent=2)
        if source == "spharm":
            L = json.load(open(Path(target_folder) / "Parametrization" / "SPHARM" / "basis_info.json"))["L_max"]
    elif cfile.exists():
        C = np.load(cfile); Z = C.reshape(C.shape[0], -1); source = "spharm"
        info = json.load(open(Path(target_folder) / "Parametrization" / "SPHARM" / "basis_info.json"))
        L = info["L_max"]
    elif X is not None:
        Z = X.reshape(X.shape[0], -1); source = "trajectories"
    else:
        return None
    T = Z.shape[0]
    mean = Z.mean(axis=0); Zc = Z - mean
    _, s, Vt = (to_cpu(v) for v in xp.linalg.svd(to_gpu(Zc), full_matrices=False))
    var = s ** 2 / max(np.sum(s ** 2), 1e-300)
    scores = Zc @ Vt.T
    n_modes = int(min(n_modes, len(s)))
    np.save(out / "modes.npy", Vt[:n_modes]); np.save(out / "scores.npy", scores[:, :n_modes]); np.save(out / "mean_shape.npy", mean)
    pd.DataFrame({"mode": np.arange(len(s)), "std": s / np.sqrt(max(T - 1, 1)), "variance_fraction": var,
                  "cumulative": np.cumsum(var)}).to_csv(out / "variance.csv", index=False)
    # mode meshes
    if source == "spharm":
        from PynamicMesh.core.parametrization import spharm_synthesize, real_sh_basis
        lm = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]
        template = icosphere(sphere_subdiv)
        Vm, Fm = spharm_synthesize(mean.reshape(-1, 3), lm, template=template)
        write_obj(d_modes / "mean_shape.obj", Vm, Fm)
        for j in range(n_modes):
            sd = s[j] / np.sqrt(max(T - 1, 1))
            for sgn, tag in ((-2, "minus2sd"), (2, "plus2sd")):
                Vj, _ = spharm_synthesize((mean + sgn * sd * Vt[j]).reshape(-1, 3), lm, template=template)
                write_obj(d_modes / f"mode{j+1}_{tag}.obj", Vj, Fm)
    elif selection is not None and selection.get("synth") is not None:   # manifold harmonics / trajectories
        synth = selection["synth"]
        Vm, Fm = synth(mean); write_obj(d_modes / "mean_shape.obj", Vm, Fm)
        for j in range(n_modes):
            sd = s[j] / np.sqrt(max(T - 1, 1))
            for sgn, tag in ((-2, "minus2sd"), (2, "plus2sd")):
                Vj, _ = synth(mean + sgn * sd * Vt[j]); write_obj(d_modes / f"mode{j+1}_{tag}.obj", Vj, Fm)
    elif source == "spharm_hd":
        pass                                           # high-degree modes: coefficients only (modes.npy)
    else:
        write_obj(d_modes / "mean_shape.obj", mean.reshape(-1, 3), F0)
        for j in range(n_modes):
            sd = s[j] / np.sqrt(max(T - 1, 1))
            for sgn, tag in ((-2, "minus2sd"), (2, "plus2sd")):
                write_obj(d_modes / f"mode{j+1}_{tag}.obj", (mean + sgn * sd * Vt[j]).reshape(-1, 3), F0)
    # distances, MDS, recurrence
    Dm = squareform(pdist(Z)); np.save(out / "distance_matrix.npy", Dm)
    mds = _classical_mds(Dm, 2); pd.DataFrame({"frame": np.arange(T), "mds1": mds[:, 0], "mds2": mds[:, 1]}).to_csv(out / "mds.csv", index=False)
    eps = np.percentile(Dm[np.triu_indices(T, 1)], 20) if T > 2 else 0
    rec = (Dm <= eps).astype(np.int8); np.save(out / "recurrence.npy", rec)
    res.update({"source": source, "variance_fraction": var[:n_modes].tolist(), "scores": scores, "distance_matrix": Dm, "mds": mds})
    # Shape-DNA from frames
    if frames is not None:
        spectra = []
        for fr in frames:
            W, M, _ = cotan_laplacian(fr["V"], fr["F"])
            ev, _ = laplacian_eigenbasis(W, M, 30, fr.get("mesh"))
            _, area = face_normals_areas(fr["V"], fr["F"])
            spectra.append(ev[1:] * area.sum())                          # scale normalised (λ·Area invariant)
        S = np.array([sp[:min(map(len, spectra))] for sp in spectra])
        Ddna = squareform(pdist(S)); np.save(out / "shape_dna_distance.npy", Ddna)
        pd.DataFrame(S, columns=[f"lambda_{k+1}" for k in range(S.shape[1])]).to_csv(out / "shape_dna_spectra.csv", index_label="frame")
        res["shape_dna"] = Ddna
    if plots:
        _plots_shape(out, s, var, scores, mds, Dm, rec, res.get("shape_dna"))
    if verbose:
        print(f"Shape space ({source}): first {n_modes} modes explain {np.cumsum(var)[n_modes-1]*100:.1f}% of the variance")
    return res


def _classical_mds(D, k=2):
    n = D.shape[0]; J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    ev, evec = np.linalg.eigh(B); order = np.argsort(ev)[::-1]
    ev, evec = np.maximum(ev[order][:k], 0), evec[:, order][:, :k]
    return evec * np.sqrt(ev)[None, :]


def _plots_shape(out, s, var, scores, mds, Dm, rec, dna):
    d = Path(out).parent / "plots"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].bar(np.arange(1, min(len(var), 12) + 1), var[:12]); axes[0, 0].set_title("variance explained per shape mode"); axes[0, 0].set_xlabel("mode")
    T = scores.shape[0]
    sc = axes[0, 1].scatter(scores[:, 0], scores[:, 1] if scores.shape[1] > 1 else np.zeros(T), c=np.arange(T), cmap="viridis")
    axes[0, 1].plot(scores[:, 0], scores[:, 1] if scores.shape[1] > 1 else np.zeros(T), "k-", lw=0.7)
    axes[0, 1].set_title("shape trajectory in the first two modes"); axes[0, 1].set_xlabel("mode 1"); axes[0, 1].set_ylabel("mode 2"); fig.colorbar(sc, ax=axes[0, 1], label="frame")
    im = axes[1, 0].imshow(Dm, cmap="magma"); axes[1, 0].set_title("shape distance matrix"); fig.colorbar(im, ax=axes[1, 0])
    axes[1, 1].imshow(rec, cmap="Greys", interpolation="nearest"); axes[1, 1].set_title("recurrence plot (20th-percentile threshold)")
    for a in (axes[1, 0], axes[1, 1]):
        a.set_xlabel("frame"); a.set_ylabel("frame")
    fig.tight_layout(); fig.savefig(d / "4_shape_space.png", dpi=200); plt.close(fig)
    if dna is not None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(dna, cmap="viridis"); ax.set_title("Shape-DNA distance (LB spectra)"); fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(d / "5_shape_dna.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- #
#  4-5. Reduced coordinates and change points
# ----------------------------------------------------------------------------- #

def binary_segmentation(Y, min_size=3, penalty=None, max_cp=6):
    """Mean-shift change points on a (T, d) series with a BIC-type penalty (Killick-style cost = Σ RSS)."""
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    T, dim = Y.shape
    if penalty is None:
        sig2 = np.var(np.diff(Y, axis=0), axis=0).mean() / 2 + 1e-12
        penalty = 2 * dim * sig2 * np.log(T)

    def cost(a, b):
        seg = Y[a:b]; return float(np.sum((seg - seg.mean(axis=0)) ** 2))
    cps = []
    segments = [(0, T)]
    while len(cps) < max_cp:
        best = None
        for (a, b) in segments:
            if b - a < 2 * min_size:
                continue
            base = cost(a, b)
            for c in range(a + min_size, b - min_size + 1):
                gain = base - cost(a, c) - cost(c, b)
                if best is None or gain > best[0]:
                    best = (gain, c, (a, b))
        if best is None or best[0] <= penalty:
            break
        gain, c, (a, b) = best
        cps.append(int(c)); segments.remove((a, b)); segments += [(a, c), (c, b)]
    return sorted(cps), penalty


def pod_coordinates(Z, energy=0.99, max_rank=8):
    """POD of a (T, d) state matrix: mean-subtracted SVD, rank by cumulative energy (≤ max_rank). Returns (a, energy fractions)."""
    Zc = Z - Z.mean(axis=0)
    U, s_, _ = np.linalg.svd(Zc, full_matrices=False)
    frac = s_ ** 2 / max(float((s_ ** 2).sum()), 1e-300)
    r = int(min(max_rank, np.searchsorted(np.cumsum(frac), energy) + 1, len(s_)))
    return U[:, :r] * s_[:r], frac


def reduced_coordinates(target_folder, out_dir, X=None, energy=0.99, max_rank=8, verbose=True, selection=None):
    """
    Low-dimensional coordinates of the sequence: POD of the aligned SPHARM coefficients (Parametrization) and of
    the vertex trajectories (Trajectories).  Written to ReducedCoordinates/ and returned as {source: a (T, r)}.
    """
    out = ensure_dir(out_dir); res = {}
    cfile = Path(target_folder) / "Parametrization" / "SPHARM" / "coeffs_consistent.npy"
    if selection is not None:                          # the quality-aware shape coordinates
        if selection["label"] != "vertex_trajectories":
            res[selection["label"]] = selection["Z"]
    elif cfile.exists():
        C = np.load(cfile); res["SPHARM_coefficients"] = C.reshape(C.shape[0], -1)
    if X is None:
        xf = Path(target_folder) / "Trajectories" / "trajectories.npy"
        X = np.load(xf) if xf.exists() else None
    if X is not None:
        res["vertex_trajectories"] = np.asarray(X).reshape(X.shape[0], -1)
    coords = {}
    for label, Z in res.items():
        if Z.shape[0] < 3:
            continue
        a, frac = pod_coordinates(Z, energy, max_rank)
        np.save(out / f"{label}_coords.npy", a)
        pd.DataFrame({"mode": np.arange(1, len(frac) + 1), "energy_fraction": frac, "cumulative": np.cumsum(frac)}).to_csv(out / f"{label}_energy.csv", index=False)
        coords[label] = a
        if verbose:
            print(f"Reduced coordinates [{label}]: rank {a.shape[1]} ({100 * frac[:a.shape[1]].sum():.1f}% of the variance)")
    return coords


def change_point_confidence(Y, cps, n_perm=500, seed=0, min_size=3):
    """
    Permutation p-value of every change point: within the segment between its neighbouring change points, the gain
    of the split (RSS reduction) is compared with the best split gain of n_perm random permutations of that segment
    (no regime structure). p = (1 + #permutations with a gain ≥ the observed one) / (1 + n_perm).
    """
    Y = np.asarray(Y, dtype=np.float64).reshape(len(Y), -1)
    rng = np.random.default_rng(seed); T = len(Y); b = [0] + sorted(cps) + [T]; pv = []

    def best_gain(seg):
        tot = np.sum((seg - seg.mean(0)) ** 2); best = 0.0
        for c in range(min_size, len(seg) - min_size + 1):
            g = tot - np.sum((seg[:c] - seg[:c].mean(0)) ** 2) - np.sum((seg[c:] - seg[c:].mean(0)) ** 2); best = max(best, g)
        return best
    for i, c in enumerate(sorted(cps)):
        seg = Y[b[i]:b[i + 2]]; k = c - b[i]
        tot = np.sum((seg - seg.mean(0)) ** 2)
        obs = tot - np.sum((seg[:k] - seg[:k].mean(0)) ** 2) - np.sum((seg[k:] - seg[k:].mean(0)) ** 2)
        null = [best_gain(seg[rng.permutation(len(seg))]) for _ in range(int(n_perm))]
        pv.append(float((1 + np.sum(np.array(null) >= obs)) / (1 + n_perm)))
    return pv


def change_point_analysis(target_folder, out_dir, X=None, plots=True, verbose=True, coords=None, n_perm=500, confidence=None):
    out = ensure_dir(out_dir)
    result = {}
    coords = coords if coords is not None else reduced_coordinates(target_folder, Path(out).parent / "ReducedCoordinates", X, verbose=False)
    for label, a in coords.items():
        if a.shape[0] >= 4:
            cps, pen = binary_segmentation(a)
            da = np.linalg.norm(smooth_derivative(a, np.arange(a.shape[0], dtype=float)), axis=1)
            cps_speed, _ = binary_segmentation(da[:, None])
            result[label] = {"mean_shift_change_points": cps, "speed_change_points": cps_speed, "penalty": float(pen)}
            if n_perm:
                result[label]["mean_shift_p_values"] = change_point_confidence(a, cps, n_perm)
    # joint detection: every reduced coordinate set and the ARAP energy, standardised, in one multivariate series
    blocks = [((a - a.mean(0)) / np.maximum(a.std(0), 1e-300)) for a in coords.values() if a.shape[0] >= 4]
    arap = Path(target_folder) / "Trajectories" / "kinematics" / "arap_energy.csv"
    if arap.exists():
        e = pd.read_csv(arap)["arap_energy_area_normalised"].values
        cps, pen = binary_segmentation(np.log(e + 1e-16)[:, None], min_size=2)
        result["arap_energy"] = {"change_points": cps, "penalty": float(pen)}
        if n_perm:
            result["arap_energy"]["p_values"] = change_point_confidence(np.log(e + 1e-16), cps, n_perm, min_size=2)
        le = np.log(e + 1e-16)
        if blocks and len(le) == blocks[0].shape[0]:
            blocks.append(((le - le.mean()) / max(le.std(), 1e-300))[:, None])
    if blocks and all(bk.shape[0] == blocks[0].shape[0] for bk in blocks):
        J = np.concatenate(blocks, axis=1)
        cps, pen = binary_segmentation(J)
        result["joint"] = {"change_points": cps, "penalty": float(pen), "signals": len(blocks)}
        if n_perm:
            result["joint"]["p_values"] = change_point_confidence(J, cps, n_perm)
    if confidence is not None:                         # change points at frames with unreliable correspondences
        low = set(int(f) for f in confidence)
        for r in result.values():
            for key in [k_ for k_ in r if k_.endswith("change_points")]:
                r[key.replace("change_points", "at_low_confidence")] = [int(c) for c in r[key] if c in low or c - 1 in low]
    with open(out / "change_points.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    if plots and result:
        fig, ax = plt.subplots(figsize=(9, 4))
        for label, r in result.items():
            if label in ("arap_energy", "joint") or label not in coords:
                continue
            a = coords[label]; ax.plot(a[:, 0] / (np.abs(a[:, 0]).max() + 1e-12), label=f"{label} a1 (normalised)")
            for c in r["mean_shift_change_points"]:
                ax.axvline(c, color="r", ls="--", alpha=0.7)
        for c in result.get("joint", {}).get("change_points", []):
            ax.axvline(c, color="k", ls=":", lw=1.5, alpha=0.8)
        ax.set_xlabel("frame"); ax.legend(fontsize=8); ax.set_title("change points of the deformation regime (red: per signal, black dotted: joint)")
        fig.tight_layout(); fig.savefig(Path(out).parent / "plots" / "6_change_points.png", dpi=200); plt.close(fig)
    if verbose:
        print(f"Change points: {json.dumps({k: v.get('mean_shift_change_points', v.get('change_points')) for k, v in result.items()})}")
    return result


# ----------------------------------------------------------------------------- #
#  6. Persistent homology
# ----------------------------------------------------------------------------- #

def persistent_homology(target_folder, out_dir, verbose=True, coords=None, prefer=None):
    coords = coords if coords is not None else reduced_coordinates(target_folder, Path(out_dir).parent / "ReducedCoordinates", verbose=False)
    a = coords.get(prefer) if prefer else None
    a = a if a is not None else coords.get("SPHARM_coefficients", coords.get("vertex_trajectories"))
    if a is None:
        return None
    try:
        from ripser import ripser
        dg = ripser(a, maxdim=1)["dgms"]
    except Exception:
        try:
            import gudhi
            rc = gudhi.RipsComplex(points=a); st = rc.create_simplex_tree(max_dimension=2); st.persistence()
            dg = [np.array([p[1] for p in st.persistence_intervals_in_dimension(0)]), np.array([p for p in st.persistence_intervals_in_dimension(1)])]
        except Exception:
            if verbose:
                print("Persistent homology skipped (install `ripser` or `gudhi`).")
            return None
    out = ensure_dir(out_dir)
    rows = [{"dimension": dim, "birth": b, "death": d, "persistence": d - b} for dim, D in enumerate(dg) for b, d in np.atleast_2d(D) if np.isfinite(d)]
    df = pd.DataFrame(rows); df.to_csv(out / "persistence.csv", index=False)
    h1 = df[df["dimension"] == 1]
    if verbose:
        print(f"Persistent homology: {len(h1)} H1 classes; most persistent loop {h1['persistence'].max() if len(h1) else 0:.3g} (periodicity indicator)")
    return df


# ----------------------------------------------------------------------------- #
#  Driver
# ----------------------------------------------------------------------------- #

def _add_change_point_times(cp, t):
    """Adds the acquisition time of every change-point frame to the change-point dict (keys ending with 'frames' /
    'frame' / 'change_points' holding frame indices get a twin '<key>_time')."""
    out = dict(cp)
    for k, v in cp.items():
        if isinstance(v, dict):
            out[k] = _add_change_point_times(v, t)
        elif isinstance(v, (list, tuple)) and k.endswith(("change_points", "frames")) and \
                all(isinstance(x, (int, np.integer)) and 0 <= int(x) < len(t) for x in v):
            out[f"{k}_time"] = [float(t[int(x)]) for x in v]
        elif isinstance(v, (int, np.integer)) and "frame" in k and 0 <= int(v) < len(t):
            out[f"{k}_time"] = float(t[int(v)])
    return out


@logged_stage("DynamicAnalysis", (0, "target_folder"), "DynamicAnalysis")
def compute_dynamic_analysis(target_folder, dt=1.0, frames=None, X=None, F0=None, hodge=True, ftle=True,
                             shape_space=True, change_points=True, homology=True, n_shape_modes=4, pod_energy=0.99,
                             shape_source="auto", shape_max_error=0.05, mh_modes="auto", spherical_spectra=True,
                             spectral_lmax=32, sht_backend="auto", hodge_nonrigid=True, cp_permutations=500,
                             shape_max_worst_error=0.15,
                             pod_max_rank=8, plots=True, verbose=True, times=None):
    """
    Descriptive analyses  of the trajectories  and the SPHARM coefficients.
    times: acquisition times of the frames (FrameTimes or array). Default: the times of the trajectories
    (Trajectories/time.npy, = k * dt unless frame times were given), so the velocity, the FTLE (1 / time unit) and the
    change points are expressed in the real time of the time-lapse whenever it is known.
    """
    target_folder = Path(target_folder)
    out = ensure_dir(target_folder / "DynamicAnalysis"); ensure_dir(out / "plots")
    if X is None:
        xfile = target_folder / "Trajectories" / "trajectories.npy"
        if xfile.exists():
            X = np.load(xfile); F0 = np.load(target_folder / "Trajectories" / "reference_faces.npy")
    report = ["# Dynamic analysis report", ""]
    results = {}
    progress(total=sum(map(bool, (hodge, ftle, shape_space, change_points, homology, spherical_spectra))) + 3, advance=0)
    if X is not None:
        T = X.shape[0]; t = np.arange(T, dtype=np.float64) * dt
        tfile = target_folder / "Trajectories" / "time.npy"
        t_given = np.asarray(getattr(times, "t", times), dtype=np.float64) if times is not None else (
            np.load(tfile) if tfile.exists() else None)
        if t_given is not None and t_given.size == T:
            t = t_given
        vfile = target_folder / "Trajectories" / "kinematics" / "velocity.npy"
        vel = np.load(vfile) if vfile.exists() else np.gradient(X, t, axis=0)
        if hodge:
            df, alpha, beta = hodge_analysis(X, F0, vel, out / "Hodge", plots, verbose); results["hodge"] = df; progress("Hodge decomposition")
            if hodge_nonrigid:                         # the same decomposition of the shape-only motion (rigid part removed)
                from PynamicMesh.core.motion_analysis import rigid_decomposition
                Xn = rigid_decomposition(X, F0, t)["nonrigid"]
                dfn, _, _ = hodge_analysis(Xn, F0, np.gradient(Xn, t, axis=0), out / "Hodge_nonrigid", False, False)
                cmp = df[["frame", "frac_curl_free", "frac_div_free", "frac_harmonic", "normal_fraction"]].merge(
                    dfn[["frame", "frac_curl_free", "frac_div_free", "frac_harmonic", "normal_fraction"]], on="frame",
                    suffixes=("_full", "_deformation"))
                cmp.to_csv(out / "Hodge_nonrigid" / "comparison.csv", index=False); results["hodge_nonrigid"] = dfn
                if plots:
                    fig, axh = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
                    for a_, suf, ttl in ((axh[0], "_full", "full motion"), (axh[1], "_deformation", "deformation only (rigid motion removed)")):
                        a_.stackplot(cmp.frame, cmp["frac_curl_free" + suf], cmp["frac_div_free" + suf], cmp["frac_harmonic" + suf],
                                     labels=["curl-free grad(α)", "div-free J grad(β)", "harmonic"]); a_.set_title(ttl); a_.set_xlabel("frame")
                    axh[0].legend(fontsize=8); axh[0].set_ylabel("energy fraction")
                    fig.tight_layout(); fig.savefig(out / "plots" / "8_hodge_full_vs_deformation.png", dpi=200); plt.close(fig)
            report += ["## Helmholtz–Hodge decomposition of the velocity field",
                       f"- mean energy fractions: curl-free {df['frac_curl_free'].mean():.3f}, divergence-free {df['frac_div_free'].mean():.3f}, harmonic {df['frac_harmonic'].mean():.3f}",
                       f"- share of kinetic energy in the normal direction (growth/shrinkage): {df['normal_fraction'].mean():.3f}", ""]
        if ftle and T >= 2:
            f_v, df = ftle_analysis(X, F0, t, out / "FTLE", plots, verbose); results["ftle"] = df; progress("FTLE")
            if not df.empty:
                report += ["## Finite-time Lyapunov exponents", f"- final mean FTLE {df['ftle_mean'].iloc[-1]:.3g}, 95th percentile {df['ftle_p95'].iloc[-1]:.3g}, max stretch ratio {df['max_stretch_ratio'].iloc[-1]:.3g}", ""]
    # shape coordinates chosen by the quality of the representation (SPHARM / high-degree SPHARM / manifold harmonics)
    selection = shape_coordinates(target_folder, X, F0, frames, shape_source, shape_max_error, mh_modes, verbose,
                                  max_worst_error=shape_max_worst_error)
    results["shape_selection"] = None if selection is None else {k_: selection[k_] for k_ in ("source", "reason", "errors")}
    progress("shape coordinates")
    if spherical_spectra and X is not None:
        from PynamicMesh.core.spherical_spectral import spherical_spectra as _spectra
        sp = _spectra(target_folder, out / "SphericalSpectra", X, F0, times=t, lmax=spectral_lmax, backend=sht_backend,
                      plots=plots, verbose=verbose)
        results["spherical_spectra"] = sp; progress("spherical spectra")
    if shape_space:
        ss = shape_space_analysis(target_folder, out / "ShapeSpace", X, F0, frames, n_shape_modes, plots=plots, verbose=verbose,
                                  selection=selection); progress("shape space")
        if ss:
            results["shape_space"] = ss
            report += ["## Shape space", f"- source: {ss['source']}; variance fractions of the first modes: " + ", ".join(f"{v:.3f}" for v in ss["variance_fraction"]), ""]
    coords = reduced_coordinates(target_folder, out / "ReducedCoordinates", X, pod_energy, pod_max_rank, verbose=verbose,
                                 selection=selection); progress("reduced coordinates")
    if coords:
        results["reduced_coordinates"] = {k: v.shape[1] for k, v in coords.items()}
        report += ["## Reduced coordinates (POD)"] + [f"- {k}: rank {v.shape[1]}" for k, v in coords.items()] + [""]
    if change_points:
        low_conf = None
        if X is not None:                              # frames with unreliable correspondences (Trajectories/tracking.csv)
            from PynamicMesh.core.motion_analysis import frame_confidence
            size = float(np.sqrt(face_normals_areas(X[0], F0)[1].sum()))
            fc = frame_confidence(target_folder, X.shape[0], size)
            low_conf = fc.frame[fc.low_confidence].tolist()
            for sub, name in (("Hodge", "hodge_energy.csv"), ("FTLE", "ftle_summary.csv")):
                f_ = out / sub / name
                if f_.exists():
                    pd.read_csv(f_).drop(columns=["confidence", "low_confidence"], errors="ignore").merge(
                        fc, on="frame", how="left").to_csv(f_, index=False)
        cp = change_point_analysis(target_folder, out / "ChangePoints", X, plots, verbose, coords=coords,
                                   n_perm=cp_permutations, confidence=low_conf); results["change_points"] = cp; progress("change points")
        if X is not None and isinstance(cp, dict) and find_frame_times(target_folder) is not None:
            cp = _add_change_point_times(cp, t)              # frames -> acquisition times
            results["change_points"] = cp
            with open(out / "ChangePoints" / "change_points.json", "w") as fh:
                json.dump(cp, fh, indent=2)
        report += ["## Change points", "```", json.dumps(cp, indent=2), "```", ""]
    if homology:
        ph = persistent_homology(target_folder, out / "Topology", verbose, coords=coords,
                                 prefer=None if selection is None else selection["label"]); progress("persistent homology")
        if ph is not None:
            results["homology"] = ph
            h1 = ph[ph["dimension"] == 1]
            report += ["## Persistent homology of the shape trajectory", f"- H1 classes: {len(h1)}; max persistence {h1['persistence'].max() if len(h1) else 0:.3g}", ""]
    if selection is not None:                          # which shape coordinates were used and why
        report += ["", "## Shape coordinates", "", f"* used: **{selection['source']}** — {selection['reason']}",
                   "* median relative reconstruction errors: " + ", ".join(
                       f"{k_} {v:.4f}" for k_, v in selection["errors"].items() if v is not None and k_ != "trajectories")]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    progress("report")
    if verbose:
        print(f"Dynamic analysis: report written to {out / 'report.md'}")
    return results