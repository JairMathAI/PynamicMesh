"""
spherical_spectral.py  —  fast spherical harmonic transforms (NVIDIA torch-harmonics on the GPU, NumPy fallback)
=================================================================================================================

Why
    The least-squares SPHARM fit of parametrization.py builds a dense (n_vertices × (L+1)²) basis matrix: practical
    up to L ≈ 25, which is enough for cells but not for shapes with long thin parts (legs, tails). An (almost)
    equi-areal sphere map sends such parts to thin strips of the sphere, and resolving a strip of width w needs a
    degree L ~ π / w: 64 - 128 for an animal. A spherical harmonic transform on a Gauss–Legendre grid costs O(L³)
    (FFT in longitude + Legendre quadrature in latitude) and runs on the GPU with torch-harmonics, so degrees of a
    few hundred are cheap.

What is here
    SphericalTransform   forward / inverse real SHT, orthonormal complex coefficients c[l, m] (m ≥ 0), Condon–Shortley
                         phase; backend 'torch' (torch_harmonics.RealSHT / InverseRealSHT, CUDA when available) or
                         'numpy' (same convention, scipy Legendre functions + FFT); 'auto' validates torch-harmonics
                         against the NumPy transform once and falls back to NumPy if they disagree.
    SphereSampler        mesh vertices on the unit sphere (a sphere map) -> barycentric resampling of vertex values on
                         the grid, and grid values back to the vertices (bilinear in (θ, φ)).
    high_degree_spharm   SPHARM of every frame up to a high degree from the aligned sphere maps of the parametrization
                         (Parametrization/SPHARM_HD/): coefficients, reconstruction error per degree (same definition
                         as the least-squares SPHARM, so both can be compared), degree energy.
    spherical_spectra    Lagrangian spectral dynamics: fields of the tracked surface (normal speed = growth, mean
                         curvature) are functions on the sphere of the reference frame for EVERY frame; their power
                         per degree P_l(t) tells which spatial scales carry the dynamics (DynamicAnalysis/SphericalSpectra).
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots

_TH = {"checked": False, "ok": False, "torch": None, "th": None, "reason": ""}


def torch_harmonics_available():
    """(available, reason): torch + torch_harmonics importable."""
    if not _TH["checked"]:
        _TH["checked"] = True
        try:
            import torch
            import torch_harmonics as th
            _TH.update(ok=True, torch=torch, th=th)
        except Exception as exc:  # noqa: BLE001
            _TH.update(ok=False, reason=f"torch-harmonics not available ({type(exc).__name__}: {exc})")
    return _TH["ok"], _TH["reason"]


def gauss_legendre_grid(nlat, nlon):
    """Colatitudes θ (from the north pole, ascending), longitudes φ = 2πk/nlon and quadrature weights of the
    Gauss–Legendre grid (the 'legendre-gauss' grid of torch-harmonics)."""
    x, w = np.polynomial.legendre.leggauss(nlat)               # x = cos θ ascending (south -> north)
    theta = np.flip(np.arccos(x)); w = np.flip(w)              # north -> south
    phi = 2 * np.pi * np.arange(nlon) / nlon
    return theta, phi, w


class SphericalTransform:
    """
    Real spherical harmonic transform on an (nlat, nlon) Gauss–Legendre grid up to degree lmax.
    forward(f) : (..., nlat, nlon) real  -> (..., lmax+1, lmax+1) complex c[l, m], m ≥ 0 (zero for m > l)
    inverse(c) : the reverse;  f(θ, φ) = Σ_l [ c_l0 Y_l0 + 2 Re Σ_{m>0} c_lm Y_lm ].
    Degree power of a field: P_l = |c_l0|² + 2 Σ_{m>0} |c_lm|²  (Parseval: Σ_l P_l = ∫ f² dΩ).
    """

    def __init__(self, lmax, nlat=None, nlon=None, backend="auto", device=None):
        self.lmax = int(lmax)
        self.nlat = int(nlat or 2 * (self.lmax + 1))           # 2x oversampling: robust to non band-limited data
        self.nlon = int(nlon or 2 * self.nlat)
        self.theta, self.phi, self.w = gauss_legendre_grid(self.nlat, self.nlon)
        self.backend = "numpy"; self.device = "cpu"
        self._P = None
        if backend in ("auto", "torch"):
            ok, why = torch_harmonics_available()
            if ok:
                try:
                    self._init_torch(device)
                    if self._validate():
                        self.backend = "torch"
                    else:
                        warnings.warn("torch-harmonics disagrees with the reference transform: NumPy backend used")
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"torch-harmonics could not be used ({exc}): NumPy backend used")
            elif backend == "torch":
                warnings.warn(f"{why}: NumPy backend used")
        if self.backend == "numpy":
            self.device = "cpu"

    # ---- NumPy backend (reference) ---------------------------------------------------------------------------
    def _legendre(self):
        """Θ_lm(θ_j) = Y_lm(θ_j, 0) (orthonormal, Condon–Shortley), table (lmax+1, lmax+1, nlat), zero for m > l."""
        if self._P is None:
            from scipy.special import sph_harm_y
            L = self.lmax
            P = np.zeros((L + 1, L + 1, self.nlat))
            for m in range(L + 1):
                ls = np.arange(m, L + 1)
                P[m:, m, :] = np.real(sph_harm_y(ls[:, None], m, self.theta[None, :], 0.0))
            self._P = P
        return self._P

    def _forward_np(self, f):
        f = np.asarray(f, dtype=np.float64)
        F = np.fft.rfft(f, axis=-1)[..., : self.lmax + 1] * (2 * np.pi / self.nlon)   # (..., nlat, m)
        P = self._legendre()                                                          # (l, m, nlat)
        return np.einsum("lmj,...jm,j->...lm", P, F, self.w)

    def _inverse_np(self, c):
        c = np.asarray(c)
        P = self._legendre()
        G = np.einsum("lmj,...lm->...jm", P, c)                                       # (..., nlat, m)
        full = np.zeros(G.shape[:-1] + (self.nlon // 2 + 1,), dtype=np.complex128)
        full[..., : self.lmax + 1] = G
        return np.fft.irfft(full, n=self.nlon, axis=-1) * self.nlon

    # ---- torch-harmonics backend (GPU) -----------------------------------------------------------------------
    def _init_torch(self, device):
        torch, th = _TH["torch"], _TH["th"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = dict(lmax=self.lmax + 1, mmax=self.lmax + 1, grid="legendre-gauss", norm="ortho")
        self._sht = th.RealSHT(self.nlat, self.nlon, **kw).to(self.device)
        self._isht = th.InverseRealSHT(self.nlat, self.nlon, **kw).to(self.device)

    def _forward_torch(self, f):
        torch = _TH["torch"]
        with torch.no_grad():
            t = torch.as_tensor(np.ascontiguousarray(f), dtype=torch.float64, device=self.device)
            return self._sht(t).cpu().numpy()

    def _inverse_torch(self, c):
        torch = _TH["torch"]
        with torch.no_grad():
            t = torch.as_tensor(np.ascontiguousarray(c), dtype=torch.complex128, device=self.device)
            return self._isht(t).cpu().numpy()

    def _validate(self):
        """torch-harmonics must reproduce the reference transform (degree powers and round trip) on a random field."""
        rng = np.random.default_rng(0)
        L = min(self.lmax, 12)
        c = np.zeros((self.lmax + 1, self.lmax + 1), dtype=np.complex128)
        for l in range(L + 1):
            c[l, : l + 1] = rng.normal(size=l + 1) + 1j * rng.normal(size=l + 1)
            c[l, 0] = c[l, 0].real
        f = self._inverse_np(c)
        ct = self._forward_torch(f)
        same_power = np.allclose(self.degree_power(ct), self.degree_power(c), rtol=1e-6, atol=1e-9)
        roundtrip = np.allclose(self._inverse_torch(ct), f, atol=1e-8 * max(1.0, np.abs(f).max()))
        return bool(same_power and roundtrip)

    # ---- public ----------------------------------------------------------------------------------------------
    def forward(self, f):
        return self._forward_torch(f) if self.backend == "torch" else self._forward_np(f)

    def inverse(self, c):
        return self._inverse_torch(c) if self.backend == "torch" else self._inverse_np(c)

    @staticmethod
    def degree_power(c):
        """P_l = |c_l0|² + 2 Σ_{m>0} |c_lm|² over the last two axes (..., l, m) -> (..., l)."""
        a = np.abs(np.asarray(c)) ** 2
        return a[..., 0] + 2 * a[..., 1:].sum(axis=-1)

    @staticmethod
    def truncate(c, L):
        c = np.array(c, copy=True); c[..., L + 1:, :] = 0
        return c


class SphereSampler:
    """
    Resampling between a mesh mapped to the unit sphere (vertices U, faces F) and the (θ, φ) grid of a transform:
    to_grid(values) interpolates vertex values barycentrically inside the spherical triangle containing every grid
    direction; to_vertices(grid) samples a grid function at the vertices (bilinear in θ, periodic in φ).
    """

    def __init__(self, U, F, theta, phi, k=16):
        U = np.asarray(U, dtype=np.float64); F = np.asarray(F, dtype=np.int64)
        U = U / np.maximum(np.linalg.norm(U, axis=1), 1e-300)[:, None]
        self.U, self.F, self.theta, self.phi = U, F, np.asarray(theta), np.asarray(phi)
        TT, PP = np.meshgrid(self.theta, self.phi, indexing="ij")
        D = np.stack([np.sin(TT) * np.cos(PP), np.sin(TT) * np.sin(PP), np.cos(TT)], axis=-1).reshape(-1, 3)
        cent = U[F].mean(axis=1); cent /= np.maximum(np.linalg.norm(cent, axis=1), 1e-300)[:, None]
        _, cand = cKDTree(cent).query(D, k=min(k, len(F)))
        cand = np.atleast_2d(cand)
        face = np.full(len(D), -1, dtype=np.int64); bary = np.zeros((len(D), 3))
        best = np.full(len(D), -np.inf)
        a, b, c = U[F[:, 0]], U[F[:, 1]], U[F[:, 2]]
        for col in range(cand.shape[1]):
            f = cand[:, col]
            A, B, C = a[f], b[f], c[f]
            n = np.cross(B - A, C - A)
            den = np.einsum("ij,ij->i", D, n)
            s = np.einsum("ij,ij->i", A, n) / np.where(np.abs(den) > 1e-300, den, 1e-300)
            P = D * s[:, None]                                         # ray / triangle-plane intersection
            v0, v1, v2 = B - A, C - A, P - A
            d00 = np.einsum("ij,ij->i", v0, v0); d01 = np.einsum("ij,ij->i", v0, v1); d11 = np.einsum("ij,ij->i", v1, v1)
            d20 = np.einsum("ij,ij->i", v2, v0); d21 = np.einsum("ij,ij->i", v2, v1)
            dn = d00 * d11 - d01 ** 2
            bv = (d11 * d20 - d01 * d21) / np.where(np.abs(dn) > 1e-300, dn, 1e-300)
            bw = (d00 * d21 - d01 * d20) / np.where(np.abs(dn) > 1e-300, dn, 1e-300)
            bu = 1 - bv - bw
            score = np.minimum(np.minimum(bu, bv), bw)                 # >= 0 inside the triangle
            better = (score > best) & (s > 0)
            face[better] = f[better]; bary[better] = np.stack([bu, bv, bw], axis=1)[better]; best[better] = score[better]
        bary = np.clip(bary, 0, None); bary /= np.maximum(bary.sum(axis=1), 1e-300)[:, None]
        self.face, self.bary = face, bary
        self.inside_fraction = float(np.mean(best > -1e-6))
        self.shape = TT.shape
        # vertex -> grid position (for to_vertices)
        th_v = np.arccos(np.clip(U[:, 2], -1, 1)); ph_v = np.mod(np.arctan2(U[:, 1], U[:, 0]), 2 * np.pi)
        j = np.clip(np.searchsorted(self.theta, th_v) - 1, 0, len(self.theta) - 2)
        self._j = j; self._tj = np.clip((th_v - self.theta[j]) / (self.theta[j + 1] - self.theta[j]), 0, 1)
        dphi = 2 * np.pi / len(self.phi)
        k0 = np.floor(ph_v / dphi).astype(np.int64) % len(self.phi)
        self._k = k0; self._tk = (ph_v - k0 * dphi) / dphi

    def to_grid(self, values):
        """Vertex values (n,) or (n, c) -> grid (nlat, nlon) or (c, nlat, nlon)."""
        v = np.asarray(values, dtype=np.float64)
        g = np.einsum("pa,pa...->p...", self.bary, v[self.F[self.face]])
        if g.ndim == 1:
            return g.reshape(self.shape)
        return np.moveaxis(g.reshape(self.shape + (g.shape[-1],)), -1, 0)

    def to_vertices(self, grid):
        """Grid (nlat, nlon) or (c, nlat, nlon) -> vertex values (n,) or (n, c)."""
        g = np.asarray(grid)
        squeeze = g.ndim == 2
        if squeeze:
            g = g[None]
        j, tj, k, tk = self._j, self._tj, self._k, self._tk
        k1 = (k + 1) % g.shape[-1]
        out = ((1 - tj) * ((1 - tk) * g[:, j, k] + tk * g[:, j, k1]) + tj * ((1 - tk) * g[:, j + 1, k] + tk * g[:, j + 1, k1]))
        return out[0] if squeeze else out.T


# ----------------------------------------------------------------------------------------------------------------- #
#  High-degree SPHARM of the frames (from the aligned sphere maps of the parametrization)
# ----------------------------------------------------------------------------------------------------------------- #
def _rel_error(V, R, p95=False):
    """RMS vertex error and the relative error used by the least-squares SPHARM (error / RMS spread of the vertices);
    with p95=True also the 95th percentile of the per-vertex relative error (the worst-represented regions)."""
    e = np.sqrt(np.sum((V - R) ** 2, axis=1))
    rms = float(np.sqrt(np.mean(e ** 2)))
    spread = float(np.sqrt(np.mean(np.sum((V - V.mean(axis=0)) ** 2, axis=1))))
    if p95:
        return rms, rms / max(spread, 1e-300), float(np.percentile(e, 95) / max(spread, 1e-300))
    return rms, rms / max(spread, 1e-300)


def auto_lmax(n_vertices, L_ls=25):
    """Degree used by default: enough to resolve features of a few vertices, at least twice the least-squares degree."""
    return int(np.clip(max(2 * L_ls, int(np.sqrt(n_vertices) / 2)), 32, 160))


def high_degree_spharm(target_folder, frames, lmax="auto", backend="auto", nlat=None, report_degrees=None,
                       L_ls=25, plots=True, verbose=True):
    """
    SPHARM of every genus-0 frame up to a high degree with a fast SHT (torch-harmonics / NumPy), using the aligned
    sphere maps of the parametrization (Parametrization/Spherical/sphere_T####.npy). Writes Parametrization/SPHARM_HD/:
      coeffs_T####.npy        (3, L+1, L+1) complex coefficients of x, y, z
      coeffs_hd_consistent.npy (T, 3, L+1, L+1) for the frames with a sphere map (dynamics input)
      reconstruction_error.csv frame, degree, rms_error, rel_error (same definition as SPHARM/reconstruction_error.csv)
      degree_energy.csv        frame, degree, energy (Σ over x, y, z of the degree power)
      hd_info.json             backend, device, degree, grid, timing
    Returns the info dict (None when no sphere map is available).
    """
    par = Path(target_folder) / "Parametrization"
    d_sph = par / "Spherical"; out = par / "SPHARM_HD"; out.mkdir(parents=True, exist_ok=True)
    idx = [i for i in range(len(frames)) if (d_sph / f"sphere_T{i:04d}.npy").exists()]
    if not idx:
        return None
    n_max = max(len(frames[i]["V"]) for i in idx)
    L = auto_lmax(n_max, L_ls) if lmax in (None, "auto") else int(lmax)
    tr = SphericalTransform(L, nlat=nlat, backend=backend)
    degrees = sorted(set([d for d in (report_degrees or [8, 16, 25, 32, 48, 64, 96, 128, 160]) if d <= L] + [L]))
    rows, energy, stack = [], [], []
    t0 = time.time()
    from PynamicMesh.core.dyn_common import pbar
    for i in pbar(idx, desc=f"High-degree SPHARM (L={L}, {tr.backend})"):
        V = np.asarray(frames[i]["V"], dtype=np.float64); F = np.asarray(frames[i]["F"], dtype=np.int64)
        U = np.load(d_sph / f"sphere_T{i:04d}.npy")
        if U.shape[0] != V.shape[0]:
            continue
        smp = SphereSampler(U, F, tr.theta, tr.phi)
        grid = smp.to_grid(V)                                         # (3, nlat, nlon)
        c = tr.forward(grid)                                           # (3, L+1, L+1)
        np.save(out / f"coeffs_T{i:04d}.npy", c.astype(np.complex64)); stack.append(c.astype(np.complex64))
        for d in degrees:
            R = smp.to_vertices(tr.inverse(SphericalTransform.truncate(c, d)))
            rms, rel, p95 = _rel_error(V, R, p95=True)
            rows.append({"frame": i, "degree": d, "rms_error": rms, "rel_error": rel, "p95_rel_error": p95})
        P = SphericalTransform.degree_power(c).sum(axis=0)
        energy += [{"frame": i, "degree": l, "energy": float(P[l])} for l in range(L + 1)]
    if not stack:
        return None
    np.save(out / "coeffs_hd_consistent.npy", np.stack(stack))
    err = pd.DataFrame(rows); err.to_csv(out / "reconstruction_error.csv", index=False)
    pd.DataFrame(energy).to_csv(out / "degree_energy.csv", index=False)
    info = {"lmax": L, "nlat": tr.nlat, "nlon": tr.nlon, "grid": "legendre-gauss", "backend": tr.backend,
            "device": tr.device, "frames": idx, "seconds": round(time.time() - t0, 2),
            "median_rel_error_at_lmax": float(err[err.degree == L].rel_error.median()),
            "convention": "orthonormal complex SH, m >= 0, Condon-Shortley phase; f = sum_l c_l0 Y_l0 + 2 Re sum_m>0 c_lm Y_lm"}
    with open(out / "hd_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    if plots:
        _plot_hd(par, err, L)
    if verbose:
        print(f"High-degree SPHARM: L = {L} on {len(idx)} frames ({tr.backend}, {tr.device}); median relative "
              f"error {info['median_rel_error_at_lmax']:.4f}")
    return info


def _plot_hd(par, err, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ls = par / "SPHARM" / "reconstruction_error.csv"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    g = err.groupby("degree").rel_error.median()
    ax.semilogy(g.index, g.values, "o-", label="fast SHT (high degree)")
    if ls.exists():
        e2 = pd.read_csv(ls).groupby("degree").rel_error.median()
        ax.semilogy(e2.index, e2.values, "s--", label="least squares (SPHARM)")
    ax.set_xlabel("degree L"); ax.set_ylabel("median relative reconstruction error"); ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"SPHARM reconstruction error vs degree (L up to {L})"); ax.legend()
    d = par / "plots"; d.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(d / "5_spharm_high_degree_error.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------- #
#  Spectral dynamics of fields on the reference sphere
# ----------------------------------------------------------------------------------------------------------------- #
def spherical_spectra(target_folder, out_dir, X, F0, times=None, lmax=32, backend="auto", plots=True, verbose=True):
    """
    The tracked mesh keeps the vertices of the reference frame, so every field on it is a function on the sphere
    map of the reference frame (Parametrization/Spherical/sphere_T0000.npy) at every time: a Lagrangian spherical
    coordinate system. For the normal speed (growth / retraction) and the mean curvature of every frame the degree
    power P_l(t) is computed with the SHT: which spatial scales grow, which scales of the shape change.
    Writes <out_dir>/: spectra.csv (field, frame, time, degree, power), summary.csv (field, frame, time, total power,
    spectral centroid l̄ = Σ l P_l / Σ P_l, dominant degree), plots. Returns the summary DataFrame or None.
    """
    from PynamicMesh.core.dyn_common import pbar, vertex_normals, cotan_laplacian
    U0f = Path(target_folder) / "Parametrization" / "Spherical" / "sphere_T0000.npy"
    if not U0f.exists():
        if verbose:
            print("Spherical spectra skipped: no sphere map of the reference frame (compute_parametrization).")
        return None
    U0 = np.load(U0f); X = np.asarray(X, dtype=np.float64); F0 = np.asarray(F0, dtype=np.int64)
    if U0.shape[0] != X.shape[1]:
        if verbose:
            print("Spherical spectra skipped: the sphere map does not match the tracked mesh.")
        return None
    T = X.shape[0]
    t = np.arange(T, dtype=np.float64) if times is None else np.asarray(times, dtype=np.float64)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tr = SphericalTransform(lmax, backend=backend)
    smp = SphereSampler(U0, F0, tr.theta, tr.phi)
    vel = np.gradient(X, t, axis=0) if T > 1 else np.zeros_like(X)
    rows, summ = [], []
    for k in pbar(range(T), desc="Spherical spectra"):
        N = vertex_normals(X[k], F0)
        W, M, _ = cotan_laplacian(X[k], F0)
        Md = np.asarray(M.diagonal()).ravel() if hasattr(M, "diagonal") else np.asarray(M).ravel()
        Hn = (W @ X[k]) / (2 * np.maximum(Md, 1e-300))[:, None]
        fields = {"normal_speed": np.einsum("ij,ij->i", vel[k], N), "mean_curvature": np.einsum("ij,ij->i", Hn, N)}
        for name, f in fields.items():
            c = tr.forward(smp.to_grid(f - f.mean()))
            P = SphericalTransform.degree_power(c)
            ls = np.arange(len(P))
            rows += [{"field": name, "frame": k, "time": float(t[k]), "degree": int(l), "power": float(P[l])} for l in ls]
            tot = float(P[1:].sum())
            summ.append({"field": name, "frame": k, "time": float(t[k]), "total_power": tot,
                         "spectral_centroid": float((ls[1:] * P[1:]).sum() / tot) if tot > 0 else np.nan,
                         "dominant_degree": int(ls[1:][np.argmax(P[1:])]) if tot > 0 else 0})
    spec = pd.DataFrame(rows); summary = pd.DataFrame(summ)
    spec.to_csv(out / "spectra.csv", index=False); summary.to_csv(out / "summary.csv", index=False)
    with open(out / "info.json", "w", encoding="utf-8") as fh:
        json.dump({"lmax": lmax, "backend": tr.backend, "device": tr.device, "grid_inside_fraction": smp.inside_fraction,
                   "fields": ["normal_speed", "mean_curvature"]}, fh, indent=2)
    if plots:
        _plot_spectra(out, spec, summary)
    if verbose:
        print(f"Spherical spectra: {T} frames, L = {lmax} ({tr.backend})")
    return summary


def _plot_spectra(out, spec, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = list(dict.fromkeys(spec.field))
    fig, axes = plt.subplots(2, len(fields), figsize=(6.5 * len(fields), 8), squeeze=False)
    for c, name in enumerate(fields):
        s = spec[(spec.field == name) & (spec.degree > 0)]
        piv = s.pivot(index="degree", columns="time", values="power")
        im = axes[0, c].imshow(np.log10(piv.values + 1e-30), aspect="auto", origin="lower", cmap="magma",
                               extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
        axes[0, c].set_title(f"{name}: log10 power per degree"); axes[0, c].set_xlabel("time"); axes[0, c].set_ylabel("degree l")
        fig.colorbar(im, ax=axes[0, c])
        sm = summary[summary.field == name]
        axes[1, c].plot(sm.time, sm.spectral_centroid, "o-", label="spectral centroid")
        axes[1, c].set_xlabel("time"); axes[1, c].set_ylabel("degree"); axes[1, c].grid(alpha=0.3)
        ax2 = axes[1, c].twinx(); ax2.plot(sm.time, sm.total_power, "s--", color="tab:red", label="total power")
        ax2.set_ylabel("total power")
        axes[1, c].set_title(f"{name}: scale (centroid) and intensity (total power)")
    d = Path(out).parent / "plots"; d.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(d / "7_spherical_spectra.png", dpi=200); plt.close(fig)
