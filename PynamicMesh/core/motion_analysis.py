"""
motion_analysis.py  —  motion, deformation and events of the tracked surface (stage "MotionAnalysis")
====================================================================================================

Uses the results already computed (Trajectories, Morse–Smale tracking, shape space, graph animation, frame times)
to describe HOW the object moves and changes. Everything is written to Results/<scene>/MotionAnalysis/:

  Confidence/        frame_confidence.csv — reliability of the correspondences of every frame (tracking.csv)
  RigidMotion/       rigid / non-rigid decomposition: translation, rotation, angular velocity, energy fractions,
                     nonrigid_trajectories.npy (shape-only motion, the reference pose)
  TrajectoryFields/  physical fields computed from the tracked mesh (complement of Physical_fields, which uses the
                     frame-to-frame maps): velocity, acceleration, normal speed, strain per frame + comparison csv
  GrowthAtlas/       cumulative normal growth and local area change of every surface point, per Morse–Smale region
  Anisotropy/        principal stretches and directions of the deformation (growth anisotropy), sliding-window FTLE
  Polarity/          migration direction vs elongation axis, front / rear asymmetry of the protrusions
  Regions/           kinematics of every Morse–Smale region of the reference frame (body vs protrusions)
  Protrusions/       protrusion kinetics: lifetimes, extension / retraction speeds, birth / death rates
  Branches/          length, extension and extension rate of every branch of the animated Reeb skeleton
  Signals/           one time series table of the dynamics (size-normalised), event timeline, synchronisation
                     (cross-correlation with lags), frequency analysis (Lomb–Scargle), DMD (modes, frequencies,
                     growth rates, forecast)
  summary.json, plots/, output_log.txt

compare_scenes(results_root) compares the scenes of a batch (Results/_comparison/).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots

from PynamicMesh.core.dyn_common import (
    logged_stage, progress, pbar, ensure_dir, xp, to_gpu, to_cpu, cotan_laplacian, face_normals_areas,
    vertex_normals, vertex_areas, kabsch_rotation, write_obj, find_frame_times,
)

MOTION_DEFAULTS = {
    "rigid": True, "fields": True, "growth_atlas": True, "anisotropy": True, "ftle_window": 3, "polarity": True,
    "regions": True, "protrusions": True, "branches": True, "signals": True, "timeline": True, "synchronization": True,
    "max_lag": None, "frequency": True, "dmd": True, "dmd_energy": 0.999, "dmd_forecast": 3, "low_confidence": 0.5,
    "flip_angle": 120.0, "dmd_delays": "auto",
    "graphs": None,                                     # graph kinds of the skeleton branches (None = every animation)
    "plots": True,
}


# --------------------------------------------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------------------------------------------- #
def _load(target_folder, rel):
    f = Path(target_folder) / rel
    if not f.exists():
        return None
    return pd.read_csv(f) if f.suffix == ".csv" else (np.load(f, allow_pickle=True) if f.suffix in (".npy", ".npz")
                                                     else json.load(open(f)))


def _times(target_folder, T, t=None):
    if t is not None and len(t) == T:
        return np.asarray(t, dtype=np.float64), "time"
    f = Path(target_folder) / "Trajectories" / "time.npy"
    if f.exists() and len(np.load(f)) == T:
        ft = find_frame_times(target_folder)
        return np.load(f), (ft.unit if ft is not None else "frame")
    return np.arange(T, dtype=np.float64), "frame"


def face_deformation(Va, Vb, F):
    """Per-face deformation gradient of the map Va -> Vb (same triangulation), in the tangent basis of Va:
    Fg (F,3,2), right Cauchy–Green C = FgᵀFg (F,2,2), tangent basis (b1, b2) of Va and its face areas."""
    tri = Va[F]; e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
    b1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-300)[:, None]
    nrm = np.cross(e1, e2); nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
    b2 = np.cross(nrm, b1)
    E0 = np.stack([np.stack([np.einsum("ij,ij->i", e1, b1), np.einsum("ij,ij->i", e1, b2)], 1),
                   np.stack([np.einsum("ij,ij->i", e2, b1), np.einsum("ij,ij->i", e2, b2)], 1)], axis=2)
    trib = Vb[F]
    Ek = np.stack([trib[:, 1] - trib[:, 0], trib[:, 2] - trib[:, 0]], axis=2)
    Fg = Ek @ np.linalg.inv(E0 + 1e-14 * np.eye(2)[None])
    C = np.transpose(Fg, (0, 2, 1)) @ Fg
    _, a = face_normals_areas(Va, F)
    return Fg, C, b1, b2, a


def _face_to_vertex(vals, F, n, w):
    acc = np.zeros(n); ws = np.zeros(n)
    for c in range(3):
        np.add.at(acc, F[:, c], vals * w); np.add.at(ws, F[:, c], w)
    return acc / np.maximum(ws, 1e-300)


def _resample_uniform(t, Y):
    """(t, Y (T, ...)) -> uniform grid with the same number of samples (linear interpolation)."""
    tu = np.linspace(t[0], t[-1], len(t))
    Yf = Y.reshape(len(t), -1)
    Yu = np.stack([np.interp(tu, t, Yf[:, j]) for j in range(Yf.shape[1])], axis=1)
    return tu, Yu.reshape((len(t),) + Y.shape[1:])


def _savefig(fig, out, name):
    d = ensure_dir(Path(out) / "plots")
    fig.tight_layout(); fig.savefig(d / name, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------------------------------- #
#  1. frame confidence (reliability of the correspondences)
# --------------------------------------------------------------------------------------------------------------- #
def frame_confidence(target_folder, T, size, low=0.5):
    """Confidence in [0, 1] of every frame from Trajectories/tracking.csv: reliable map fraction x surface
    closeness (p95 distance to the true surface relative to the size). Frames below `low` x median are flagged."""
    trk = _load(target_folder, "Trajectories/tracking.csv")
    c = np.ones(T)
    if trk is not None and len(trk) == T:
        rel = trk["reliable_fraction"].to_numpy(float) if "reliable_fraction" in trk else np.ones(T)
        d = trk["p95_surface_distance"].to_numpy(float) if "p95_surface_distance" in trk else np.zeros(T)
        c = np.clip(np.nan_to_num(rel, nan=1.0), 0, 1) * np.exp(-np.nan_to_num(d, nan=0.0) / (0.02 * size))
        c[0] = 1.0
    med = float(np.median(c[1:])) if T > 1 else 1.0
    flag = c < low * med
    return pd.DataFrame({"frame": np.arange(T), "confidence": c, "low_confidence": flag})


# --------------------------------------------------------------------------------------------------------------- #
#  2. rigid / non-rigid decomposition
# --------------------------------------------------------------------------------------------------------------- #
def rigid_decomposition(X, F0, t):
    """
    Best rigid motion of every frame w.r.t. the reference (area-weighted Kabsch): X_t ≈ (X_0 - c_0) R_tᵀ + c_t.
    Returns a dict: R (T,3,3), centroid (T,3), nonrigid (T,n,3) = the shape of every frame in the reference pose,
    and a per-frame table: translation, rotation angle, angular velocity, rigid and deformation displacement
    (RMS) and the deformation fraction of the motion energy.
    """
    T, n, _ = X.shape
    m = np.asarray(vertex_areas(X[0], F0), dtype=np.float64); m = m / m.sum()
    c = np.einsum("tni,n->ti", X, m)
    P0 = X[0] - c[0]
    R = np.zeros((T, 3, 3)); Xn = np.zeros_like(X); rows = []
    for k in range(T):
        Y = X[k] - c[k]
        Rk = kabsch_rotation(P0, Y, m)                     # P0 @ Rkᵀ ≈ Y
        R[k] = Rk
        Xn[k] = Y @ Rk + c[0]                              # back to the reference pose: shape only
    for k in range(T):
        rigid = P0 @ R[k].T + c[k]
        d_tot = X[k] - X[0]; d_rig = rigid - X[0]; d_def = X[k] - rigid
        e_rig = float(np.sum(m * np.sum(d_rig ** 2, 1))); e_def = float(np.sum(m * np.sum(d_def ** 2, 1)))
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R[k]) - 1) / 2, -1, 1))))
        if k > 0:
            dR = R[k] @ R[k - 1].T; dt = max(t[k] - t[k - 1], 1e-12)
            th = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
            axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            axis = axis / max(np.linalg.norm(axis), 1e-300)
            w = th / dt
            v = np.linalg.norm(c[k] - c[k - 1]) / dt
            step = float(np.degrees(th))
        else:
            w, v, axis, step = 0.0, 0.0, np.zeros(3), 0.0
        rows.append({"frame": k, "time": float(t[k]), "translation": float(np.linalg.norm(c[k] - c[0])),
                     "rotation_deg": ang, "angular_speed_deg": float(np.degrees(w)), "rotation_axis_x": axis[0],
                     "rotation_axis_y": axis[1], "rotation_axis_z": axis[2], "centroid_speed": float(v),
                     "frame_to_frame_rotation_deg": step,
                     "rigid_rms": float(np.sqrt(e_rig)), "deformation_rms": float(np.sqrt(e_def)),
                     "deformation_fraction": (e_def / (e_rig + e_def)) if (k > 0 and e_rig + e_def > 1e-12 * float(np.sum(m * np.sum(P0 ** 2, 1)))) else 0.0})
    return {"R": R, "centroid": c, "nonrigid": Xn, "table": pd.DataFrame(rows), "weights": m}


# --------------------------------------------------------------------------------------------------------------- #
#  3. physical fields from the trajectories (complement of Physical_fields)
# --------------------------------------------------------------------------------------------------------------- #
def trajectory_fields(target_folder, out, X, F0, t, nonrigid=None):
    """
    Physical fields of the TRACKED mesh (fixed triangulation, robust tracking + rigidity projection), the same
    quantities as Physical_fields/ (frame-to-frame maps) but from one consistent source: velocity, acceleration,
    normal speed, Green–Lagrange strain w.r.t. the reference (norm, area strain), deformation-only velocity.
    Per frame <out>/frame_####.npz (vertex fields), global_metrics.csv and comparison.csv (vs Physical_fields).
    """
    out = ensure_dir(out)
    T, n, _ = X.shape
    vel = np.gradient(X, t, axis=0) if T > 1 else np.zeros_like(X)
    acc = np.gradient(vel, t, axis=0) if T > 2 else np.zeros_like(X)
    veln = np.gradient(nonrigid, t, axis=0) if nonrigid is not None and T > 1 else None
    rows = []
    for k in pbar(range(T), desc="Trajectory fields"):
        N = vertex_normals(X[k], F0)
        Fg, C, _, _, a0 = face_deformation(X[0], X[k], F0)
        E = 0.5 * (C - np.eye(2)[None])                                     # Green–Lagrange strain
        e_norm = np.sqrt(np.einsum("fij,fij->f", E, E))
        area_strain = np.sqrt(np.maximum(np.linalg.det(C), 1e-300)) - 1
        fields = {"velocity": np.linalg.norm(vel[k], axis=1), "acceleration": np.linalg.norm(acc[k], axis=1),
                  "normal_speed": np.einsum("ij,ij->i", vel[k], N),
                  "strain_norm": _face_to_vertex(e_norm, F0, n, a0), "area_strain": _face_to_vertex(area_strain, F0, n, a0)}
        if veln is not None:
            fields["deformation_velocity"] = np.linalg.norm(veln[k], axis=1)
        np.savez_compressed(out / f"frame_{k:04d}.npz", **fields, velocity_vectors=vel[k].astype(np.float32))
        w = np.asarray(vertex_areas(X[k], F0)); w = w / w.sum()
        rows.append({"frame": k, "time": float(t[k]), **{f"mean_{key}": float(np.sum(w * np.abs(v))) for key, v in fields.items()},
                     "max_velocity": float(fields["velocity"].max())})
    gm = pd.DataFrame(rows); gm.to_csv(out / "global_metrics.csv", index=False)
    # comparison with the frame-to-frame physical fields
    pf = _load(target_folder, "Physical_fields/global_physical_metrics.csv")
    if pf is not None and "mean_speed" in pf:
        k = pf["Time_Step"].to_numpy(int) if "Time_Step" in pf else np.arange(1, len(pf) + 1)
        ok = k < T
        cmp = pd.DataFrame({"frame": k[ok], "mean_speed_frame_to_frame": pf["mean_speed"].to_numpy(float)[ok],
                            "mean_speed_trajectories": gm["mean_velocity"].to_numpy()[k[ok]]})
        cmp["ratio"] = cmp.mean_speed_trajectories / cmp.mean_speed_frame_to_frame.replace(0, np.nan)
        cmp.to_csv(out / "comparison.csv", index=False)
    return gm


FIELD_PAIRS = {"velocity": "speed", "acceleration": "acceleration", "normal_flow": "normal flow",
               "strain": "strain (edges)", "area_strain": "area strain"}


def field_comparison(target_folder, out, X, F0, t):
    """
    Frame-to-frame physical fields (Physical_fields/, from the functional-map correspondences) vs the SAME fields of
    the tracked mesh: every transition i-1 -> i is evaluated with the functions of physic_model (displacement /
    velocity, edge strain, area strain, normal flow, acceleration) on the tracked mesh with the identity correspondence
    (same definitions, units and time intervals — only the source of the correspondences differs). The frame-to-frame
    field is sampled at the tracked vertices (nearest vertex of the frame mesh).
    Writes <out>/comparison/frame_####.npz ({field}_frame_to_frame, {field}_trajectories) and
    <out>/comparison_fields.csv (per frame and field: area-weighted mean |value| of both sources, ratio, correlation,
    relative difference). Returns the table (None without frame-to-frame fields).
    """
    from scipy.spatial import cKDTree
    from PynamicMesh.core.physic_model import (compute_displacement_velocity, compute_finite_element_strain,
                                               compute_area_strain, compute_flow_decomposition)
    target_folder = Path(target_folder); pf = target_folder / "Physical_fields"
    T, n, _ = X.shape
    if T < 2 or not any((pf / f"frame_{k:04d}.npz").exists() for k in range(1, T)):
        return None
    d = ensure_dir(Path(out) / "comparison")
    ident = np.arange(n); prev_disp = dt_prev = None; rows = []
    for k in pbar(range(1, T), desc="Field comparison"):
        dt = float(t[k] - t[k - 1]) or 1.0
        v_mag, disp = compute_displacement_velocity(X[k - 1], X[k], ident, dt=dt)
        nflow, _ = compute_flow_decomposition(X[k], F0, disp)
        acc = (np.linalg.norm(disp / dt - prev_disp / dt_prev, axis=1) / (0.5 * (dt + dt_prev))
               if prev_disp is not None else np.zeros(n))
        traj = {"velocity": v_mag, "acceleration": acc, "normal_flow": nflow,
                "strain": compute_finite_element_strain(X[k - 1], X[k], F0, ident),
                "area_strain": compute_area_strain(X[k - 1], X[k], F0, ident)}
        prev_disp, dt_prev = disp, dt
        f = pf / f"frame_{k:04d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        nn = cKDTree(np.asarray(z["vertices"], float)).query(X[k])[1]
        w = np.asarray(vertex_areas(X[k], F0)); w = w / w.sum()
        save = {}
        for key, name in FIELD_PAIRS.items():
            if key not in z.files:
                continue
            a = np.asarray(z[key], float)[nn]; b = np.asarray(traj[key], float)
            save[f"{key}_frame_to_frame"] = a.astype(np.float32); save[f"{key}_trajectories"] = b.astype(np.float32)
            ma, mb = float(np.sum(w * np.abs(a))), float(np.sum(w * np.abs(b)))
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
            rel = float(np.median(np.abs(b - a)) / max(np.median(np.abs(a)), 1e-300))
            rows.append({"frame": k, "time": float(t[k]), "field": key, "mean_abs_frame_to_frame": ma,
                         "mean_abs_trajectories": mb, "ratio": mb / ma if ma > 0 else np.nan,
                         "correlation": corr, "rel_difference": rel})
        np.savez_compressed(d / f"frame_{k:04d}.npz", **save)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "comparison_fields.csv", index=False)
    return df


# --------------------------------------------------------------------------------------------------------------- #
#  4. growth atlas, anisotropy, sliding-window FTLE
# --------------------------------------------------------------------------------------------------------------- #
def growth_atlas(out, X, F0, t, labels0=None):
    """Cumulative normal growth ∫ v_n dt and local area change log(A_t / A_0) of every reference point; per
    Morse–Smale region of the reference frame (labels0) the mean of both over time."""
    out = ensure_dir(out)
    T, n, _ = X.shape
    vel = np.gradient(X, t, axis=0) if T > 1 else np.zeros_like(X)
    A0 = np.asarray(vertex_areas(X[0], F0)); cum = np.zeros(n); G = np.zeros((T, n)); LA = np.zeros((T, n))
    for k in range(T):
        vn = np.einsum("ij,ij->i", vel[k], vertex_normals(X[k], F0))
        if k > 0:
            cum = cum + 0.5 * (vn + vn_prev) * (t[k] - t[k - 1])
        vn_prev = vn
        G[k] = cum; LA[k] = np.log(np.maximum(np.asarray(vertex_areas(X[k], F0)), 1e-300) / np.maximum(A0, 1e-300))
    np.save(out / "cumulative_normal_growth.npy", G.astype(np.float32)); np.save(out / "log_area_change.npy", LA.astype(np.float32))
    rows = []
    if labels0 is not None and len(labels0) == n:
        for r in np.unique(labels0):
            msk = labels0 == r; w = A0[msk] / A0[msk].sum()
            for k in range(T):
                rows.append({"region": int(r), "frame": k, "time": float(t[k]), "cumulative_growth": float(np.sum(w * G[k, msk])),
                             "log_area_change": float(np.sum(w * LA[k, msk])), "region_area_fraction": float(A0[msk].sum() / A0.sum())})
    reg = pd.DataFrame(rows)
    if len(reg):
        reg.to_csv(out / "regions.csv", index=False)
    return G, LA, reg


def growth_anisotropy(out, X, F0, t, window=3):
    """
    Principal stretches λ1 ≥ λ2 of the deformation reference -> frame t (shape-only motion), anisotropy
    log(λ1/λ2) and the principal stretch direction (3-D, current frame) of every face; the sliding-window FTLE
    σ_k = ln sqrt(λmax C_{k→k+w}) / (t_{k+w} - t_k) shows WHEN separating regions form (the FTLE of the Dynamic
    Analysis integrates from the reference frame).
    """
    out = ensure_dir(out)
    T, n, _ = X.shape
    rows, wrows = [], []
    for k in pbar(range(1, T), desc="Growth anisotropy"):
        Fg, C, b1, b2, a0 = face_deformation(X[0], X[k], F0)
        lam, vec = np.linalg.eigh(C)                                         # ascending
        l1 = np.sqrt(np.maximum(lam[:, 1], 1e-300)); l2 = np.sqrt(np.maximum(lam[:, 0], 1e-300))
        aniso = np.log(l1 / l2)
        d = np.einsum("fij,fj->fi", Fg, vec[:, :, 1]); d /= np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
        np.savez_compressed(out / f"anisotropy_T{k:04d}.npz", lambda1=l1.astype(np.float32), lambda2=l2.astype(np.float32),
                            anisotropy=aniso.astype(np.float32), direction=d.astype(np.float32),
                            anisotropy_vertex=_face_to_vertex(aniso, F0, n, a0).astype(np.float32))
        rows.append({"frame": k, "time": float(t[k]), "mean_anisotropy": float(np.average(aniso, weights=a0)),
                     "p95_anisotropy": float(np.percentile(aniso, 95)), "mean_lambda1": float(np.average(l1, weights=a0)),
                     "mean_lambda2": float(np.average(l2, weights=a0))})
    for k in range(0, T - window):
        _, C, _, _, a0 = face_deformation(X[k], X[k + window], F0)
        s = np.log(np.sqrt(np.maximum(np.linalg.eigvalsh(C)[:, -1], 1e-300))) / max(t[k + window] - t[k], 1e-12)
        np.save(out / f"ftle_window_T{k:04d}.npy", _face_to_vertex(s, F0, n, a0).astype(np.float32))
        wrows.append({"frame_start": k, "frame_end": k + window, "time_start": float(t[k]), "time_end": float(t[k + window]),
                      "ftle_mean": float(np.average(s, weights=a0)), "ftle_p95": float(np.percentile(s, 95)), "ftle_max": float(s.max())})
    an = pd.DataFrame(rows); an.to_csv(out / "anisotropy_summary.csv", index=False)
    fw = pd.DataFrame(wrows)
    if len(fw):
        fw.to_csv(out / "ftle_window_summary.csv", index=False)
    return an, fw


# --------------------------------------------------------------------------------------------------------------- #
#  5. polarity, regional kinematics
# --------------------------------------------------------------------------------------------------------------- #
def polarity(out, X, t, frames=None, ms_seq=None):
    """Migration direction (centroid velocity) vs elongation axis (first principal axis of the shape): angle between
    them; front / rear asymmetry of the protrusion tips (Morse–Smale maxima ahead of / behind the centroid along
    the migration direction)."""
    out = ensure_dir(out)
    T = X.shape[0]
    c = X.mean(axis=1)
    v = np.gradient(c, t, axis=0) if T > 1 else np.zeros_like(c)
    rows = []
    ms_by_frame = {int(m.frame): m for m in (ms_seq or [])}
    for k in range(T):
        Y = X[k] - c[k]
        _, s, Vt = np.linalg.svd(Y, full_matrices=False)
        ax = Vt[0]; vk = v[k]; sp = np.linalg.norm(vk)
        ang = float(np.degrees(np.arccos(np.clip(abs(ax @ vk) / max(sp, 1e-300), 0, 1)))) if sp > 0 else np.nan
        row = {"frame": k, "time": float(t[k]), "speed": float(sp), "elongation": float(s[0] / max(s[1], 1e-300)),
               "angle_axis_motion_deg": ang}
        m = ms_by_frame.get(k)
        if m is not None and frames is not None and k < len(frames) and sp > 0:
            V = np.asarray(frames[k]["V"]); tips = V[np.asarray(list(m.maxima), dtype=int)]
            proj = (tips - V.mean(axis=0)) @ (vk / sp)
            row.update({"n_tips": int(len(tips)), "front_tips": int((proj > 0).sum()),
                        "front_fraction": float((proj > 0).mean()) if len(tips) else np.nan})
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(out / "polarity.csv", index=False)
    return df


def regional_kinematics(out, X, Xn, F0, t, labels0):
    """Mean speed, deformation-only speed and area change of every Morse–Smale region of the reference frame."""
    out = ensure_dir(out)
    T, n, _ = X.shape
    if labels0 is None or len(labels0) != n:
        return None
    vel = np.linalg.norm(np.gradient(X, t, axis=0), axis=2) if T > 1 else np.zeros((T, n))
    veln = np.linalg.norm(np.gradient(Xn, t, axis=0), axis=2) if T > 1 else np.zeros((T, n))
    A0 = np.asarray(vertex_areas(X[0], F0)); rows = []
    for r in np.unique(labels0):
        msk = labels0 == r; w = A0[msk] / A0[msk].sum()
        for k in range(T):
            Ak = np.asarray(vertex_areas(X[k], F0))[msk].sum()
            rows.append({"region": int(r), "frame": k, "time": float(t[k]), "mean_speed": float(np.sum(w * vel[k, msk])),
                         "mean_deformation_speed": float(np.sum(w * veln[k, msk])), "area_ratio": float(Ak / A0[msk].sum())})
    df = pd.DataFrame(rows); df.to_csv(out / "regional_kinematics.csv", index=False)
    return df


# --------------------------------------------------------------------------------------------------------------- #
#  6. protrusion kinetics, branches of the skeleton
# --------------------------------------------------------------------------------------------------------------- #
def protrusion_kinetics(target_folder, out, t, size):
    """Lifetimes, extension / retraction speeds (normal growth per time, size-normalised) and birth / death rates
    of the tracked Morse–Smale regions (Region_tracking)."""
    tracks = _load(target_folder, "MSComplexAnalysis/Region_tracking/region_tracks.csv")
    summ = _load(target_folder, "MSComplexAnalysis/Region_tracking/tracking_summary.csv")
    if tracks is None or not len(tracks):
        return None
    out = ensure_dir(out)
    tr = tracks.copy()
    f0 = tr.first_frame.clip(lower=0).astype(int).clip(upper=len(t) - 1); f1 = tr.last_frame.astype(int).clip(upper=len(t) - 1)
    tr["lifetime_time"] = t[f1.to_numpy()] - t[f0.to_numpy()]
    dur = tr.lifetime_time.replace(0, np.nan)
    if "mean_normal_growth" in tr:
        tr["extension_speed"] = tr.mean_normal_growth * tr.n_frames / dur
        tr["extension_speed_rel"] = tr.extension_speed / size
        tr["behaviour"] = np.where(tr.extension_speed > 0, "extending", np.where(tr.extension_speed < 0, "retracting", "stable"))
    tr.to_csv(out / "protrusion_kinetics.csv", index=False)
    stats = {"n_tracks": int(len(tr)), "mean_lifetime": float(np.nanmean(tr.lifetime_time)),
             "median_lifetime": float(np.nanmedian(tr.lifetime_time)),
             "fraction_born_during": float(np.mean(tr.born)) if "born" in tr else None,
             "fraction_died_during": float(np.mean(tr.died)) if "died" in tr else None,
             "mean_extension_speed_rel": float(np.nanmean(tr.extension_speed_rel)) if "extension_speed_rel" in tr else None}
    if summ is not None and len(summ):
        span = max(t[-1] - t[0], 1e-12)
        for c in ("n_birth", "n_death", "n_split", "n_merge"):
            if c in summ:
                stats[f"{c[2:]}_rate"] = float(summ[c].sum() / span)
    with open(out / "protrusion_statistics.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    return tr, stats


def branch_tracking(target_folder, out, graphs=None):
    """Branches of every animated Reeb skeleton (GraphAnimation): maximal paths between nodes of degree ≠ 2; length,
    relative extension and extension rate over the animation steps."""
    import networkx as nx
    ga = Path(target_folder) / "GraphAnimation"
    if not ga.is_dir():
        return None
    out = ensure_dir(out); allrows = []
    kinds = None if graphs is None else tuple([graphs] if isinstance(graphs, str) else graphs)
    for d in sorted(p for p in ga.iterdir() if p.is_dir() and (p / "animation.npz").exists()):
        if kinds is not None and d.name.split("_T")[0] not in kinds:      # graph kind of the animation
            continue
        z = np.load(d / "animation.npz", allow_pickle=True)
        P, E, tt = z["positions"], z["edges"].astype(int), z["times"]
        G = nx.Graph(); G.add_edges_from(map(tuple, E))
        junction = {v for v in G.nodes if G.degree(v) != 2}
        seen, branches = set(), []
        for u in junction:
            for w in G.neighbors(u):
                if (u, w) in seen:
                    continue
                path = [u, w]; seen.add((u, w)); seen.add((w, u))
                while path[-1] not in junction:
                    nxt = [x for x in G.neighbors(path[-1]) if x != path[-2]]
                    if not nxt:
                        break
                    seen.add((path[-1], nxt[0])); seen.add((nxt[0], path[-1])); path.append(nxt[0])
                branches.append(path)
        for b_id, path in enumerate(branches):
            L = np.sum(np.linalg.norm(np.diff(P[:, path], axis=1), axis=2), axis=1)
            rate = np.gradient(L, tt) if len(tt) > 1 else np.zeros_like(L)
            for s in range(len(tt)):
                allrows.append({"animation": d.name, "branch": b_id, "n_nodes": len(path), "start_node": int(path[0]),
                                "end_node": int(path[-1]), "step": s, "time": float(tt[s]), "length": float(L[s]),
                                "extension": float(L[s] / max(L[0], 1e-300) - 1), "extension_rate": float(rate[s])})
    if not allrows:
        return None
    df = pd.DataFrame(allrows); df.to_csv(out / "branches.csv", index=False)
    return df


# --------------------------------------------------------------------------------------------------------------- #
#  7. signals, timeline, synchronisation, frequencies, DMD
# --------------------------------------------------------------------------------------------------------------- #
def signal_table(target_folder, t, size, rigid=None, fields=None, aniso=None, polar=None):
    """One time series per aspect of the dynamics (size-normalised speeds: size units per time)."""
    T = len(t); S = pd.DataFrame({"frame": np.arange(T), "time": t})
    if rigid is not None:
        S["centroid_speed_rel"] = rigid.centroid_speed.to_numpy() / size
        S["angular_speed_deg"] = rigid.angular_speed_deg.to_numpy()
        S["deformation_fraction"] = rigid.deformation_fraction.to_numpy()
    if fields is not None:
        S["deformation_speed_rel"] = fields.get("mean_deformation_velocity", fields["mean_velocity"]).to_numpy() / size
        S["mean_strain"] = fields.mean_strain_norm.to_numpy()
        S["normal_speed_rel"] = fields.mean_normal_speed.to_numpy() / size
    sc = _load(target_folder, "DynamicAnalysis/ShapeSpace/scores.npy")
    if sc is not None and len(sc) == T and T > 1:
        S["shape_change_speed"] = np.linalg.norm(np.gradient(np.asarray(sc), t, axis=0), axis=1)
        for j in range(min(2, sc.shape[1])):
            S[f"shape_mode{j + 1}"] = np.asarray(sc)[:, j]
    bg = _load(target_folder, "Basic_Geometry/features_computed.csv")
    if bg is not None and len(bg) == T and T > 1:
        for col in ("area", "volume"):
            if col in bg:
                v = bg[col].to_numpy(float); S[f"{col}_growth_rate"] = np.gradient(v, t) / np.maximum(np.abs(v), 1e-300)
    hd = _load(target_folder, "DynamicAnalysis/Hodge/hodge_energy.csv")
    if hd is not None and len(hd) == T:
        S["curl_free_fraction"] = hd.frac_curl_free.to_numpy(); S["div_free_fraction"] = hd.frac_div_free.to_numpy()
    summ = _load(target_folder, "MSComplexAnalysis/Region_tracking/tracking_summary.csv")
    if summ is not None and "Time_Step" in summ:
        for c in ("n_regions_curr", "n_birth", "n_death"):
            if c in summ:
                v = np.zeros(T); k = summ.Time_Step.to_numpy(int); ok = k < T; v[k[ok]] = summ[c].to_numpy(float)[ok]
                if c == "n_regions_curr" and T:
                    v[0] = summ.n_regions_prev.iloc[0] if "n_regions_prev" in summ else v[1]
                S[{"n_regions_curr": "n_protrusions", "n_birth": "protrusion_births", "n_death": "protrusion_deaths"}[c]] = v
    if aniso is not None and len(aniso):
        v = np.zeros(T); v[aniso.frame.to_numpy(int)] = aniso.mean_anisotropy.to_numpy(); S["mean_anisotropy"] = v
    if polar is not None and "angle_axis_motion_deg" in polar:
        S["axis_motion_angle_deg"] = polar.angle_axis_motion_deg.to_numpy()
    return S


def event_timeline(target_folder, t):
    """All the discrete events of the sequence in one table: splits / merges of the surface and of the Reeb graph,
    cycles, protrusion births / deaths / splits / merges, regime changes (change points)."""
    ev = []
    ce = _load(target_folder, "Basic_Geometry/component_events.csv")
    if ce is not None:
        for _, r in ce.iterrows():
            ev.append({"frame": int(r["frame"]), "source": "surface", "event": str(r.get("event", "component change")), "detail": ""})
    tp = _load(target_folder, "Graph_analysis/topology_per_frame.csv")
    if tp is not None:
        for _, r in tp.iterrows():
            if r.get("Delta_Betti_0", 0) not in (0, np.nan) and str(r.get("Component_Event", "")) not in ("", "nan"):
                ev.append({"frame": int(r["Frame"]), "source": "Reeb graph", "event": str(r["Component_Event"]), "detail": ""})
            if r.get("Delta_Betti_1", 0) and not np.isnan(r.get("Delta_Betti_1", 0)) and r["Delta_Betti_1"] != 0:
                ev.append({"frame": int(r["Frame"]), "source": "Reeb graph",
                           "event": "cycle created" if r["Delta_Betti_1"] > 0 else "cycle removed", "detail": f"{int(r['Delta_Betti_1']):+d}"})
    summ = _load(target_folder, "MSComplexAnalysis/Region_tracking/tracking_summary.csv")
    if summ is not None:
        for _, r in summ.iterrows():
            for c, name in (("n_birth", "protrusion birth"), ("n_death", "protrusion death"), ("n_split", "protrusion split"), ("n_merge", "protrusion merge")):
                if c in summ and r[c] > 0:
                    ev.append({"frame": int(r["Time_Step"]), "source": "protrusions", "event": name, "detail": f"x{int(r[c])}"})
    rd = _load(target_folder, "ReebDynamics/events.csv")              # time-varying Reeb graph (events between frames)
    if rd is not None and len(rd) and len(t) > 1:
        for _, r in rd.iterrows():
            k = int(np.clip(np.searchsorted(t, float(r.time), side="right") - 1, 0, len(t) - 1))
            ev.append({"frame": k, "source": "Reeb dynamics", "event": str(r.event).split(" (")[0], "detail": f"t={float(r.time):.4g}"})
    cp = _load(target_folder, "DynamicAnalysis/ChangePoints/change_points.json")
    if cp:
        for src, d in cp.items():
            if not isinstance(d, dict):
                continue
            for key, val in d.items():
                if key.endswith("change_points") and isinstance(val, list):
                    for c in val:
                        ev.append({"frame": int(c), "source": "regime change", "event": key.replace("_", " "), "detail": src})
    df = pd.DataFrame(ev, columns=["frame", "source", "event", "detail"])
    if len(df):
        df = df[df.frame < len(t)].copy(); df.insert(1, "time", t[df.frame.to_numpy(int)]); df = df.sort_values(["frame", "source"])
    return df


def synchronization(S, max_lag=None):
    """Normalised cross-correlation of every pair of signals for lags −L..L (uniform resampling first): the maximum
    |correlation| and its lag (positive lag: the first signal LEADS the second)."""
    cols = [c for c in S.columns if c not in ("frame", "time") and S[c].std() > 0]
    if len(cols) < 2 or len(S) < 4:
        return None, None
    tu, Y = _resample_uniform(S.time.to_numpy(float), S[cols].to_numpy(float))
    Y = (Y - Y.mean(0)) / np.maximum(Y.std(0), 1e-300)
    T = len(tu); L = int(max_lag if max_lag is not None else max(1, T // 4)); dt = tu[1] - tu[0] if T > 1 else 1.0
    best = np.zeros((len(cols), len(cols))); lag = np.zeros_like(best)
    for i in range(len(cols)):
        for j in range(len(cols)):
            cc = []
            for l in range(-L, L + 1):
                a, b = (Y[:T - l, i], Y[l:, j]) if l >= 0 else (Y[-l:, i], Y[:T + l, j])
                cc.append(np.mean(a * b) if len(a) > 2 else 0.0)
            cc = np.array(cc); k = int(np.argmax(np.abs(cc)))
            best[i, j] = cc[k]; lag[i, j] = (k - L) * dt
    return pd.DataFrame(best, index=cols, columns=cols), pd.DataFrame(lag, index=cols, columns=cols)


def frequency_analysis(S):
    """Lomb–Scargle periodogram of every signal (works with irregular frame times): dominant period and its relative
    power (fraction of the periodogram), next to the persistent homology which only says WHETHER the motion is cyclic."""
    from scipy.signal import lombscargle
    t = S.time.to_numpy(float); T = len(t)
    if T < 5:
        return None, None
    span = t[-1] - t[0]; dtm = np.median(np.diff(t))
    freqs = np.linspace(1.0 / span, 0.5 / dtm, 200)
    rows, spec = [], {"frequency": freqs}
    for c in S.columns:
        if c in ("frame", "time") or S[c].std() == 0:
            continue
        y = S[c].to_numpy(float); y = y - y.mean()
        p = lombscargle(t, y, 2 * np.pi * freqs, normalize=True)
        spec[c] = p; k = int(np.argmax(p))
        rows.append({"signal": c, "dominant_period": float(1 / freqs[k]), "dominant_frequency": float(freqs[k]),
                     "peak_power": float(p[k]), "power_fraction": float(p[k] / max(p.sum(), 1e-300))})
    return pd.DataFrame(rows), pd.DataFrame(spec)


def dmd_analysis(out, Xn, F0, t, energy=0.999, forecast=3, delays="auto"):
    """
    Dynamic Mode Decomposition of the shape-only motion (uniformly resampled), with time-delay (Hankel) embedding:
    the motion is first reduced by POD (a_k = U_rᵀ (x_k − x̄)), d consecutive coordinate vectors are stacked into one
    state h_k = [a_k; …; a_{k+d−1}] and h_{k+1} ≈ A h_k is fitted (exact DMD). The delays make oscillations visible
    even when one spatial pattern only oscillates in place (a standing wave has rank 1 and plain DMD cannot
    represent it). Eigenvalues μ_j -> continuous rates λ_j = ln μ_j / Δt: frequency Im λ / 2π and growth (Re λ > 0)
    or decay rate. Writes the modes (frequency, period, growth rate, amplitude, energy), the reconstruction error,
    the leading modes as displacement meshes and a forecast of the next `forecast` steps.
    """
    out = ensure_dir(out)
    T, n, _ = Xn.shape
    if T < 6:
        return None
    tu, Xu = _resample_uniform(t, Xn); dt = tu[1] - tu[0]
    D = Xu.reshape(T, -1).T                                                    # (3n, T)
    mean = D.mean(axis=1, keepdims=True); Dc = D - mean
    U, s, _ = (to_cpu(a) for a in xp.linalg.svd(to_gpu(Dc), full_matrices=False))
    r = int(np.searchsorted(np.cumsum(s ** 2) / max(np.sum(s ** 2), 1e-300), energy) + 1); r = max(1, min(r, len(s), T - 2))
    U = U[:, :r]; A = U.T @ Dc                                                 # POD coordinates (r, T)
    d = int(min(max(2, T // 4), 8)) if delays == "auto" else max(1, int(delays))
    d = min(d, T - 3)
    H = np.vstack([A[:, k:T - d + 1 + k] for k in range(d)])                  # Hankel (r d, T − d + 1)
    H1, H2 = H[:, :-1], H[:, 1:]
    Uh, sh, Vh = np.linalg.svd(H1, full_matrices=False)
    q = int(np.searchsorted(np.cumsum(sh ** 2) / max(np.sum(sh ** 2), 1e-300), energy) + 1); q = max(2, min(q, len(sh)))
    Uh, sh, Vh = Uh[:, :q], sh[:q], Vh[:q]
    At = Uh.T @ H2 @ Vh.T / sh[None, :]
    mu, W = np.linalg.eig(At)
    Phi_h = H2 @ Vh.T / sh[None, :] @ W                                       # exact DMD modes in delay space
    b = np.linalg.lstsq(Phi_h, H[:, 0].astype(complex), rcond=None)[0]
    lam = np.log(mu.astype(complex)) / dt
    Phi = U @ Phi_h[:r]                                                        # spatial modes (first delay block)
    steps = np.arange(T)
    rec = (Phi @ (b[:, None] * mu[:, None] ** steps[None, :])).real + mean
    err = float(np.sqrt(np.mean((rec - D) ** 2)) / max(np.sqrt(np.mean(Dc ** 2)), 1e-300))
    energy_j = np.abs(b) ** 2 * np.sum(np.abs(Phi) ** 2, axis=0)
    modes = pd.DataFrame({"mode": np.arange(len(mu)), "frequency": lam.imag / (2 * np.pi),
                          "period": np.where(np.abs(lam.imag) > 1e-12, 2 * np.pi / np.maximum(np.abs(lam.imag), 1e-300), np.inf),
                          "growth_rate": lam.real, "abs_eigenvalue": np.abs(mu), "amplitude": np.abs(b),
                          "energy_fraction": energy_j / max(energy_j.sum(), 1e-300)}).sort_values("energy_fraction", ascending=False)
    modes.to_csv(out / "dmd_modes.csv", index=False)
    d_m = ensure_dir(out / "dmd_mode_meshes")
    base = mean.reshape(n, 3)
    for jm in modes["mode"].to_numpy()[:4]:
        disp = (Phi[:, jm] * b[jm]).real.reshape(n, 3)
        sc = 0.05 * np.ptp(base, axis=0).max() / max(np.abs(disp).max(), 1e-300)
        write_obj(d_m / f"dmd_mode{int(jm)}_plus.obj", base + sc * disp, F0); write_obj(d_m / f"dmd_mode{int(jm)}_minus.obj", base - sc * disp, F0)
    if forecast:
        fut = np.arange(T, T + int(forecast))
        pred = (Phi @ (b[:, None] * mu[:, None] ** fut[None, :])).real + mean
        np.save(out / "dmd_forecast.npy", pred.T.reshape(len(fut), n, 3).astype(np.float32))
        np.save(out / "dmd_forecast_times.npy", tu[-1] + dt * (fut - T + 1))
    info = {"pod_rank": r, "delays": d, "dmd_rank": q, "dt": float(dt), "reconstruction_rel_error": err, "n_forecast": int(forecast)}
    with open(out / "dmd_info.json", "w") as fh:
        json.dump(info, fh, indent=2)
    return modes, info


# --------------------------------------------------------------------------------------------------------------- #
#  plots
# --------------------------------------------------------------------------------------------------------------- #
def _region_colors(regions):
    """One colour per region from a colormap (+ the ScalarMappable for a colorbar: a legend with many regions would
    be unreadable)."""
    regions = sorted(regions)
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(len(regions) - 1, 1))
    return {r: cmap(norm(i)) for i, r in enumerate(regions)}, matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), regions


def _region_figure(df, cols, titles, panel_w=6.0):
    """Figure with one panel per column of `df` (one line per Morse–Smale region) and the region colour scale in its
    own narrow grid column (a colorbar stealing space from the panels would be drawn over the last one)."""
    colors, sm, regions = _region_colors(df.region.unique())
    n = len(cols)
    fig = plt.figure(figsize=(panel_w * n + 1.2, 4.6))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.045], wspace=0.28)
    axes = [fig.add_subplot(gs[0, k]) for k in range(n)]
    for a, col, ttl in zip(axes, cols, titles):
        for r in regions:
            g = df[df.region == r]; a.plot(g["time"], g[col], "-", color=colors[r], lw=1.2)
        a.set_title(ttl); a.set_xlabel("time"); a.grid(alpha=0.3)
    cax = fig.add_subplot(gs[0, n])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Morse–Smale region of the reference frame")
    if len(regions) <= 12:
        cb.set_ticks(range(len(regions))); cb.set_ticklabels([str(r) for r in regions])
    else:
        cb.set_ticks([0, len(regions) - 1]); cb.set_ticklabels(["first", "last"])
    return fig


def _plots_extra(out, conf, rig, fields, cmp_, reg, polar, regk, spec):
    """Plots of the tables that plots 1-9 do not show (confidence, field comparison, regions, polarity, spectra)."""
    if conf is not None and len(conf):
        t = rig.time if rig is not None and len(rig) == len(conf) else conf.frame
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(t, conf.confidence, "o-", color="tab:blue", label="confidence of the correspondences")
        lo = conf.low_confidence.to_numpy(bool)
        if lo.any():
            ax.scatter(np.asarray(t)[lo], conf.confidence[lo], s=80, facecolors="none", edgecolors="tab:red", lw=2, label="low confidence")
        if "possible_symmetry_flip" in conf:
            for k_, f_ in enumerate(conf.possible_symmetry_flip.to_numpy(bool)):
                if f_:
                    ax.axvline(np.asarray(t)[k_], color="tab:orange", ls="--", lw=1.5)
            ax.plot([], [], color="tab:orange", ls="--", label="probable symmetric flip of the maps")
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("time"); ax.set_ylabel("confidence"); ax.legend(fontsize=8)
        ax.set_title("reliability of every frame (tracking diagnostics)")
        _savefig(fig, out, "10_frame_confidence.png")
    if fields is not None and len(fields):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        if cmp_ is not None and len(cmp_):
            tt = fields.set_index("frame").time.reindex(cmp_.frame).to_numpy()
            ax[0].plot(tt, cmp_.mean_speed_frame_to_frame, "s--", label="frame-to-frame maps (Physical_fields)")
            ax[0].plot(tt, cmp_.mean_speed_trajectories, "o-", label="trajectories (from_trajectories)")
            ax[0].set_title("mean speed: two sources"); ax[0].legend(fontsize=8)
            ax[1].plot(tt, cmp_.ratio, "o-", color="tab:purple"); ax[1].axhline(1, color="gray", lw=0.8)
            ax[1].set_title("ratio trajectories / frame-to-frame")
        else:
            ax[0].plot(fields.time, fields.mean_velocity, "o-"); ax[0].set_title("mean speed (trajectories)")
            ax[1].axis("off")
        ax[2].plot(fields.time, fields.mean_strain_norm, "o-", label="strain norm (Green–Lagrange)")
        ax[2].plot(fields.time, fields.mean_area_strain, "s--", label="area strain")
        ax2 = ax[2].twinx(); ax2.plot(fields.time, fields.mean_normal_speed, "^:", color="tab:green", label="|normal speed|")
        ax2.set_ylabel("|normal speed|")
        h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax[2].legend(h1 + h2, l1 + l2, fontsize=8); ax[2].set_title("deformation of the tracked mesh")
        for a in ax:
            a.set_xlabel("time")
        _savefig(fig, out, "11_trajectory_vs_frame_fields.png")
    if reg is not None and len(reg):
        fig = _region_figure(reg, ["cumulative_growth", "log_area_change"],
                             ["cumulative normal growth per region", "local area change log(A / A0) per region"])
        _savefig(fig, out, "12_growth_atlas_regions.png")
    if polar is not None and len(polar):
        fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
        ax[0, 0].plot(polar.time, polar.speed, "o-"); ax[0, 0].set_title("speed of the centroid")
        ax[0, 1].plot(polar.time, polar.angle_axis_motion_deg, "o-", color="tab:red"); ax[0, 1].set_ylim(0, 92)
        ax[0, 1].set_title("angle elongation axis / direction of motion (0 = moves along its long axis)")
        if "front_fraction" in polar:
            ax[1, 0].plot(polar.time, polar.front_fraction, "o-", color="tab:green"); ax[1, 0].axhline(0.5, color="gray", lw=0.8)
            ax[1, 0].set_ylim(-0.05, 1.05); ax[1, 0].set_title("fraction of protrusion tips ahead of the centroid")
        else:
            ax[1, 0].axis("off")
        ax[1, 1].plot(polar.time, polar.elongation, "o-", color="tab:brown"); ax[1, 1].set_title("elongation (first / second principal axis)")
        for a in ax.ravel():
            a.set_xlabel("time"); a.grid(alpha=0.3)
        _savefig(fig, out, "13_polarity.png")
    if regk is not None and len(regk):
        fig = _region_figure(regk, ["mean_speed", "mean_deformation_speed", "area_ratio"],
                             ["mean speed per region", "deformation-only speed per region", "area ratio A / A0 per region"])
        _savefig(fig, out, "14_regional_kinematics.png")
    if spec is not None and len(spec) and spec.shape[1] > 1:
        cols = [c for c in spec.columns if c != "frequency"]
        P = spec[cols].to_numpy(float).T
        P = P / np.maximum(P.max(axis=1, keepdims=True), 1e-300)        # each signal normalised to its own peak
        per = 1.0 / spec.frequency.to_numpy(float)
        fig, ax = plt.subplots(figsize=(11, 0.32 * len(cols) + 2.2))
        im = ax.imshow(P, aspect="auto", cmap="magma", extent=[0, len(per), len(cols) - 0.5, -0.5], interpolation="nearest")
        ticks = np.linspace(0, len(per) - 1, 8).astype(int)
        ax.set_xticks(ticks + 0.5); ax.set_xticklabels([f"{per[i]:.3g}" for i in ticks])
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
        ax.set_xlabel("period (time units; long periods on the left)"); ax.set_title("periodogram of every signal (normalised per signal)")
        fig.colorbar(im, ax=ax, label="relative power")
        _savefig(fig, out, "15_periodograms.png")


def _plot_field_comparison(out, df):
    """16: every field shared by the frame-to-frame and the trajectory physical fields — mean |value| of both sources
    over time (left) and their agreement: correlation of the per-vertex values and relative difference (right)."""
    if df is None or not len(df):
        return
    keys = [k for k in FIELD_PAIRS if k in set(df.field)]
    fig, ax = plt.subplots(len(keys), 2, figsize=(14, 2.8 * len(keys) + 0.6), squeeze=False)
    for r, key in enumerate(keys):
        g = df[df.field == key].sort_values("time")
        a = ax[r, 0]
        a.plot(g.time, g.mean_abs_frame_to_frame, "s--", ms=3, label="frame-to-frame maps (Physical_fields)")
        a.plot(g.time, g.mean_abs_trajectories, "o-", ms=3, label="trajectories (tracked mesh)")
        a.set_title(f"{FIELD_PAIRS[key]}: mean |value|"); a.grid(alpha=0.3)
        if r == 0:
            a.legend(fontsize=8)
        b = ax[r, 1]
        b.plot(g.time, g.correlation, "o-", color="tab:green", ms=3, label="correlation per vertex")
        b.set_ylim(-1.05, 1.05); b.axhline(0, color="gray", lw=0.6)
        b2 = b.twinx(); b2.plot(g.time, g.rel_difference, "^:", color="tab:red", ms=3, label="relative difference")
        b2.set_ylabel("median |diff| / median |f2f|", fontsize=8)
        b.set_title(f"{FIELD_PAIRS[key]}: agreement of the two sources"); b.grid(alpha=0.3)
        if r == 0:
            h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels(); b.legend(h1 + h2, l1 + l2, fontsize=8)
    for a in ax[-1]:
        a.set_xlabel("time")
    fig.suptitle("physical fields: frame-to-frame maps vs tracked mesh (same definitions, transition i-1 -> i)")
    _savefig(fig, out, "16_field_comparison.png")


def _plots(out, rigid, S, events, sync, lags, freq, modes, an, fw, prot, branches, conf=None):
    if rigid is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(rigid.time, rigid.translation, "o-"); ax[0].set_title("translation of the centroid"); ax[0].set_xlabel("time")
        ax[1].plot(rigid.time, rigid.rotation_deg, "o-", label="rotation"); ax[1].plot(rigid.time, rigid.angular_speed_deg, "s--", label="angular speed (deg / time)")
        ax[1].set_title("rotation of the object"); ax[1].legend(); ax[1].set_xlabel("time")
        ax[2].plot(rigid.time, rigid.deformation_fraction, "o-", color="tab:red"); ax[2].set_ylim(0, 1.05)
        ax[2].set_title("deformation fraction of the motion (0 = rigid)"); ax[2].set_xlabel("time")
        _savefig(fig, out, "1_rigid_motion.png")
    if S is not None:
        cols = [c for c in S.columns if c not in ("frame", "time")]
        fig, axes = plt.subplots(int(np.ceil(len(cols) / 3)), 3, figsize=(15, 2.6 * np.ceil(len(cols) / 3)), squeeze=False)
        low_t = [] if conf is None or len(conf) != len(S) else S.time[conf.low_confidence.to_numpy(bool)].tolist()
        for a, c in zip(axes.ravel(), cols):
            a.plot(S.time, S[c], "o-", ms=3); a.set_title(c, fontsize=9)
            for tt in low_t:                                      # frames with unreliable correspondences
                a.axvspan(tt - 0.5 * np.median(np.diff(S.time)), tt + 0.5 * np.median(np.diff(S.time)), color="tab:red", alpha=0.12)
            if events is not None and len(events):
                for tt in events.time.unique():
                    a.axvline(tt, color="gray", lw=0.6, alpha=0.5)
        for a in axes.ravel()[len(cols):]:
            a.axis("off")
        if low_t:
            fig.suptitle("grey lines: events | red bands: frames with low correspondence confidence", fontsize=10)
        _savefig(fig, out, "2_signals_with_events.png")
    if events is not None and len(events):
        src = list(dict.fromkeys(events.source)); fig, ax = plt.subplots(figsize=(13, 1.5 + 1.1 * len(src)))
        for i, sname in enumerate(src):
            e = events[events.source == sname]; ax.scatter(e.time, np.full(len(e), i), s=50, label=sname, zorder=3)
            # one label per time (events at the same time merged), staggered vertically; long rows: markers only
            groups = e.groupby("time")
            if len(groups) <= 40:
                for n_, (tt, g) in enumerate(groups):
                    cnt = g.event.value_counts()
                    short = {"mean shift change points": "shift", "speed change points": "speed", "change points": "change"}
                    txt = ", ".join((f"{short.get(ev, ev.replace('protrusion ', ''))} x{c}" if c > 1
                                     else short.get(ev, ev.replace("protrusion ", ""))) for ev, c in cnt.items())
                    ax.annotate(txt, (tt, i), fontsize=6, xytext=(3, 5 + 8 * (n_ % 4)), textcoords="offset points",
                                ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        ax.set_yticks(range(len(src))); ax.set_yticklabels(src); ax.set_ylim(-0.5, len(src) - 0.1)
        ax.set_xlabel("time"); ax.set_title("event timeline (details in Signals/event_timeline.csv)")
        _savefig(fig, out, "3_event_timeline.png")
    if sync is not None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
        im = ax[0].imshow(sync.values, cmap="RdBu_r", vmin=-1, vmax=1); ax[0].set_title("max cross-correlation")
        im2 = ax[1].imshow(lags.values, cmap="PuOr"); ax[1].set_title("lag of the maximum (row leads column if > 0)")
        for a, i_ in ((ax[0], im), (ax[1], im2)):
            a.set_xticks(range(len(sync))); a.set_xticklabels(sync.columns, rotation=90, fontsize=7)
            a.set_yticks(range(len(sync))); a.set_yticklabels(sync.index, fontsize=7); fig.colorbar(i_, ax=a)
        _savefig(fig, out, "4_synchronization.png")
    if freq is not None and len(freq):
        fig, ax = plt.subplots(figsize=(9, 0.35 * len(freq) + 1.5))
        ax.barh(freq.signal, freq.dominant_period, color=plt.cm.viridis(freq.power_fraction / max(freq.power_fraction.max(), 1e-300)))
        ax.set_xlabel("dominant period (time units)"); ax.set_title("dominant periods (colour = relative power)")
        _savefig(fig, out, "5_periods.png")
    if modes is not None and len(modes):
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(modes.frequency, modes.growth_rate, s=40 + 400 * modes.energy_fraction, c=modes.energy_fraction, cmap="plasma")
        ax.axhline(0, color="gray", lw=0.8); ax.set_xlabel("frequency (1 / time)"); ax.set_ylabel("growth rate (1 / time; < 0 decays)")
        ax.set_title("DMD modes of the shape-only motion (size = energy)"); fig.colorbar(sc, ax=ax, label="energy fraction")
        _savefig(fig, out, "6_dmd_modes.png")
    if an is not None and len(an):
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(an.time, an.mean_anisotropy, "o-", label="mean"); ax[0].plot(an.time, an.p95_anisotropy, "s--", label="95th percentile")
        ax[0].set_title("growth anisotropy log(λ1 / λ2)"); ax[0].legend(); ax[0].set_xlabel("time")
        if fw is not None and len(fw):
            ax[1].plot(fw.time_start, fw.ftle_mean, "o-", label="mean"); ax[1].plot(fw.time_start, fw.ftle_p95, "s--", label="95th percentile")
            ax[1].set_title("sliding-window FTLE"); ax[1].legend(); ax[1].set_xlabel("window start time")
        _savefig(fig, out, "7_anisotropy_ftle_window.png")
    if prot is not None and len(prot):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(prot.lifetime_time.dropna(), bins=15, color="tab:blue"); ax[0].set_title("protrusion lifetimes"); ax[0].set_xlabel("lifetime")
        if "extension_speed_rel" in prot:
            ax[1].hist(prot.extension_speed_rel.dropna(), bins=15, color="tab:green"); ax[1].axvline(0, color="k", lw=0.8)
            ax[1].set_title("extension (> 0) / retraction (< 0) speed (size / time)")
        _savefig(fig, out, "8_protrusion_kinetics.png")
    if branches is not None and len(branches):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for (anim, b), g in branches.groupby(["animation", "branch"]):
            ax.plot(g.time, g.extension, lw=1)
        ax.set_title("extension of the skeleton branches (L / L0 - 1)"); ax.set_xlabel("time")
        _savefig(fig, out, "9_branch_extension.png")


# --------------------------------------------------------------------------------------------------------------- #
#  stage driver
# --------------------------------------------------------------------------------------------------------------- #
@logged_stage("MotionAnalysis", (0, "target_folder"), "MotionAnalysis")
def compute_motion_analysis(target_folder, frames=None, X=None, F0=None, times=None, verbose=True, **params):
    """Stage driver (see the module docstring); `params` override MOTION_DEFAULTS."""
    p = {**MOTION_DEFAULTS, **params}
    target_folder = Path(target_folder); out = ensure_dir(target_folder / "MotionAnalysis"); ensure_dir(out / "plots")
    if X is None:
        xf = target_folder / "Trajectories" / "trajectories.npy"
        if not xf.exists():
            print("Motion analysis skipped: no trajectories (compute_trajectories)."); return None
        X = np.load(xf); F0 = np.load(target_folder / "Trajectories" / "reference_faces.npy")
    X = np.asarray(X, dtype=np.float64); F0 = np.asarray(F0, dtype=np.int64); T, n, _ = X.shape
    t, unit = _times(target_folder, T, None if times is None else getattr(times, "t", times))
    size = float(np.sqrt(face_normals_areas(X[0], F0)[1].sum()))
    steps = [k for k in ("rigid", "fields", "growth_atlas", "anisotropy", "polarity", "regions", "protrusions", "branches", "signals", "dmd") if p[k]]
    progress(total=len(steps) + 2, advance=0)
    summary = {"n_frames": T, "time_unit": unit, "size": size}
    conf = frame_confidence(target_folder, T, size, p["low_confidence"])
    ensure_dir(out / "Confidence"); conf.to_csv(out / "Confidence" / "frame_confidence.csv", index=False)
    summary["low_confidence_frames"] = conf.frame[conf.low_confidence].tolist(); progress("frame confidence")
    ms_seq, labels0 = None, None
    if (target_folder / "MSComplexAnalysis" / "MS_Complex").is_dir():
        try:
            from PynamicMesh.core.SMComplex import load_ms_sequence
            ms_seq = load_ms_sequence(target_folder)
            m0 = [m for m in ms_seq if int(m.frame) == 0]
            labels0 = np.asarray(m0[0].label_max) if m0 else None
        except Exception as exc:  # noqa: BLE001
            print(f"[Warning] Morse–Smale results not usable: {exc}")
    rig = fields = an = fw = polar = prot = branches = S = events = sync = lags = freq = modes = None
    reg = regk = spec = cmp_ = fcmp = None
    Xn = X
    if p["rigid"]:
        rd = rigid_decomposition(X, F0, t); rig = rd["table"]; Xn = rd["nonrigid"]
        # a jump of the rigid orientation between consecutive frames is (almost always) a symmetric flip of the
        # correspondences (front / back, left / right) and not a real rotation: flagged with the frame confidence
        flips = rig.frame[rig.frame_to_frame_rotation_deg > p["flip_angle"]].tolist()
        conf["possible_symmetry_flip"] = conf.frame.isin(flips)
        conf.to_csv(out / "Confidence" / "frame_confidence.csv", index=False)
        summary["possible_symmetry_flips"] = flips
        if flips and verbose:
            print(f"[Warning] frames {flips}: the orientation jumps by more than {p['flip_angle']:.0f} deg between frames - "
                  "probably a symmetric flip of the correspondences (use the symmetry options of the functional maps)")
        d = ensure_dir(out / "RigidMotion"); rig.to_csv(d / "rigid_motion.csv", index=False)
        np.save(d / "nonrigid_trajectories.npy", Xn.astype(np.float32)); np.save(d / "rotations.npy", rd["R"])
        summary.update({"mean_deformation_fraction": float(rig.deformation_fraction[1:].mean()) if T > 1 else 0.0,
                        "total_rotation_deg": float(rig.rotation_deg.iloc[-1]), "net_translation_rel": float(rig.translation.iloc[-1] / size)})
        progress("rigid / non-rigid decomposition")
    if p["fields"]:
        fields = trajectory_fields(target_folder, target_folder / "Physical_fields" / "from_trajectories", X, F0, t, Xn if p["rigid"] else None)
        fcmp = field_comparison(target_folder, target_folder / "Physical_fields" / "from_trajectories", X, F0, t)
        fields.to_csv(out / "trajectory_fields_global.csv", index=False); progress("trajectory fields")
    if p["growth_atlas"]:
        _, _, reg = growth_atlas(out / "GrowthAtlas", X, F0, t, labels0); progress("growth atlas")
        reg = reg if reg is not None and len(reg) else None
    if p["anisotropy"]:
        an, fw = growth_anisotropy(out / "Anisotropy", Xn, F0, t, int(p["ftle_window"]))
        if len(an):
            summary["mean_anisotropy_final"] = float(an.mean_anisotropy.iloc[-1])
        progress("growth anisotropy")
    if p["polarity"]:
        polar = polarity(out / "Polarity", X, t, frames, ms_seq)
        if "front_fraction" in polar:
            summary["mean_front_fraction"] = float(polar.front_fraction.mean())
        progress("polarity")
    if p["regions"] and labels0 is not None:
        regk = regional_kinematics(out / "Regions", X, Xn, F0, t, labels0); progress("regional kinematics")
    if p["protrusions"]:
        r = protrusion_kinetics(target_folder, out / "Protrusions", t, size)
        if r is not None:
            prot, st = r; summary["protrusions"] = st
        progress("protrusion kinetics")
    if p["branches"]:
        branches = branch_tracking(target_folder, out / "Branches", p.get("graphs")); progress("skeleton branches")
    if p["signals"]:
        d = ensure_dir(out / "Signals")
        S = signal_table(target_folder, t, size, rig, fields, an, polar); S.to_csv(d / "signals.csv", index=False)
        S.merge(conf, on="frame").to_csv(d / "signals_with_confidence.csv", index=False)
        if p["timeline"]:
            events = event_timeline(target_folder, t); events.to_csv(d / "event_timeline.csv", index=False)
        if p["synchronization"]:
            sync, lags = synchronization(S, p["max_lag"])
            if sync is not None:
                sync.to_csv(d / "cross_correlation.csv"); lags.to_csv(d / "cross_correlation_lag.csv")
        if p["frequency"]:
            freq, spec = frequency_analysis(S)
            if freq is not None:
                freq.to_csv(d / "periods.csv", index=False); spec.to_csv(d / "periodograms.csv", index=False)
                summary["dominant_periods"] = dict(zip(freq.signal, freq.dominant_period.round(6)))
        progress("signals, timeline, synchronisation, frequencies")
    if p["dmd"]:
        r = dmd_analysis(out / "Signals" / "DMD", Xn, F0, t, p["dmd_energy"], int(p["dmd_forecast"]), p["dmd_delays"])
        if r is not None:
            modes, info = r; summary["dmd"] = info
        progress("dynamic mode decomposition")
    if p["plots"]:
        cf = target_folder / "Physical_fields" / "from_trajectories" / "comparison.csv"
        cmp_ = pd.read_csv(cf) if p["fields"] and cf.exists() else None
        _plots(out, rig, S, events, sync, lags, freq, modes, an, fw, prot, branches, conf=conf)
        _plots_extra(out, conf, rig, fields, cmp_, reg, polar, regk, spec)
        _plot_field_comparison(out, fcmp)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    progress("plots and summary")
    if verbose:
        print(f"Motion analysis: {T} frames; results in {out}")
    return summary


# --------------------------------------------------------------------------------------------------------------- #
#  comparison of the scenes of a batch
# --------------------------------------------------------------------------------------------------------------- #
def scene_features(scene_dir):
    """Size-normalised descriptors of one scene (for phenotyping): geometry, motion, protrusions, spectra."""
    d = Path(scene_dir); f = {}
    bg = _load(d, "Basic_Geometry/features_computed.csv")
    if bg is not None and len(bg):
        for c in ("sphericity", "convexity"):
            if c in bg:
                f[f"{c}_mean"] = float(bg[c].mean())
        if "area" in bg:
            f["area_change_rel"] = float(bg.area.iloc[-1] / bg.area.iloc[0] - 1)
        if "volume" in bg and bg.volume.notna().all():
            f["volume_change_rel"] = float(bg.volume.iloc[-1] / bg.volume.iloc[0] - 1)
    sm = _load(d, "MotionAnalysis/summary.json")
    if sm:
        for k in ("mean_deformation_fraction", "total_rotation_deg", "net_translation_rel", "mean_anisotropy_final", "mean_front_fraction"):
            if sm.get(k) is not None:
                f[k] = float(sm[k])
        for k, v in (sm.get("protrusions") or {}).items():
            if isinstance(v, (int, float)) and v is not None:
                f[f"protrusion_{k}"] = float(v)
    S = _load(d, "MotionAnalysis/Signals/signals.csv")
    if S is not None:
        for c in ("centroid_speed_rel", "deformation_speed_rel", "shape_change_speed", "mean_strain"):
            if c in S:
                f[f"{c}_mean"] = float(S[c].mean())
    de = _load(d, "Parametrization/SPHARM/spharm_degree_energy.csv")
    if de is not None and "energy" in de and "degree" in de:
        e = de.groupby("degree").energy.mean(); e = e / max(e[e.index > 0].sum(), 1e-300)
        for l in (1, 2, 3, 4):
            if l in e.index:
                f[f"spharm_energy_l{l}"] = float(e[l])
    dna = _load(d, "DynamicAnalysis/ShapeSpace/shape_dna_spectra.csv")
    if dna is not None and len(dna):
        lam = dna.drop(columns=[c for c in dna.columns if c == "frame"]).iloc[0].to_numpy(float)
        for k in range(min(5, len(lam))):
            f[f"shape_dna_{k + 1}"] = float(lam[k])
    return f


def compare_scenes(results_root, scenes=None, plots=True, verbose=True):
    """Phenotyping across the scenes of a batch: a table of size-normalised descriptors of every scene, the
    z-scored distance matrix, hierarchical clustering and a 2-D MDS map (Results/_comparison/)."""
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist, squareform
    root = Path(results_root)
    names = scenes or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    rows = {s: scene_features(root / s) for s in names if (root / s).is_dir()}
    rows = {k: v for k, v in rows.items() if v}
    if len(rows) < 2:
        return None
    out = ensure_dir(root / "_comparison")
    tab = pd.DataFrame(rows).T
    tab.to_csv(out / "scene_features.csv")
    Z = tab.dropna(axis=1, how="any")
    Z = Z.loc[:, Z.std() > 0]
    if Z.shape[1] == 0:
        return tab
    Zs = (Z - Z.mean()) / Z.std()
    D = squareform(pdist(Zs.values)); pd.DataFrame(D, index=Z.index, columns=Z.index).to_csv(out / "scene_distance.csv")
    if plots and len(Z) >= 2:
        n = len(Z); J = np.eye(n) - 1 / n; B = -0.5 * J @ (D ** 2) @ J; ev, vec = np.linalg.eigh(B); o = np.argsort(ev)[::-1]
        mds = vec[:, o[:2]] * np.sqrt(np.maximum(ev[o[:2]], 0))
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
        if n >= 3:
            dendrogram(linkage(Zs.values, "ward"), labels=list(Z.index), ax=ax[0]); ax[0].set_title("hierarchical clustering (Ward)")
        else:
            ax[0].axis("off")
        ax[1].scatter(mds[:, 0], mds[:, 1] if mds.shape[1] > 1 else np.zeros(n))
        for i, nm in enumerate(Z.index):
            ax[1].annotate(nm, (mds[i, 0], mds[i, 1] if mds.shape[1] > 1 else 0))
        ax[1].set_title("scenes in descriptor space (MDS)")
        im = ax[2].imshow(Zs.values.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2); ax[2].set_yticks(range(Zs.shape[1]))
        ax[2].set_yticklabels(Zs.columns, fontsize=7); ax[2].set_xticks(range(n)); ax[2].set_xticklabels(Z.index, rotation=45)
        ax[2].set_title("descriptors (z-scores)"); fig.colorbar(im, ax=ax[2])
        fig.tight_layout(); fig.savefig(out / "scene_comparison.png", dpi=200); plt.close(fig)
    if verbose:
        print(f"Scene comparison: {len(Z)} scenes, {Z.shape[1]} descriptors -> {out}")
    return tab
