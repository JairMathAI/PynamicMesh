"""
lineage_assembly.py  —  the standard results of the pipeline for a WHOLE scene that divides / fuses
=================================================================================================

In lineage mode every track (one surface between two topology events) is analysed by the whole pipeline in
Results/<scene>/TopologyEvents/Results/<scene>_track<k>/. This module builds, from those tracks and the topology events,
the results of the WHOLE scene in the usual folders of Results/<scene>/ — so the scene is read, plotted and animated
exactly like a regular scene:

  Transform_Matrices   scene point maps FMV_T{k}_T{k-1} (track maps inside a track — or nearest points where a track
                       map is flipped —, daughter -> mother and fused -> parent maps at the events; new material is
                       matched to the nearest previous point and listed in NEW_T{k}.npy) and scene functional maps
                       FMC_T{k-1}_T{k}: C = Phi_k^T M_k Pi Phi_{k-1} in the Laplace–Beltrami eigenbasis of the WHOLE frame.
                       For a frame of several surfaces the spectrum is the union of the spectra of its surfaces (the zero
                       eigenvalue repeats once per surface), so a division shows as the mother's modes redistributed
                       into the daughters' modes, with maps of the same size before and after.
  Diagonal_analysis    heatmaps, diagonal metrics, similarity and animation of the scene functional maps.
  Physical_fields      all the physical fields (computing_fields of the physical model) on the scene with the scene maps
                       (+ 'new_material' and 'track' per vertex), its plots, viewer and GIFs.
  Reeb_Graphs          the Reeb graphs of all the cells of a frame in one graph (scene vertices / positions) + scalars.
  MSComplexAnalysis    a Morse–Smale complex per frame (MS_T####.pkl) merging the complexes of the cells, the per-vertex
                       arrays, the merged graphs (MS_Graphs) and the critical point / protrusion tables.
  Graph_analysis       graph time analysis and similarity of the scene graphs.
  Trajectories         the material of the scene followed through the events by a particle system on the surfaces:
                       Lagrangian advection (observed displacement field + projection on the surface), tangential
                       relaxation, births / deaths (material vs numerical re-seeding), permanent cuts, remesh-and-
                       transfer at every division / fusion and adaptive re-seeding, anchored continuous trajectories
                       (material_cell.npy: cell of every particle; cut_faces.npy: faces cut; interpolation with the
                       configured schemes).
  GraphAnimation       computed on the scene trajectories (node anchors saved: the viewer splits / merges the graph).
  DynamicAnalysis, MotionAnalysis (+ Physical_fields/from_trajectories), ReebDynamics
                       Lagrangian analyses (fixed connectivity): computed per cell and TRANSFERRED onto the scene
                       particles (nearest vertex of the cell), display values on existing elements only, cumulative
                       quantities accumulated along the lineage; tables combined by the meaning of every column
                       (sum / area-weighted mean / max / min, per-cell descriptors kept per cell) + cells_<table>.csv.
  Parametrization      the cells' sphere maps and SPHARM placed in the scene; per-frame tables (scene_<table>.csv).
  TopologyEvents/      material tracking (Lineage_tracking), graph surgery (Graph_surgery) and the shape / pinch plots.
"""
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PynamicMesh.core.dyn_common import logged_stage, progress, pbar, ensure_dir, load_frames

# advanced options of the scene material and of the transfer, settable from topology_params (config.yaml:
# Topology_Events/topology_params); missing keys keep the defaults of _particles / transfer_stage
PARTICLE_OPTIONS = ("event_remesh", "adaptive_remesh", "fold_tol", "drift_tol", "drift_share", "k_nn", "relax",
                    "death_factor")
TRANSFER_OPTIONS = ("hold",)

SCENE_STAGES = ("transform", "diagonal", "physics", "graphs", "mscomplex", "trajectories", "graph_animation",
                "dynamic_analysis", "motion_analysis", "tables")


# --------------------------------------------------------------------------------------------------------------- #
#  context
# --------------------------------------------------------------------------------------------------------------- #
def _keep(F, vlab, comp):
    """Frame vertices of one piece, in the order of the exported track mesh (same rule as topology_events.submesh)."""
    vm = vlab == comp
    return np.unique(F[vm[F].all(axis=1)])


def _read(path):
    """A CSV table, or an empty table when the file is missing or holds no data (an empty CSV written on Windows still
    contains a line break)."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _context(target_folder, mesh_folder, loader):
    te = Path(target_folder) / "TopologyEvents"
    frames = load_frames(mesh_folder, loader)
    comps = pd.read_csv(te / "components.csv")
    vl = np.load(te / "vertex_labels.npz")
    lin = pd.read_csv(te / "lineage.csv")
    tr = _read(te / "transitions.csv")
    T = len(frames)
    t = np.asarray([float(comps[comps.frame == k].time.iloc[0]) if (comps.frame == k).any() else k for k in range(T)])
    keeps = {}
    for r in comps.itertuples():
        keeps[(int(r.frame), int(r.track))] = _keep(frames[int(r.frame)]["F"], vl[f"frame_{int(r.frame):04d}"], int(r.component))
    return {"target": Path(target_folder), "te": te, "frames": frames, "comps": comps, "lin": lin, "tr": tr, "t": t, "T": T,
            "keeps": keeps, "scene": Path(mesh_folder).name, "mesh_folder": Path(mesh_folder),
            "start": {int(r.track): int(r.start_frame) for r in lin.itertuples()},
            "end": {int(r.track): int(r.end_frame) for r in lin.itertuples()},
            "res": te / "Results", "divs": _read(te / "divisions.csv"), "fus": _read(te / "fusions.csv")}


def _track_dir(ctx, tid):
    return ctx["res"] / f"{ctx['scene']}_track{tid:02d}"


def _tracks_at(ctx, k):
    return sorted(tid for (fr, tid) in ctx["keeps"] if fr == k)


def _vertex_areas(V, F):
    fa = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    a = np.zeros(len(V)); np.add.at(a, F.ravel(), np.repeat(fa / 3.0, 3))
    return a


def _event_lines(ax, ctx):
    tr = ctx["tr"]
    for r in (tr.itertuples() if len(tr) else []):
        if r.event in ("division", "fusion"):
            ax.axvline(ctx["t"][int(r.frame)], color="tab:red" if r.event == "division" else "tab:purple", alpha=0.35, lw=1)


# --------------------------------------------------------------------------------------------------------------- #
#  scene point maps (+ FMV files) and functional maps
# --------------------------------------------------------------------------------------------------------------- #
def scene_maps(ctx, mode="auto"):
    """
    Vertex of frame k -> vertex of frame k-1 (-1 = new material) for every k >= 1. Inside a track: the track's
    functional-map point map; with mode 'auto' the nearest points when the track map moves the points more than twice
    as far as the nearest-point map (a symmetric flip — the frames are aligned and move little between frames).
    Across a division / fusion: the daughter -> mother / fused -> parent maps of the topology events.
    """
    from scipy.spatial import cKDTree
    te, frames, keeps, T = ctx["te"], ctx["frames"], ctx["keeps"], ctx["T"]
    divs, fus = ctx["divs"], ctx["fus"]
    maps, source = {}, []
    for k in pbar(range(1, T), desc="Scene correspondences"):
        m = np.full(len(frames[k]["V"]), -1, dtype=np.int64)
        for tid in _tracks_at(ctx, k):
            if (k - 1, tid) not in keeps:
                continue
            j = k - ctx["start"][tid]
            fmv = _track_dir(ctx, tid) / "Transform_Matrices" / f"FMV_T{j:04d}_T{j - 1:04d}.npy"
            kc, kp = keeps[(k, tid)], keeps[(k - 1, tid)]
            Vc, Vp = frames[k]["V"][kc], frames[k - 1]["V"][kp]
            nn = cKDTree(Vp).query(Vc)[1]
            p2p = np.load(fmv) if (fmv.exists() and mode != "nearest") else None
            how = "functional map"
            if p2p is None or len(p2p) != len(kc) or p2p.max() >= len(kp):
                p2p, how = nn, "nearest points"
            elif mode == "auto":
                if np.mean(np.linalg.norm(Vc - Vp[p2p], axis=1)) > 2.0 * np.mean(np.linalg.norm(Vc - Vp[nn], axis=1)) + 1e-12:
                    p2p, how = nn, "nearest points (functional map inconsistent)"
            m[kc] = kp[p2p]
            source.append({"frame": k, "track": tid, "map": how, "mean_displacement": float(np.mean(np.linalg.norm(Vc - Vp[p2p], axis=1)))})
        for r in (divs.itertuples() if len(divs) else []):
            if int(r.frame) != k:
                continue
            km = keeps[(k - 1, int(r.mother_track))]
            for i, dt in enumerate(eval(r.daughter_tracks)):
                f = te / "Divisions" / f"division_{int(r.division):02d}" / f"daughter_{i}_to_mother.npy"
                if f.exists() and (k, dt) in keeps:
                    d2m = np.load(f); kd = keeps[(k, dt)]
                    if len(d2m) == len(kd):
                        m[kd] = np.where(d2m >= 0, km[np.clip(d2m, 0, len(km) - 1)], -1)
                        source.append({"frame": k, "track": dt, "map": f"division {int(r.division)}", "mean_displacement": np.nan})
        for r in (fus.itertuples() if len(fus) else []):
            if int(r.frame) != k:
                continue
            fd = te / "Fusions" / f"fusion_{int(r.fusion):02d}"
            if not (fd / "child_to_parent.npy").exists():
                continue
            part = np.load(fd / "child_partition.npy"); c2p = np.load(fd / "child_to_parent.npy")
            kc = keeps[(k, int(r.child_track))]
            if len(c2p) != len(kc):
                continue
            for i, pt in enumerate(eval(r.parent_tracks)):
                sel = (part == i + 1) & (c2p >= 0); kp = keeps[(k - 1, pt)]
                m[kc[sel]] = kp[np.clip(c2p[sel], 0, len(kp) - 1)]
            source.append({"frame": k, "track": int(r.child_track), "map": f"fusion {int(r.fusion)}", "mean_displacement": np.nan})
        maps[k] = m
    return maps, pd.DataFrame(source)


def write_transform(ctx, maps, sources, k_basis=None):
    """FMV (point maps, new material matched to the nearest previous point), NEW (new-material masks) and FMC (scene
    functional maps in the eigenbasis of the whole frame) in Results/<scene>/Transform_Matrices."""
    from scipy.spatial import cKDTree
    from pyFM.mesh import TriMesh
    frames, T = ctx["frames"], ctx["T"]
    d = ensure_dir(ctx["target"] / "Transform_Matrices")
    sources.to_csv(d / "scene_map_sources.csv", index=False)
    filled = {}
    for k, m in maps.items():
        f = m.copy(); new = m < 0
        if new.any():
            f[new] = cKDTree(frames[k - 1]["V"]).query(frames[k]["V"][new])[1]
        filled[k] = f
        np.save(d / f"FMV_T{k:04d}_T{k - 1:04d}.npy", f); np.save(d / f"NEW_T{k:04d}.npy", new)
    # scene functional maps: DIRECT SUM of the cells' function spaces, L2(A u B) = L2(A) + L2(B), with the cells' own
    # Laplace–Beltrami bases. Inside a cell the block IS the cell's own functional map (FMC of its track); at a division
    # the mother's space maps into the daughters' spaces (a column of blocks C_i = Phi_Di^T M_Di Pi_i Phi_M), at a fusion
    # the parents' spaces into the fused cell's space (a row of blocks).
    blocks_k, fmcs = {}, {}

    def track_fmc(c, j):
        f = _track_dir(ctx, c) / "Transform_Matrices" / f"FMC_T{j - 1:04d}_T{j:04d}.npy"
        return np.load(f) if f.exists() else None

    def size_at(c, k):
        """Basis size of cell c at frame k: the size its own functional maps use there (they may differ from pair to
        pair: ZoomOut-refined maps grow, ICP-refined ones keep the initial size)."""
        j = k - ctx["start"][c]
        A = track_fmc(c, j)                                   # map into frame j: rows
        if A is not None:
            return int(A.shape[0])
        B = track_fmc(c, j + 1)                               # map out of frame j: columns
        if B is not None:
            return int(B.shape[1])
        return int(k_basis or 30)
    cache = {}

    def basis(k, tid, K):
        if (k, tid, K) not in cache:
            kp = ctx["keeps"][(k, tid)]; Fk = frames[k]["F"]
            m = np.zeros(len(frames[k]["V"]), bool); m[kp] = True
            loc = -np.ones(len(frames[k]["V"]), np.int64); loc[kp] = np.arange(len(kp))
            mesh = TriMesh(np.asarray(frames[k]["V"][kp], float), loc[Fk[m[Fk].all(1)]])
            mesh.process(k=K, intrinsic=False, verbose=False)
            cache[(k, tid, K)] = (np.asarray(mesh.eigenvectors[:, :K]), np.asarray(mesh.A.diagonal()).ravel())
        return cache[(k, tid, K)]
    # ONE basis size for the whole scene (the largest the cells' maps reach): the cells' maps may have different
    # sizes (ZoomOut-refined maps grow, ICP-refined maps keep the initial size); a smaller map is placed in the top-left
    # of its block (the higher modes it does not determine stay 0) and the event blocks use the full basis
    Ks = [int(max(np.load(f, mmap_mode="r").shape)) for tid in ctx["start"]
          for f in (_track_dir(ctx, tid) / "Transform_Matrices").glob("FMC_T*.npy")]
    K_scene = max(Ks) if Ks else int(k_basis or 30)
    for k in pbar(range(1, T), desc="Scene functional maps"):
        cur, prv = _tracks_at(ctx, k), _tracks_at(ctx, k - 1)
        own = {}
        for c in cur:
            if c in prv:
                A = track_fmc(c, k - ctx["start"][c])
                if A is not None and A.shape == (K_scene, K_scene):
                    own[c] = A                      # the cell's own full-size map, unchanged
                # a smaller map (ICP-refined, initial size) is NOT padded: its block is computed at full size by the
                # pull-back of the same point correspondence (below), so every transition has the same resolution
        rsz = [K_scene] * len(cur)
        csz = [K_scene] * len(prv)
        roff = np.cumsum([0] + rsz); coff = np.cumsum([0] + csz)
        C = np.zeros((int(roff[-1]), int(coff[-1])))
        owner = np.full(len(frames[k - 1]["V"]), -1, np.int64); local = np.full(len(frames[k - 1]["V"]), -1, np.int64)
        for c in prv:
            kp = ctx["keeps"][(k - 1, c)]; owner[kp] = c; local[kp] = np.arange(len(kp))
        layout = []
        for a, c in enumerate(cur):
            if c in prv:                                                   # same cell: its own functional map
                b = prv.index(c)
                if own.get(c) is not None:
                    blk = own[c]; how = "track functional map"
                else:
                    Pc, ac = basis(k, c, rsz[a]); Pp, _ = basis(k - 1, c, csz[b])
                    kc = ctx["keeps"][(k, c)]; mp = maps[k][kc]; ok = (mp >= 0) & (owner[np.clip(mp, 0, None)] == c)
                    blk = (Pc[ok] * ac[ok, None]).T @ Pp[local[mp[ok]]]; how = "projection"
                C[roff[a]:roff[a + 1], coff[b]:coff[b + 1]] = blk
                layout.append({"row_cell": c, "col_cell": c, "kind": how, "rows": [int(roff[a]), int(roff[a + 1])],
                               "cols": [int(coff[b]), int(coff[b + 1])]})
                continue
            kc = ctx["keeps"][(k, c)]; mp = maps[k][kc]                    # a cell born at an event
            Pc, ac = basis(k, c, rsz[a])
            for b, p_ in enumerate(prv):
                ok = (mp >= 0) & (owner[np.clip(mp, 0, None)] == p_)
                if not ok.any():
                    continue
                Pp, _ = basis(k - 1, p_, csz[b])
                C[roff[a]:roff[a + 1], coff[b]:coff[b + 1]] = (Pc[ok] * ac[ok, None]).T @ Pp[local[mp[ok]]]
                layout.append({"row_cell": c, "col_cell": p_, "kind": "event projection", "rows": [int(roff[a]), int(roff[a + 1])],
                               "cols": [int(coff[b]), int(coff[b + 1])]})
        fmcs[k] = C; blocks_k[k] = layout
        np.save(d / f"FMC_T{k - 1:04d}_T{k:04d}.npy", C)
    with open(d / "scene_fm_blocks.json", "w") as fh:
        json.dump({str(k): v for k, v in blocks_k.items()}, fh, indent=1)
    return filled, (fmcs, blocks_k)


def _block_heatmap(C, k, layout, path, ctx):
    """Heatmap of a scene functional map made of cell blocks: block boundaries drawn and labelled with the cells, the
    event (division: the mother's modes go to every daughter — one staircase per daughter; fusion: every parent's
    modes go to the fused cell) named in the title."""
    import matplotlib.pyplot as plt
    ev = ""
    tr = ctx["tr"]
    if len(tr):
        e = tr[(tr.frame == k) & tr.event.isin(["division", "fusion"])]
        if len(e):
            ev = "  —  " + ", ".join(f"{r.event} {r.parent_tracks} -> {r.child_tracks}" for r in e.itertuples())
    fig, ax = plt.subplots(figsize=(10, 9))
    lim = float(np.abs(C).max()) or 1.0
    im = ax.imshow(C, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    rows = sorted({tuple(b["rows"]) + (b["row_cell"],) for b in layout}); cols = sorted({tuple(b["cols"]) + (b["col_cell"],) for b in layout})
    for r0, r1, c in rows:
        if r0 > 0:
            ax.axhline(r0 - 0.5, color="k", lw=0.8)
        ax.text(-1, (r0 + r1) / 2, f"cell {c}", ha="right", va="center", fontsize=8)
    for c0, c1, c in cols:
        if c0 > 0:
            ax.axvline(c0 - 0.5, color="k", lw=0.8)
        ax.text((c0 + c1) / 2, C.shape[0] + 1, f"cell {c}", ha="center", va="top", fontsize=8)
    ax.set_title(f"Transformation Heatmap Mesh T{k - 1} \u2192 Mesh T{k} (cell blocks){ev}", fontsize=10)
    ax.set_xlabel(f"Mesh T{k - 1} Eigenfunctions", labelpad=14); ax.set_ylabel(f"Mesh T{k} Eigenfunctions", labelpad=30)
    ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def write_displacements(ctx, filled):
    """Trajectories/displacements of the scene (same files as the trajectories stage): backward_T{i}_T{i-1}
    (V_{i-1}[p2p] - V_i, every vertex of frame i), forward_T{i-1}_T{i} (mean displacement towards the vertices of
    frame i matched to every vertex of frame i-1, inpainted where none is) and forward_hit, from the scene maps."""
    from PynamicMesh.core.trajectories import _forward_field
    frames = ctx["frames"]; d = ensure_dir(ctx["target"] / "Trajectories" / "displacements")
    for i in range(1, ctx["T"]):
        m = np.asarray(filled[i], dtype=np.int64)
        Vp, Fp, Vi = frames[i - 1]["V"], frames[i - 1]["F"], frames[i]["V"]
        np.save(d / f"backward_T{i:04d}_T{i - 1:04d}.npy", Vp[m] - Vi)
        fwd, hit = _forward_field(Vp, Fp, Vi, m)
        np.save(d / f"forward_T{i - 1:04d}_T{i:04d}.npy", fwd); np.save(d / f"forward_hit_T{i - 1:04d}_T{i:04d}.npy", hit)


def write_diagonal(ctx, fm):
    """The diagonal analysis of the pipeline on the scene functional maps: heatmaps of the whole (block) maps; the
    diagonal metrics of every block (a cell's own map, or an event block mother -> daughter / parent -> fused cell),
    averaged per transition; the similarity of consecutive maps of the same size."""
    from PynamicMesh.core import pipelines as PL
    fmcs, layout = fm
    d = ensure_dir(ctx["target"] / "Diagonal_analysis")
    inertia, decay, cdf, sim, heat, rows, events, nblocks = [], [], [], [], [], [], [], []
    prev = None
    for k in sorted(fmcs):
        C = fmcs[k]
        h = d / f"FM_{k - 1}{k}.png"
        lay = layout.get(k, [])
        _block_heatmap(C, k, lay, h, ctx)                         # cell blocks marked (same size for the GIF)
        heat.append(h)
        mets, wts = [], []
        for b in layout.get(k, []):
            blk = C[b["rows"][0]:b["rows"][1], b["cols"][0]:b["cols"][1]]
            if np.any(blk):
                a_, d_, c_ = PL.Diagonal_metrics(blk); mets.append((a_, d_, c_)); wts.append(float(np.sum(blk ** 2)))
                rows.append({"transition": f"FM_{k - 1}{k}", "row_cell": b["row_cell"], "col_cell": b["col_cell"],
                             "kind": b["kind"], "inertia": float(a_), "decay": float(d_)})
        if not mets:
            mets = [PL.Diagonal_metrics(C)]; wts = [1.0]
        # the whole-matrix metrics are defined on square cell blocks: the scene value is the ENERGY-weighted mean of
        # the block metrics (|C_b|_F^2: the share of the map carried by every block); one block = the plain metric
        wv = np.asarray(wts, float); wv = wv / wv.sum() if wv.sum() > 0 else np.full(len(wv), 1.0 / len(wv))
        inertia.append(float(np.dot(wv, [m[0] for m in mets]))); decay.append(float(np.dot(wv, [m[1] for m in mets])))
        cdf.append(mets[int(np.argmax(wv))][2])
        tr_ = ctx["tr"]
        ev_ = tr_[(tr_.frame == k) & tr_.event.isin(["division", "fusion"])] if len(tr_) else []
        events.append(";".join(sorted(set(ev_.event))) if len(ev_) else "")
        nblocks.append(len(mets))
        if prev is not None:
            sim.append(list(PL.compute_heatmap_similarity(prev, C)) if prev.shape == C.shape else [np.nan] * 5)
        prev = C
    PL.plot_diagonal(inertia, decay, cdf, d / "Diagonal_metrics.png")
    with open(d / "Diagonal_Metrics.csv", "w", encoding="utf-8") as fh:
        fh.write("Timestep,Map,Inertia_Metric,Decay_Metric,Event,N_blocks\n")
        for i, (a, b) in enumerate(zip(inertia, decay)):
            fh.write(f"{i + 1},FM_{i}{i + 1},{a:.6f},{b:.6f},{events[i]},{nblocks[i]}\n")
    pd.DataFrame(rows).to_csv(d / "Diagonal_Metrics_blocks.csv", index=False)
    if sim:
        good = [v for v in sim if np.isfinite(v[0])]
        if good:
            PL.plot_similarity_metrics(good, d / "Cross_Heatmap_Similarity.png")
        with open(d / "Cross_Heatmap_Similarity.csv", "w", encoding="utf-8") as fh:
            fh.write("Comparison,JSD,Pearson_Corr,Spearman_Corr,Manhattan_Dist,Euclidean_Dist\n")
            for i, v in enumerate(sim):
                fh.write(f"FM_{i}{i + 1}_vs_FM_{i + 1}{i + 2}," + ",".join(f"{x:.6f}" for x in v) + "\n")
    if heat:
        PL.headmap_gif(heat, ctx["target"])


def write_physics(ctx, maps, times=None):
    """All the fields of the physical model on the scene (computing_fields with the scene maps)."""
    from PynamicMesh.core.physic_model import computing_fields
    from PynamicMesh.utils.tools import load_aligned_mesh
    out = ensure_dir(ctx["target"] / "Physical_fields")
    t = ctx["t"]
    computing_fields(str(ctx["mesh_folder"]), str(ctx["target"] / "Transform_Matrices"), str(out), single_file=False,
                     loader=load_aligned_mesh, plot=True, times=t if times is None else times)
    for k in range(ctx["T"]):                              # new material and the cell of every vertex
        f = out / f"frame_{k:04d}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f, allow_pickle=True))
        n = len(ctx["frames"][k]["V"])
        tr_ = np.full(n, -1, dtype=np.int32)
        for tid in _tracks_at(ctx, k):
            tr_[ctx["keeps"][(k, tid)]] = tid
        z["new_material"] = (maps[k] < 0) if k in maps else np.zeros(n, bool)
        z["track"] = tr_
        np.savez(f, **z)


# --------------------------------------------------------------------------------------------------------------- #
#  graphs: Reeb, Morse–Smale (complexes and graphs), surgery
# --------------------------------------------------------------------------------------------------------------- #
def _merge_graphs(ctx, k, sub, pat, region_labels=None):
    """One graph with the nodes of all the tracks of frame k (scene vertices); region_labels: per-track arrays of the
    region (critical vertex) of every vertex, to attach the region of every Morse–Smale node."""
    import networkx as nx
    G = nx.Graph(); found = False
    for tid in _tracks_at(ctx, k):
        j = k - ctx["start"][tid]
        f = _track_dir(ctx, tid) / sub / pat.format(j)
        if not f.exists():
            continue
        with open(f, "rb") as fh:
            g = pickle.load(fh)
        found = True; kp = ctx["keeps"][(k, tid)]
        ids = {nd: tid * 100000 + i for i, nd in enumerate(g.nodes())}
        lab = region_labels.get(tid) if region_labels else None
        for nd, data in g.nodes(data=True):
            dd = dict(data); dd["track"] = tid; dd["track_node"] = nd
            if "vertices" in dd:
                v = np.asarray(dd["vertices"], dtype=np.int64); dd["vertices"] = kp[v[(v >= 0) & (v < len(kp))]]
            if "vertex" in dd and np.ndim(dd["vertex"]) == 0 and 0 <= int(dd["vertex"]) < len(kp):
                v = int(dd["vertex"])
                if lab is not None and "vertices" not in dd:
                    dd["vertices"] = kp[np.flatnonzero(lab == v)]
                dd["vertex"] = int(kp[v])
            G.add_node(ids[nd], **dd)
        for a, b, data in g.edges(data=True):
            G.add_edge(ids[a], ids[b], **data)
    return G if found else None


def write_center_lineage(ctx):
    """Area-weighted centre of every cell at every frame, with its parents (divisions / fusions):
    TopologyEvents/center_lineage.csv — drawn by the Morse–Smale graph viewer as the centre lineage."""
    rows = []
    par = {int(r.track): str(r.parents) for r in ctx["lin"].itertuples()}
    for (k, tid), kp in sorted(ctx["keeps"].items()):
        V, Fk = ctx["frames"][k]["V"], ctx["frames"][k]["F"]
        m = np.zeros(len(V), bool); m[kp] = True; Fs = Fk[m[Fk].all(1)]
        a = 0.5 * np.linalg.norm(np.cross(V[Fs[:, 1]] - V[Fs[:, 0]], V[Fs[:, 2]] - V[Fs[:, 0]]), axis=1)
        c = (V[Fs].mean(1) * a[:, None]).sum(0) / max(a.sum(), 1e-300)
        p_ = [int(x) for x in __import__("re").findall(r"\d+", par.get(tid, "[]"))]
        rows.append({"frame": k, "time": float(ctx["t"][k]), "cell": tid, "x": c[0], "y": c[1], "z": c[2],
                     "parents": ";".join(map(str, p_))})
    pd.DataFrame(rows).to_csv(ctx["te"] / "center_lineage.csv", index=False)


def write_reeb(ctx):
    frames, T = ctx["frames"], ctx["T"]
    d = ensure_dir(ctx["target"] / "Reeb_Graphs"); graphs = {}
    for k in pbar(range(T), desc="Scene Reeb graphs"):
        G = _merge_graphs(ctx, k, "Reeb_Graphs", "Reeb_T{:04d}.pkl")
        if G is None:
            continue
        scal = np.full(len(frames[k]["V"]), np.nan)
        for tid in _tracks_at(ctx, k):
            sf = _track_dir(ctx, tid) / "Reeb_Graphs" / f"Scalar_T{k - ctx['start'][tid]:04d}.npy"
            kp = ctx["keeps"][(k, tid)]
            if sf.exists():
                s = np.load(sf)
                if len(s) == len(kp):
                    scal[kp] = s
        with open(d / f"Reeb_T{k:04d}.pkl", "wb") as fh:
            pickle.dump(G, fh)
        np.save(d / f"Scalar_T{k:04d}.npy", scal)
        graphs[k] = G
    return graphs


def _merge_ms(objs, keeps, n, frame):
    """One MSComplex for a frame from the complexes of its cells (indices re-mapped to the scene vertices)."""
    import dataclasses
    base = max(objs, key=lambda o: o[1].n_vertices)[1]
    vec = lambda name, fill, dtype: np.full(n, fill, dtype=dtype)                       # noqa: E731
    out = {"scalar": vec("scalar", np.nan, float), "label_max": vec(0, -1, np.int64), "label_min": vec(0, -1, np.int64),
           "cell": vec(0, -1, np.int64), "label_max_raw": vec(0, -1, np.int64), "label_min_raw": vec(0, -1, np.int64),
           "cp_type_raw": vec(0, 0, np.int64), "cp_multiplicity": vec(0, 0, np.int64), "vertex_area": vec(0, 0.0, float),
           "hull_depth": vec(0, 0.0, float), "core_label": vec(0, -1, np.int64)}
    lists = {"maxima": [], "minima": [], "saddles": [], "boundary_edges": []}
    crit, prot, hull = [], [], []
    cell_off = 0
    for tid, o in objs:
        kp = keeps[tid]
        rem = lambda a: np.where(np.asarray(a) >= 0, kp[np.clip(np.asarray(a), 0, len(kp) - 1)], -1)  # noqa: E731
        for key in ("scalar", "cp_type_raw", "cp_multiplicity", "vertex_area", "hull_depth"):
            if hasattr(o, key) and getattr(o, key) is not None and len(getattr(o, key)) == len(kp):
                out[key][kp] = getattr(o, key)
        for key in ("label_max", "label_min", "label_max_raw", "label_min_raw", "core_label"):
            if hasattr(o, key) and getattr(o, key) is not None and len(getattr(o, key)) == len(kp):
                out[key][kp] = rem(getattr(o, key))
        if getattr(o, "cell", None) is not None and len(o.cell) == len(kp):
            c = np.asarray(o.cell); out["cell"][kp] = np.where(c >= 0, c + cell_off, -1); cell_off += int(c.max()) + 1 if len(c) else 0
        for key in ("maxima", "minima", "saddles"):
            lists[key].append(rem(getattr(o, key, np.zeros(0, np.int64))))
        be = np.asarray(getattr(o, "boundary_edges", np.zeros((0, 2), np.int64)))
        if be.size:
            lists["boundary_edges"].append(rem(be))
        for df, acc, cols in ((getattr(o, "critical", None), crit, ("vertex",)),
                              (getattr(o, "protrusions", None), prot, ("protrusion_id", "tip_vertex")),
                              (getattr(o, "hull_candidates", None), hull, ("vertex",))):
            if isinstance(df, pd.DataFrame) and len(df):
                df = df.copy()
                for ccol in cols:
                    if ccol in df:
                        df[ccol] = rem(df[ccol].to_numpy(np.int64))
                df["track"] = tid; acc.append(df)
    fields = dict(out)
    for key in ("maxima", "minima", "saddles"):
        fields[key] = np.concatenate(lists[key]) if lists[key] else np.zeros(0, np.int64)
    fields["boundary_edges"] = np.concatenate(lists["boundary_edges"]) if lists["boundary_edges"] else np.zeros((0, 2), np.int64)
    fields["critical"] = pd.concat(crit, ignore_index=True) if crit else base.critical.iloc[:0]
    fields["protrusions"] = pd.concat(prot, ignore_index=True) if prot else base.protrusions.iloc[:0]
    fields["hull_candidates"] = pd.concat(hull, ignore_index=True) if hull else base.hull_candidates.iloc[:0]
    meta = dict(getattr(base, "meta", {}) or {}); meta["n_cells_scene"] = len(objs); meta["tracks"] = [t for t, _ in objs]
    fields.update(frame=frame, n_vertices=n, meta=meta)
    valid = {f.name for f in dataclasses.fields(base)}
    return dataclasses.replace(base, **{k: v for k, v in fields.items() if k in valid})


def write_mscomplex(ctx):
    """Morse–Smale complexes, per-vertex arrays, graphs and tables of the scene (MSComplexAnalysis)."""
    from PynamicMesh.core.SMComplex import MSComplex
    frames, T = ctx["frames"], ctx["T"]
    root = ctx["target"] / "MSComplexAnalysis"; dc = ensure_dir(root / "MS_Complex")
    graphs, any_ms = {}, False
    for k in pbar(range(T), desc="Scene Morse-Smale complexes"):
        objs, labels, keeps = [], {}, {}
        for tid in _tracks_at(ctx, k):
            f = _track_dir(ctx, tid) / "MSComplexAnalysis" / "MS_Complex" / f"MS_T{k - ctx['start'][tid]:04d}.pkl"
            if f.exists():
                o = MSComplex.from_pickle(f)
                if o.n_vertices == len(ctx["keeps"][(k, tid)]):
                    objs.append((tid, o)); labels[tid] = np.asarray(o.label_max); keeps[tid] = ctx["keeps"][(k, tid)]
        if not objs:
            continue
        any_ms = True
        ms = _merge_ms(objs, keeps, len(frames[k]["V"]), k)
        ms.to_pickle(dc / f"MS_T{k:04d}.pkl")
        np.save(dc / f"Labels_T{k:04d}.npy", ms.label_max); np.save(dc / f"Core_T{k:04d}.npy", ms.core_label)
        np.save(dc / f"HullDepth_T{k:04d}.npy", ms.hull_depth); np.save(dc / f"Scalar_T{k:04d}.npy", ms.scalar)
        G = _merge_graphs(ctx, k, "MSComplexAnalysis/MS_Graphs", "MSGraph_T{:04d}.pkl", labels)
        if G is not None:
            ensure_dir(root / "MS_Graphs")
            with open(root / "MS_Graphs" / f"MSGraph_T{k:04d}.pkl", "wb") as fh:
                pickle.dump(G, fh)
            graphs[k] = G
    if not any_ms:
        return graphs
    # the reports of the Morse–Smale stage on the scene complexes (critical points, protrusions: tables and plots)
    from PynamicMesh.core import SMComplex as SM
    with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
        SM.critical_points_report(ctx["target"], single_file=False)
    # region tracking: the cells' tables on the scene timeline (counts summed per transition, ids unique per cell)
    rt = ensure_dir(root / "Region_tracking"); cat = {}
    for tid in sorted(ctx["start"]):
        sd = _track_dir(ctx, tid) / "MSComplexAnalysis" / "Region_tracking"
        for f in (sorted(sd.glob("*.csv")) if sd.is_dir() else []):
            df = _read(f)
            if not len(df):
                continue
            df = df.copy(); st = ctx["start"][tid]
            for c in df.columns:
                if c in ("Time_Step", "first_frame", "last_frame", "origin_frame") and pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c] + st
                elif c in ("track_id", "curr_track_id") and pd.api.types.is_integer_dtype(df[c]):
                    df[c] = np.where(df[c] >= 0, df[c] + tid * 100000, df[c])
            if "Transition" in df and "Time_Step" in df:
                df["Transition"] = [f"T{int(k) - 1:04d}->T{int(k):04d}" if k > 0 else str(tr_)
                                    for k, tr_ in zip(df["Time_Step"], df["Transition"])]
            df["cell"] = tid
            cat.setdefault(f.name, []).append(df)
    for name, dfs in cat.items():
        df = pd.concat(dfs, ignore_index=True)
        if name == "tracking_summary.csv":                       # one row per scene transition
            cnt = [c for c in df.columns if c.startswith("n_")]
            mean = [c for c in df.columns if c.startswith("mean_") or c.endswith("_fraction_prev")]
            g = df.groupby("Time_Step")
            df = pd.concat([g["Transition"].first(), g[cnt].sum(), g[mean].mean()], axis=1).reset_index()
        elif name in ("fate_transition_counts.csv", "long_fate_counts.csv"):
            key = df.columns[0]
            df = df.drop(columns=["cell"]).groupby(key).sum(numeric_only=True).reset_index()
        df.to_csv(rt / name, index=False)
    for tid in sorted(ctx["start"]):                             # fate meanings (same for every cell)
        fm_ = _track_dir(ctx, tid) / "MSComplexAnalysis" / "Region_tracking" / "fate_meanings.json"
        if fm_.exists():
            shutil.copy2(fm_, rt / "fate_meanings.json"); break
    if (rt / "tracking_summary.csv").exists() and (rt / "region_lineage.csv").exists():
        try:
            with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
                SM.tracking_report(ctx["target"], single_file=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[Scene Morse-Smale] tracking report: {exc}")
    return graphs


def _plot_ms_counts(ctx, root):
    cp = _read(root / "critical_points.csv")
    if not len(cp) or "frame" not in cp:
        return
    num = [c for c in cp.columns if c not in ("cell", "track", "frame") and pd.api.types.is_numeric_dtype(cp[c])][:4]
    if not num:
        return
    tot = cp.groupby("frame")[num].sum()
    fig, axs = plt.subplots(len(num), 1, figsize=(12, 2.4 * len(num) + 0.6), sharex=True, squeeze=False)
    for a, c in zip(axs[:, 0], num):
        a.plot(ctx["t"][tot.index.to_numpy(int)], tot[c], "k-o", ms=3, label="scene")
        for tid, g in cp.groupby("cell"):
            a.plot(ctx["t"][g.frame.to_numpy(int)], g[c], "-", lw=1, alpha=0.8, label=f"cell {tid}")
        _event_lines(a, ctx); a.set_ylabel(c, fontsize=8); a.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7, ncol=6); axs[0, 0].set_title("Morse–Smale critical points of the scene (all cells)")
    axs[-1, 0].set_xlabel("time")
    ensure_dir(root / "plots")
    fig.tight_layout(); fig.savefig(root / "plots" / "scene_critical_points.png", dpi=200); plt.close(fig)


def graph_surgery(ctx, maps, graphs_by_kind):
    """For every division / fusion and graph kind: overlap of the nodes before and after the event (through the scene
    maps) and the nodes cut by the event (nodes before whose vertices go to several cells after)."""
    d = ensure_dir(ctx["te"] / "Graph_surgery"); summary = []
    events = []
    for r in (ctx["divs"].itertuples() if len(ctx["divs"]) else []):
        events.append(("division", int(r.division), int(r.frame), [int(r.mother_track)], eval(r.daughter_tracks)))
    for r in (ctx["fus"].itertuples() if len(ctx["fus"]) else []):
        events.append(("fusion", int(r.fusion), int(r.frame), eval(r.parent_tracks), [int(r.child_track)]))
    for kind, G_by_frame in graphs_by_kind.items():
        for ev, idx, k, before, after in events:
            Gb, Ga = G_by_frame.get(k - 1), G_by_frame.get(k)
            if Gb is None or Ga is None:
                continue
            owner = {}
            for nd, data in Gb.nodes(data=True):
                if data.get("track") in before:
                    for v in np.asarray(data.get("vertices", []), dtype=np.int64):
                        owner[int(v)] = nd
            rows, dest = [], {}
            for nd, data in Ga.nodes(data=True):
                if data.get("track") not in after:
                    continue
                vs = np.asarray(data.get("vertices", []), dtype=np.int64)
                if not len(vs):
                    continue
                src = maps[k][vs]; src = src[src >= 0]
                if not len(src):
                    rows.append({"node_after": nd, "track_after": data["track"], "node_before": -1, "track_before": -1,
                                 "overlap": 1.0, "note": "new material"}); continue
                own = pd.Series([owner.get(int(v), -1) for v in src]).value_counts(normalize=True)
                for nb, frac in own.items():
                    tb = Gb.nodes[nb]["track"] if nb in Gb.nodes else -1
                    rows.append({"node_after": nd, "track_after": data["track"], "node_before": int(nb), "track_before": tb,
                                 "overlap": float(frac), "note": "" if nb != -1 else "outside the graph"})
                    if nb != -1 and frac > 0.1:
                        dest.setdefault(nb, set()).add(data["track"])
            df = pd.DataFrame(rows)
            cut = [int(nb) for nb, s in dest.items() if len(s) > 1]
            if len(df):
                df["node_before_cut_by_event"] = df.node_before.isin(cut)
            df.to_csv(d / f"{kind}_{ev}_{idx:02d}.csv", index=False)
            summary.append({"graph": kind, "event": ev, "index": idx, "frame": k,
                            "nodes_before": sum(1 for _, x in Gb.nodes(data=True) if x.get("track") in before),
                            "nodes_after": sum(1 for _, x in Ga.nodes(data=True) if x.get("track") in after),
                            "nodes_cut": len(cut), "rows": len(df)})
    pd.DataFrame(summary).to_csv(d / "summary.csv", index=False)


def write_graph_analysis(ctx, flags):
    """Graph time analysis and similarity of the scene graphs (the usual graph analyses)."""
    from PynamicMesh.core import pipelines as PL
    for folder, out, tag in ((ctx["target"] / "Reeb_Graphs", None, None),
                             (ctx["target"] / "MSComplexAnalysis" / "MS_Graphs", ctx["target"] / "MSComplexAnalysis" / "Graph_analysis", "MSGraph")):
        if not (folder.is_dir() and any(folder.glob("*.pkl"))):
            continue
        if flags.get("time_graph_analysis", True):
            csv = PL.graph_time_analysis(folder, single_file=False, analysis_folder=out)
            if csv:
                PL.plot_dynamic_graph_analysis(csv, single_file=False)
        if flags.get("graph_sim", True) and tag is None:
            csv = PL.graph_similarity(reeb_folder_path=folder, metrics_list=flags.get("graph_metrics"), single_file=False,
                                      analysis_folder=out, tag=tag)
            if csv:
                PL.plot_graph_similarity(csv, single_file=False)


# --------------------------------------------------------------------------------------------------------------- #
#  material tracking (TopologyEvents/Lineage_tracking) and scene trajectories (Trajectories)
# --------------------------------------------------------------------------------------------------------------- #
def material_tracking(ctx, maps):
    """Origin (vertex of the first frame, -1 = created later), birth frame and cell of every vertex of every frame."""
    frames, keeps, T = ctx["frames"], ctx["keeps"], ctx["T"]
    d = ensure_dir(ctx["te"] / "Lineage_tracking")
    origin = np.full(len(frames[0]["V"]), -1, dtype=np.int64)
    for (fr, tid), kp in keeps.items():
        if fr == 0:
            origin[kp] = kp
    O, B, TR, rows = {0: origin}, {0: np.zeros(len(origin), np.int64)}, {}, []
    V0, F0 = frames[0]["V"], frames[0]["F"]
    l0 = float(np.mean(np.linalg.norm(V0[F0[:, 0]] - V0[F0[:, 1]], axis=1)))
    for k in range(T):
        tr_ = np.full(len(frames[k]["V"]), -1, dtype=np.int64)
        for tid in _tracks_at(ctx, k):
            tr_[keeps[(k, tid)]] = tid
        TR[k] = tr_
        if k >= 1:
            m = maps[k]; ok = m >= 0
            O[k] = np.where(ok, O[k - 1][np.clip(m, 0, None)], -1); B[k] = np.where(ok, B[k - 1][np.clip(m, 0, None)], k)
        fa = _vertex_areas(frames[k]["V"], frames[k]["F"]); live = TR[k] >= 0; tot = fa[live].sum() or 1.0
        Fk = frames[k]["F"]; ee = np.concatenate([Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]]])
        oa, ob = O[k][ee[:, 0]], O[k][ee[:, 1]]; good = (oa >= 0) & (ob >= 0)
        rows.append({"frame": k, "time": float(ctx["t"][k]),
                     "original_material_area_fraction": float(fa[(O[k] >= 0) & live].sum() / tot),
                     "new_material_area_fraction": float(fa[(O[k] < 0) & live].sum() / tot),
                     "mean_material_age": float(np.mean(k - B[k][live])) if live.any() else np.nan,
                     "material_coherence": float(np.mean(np.linalg.norm(V0[oa[good]] - V0[ob[good]], axis=1)) / max(l0, 1e-300)) if good.any() else np.nan})
    np.savez_compressed(d / "material.npz", **{f"origin_T{k:04d}": O[k] for k in range(T)},
                        **{f"birth_T{k:04d}": B[k] for k in range(T)}, **{f"track_T{k:04d}": TR[k] for k in range(T)})
    pd.DataFrame(rows).to_csv(d / "material_summary.csv", index=False)


def track_fidelity(ctx, trj):
    """Mean distance of every track's tracked mesh to the real surface of each of its frames, relative to the size of
    the surface (max over the frames): a tracked mesh that does not follow the surface (too rigid for a large
    deformation) cannot hand its material over at an event."""
    from scipy.spatial import cKDTree
    out = {}
    for tid, (X, _) in trj.items():
        worst = 0.0
        for j in range(X.shape[0]):
            k = ctx["start"][tid] + j
            if (k, tid) not in ctx["keeps"]:
                continue
            Vs = ctx["frames"][k]["V"][ctx["keeps"][(k, tid)]]
            size = float(np.ptp(Vs, 0).max()) or 1.0
            worst = max(worst, float(cKDTree(Vs).query(X[j])[0].mean()) / size)
        out[int(tid)] = worst
    return out


def _ordered_loop(e_):
    """Vertices of a boundary loop in order, following its directed edges (a -> b); None if it is not a simple cycle."""
    nxt = {int(a): int(b) for a, b in e_}
    if len(nxt) != len(e_):
        return None
    start = int(e_[0, 0]); out = [start]; cur = nxt[start]
    while cur != start and len(out) <= len(e_):
        out.append(cur); cur = nxt.get(cur, start)
    return np.asarray(out) if len(out) == len(e_) else None


def _zip_tube(A, B, pos):
    """
    Triangle strip (a tube: the 1-handle of a fusion) joining two boundary loops A and B, both ordered along their
    directed boundary edges. A is walked forward and B backward (facing holes have opposite orientations) from the
    closest pair of points; at each step the loop whose next point has the smaller arc-length fraction advances, giving
    the face (a_i+1, a_i, b) or (b_j, b_j-1, a) — consistently oriented with both rims.
    """
    j0 = int(np.argmin(np.linalg.norm(pos[B] - pos[A[0]], axis=1)))
    B = np.roll(B[::-1], -int(np.flatnonzero(B[::-1] == B[j0])[0]))     # backward, starting at the closest point

    def frac(L):
        d = np.linalg.norm(np.diff(pos[np.r_[L, L[:1]]], axis=0), axis=1)
        return np.r_[0.0, np.cumsum(d)] / max(d.sum(), 1e-300)
    fa, fb = frac(A), frac(B)
    i = j = 0; nA, nB = len(A), len(B); faces = []
    while i < nA or j < nB:
        a, a1 = A[i % nA], A[(i + 1) % nA]; b, b1 = B[j % nB], B[(j + 1) % nB]
        if j >= nB or (i < nA and fa[i + 1] <= fb[j + 1]):
            faces.append([a1, a, b]); i += 1
        else:
            faces.append([b, b1, a]); j += 1
    return np.asarray(faces, np.int64)


def _particles(ctx, maps, death_factor=2.5, k_nn=8, relax=2, event_remesh=True, adaptive_remesh=True,
               fold_tol=1.5, drift_tol=0.75, drift_share=0.01):
    """
    Particle system of the scene (material with a lifecycle), advected in a Lagrangian way. Particles start as the
    vertices of the first frame and live at CONTINUOUS positions on the surfaces: at every frame each particle is moved
    by the observed displacement field (displacements V_k[u] - V_{k-1}[map(u)] of the scene maps, inverse-distance
    weighted over its k_nn nearest mapped points) and projected onto the closest point of the new surface, so the
    material neither collapses onto vertices nor leaves the surface; its cell is the cell of the triangle it lands on.
    A particle DIES when the nearest mapped point is farther than death_factor x the median edge length (no material
    of the new frame comes from it: the contact surfaces annihilated at a fusion). NEW particles are born at the
    new-material vertices of the scene maps and, at a division / fusion, as DISK CAPS: the surgery cuts the material
    mesh along a circle (faces of particles in different cells), and every boundary loop left open is closed by a disk
    — one new particle at the loop's centre, projected on the surface, joined to every rim edge (the 2-handle of a
    division caps both daughters; at a fusion the holes left by the annihilated contact are closed the same way). Faces: the first frame's faces, the faces joining
    newborn particles to alive ones, and at a fusion the seam faces joining material of different parents. After every
    projection the particles are relaxed tangentially. event_remesh (default): at a division / fusion the particles of
    the parent cells end and every child cell is RE-SEEDED from its own frame surface (one particle per vertex, its
    triangles) — the standard remesh-and-transfer treatment of topology changes in explicit surface tracking: the
    children are exactly their closed surfaces (no seam, tube or interior sheet); continuity of the trajectories comes
    from the anchoring (_anchor: a newborn follows its nearest parent particle before its birth, a parent particle its
    nearest child particle after its end). adaptive_remesh (default): the same re-seeding for any cell whose material
    mesh stops representing its surface (Lagrangian–Eulerian remeshing): a material face farther than fold_tol x the
    median edge from the cell's real surface (a sheet / fold through the interior, e.g. a waist that widens much faster
    than the material can follow after a fusion), or more than drift_share of the faces farther than drift_tol x the
    edge. The particles are (relax passes of Laplacian smoothing over the faces not cut —
    never across a division or an old cell boundary — each followed by the projection on the surface), which keeps the
    material mesh regular without moving material off the surface.
    Returns positions (T, N, 3) with NaN outside the lifetime, cell (T, N) (-1 = not alive), birth, death (frame,
    T = alive at the end), faces F (over the particles) and the frame each face appears.
    """
    import trimesh
    from scipy.spatial import cKDTree
    frames, keeps, T = ctx["frames"], ctx["keeps"], ctx["T"]
    remesh_log = []
    bkind, dkind = {}, {}            # why every particle is born / dies: 'material' (created / annihilated material)
    #                                  or 'remesh' (numerical re-seeding of a cell at an event or by the adaptive check)
    fusion_frames = set(int(r.frame) for r in ctx["fus"].itertuples()) if len(ctx["fus"]) else set()
    event_frames = fusion_frames | (set(int(r.frame) for r in ctx["divs"].itertuples()) if len(ctx["divs"]) else set())

    def labels(k):
        tr_ = np.full(len(frames[k]["V"]), -1, np.int64)
        for tid in _tracks_at(ctx, k):
            tr_[keeps[(k, tid)]] = tid
        return tr_
    V0 = frames[0]["V"]
    live0 = np.unique(np.concatenate([keeps[(0, tid)] for tid in _tracks_at(ctx, 0)]))
    loc0 = -np.ones(len(V0), np.int64); loc0[live0] = np.arange(len(live0))
    F0 = frames[0]["F"]
    faces = [loc0[F0[(loc0[F0] >= 0).all(1)]]]; face_birth = [np.zeros(len(faces[0]), np.int64)]
    pos = V0[live0].copy(); alive = np.ones(len(pos), bool); cell = labels(0)[live0]
    cut_state = np.zeros(len(faces[0]), bool)                 # faces cut so far (a surgery is permanent)

    def relax_positions(surf, lab):
        from scipy.sparse import coo_matrix
        Fall = np.concatenate(faces)
        ok = ~cut_state & alive[Fall].all(1)
        if not ok.any():
            return
        E = np.concatenate([Fall[ok][:, [0, 1]], Fall[ok][:, [1, 2]], Fall[ok][:, [2, 0]]])
        n = len(pos)
        A = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        A.data[:] = 1.0
        deg = np.asarray(A.sum(1)).ravel(); use = (deg > 0) & alive
        for _ in range(int(relax)):
            avg = (A @ pos) / np.maximum(deg, 1)[:, None]
            moved = pos[use] + 0.5 * (avg[use] - pos[use])
            proj, _, tri = trimesh.proximity.closest_point(surf, moved)
            pos[use] = proj; cell[use] = lab[surf.faces[tri, 0]]
    birth = [0] * len(pos); death = [T] * len(pos)
    pos_rows, cell_rows = [], []
    for k in pbar(range(T), desc="Scene particles"):
        V, Fk = frames[k]["V"], frames[k]["F"]; lab = labels(k)
        if k > 0:
            Vp, Fp = frames[k - 1]["V"], frames[k - 1]["F"]
            h = float(np.median(np.linalg.norm(Vp[Fp[:, 0]] - Vp[Fp[:, 1]], axis=1)))
            m = maps[k]; cv = np.flatnonzero(m >= 0)
            remesh_now = event_remesh and k in event_frames
            children = set()
            if remesh_now:                                          # remesh-and-transfer: the parents end here
                ev = ctx["tr"][(ctx["tr"].frame == k) & ctx["tr"].event.isin(["division", "fusion"])] if len(ctx["tr"]) else []
                parents = set()
                for r_ in (ev.itertuples() if len(ev) else []):
                    parents |= set(int(x) for x in eval(r_.parent_tracks)); children |= set(int(x) for x in eval(r_.child_tracks))
                for pid in np.flatnonzero(alive & np.isin(cell, list(parents))):
                    alive[pid] = False; death[pid] = k; cell[pid] = -1; dkind[int(pid)] = "remesh"
            ids = np.flatnonzero(alive); prev_cell = cell.copy()
            if len(ids) and len(cv):
                Pm = Vp[m[cv]]; Dm = V[cv] - Pm
                kk = min(k_nn, len(cv))
                dist, j = cKDTree(Pm).query(pos[ids], k=kk)
                dist = dist.reshape(len(ids), kk); j = j.reshape(len(ids), kk)
                dead = dist[:, 0] > death_factor * h
                for pid in ids[dead]:
                    alive[pid] = False; death[pid] = k; cell[pid] = -1; dkind[int(pid)] = "material"
                keep_ = ~dead; ids2 = ids[keep_]
                w = 1.0 / np.maximum(dist[keep_], 1e-12) ** 2; w /= w.sum(1, keepdims=True)
                moved = pos[ids2] + np.einsum("ij,ijk->ik", w, Dm[j[keep_]])
                surf = trimesh.Trimesh(V, Fk, process=False)
                if not event_remesh and any(0 <= k - f_ <= 2 for f_ in fusion_frames):   # (remesh: parents already ended)
                    # fusion: the facing membranes of the parents end up INSIDE the fused solid and are annihilated
                    # (inside / outside test on the closed fused surface, deeper than half an edge)
                    try:
                        inside = surf.contains(moved)
                        if inside.any():
                            depth = trimesh.proximity.closest_point(surf, moved[inside])[1]
                            gone = np.flatnonzero(inside)[depth > 0.5 * h]
                            for pid in ids2[gone]:
                                alive[pid] = False; death[pid] = k; cell[pid] = -1; dkind[int(pid)] = "material"
                            kept = np.ones(len(ids2), bool); kept[gone] = False
                            ids2, moved = ids2[kept], moved[kept]
                    except Exception:  # noqa: BLE001 - mesh not watertight: no interior test
                        pass
                proj, _, tri = trimesh.proximity.closest_point(surf, moved)
                pos[ids2] = proj
                cell[ids2] = lab[Fk[tri, 0]]
            hk = float(np.median(np.linalg.norm(V[Fk[:, 0]] - V[Fk[:, 1]], axis=1)))
            ids = np.flatnonzero(alive)
            new_v = np.flatnonzero((m < 0) & (lab >= 0))                # new material of the scene maps
            if remesh_now and children:                                 # every child re-seeded from its frame surface
                new_v = np.union1d(new_v, np.flatnonzero(np.isin(lab, list(children))))
            nb_start = len(pos)
            if len(new_v):                                     # newborn particles (new material)
                pos = np.concatenate([pos, V[new_v]]); alive = np.concatenate([alive, np.ones(len(new_v), bool)])
                cell = np.concatenate([cell, lab[new_v]]); prev_cell = np.concatenate([prev_cell, np.full(len(new_v), -1)])
                for i_, v_ in enumerate(new_v):
                    bkind[nb_start + i_] = "remesh" if (remesh_now and int(lab[v_]) in children) else "material"
                birth += [k] * len(new_v); death += [T] * len(new_v)
            if len(new_v) or k in fusion_frames:
                # owner of every vertex of the frame: its newborn particle, else the nearest alive particle
                ids = np.flatnonzero(alive)
                _, jn = cKDTree(pos[ids]).query(V); owner = ids[jn]
                if len(new_v):
                    owner[new_v] = np.arange(nb_start, nb_start + len(new_v))
                fo = owner[Fk]
                distinct = (fo[:, 0] != fo[:, 1]) & (fo[:, 1] != fo[:, 2]) & (fo[:, 0] != fo[:, 2])
                add = np.zeros(len(Fk), bool)
                if len(new_v):
                    add |= distinct & (fo >= nb_start).any(1)          # faces joining newborn particles
                if k in fusion_frames and not remesh_now:              # the seam: material of different parents
                    pc = prev_cell[fo]
                    add |= distinct & (pc.min(1) != pc.max(1)) & (pc.min(1) >= 0)
                if add.any():
                    faces.append(fo[add]); face_birth.append(np.full(int(add.sum()), k, np.int64))
                    cut_state = np.concatenate([cut_state, np.zeros(int(add.sum()), bool)])
            # permanent surgery: faces whose particles died or are in different cells stay cut
            Fall = np.concatenate(faces)
            cut_state |= ~(alive[Fall].all(1) & (cell[Fall].min(1) == cell[Fall].max(1)))
            if True:                                            # rims left by the surgery or by dying material: caps
                                                                # (tubes between different parents at a fusion)
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import connected_components
                act = Fall[~cut_state]
                de = np.concatenate([act[:, [0, 1]], act[:, [1, 2]], act[:, [2, 0]]])
                key = de[:, 0] * len(pos) + de[:, 1]; rkey = de[:, 1] * len(pos) + de[:, 0]
                bnd = de[~np.isin(key, rkey)]                       # directed boundary edges (a -> b)
                if len(bnd) >= 3:
                    n_ = len(pos)
                    g = coo_matrix((np.ones(len(bnd)), (bnd[:, 0], bnd[:, 1])), shape=(n_, n_))
                    _, comp = connected_components(g, directed=False)
                    surf = trimesh.Trimesh(V, Fk, process=False)
                    new_faces = []
                    loops = [bnd[comp[bnd[:, 0]] == c_] for c_ in np.unique(comp[bnd[:, 0]])]
                    loops = [e_ for e_ in loops if len(e_) >= 3]
                    if k in fusion_frames and len(loops) >= 2 and not (event_remesh and k in event_frames):
                        # the 1-handle of a fusion: a tube between the holes of different parents (closest pairs)
                        info_ = []
                        for e_ in loops:
                            L = _ordered_loop(e_)
                            pc_ = prev_cell[np.unique(e_)]; pc_ = pc_[pc_ >= 0]
                            info_.append((L, int(np.bincount(pc_).argmax()) if len(pc_) else -1, pos[np.unique(e_)].mean(0)))
                        used = set()
                        order = sorted(((np.linalg.norm(info_[a][2] - info_[b][2]), a, b) for a in range(len(loops))
                                        for b in range(a + 1, len(loops))
                                        if info_[a][1] != info_[b][1] and info_[a][1] >= 0 and info_[b][1] >= 0
                                        and info_[a][0] is not None and info_[b][0] is not None))
                        for _, a, b in order:
                            if a in used or b in used:
                                continue
                            new_faces.append(_zip_tube(info_[a][0], info_[b][0], pos)); used |= {a, b}
                        loops = [e_ for q_, e_ in enumerate(loops) if q_ not in used]
                    for e_ in loops:
                        loop = np.unique(e_)
                        cp, _, tri = trimesh.proximity.closest_point(surf, pos[loop].mean(axis=0)[None, :])
                        nid = len(pos)
                        pos = np.concatenate([pos, cp]); alive = np.concatenate([alive, [True]])
                        cell = np.concatenate([cell, [lab[Fk[tri[0], 0]]]]); birth.append(k); death.append(T)
                        bkind[nid] = "material"                    # new membrane closing a hole
                        new_faces.append(np.c_[e_[:, 1], e_[:, 0], np.full(len(e_), nid)])
                    if new_faces:
                        nf = np.concatenate(new_faces)
                        faces.append(nf); face_birth.append(np.full(len(nf), k, np.int64))
                        cut_state = np.concatenate([cut_state, np.zeros(len(nf), bool)])
            if relax and k > 0:
                relax_positions(trimesh.Trimesh(V, Fk, process=False), lab)
            if adaptive_remesh and k > 0:
                # Lagrangian–Eulerian remeshing: a cell whose material mesh left its real surface is re-seeded
                Fall = np.concatenate(faces); act = np.flatnonzero(~cut_state)
                hk_ = float(np.median(np.linalg.norm(V[Fk[:, 0]] - V[Fk[:, 1]], axis=1)))
                reseed = []
                born_now = np.asarray(birth) == k
                for c in sorted(set(lab[lab >= 0].tolist())):
                    in_c = alive & (cell == c)
                    if in_c.any() and born_now[:len(in_c)][in_c].all():
                        continue                                    # (re-)seeded in this very frame: faithful
                    fc_ = act[(cell[Fall[act]] == c).all(1)]
                    if not len(fc_):
                        continue
                    kc_ = ctx["keeps"].get((k, c))
                    if kc_ is None:
                        continue
                    mloc = np.zeros(len(V), bool); mloc[kc_] = True
                    Fc_ = Fk[mloc[Fk].all(1)]
                    surf_c = trimesh.Trimesh(V, Fc_, process=False)
                    dd = trimesh.proximity.closest_point(surf_c, pos[Fall[fc_]].mean(axis=1))[1]
                    if dd.max() > fold_tol * hk_ or np.mean(dd > drift_tol * hk_) > drift_share:
                        reseed.append(c)
                for c in reseed:
                    for pid in np.flatnonzero(alive & (cell == c)):
                        alive[pid] = False; death[pid] = k; cell[pid] = -1; dkind[int(pid)] = "remesh"
                    kc_ = ctx["keeps"][(k, c)]
                    nid = np.arange(len(pos), len(pos) + len(kc_))
                    pos = np.concatenate([pos, V[kc_]]); alive = np.concatenate([alive, np.ones(len(kc_), bool)])
                    for i_ in range(len(kc_)):
                        bkind[len(pos) - len(kc_) + i_] = "remesh"
                    cell = np.concatenate([cell, np.full(len(kc_), c)]); birth += [k] * len(kc_); death += [T] * len(kc_)
                    loc_ = -np.ones(len(V), np.int64); loc_[kc_] = nid
                    Fc_ = loc_[Fk[(loc_[Fk] >= 0).all(1)]]
                    faces.append(Fc_); face_birth.append(np.full(len(Fc_), k, np.int64))
                    cut_state = np.concatenate([cut_state, np.zeros(len(Fc_), bool)])
                if reseed:
                    Fall = np.concatenate(faces)
                    cut_state |= ~(alive[Fall].all(1) & (cell[Fall].min(1) == cell[Fall].max(1)))
                    remesh_log.append((k, reseed))
        P = np.full((len(pos), 3), np.nan); P[alive] = pos[alive]
        C = np.where(alive, cell, -1)
        pos_rows.append(P); cell_rows.append(C.copy())
    N = len(pos)
    X = np.full((T, N, 3), np.nan); CELL = np.full((T, N), -1, np.int64)
    for k in range(T):
        X[k, :len(pos_rows[k])] = pos_rows[k]; CELL[k, :len(cell_rows[k])] = cell_rows[k]
    n_ = len(birth)
    ctx["_particle_kinds"] = (np.array([bkind.get(i, "material" if birth[i] > 0 else "initial") for i in range(n_)], dtype=object),
                              np.array([dkind.get(i, "material" if death[i] < T else "") for i in range(n_)], dtype=object))
    return X, CELL, np.asarray(birth), np.asarray(death), np.concatenate(faces), np.concatenate(face_birth)


def _anchor(XL, CELL, birth, death):
    """
    Finite positions for every particle at every frame without floating material: before its birth a particle follows
    the nearest OLDER particle alive at its birth frame, after its death the nearest particle that survives it — so a
    particle that is not alive always coincides with real material (it is never seen on its own, adds no spurious motion
    to the analyses and its faces are masked). Births are resolved in increasing order and deaths in decreasing order,
    so anchors that are themselves anchored are already complete.
    """
    from scipy.spatial import cKDTree
    X = XL.copy(); T, N = CELL.shape
    alive = CELL >= 0
    for b in sorted(set(int(x) for x in birth if x > 0)):
        P = np.flatnonzero((birth == b) & alive[b])
        Q = np.flatnonzero((birth < b) & alive[b - 1])            # the material just before: (re-seeded) parents included
        if len(Q) and len(P) and b < T:
            _, j = cKDTree(XL[b - 1, Q]).query(XL[b, P]); X[:b, P] = X[:b, Q[j]]
    for d in sorted(set(int(x) for x in death if x < T), reverse=True):
        P = np.flatnonzero((death == d) & alive[d - 1]) if d >= 1 else np.zeros(0, np.int64)
        Q = np.flatnonzero((death > d) & alive[d]) if d >= 1 else np.zeros(0, np.int64)   # survivors and newborns at d
        if len(Q) and len(P) and d >= 1:
            _, j = cKDTree(XL[d, Q]).query(XL[d - 1, P]); X[d:, P] = X[d:, Q[j]]
    for k in range(1, T):                                     # anything left: hold
        miss = np.isnan(X[k, :, 0]); X[k, miss] = X[k - 1, miss]
    for k in range(T - 2, -1, -1):
        miss = np.isnan(X[k, :, 0]); X[k, miss] = X[k + 1, miss]
    return np.nan_to_num(X)


def write_trajectories(ctx, maps, method="maps", substeps=4, schemes=("hermite_catmull_rom",)):
    """
    Scene trajectories with a particle lifecycle (_particles): material conserved inside a cell, particles that die
    (contact surfaces at a fusion) and particles born (furrow, bridge) at the events, faces cut when their particles
    are in different cells or not alive. Writes the usual Trajectories files (trajectories.npy with held positions for
    the analyses, reference_faces.npy, time.npy, tracking.csv, kinematics/velocity.npy, info.json) plus
    trajectories_lifecycle.npy (NaN outside the lifetimes), material_cell.npy, particle_birth / particle_death.npy,
    cut_faces.npy and the sub-frame interpolation (interpolation/hermite_catmull_rom.npy, linear.npy,
    query_times.npy, cut_faces.npy): inside a lifetime Catmull–Rom, an event takes effect at the middle of its interval.
    """
    frames, T, t = ctx["frames"], ctx["T"], ctx["t"]
    d = ensure_dir(ctx["target"] / "Trajectories")
    XL, CELL, birth, death, F, fbirth = _particles(ctx, maps, **{k: v for k, v in ctx.get("opts", {}).items()
                                                                  if k in PARTICLE_OPTIONS and v is not None})
    ever = (CELL >= 0).any(axis=0)                     # particles never alive (no position at any frame): removed
    if not ever.all():
        new_id = -np.ones(len(ever), np.int64); new_id[ever] = np.arange(int(ever.sum()))
        keepf = ever[F].all(axis=1)
        XL, CELL, birth, death = XL[:, ever], CELL[:, ever], birth[ever], death[ever]
        if "_particle_kinds" in ctx:
            ctx["_particle_kinds"] = tuple(a[ever] for a in ctx["_particle_kinds"])
        F, fbirth = new_id[F[keepf]], fbirth[keepf]
    used = np.unique(F)                                  # particles in no face (isolated) are dropped
    if len(used) < XL.shape[1]:
        re_ = -np.ones(XL.shape[1], np.int64); re_[used] = np.arange(len(used))
        XL, CELL, birth, death, F = XL[:, used], CELL[:, used], birth[used], death[used], re_[F]
    X = _anchor(XL, CELL, birth, death)
    alive = CELL >= 0
    fa = alive[:, F].all(axis=2)
    same = (CELL[:, F].min(axis=2) == CELL[:, F].max(axis=2))
    born = np.arange(T)[:, None] >= fbirth[None, :]
    broken = ~(fa & same) & born
    # a surgery is permanent: once cut (after its birth) a face stays cut; before its birth a face does not exist
    cut = np.maximum.accumulate(broken, axis=0) | ~born
    np.save(d / "trajectories.npy", X); np.save(d / "trajectories_lifecycle.npy", XL)
    np.save(d / "reference_faces.npy", F); np.save(d / "time.npy", t); np.save(d / "material_cell.npy", CELL)
    np.save(d / "particle_birth.npy", birth); np.save(d / "particle_death.npy", death); np.save(d / "cut_faces.npy", cut)
    with open(d / "frame_names.txt", "w") as fh:
        fh.write("\n".join(f["name"] for f in frames))
    kin = ensure_dir(d / "kinematics")
    vel = np.gradient(X, t, axis=0) if T > 1 else np.zeros_like(X)
    vel[~alive] = 0.0
    np.save(kin / "velocity.npy", vel)
    # sub-frame interpolation with the schemes of the trajectories stage (interpolate_sequence on the anchored
    # positions); the cut masks of an event interval switch at its middle
    from PynamicMesh.core.trajectories import interpolate_sequence
    if (d / "interpolation").is_dir():                        # only the schemes of this run
        shutil.rmtree(d / "interpolation")
    ip = ensure_dir(d / "interpolation")
    q = [t[0]] + [t[k - 1] + (t[k] - t[k - 1]) * s_ / substeps for k in range(1, T) for s_ in range(1, substeps + 1)]
    q = np.asarray(q); S = len(q)
    cut_s = np.zeros((S, len(F)), bool); cut_s[0] = cut[0]; idx = 1
    for k in range(1, T):
        for s_ in range(1, substeps + 1):
            cut_s[idx] = cut[k - 1] if s_ / substeps < 0.5 else cut[k]; idx += 1
    written = []
    for sc in (schemes or ("hermite_catmull_rom",)):
        try:
            Y = np.asarray(interpolate_sequence(X, F, sc, q, t=t), dtype=float)
            if Y.shape[0] == S:
                np.save(ip / f"{sc}.npy", Y); written.append(sc)
        except Exception as exc:  # noqa: BLE001 - a scheme that cannot run here is skipped
            print(f"[Scene trajectories] interpolation scheme {sc} skipped: {exc}")
    np.save(ip / "query_times.npy", q); np.save(ip / "cut_faces.npy", cut_s)
    bk, dk = ctx.get("_particle_kinds", (np.full(len(birth), "material", dtype=object), np.full(len(death), "material", dtype=object)))
    np.save(d / "particle_birth_kind.npy", bk.astype(str)); np.save(d / "particle_death_kind.npy", dk.astype(str))
    rows = []
    for k in range(T):
        b_, d_ = (birth == k) & (k > 0), death == k
        rows.append({"frame": k, "covered_fraction": 1.0, "reliable_fraction": float(alive[k].mean()), "inpainted": 0,
                     "alive_particles": int(alive[k].sum()), "born": int(b_.sum()), "died": int(d_.sum()),
                     # created / annihilated material vs numerical re-seeding of a cell (remesh)
                     "born_material": int((b_ & (bk == "material")).sum()), "born_remesh": int((b_ & (bk == "remesh")).sum()),
                     "died_material": int((d_ & (dk == "material")).sum()), "died_remesh": int((d_ & (dk == "remesh")).sum())})
    pd.DataFrame(rows).to_csv(d / "tracking.csv", index=False)
    info = {"n_frames": T, "n_reference_vertices": int(X.shape[1]), "dt": float(np.median(np.diff(t))) if T > 1 else 1.0,
            "times": t.tolist(), "trajectory_method": "scene particles (lifecycle through the scene maps)",
            "particles_initial": int((birth == 0).sum()), "particles_born": int((birth > 0).sum()),
            "particles_died": int((death < T).sum()), "cut_faces_per_frame": cut.sum(1).tolist(),
            "schemes": written, "substeps": substeps,
            "events": ctx["tr"][["frame", "event"]].to_dict("records") if len(ctx["tr"]) else []}
    with open(d / "info.json", "w") as fh:
        json.dump(info, fh, indent=2)
    ensure_dir(d / "plots")
    spd = np.linalg.norm(vel, axis=2)
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for tid in sorted(set(np.unique(CELL)) - {-1}):
        ax[0].plot(t, np.nanmean(np.where(CELL == tid, spd, np.nan), axis=1), "-o", ms=3, label=f"cell {tid}")
        cnt = (CELL == tid).sum(1).astype(float); cnt[cnt == 0] = np.nan      # only during the cell's lifetime
        ax[1].plot(t, cnt, "-o", ms=3)
    bm = np.array([r["born_material"] for r in rows]); br = np.array([r["born_remesh"] for r in rows])
    dm = np.array([r["died_material"] for r in rows]); dr = np.array([r["died_remesh"] for r in rows])
    ax[2].bar(t, bm, color="tab:green", label="created material")
    ax[2].bar(t, -dm, color="tab:red", label="annihilated material")
    ax[2].bar(t, br, bottom=bm, color="lightgray", hatch="//", edgecolor="gray", label="re-seeded (remesh, numerical)")
    ax[2].bar(t, -dr, bottom=-dm, color="lightgray", hatch="//", edgecolor="gray")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("mean speed"); ax[1].set_ylabel("particles per cell"); ax[2].set_ylabel("born (+) / died (-)")
    ax[2].set_xlabel("time"); ax[0].legend(fontsize=7, ncol=6); ax[2].legend(fontsize=7)
    ax[0].set_title("scene particles: material followed through the divisions and fusions")
    for a in ax:
        _event_lines(a, ctx); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(d / "plots" / "scene_particles.png", dpi=200); plt.close(fig)
    return X, F


# --------------------------------------------------------------------------------------------------------------- #
#  Lagrangian analyses of the cells transferred onto the scene particles (DynamicAnalysis, MotionAnalysis, ReebDynamics)
# --------------------------------------------------------------------------------------------------------------- #
_FRAME_RE = __import__("re").compile(r"(_T|frame_|_t)(\d{4})")
_FRAME_COLS = ("frame", "timestep", "time_step", "t_index", "frame_index")


def _time_hold(A):
    """Fill the elements without a value (NaN) of a time series A (time first) with the value of the SAME element at the
    nearest earlier frame that has one, else at the nearest later frame (forward then backward fill along time).
    Frames that no cell defines (surgeries) and the first frame of material created at an event or a re-seeding then
    keep a value, for a continuous display; elements with a value are unchanged."""
    A = np.asarray(A, dtype=float)
    if A.shape[0] < 2 or np.isfinite(A).all():
        return A
    T = A.shape[0]; ok = np.isfinite(A)
    ar = np.broadcast_to(np.arange(T).reshape((T,) + (1,) * (A.ndim - 1)), A.shape)
    fi = np.where(ok, ar, 0); fi = np.maximum.accumulate(fi, axis=0)                       # last valid frame
    bi = np.where(ok, ar, T - 1); bi = np.minimum.accumulate(bi[::-1], axis=0)[::-1]      # next valid frame
    ff = np.take_along_axis(A, fi, axis=0); bf = np.take_along_axis(A, bi, axis=0)
    return np.where(ok, A, np.where(np.isfinite(ff), ff, bf))


_CUMULATIVE = ("cumulative_", "log_area_change", "nonrigid_displacement")


def _lineage_accumulate(S, scene_traj):
    """
    A CUMULATIVE quantity (time first, particles second) accumulated along the LINEAGE: every cell accumulates from its
    own birth, so at a division / fusion the new cell restarts from 0; the scene value is the cell's own value plus
    what the material inherited. A particle's inheritance is fixed when it first appears, from the nearest particle of
    the previous frame: re-seeded within the SAME cell (adaptive re-seeding: the cell's accumulation continues) -> that
    particle's inheritance; created at an event (a new cell starting from 0) -> that particle's TOTAL accumulated value.
    Particles alive at the first frame inherit nothing. Values are continuous through every event.
    """
    from scipy.spatial import cKDTree
    if scene_traj is None:
        return S
    XL, CELL, _, _ = scene_traj
    T, N = CELL.shape
    if S.ndim < 2 or S.shape[0] != T or S.shape[1] != N:
        return S
    out = np.array(S, dtype=float)
    inh = np.zeros((N,) + S.shape[2:]); seen = CELL[0] >= 0
    for k in range(1, T):
        al = CELL[k] >= 0
        new = np.flatnonzero(al & ~seen)
        if len(new):
            q_all = np.flatnonzero(CELL[k - 1] >= 0)
            if len(q_all):
                _, j = cKDTree(XL[k - 1, q_all]).query(XL[k, new]); q = q_all[j]
                same = CELL[k - 1, q] == CELL[k, new]
                tot = np.nan_to_num(out[k - 1, q])
                inh[new] = np.where(same.reshape((-1,) + (1,) * (S.ndim - 2)), inh[q], tot)
            seen[new] = True
        out[k, al] = S[k, al] + inh[al]
    return out


def _exist_fill(H, frames_k, scene_traj):
    """
    Values of a series H (series frame first; frames_k: the scene frame of every row) restricted to the elements that
    EXIST: the element axis (particles: N, or faces of the scene mesh) is found on axis 1 or 2; at every frame the
    elements that do not exist (particles not alive, cut faces) are emptied, and existing elements still without a
    value (material created at an event / a re-seeding that has no value at any frame of its own) take the value of the
    NEAREST element in space at the nearest frame that has values. Other arrays are returned unchanged.
    """
    from scipy.spatial import cKDTree
    if scene_traj is None:
        return H
    XL, CELL, F, cut = scene_traj
    N, nF = CELL.shape[1], len(F)
    ax = next((a for a in (1, 2) if H.ndim > a and H.shape[a] in (N, nF)), None)
    if ax is None:
        return H
    H2 = np.moveaxis(np.array(H, dtype=float), ax, 1)              # (frames, elements, ...)
    face = H2.shape[1] == nF and nF != N
    flat = H2.reshape(H2.shape[0], H2.shape[1], -1)
    exist = [(~cut[k]) if face else (CELL[k] >= 0) for k in frames_k]

    def where_(k):
        P = XL[k]
        return P[F].mean(axis=1) if face else P
    for i in range(len(frames_k)):
        flat[i][~exist[i]] = np.nan
    valued = [exist[i] & np.isfinite(flat[i]).all(axis=1) for i in range(len(frames_k))]
    for i, k in enumerate(frames_k):
        need = np.flatnonzero(exist[i] & ~valued[i])
        if not len(need):
            continue
        for d in range(0, len(frames_k)):                          # the nearest frame with values
            cand = [j for j in (i, i + d, i - d) if 0 <= j < len(frames_k) and valued[j].any()]
            cand = [j for j in cand if j != i or d == 0]
            if cand:
                j = cand[0]; src = np.flatnonzero(valued[j])
                pa, pb = where_(frames_k[j])[src], where_(k)[need]
                okb = np.isfinite(pb).all(axis=1)
                if okb.any() and len(src):
                    _, nn = cKDTree(pa).query(pb[okb]); flat[i][need[okb]] = flat[j][src[nn]]
                break
    return np.moveaxis(flat.reshape(H2.shape), 1, ax)


def _hold_series(out_root, series, log, scene_traj=None):
    """Temporal hold (_time_hold) of every per-frame series of a transferred stage (series: file pattern with '{:04d}'
    -> frames). .npy series and every float key of .npz series (never the mesh 'vertices' / 'faces') whose shape is the
    same at every frame; the files are rewritten only where values were filled, and the log says how many."""
    out_root = Path(out_root)
    for pat, ks in series.items():
        files = [out_root / pat.format(k) for k in range(min(ks), max(ks) + 1)]
        if len(files) < 2 or not all(f.exists() for f in files):
            continue
        try:
            if files[0].suffix == ".npy":
                arrs = [np.load(f, allow_pickle=True) for f in files]
                if all(np.issubdtype(a.dtype, np.floating) for a in arrs) and len({a.shape for a in arrs}) == 1:
                    S = np.stack(arrs); H = _exist_fill(_time_hold(S), list(range(min(ks), max(ks) + 1)), scene_traj)
                    n_fill = int(np.sum(~np.isfinite(S) & np.isfinite(H)))
                    if n_fill or not np.array_equal(np.isfinite(S), np.isfinite(H)):
                        for f, a in zip(files, H):
                            np.save(f, a)
                        log.append(f"scene {pat}: {n_fill} values held from the nearest frame of the same element")
            else:
                zs = [dict(np.load(f, allow_pickle=True)) for f in files]
                held, changed = 0, False
                for kk in list(zs[0]):
                    if kk in ("vertices", "V", "points", "faces", "F"):
                        continue
                    arrs = [z.get(kk) for z in zs]
                    if any(a is None for a in arrs) or len({np.shape(a) for a in arrs}) != 1 \
                            or not all(np.issubdtype(np.asarray(a).dtype, np.floating) for a in arrs):
                        continue
                    S = np.stack(arrs); H = _exist_fill(_time_hold(S), list(range(min(ks), max(ks) + 1)), scene_traj)
                    n_fill = int(np.sum(~np.isfinite(S) & np.isfinite(H)))
                    if n_fill or not np.array_equal(np.isfinite(S), np.isfinite(H)):
                        held += n_fill; changed = True
                        for z, a in zip(zs, H):
                            z[kk] = a
                if changed:
                    for f, z in zip(files, zs):
                        np.savez_compressed(f, **z)
                    log.append(f"scene {pat}: {held} values filled (existing elements only; non-existing ones emptied)")
        except Exception as exc:  # noqa: BLE001 - never stops the stage
            log.append(f"scene {pat}: hold FAILED ({type(exc).__name__}: {exc})")


# per-cell descriptors: every cell has its own axes / modes / embedding, a value of the union has no meaning
_PER_CELL_TABLES = {"rigid_motion.csv", "polarity.csv", "signals.csv", "signals_with_confidence.csv", "mds.csv",
                    "shape_dna_spectra.csv", "event_timeline.csv"}
_ID_COLUMNS = {"region", "mode", "branch", "track", "track_id", "signal", "dimension", "animation", "frame_start",
               "frame_end", "other", "kind", "node", "label", "field", "degree", "order", "l", "m", "band", "harmonic"}


# index / position columns: never plotted or aggregated as quantities (sub-frame sample counters, vertex numbers,
# parent identifiers)
_INDEX_COLUMNS = {"sample", "step", "frame_start", "frame_end", "extremum", "saddle", "vertex", "parent_track",
                  "start_node", "end_node", "first_frame", "last_frame", "origin_frame"}


def _is_key_column(df, c):
    """A key (identity) column: a known key name, the members of a pair (track_a / track_b, branch_a / branch_b ...:
    a contact between two objects), or any NON-numeric column (text is read as 'object' by pandas < 3 and as the
    'string' dtype by pandas >= 3 — both are keys)."""
    cl = c.lower()
    if cl in _INDEX_COLUMNS:
        return False
    return (cl in _ID_COLUMNS or cl.startswith(("track_", "branch_", "region_", "node_"))
            or not pd.api.types.is_numeric_dtype(df[c]))


def _cell_areas(ctx):
    """Surface area of every cell at every frame {(frame, cell): area} (weights of the intensive quantities)."""
    if "_cell_areas" not in ctx:
        out = {}
        for (k, c), kp in ctx["keeps"].items():
            V, Fk = ctx["frames"][k]["V"], ctx["frames"][k]["F"]
            m = np.zeros(len(V), bool); m[kp] = True; Fs = Fk[m[Fk].all(1)]
            out[(k, c)] = float(0.5 * np.linalg.norm(np.cross(V[Fs[:, 1]] - V[Fs[:, 0]], V[Fs[:, 2]] - V[Fs[:, 0]]), axis=1).sum())
        ctx["_cell_areas"] = out
    return ctx["_cell_areas"]


def _column_rule(col):
    """How a per-cell column combines into ONE value for the union of the cells at a frame:
    'wmean' area-weighted mean (intensive: means, fractions, ratios, rates), 'sum' (extensive: counts, energies,
    totals, areas, volumes), 'max' (maxima; percentiles -> the largest cell percentile, an upper bound), 'min' (minima,
    confidences: the least reliable cell)."""
    c = col.lower()
    if c.startswith(("max", "p9", "peak")) or c.endswith(("_max", "_p95", "_p99")):
        return "max"                                            # (max_stretch_ratio is a maximum, not a ratio mean)
    if c.startswith(("mean", "avg", "frac", "ratio", "rel_", "median")) or "fraction" in c or "ratio" in c or c.endswith("_mean"):
        return "wmean"
    if c.startswith("min") or c.endswith("_min") or "confidence" in c:
        return "min"
    if c.startswith(("n_", "num_", "count", "e_", "total", "sum_")) or c in ("n", "area", "volume", "energy") \
            or c.endswith(("_area", "_volume", "_energy", "_count")):
        return "sum"
    return "wmean"


def _aggregate_cells(cat, name, areas, starts=None):
    """
    The scene value of a per-frame table of the cells (one row per cell and frame), by the meaning of every column
    (_column_rule; weights of the intensive ones: the cell areas at that frame), plus n_cells. Tables that do NOT
    combine — per-cell descriptors (_PER_CELL_TABLES: own axes, modes, embeddings), tables with other key columns
    (regions, modes, branches...) or without a frame column — are returned as None (kept one row per cell).
    """
    fc = next((c for c in cat.columns if c.lower() in _FRAME_COLS), None)
    if fc is None:
        return None, "one row per cell (no frame column)"
    if name in _PER_CELL_TABLES:
        return None, "one row per cell (per-cell descriptor: no value for the union of the cells)"
    ids = [c for c in cat.columns if c not in ("cell", fc) and _is_key_column(cat, c)]
    if ids:
        return None, f"one row per cell (other key columns: {', '.join(ids)})"
    if starts:
        # a cell born at a division / fusion: the per-cell pipeline sets the confidence of its FIRST frame to 1 by
        # convention (no previous frame); that frame is the event transition itself — no confidence for the scene
        born = np.array([int(k) == int(starts.get(int(c), -1)) and int(starts.get(int(c), 0)) > 0
                         for k, c in zip(cat[fc], cat["cell"])])
        if born.any():
            cat = cat.copy()
            for c in cat.columns:
                if "confidence" in c.lower() and pd.api.types.is_numeric_dtype(cat[c]) \
                        and not pd.api.types.is_bool_dtype(cat[c]):          # (low_confidence flags stay as they are)
                    cat[c] = cat[c].astype(float); cat.loc[born, c] = np.nan
    w = np.array([areas.get((int(k), int(c)), np.nan) for k, c in zip(cat[fc], cat["cell"])], float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    cols = [c for c in cat.columns if c not in ("cell", fc, "time") and pd.api.types.is_numeric_dtype(cat[c])
            and c.lower() not in _INDEX_COLUMNS]
    rows = []
    for k, g in cat.assign(_w=w).groupby(fc, sort=True):
        r = {fc: int(k)}
        if "time" in g:
            r["time"] = float(g["time"].iloc[0])
        r["n_cells"] = int(g["cell"].nunique())
        for c in cols:
            v = g[c].to_numpy(float); ok = np.isfinite(v)
            if not ok.any():
                r[c] = np.nan; continue
            rule = _column_rule(c)
            if rule == "sum":
                r[c] = float(v[ok].sum())
            elif rule == "max":
                r[c] = float(v[ok].max())
            elif rule == "min":
                r[c] = float(v[ok].min())
            else:
                ww = g["_w"].to_numpy(float)[ok]; r[c] = float((v[ok] * ww).sum() / ww.sum())
        rows.append(r)
    rules = {}
    for c in cols:
        rules.setdefault(_column_rule(c), []).append(c)
    how = "per frame over the cells: " + "; ".join(f"{k} [{', '.join(v)}]" for k, v in sorted(rules.items()))
    return pd.DataFrame(rows), how


def transfer_stage(ctx, stage, scene_traj, hold=True):
    """
    A Lagrangian analysis (fixed connectivity, defined between topological surgeries) of every cell onto the scene: the
    per-cell results are computed on a mesh whose connectivity is fixed for the cell's lifetime, so they carry no split /
    merge artefacts; every scene particle takes the value of the nearest vertex of its own cell's tracked mesh (faces:
    the nearest face), file names are renumbered to scene frames, transitions through an event (not defined by any
    cell) stay empty. Stacked arrays (frames x vertices ...) are assembled frame by frame; static per-cell arrays are
    kept per cell (cells/cell_XX/); tables are written per frame (mean over the cells, same name) and concatenated with
    a 'cell' column (cells_<name>.csv).
    """
    from scipy.spatial import cKDTree
    XL, CELL, F, cut = scene_traj
    T, N = CELL.shape
    if (ctx["target"] / stage).is_dir():                    # the scene's copy only (never a cell's results): no stale
        shutil.rmtree(ctx["target"] / stage)                  # files of an earlier run survive
    out_root = ensure_dir(ctx["target"] / stage)
    per_frame, stacked, tables = {}, {}, {}
    log = []

    def scene_frame_faces(k, c):
        live = F[~cut[k]]
        return live[(CELL[k][live] == c).all(1)]
    for tid in sorted(ctx["start"]):
        sd = _track_dir(ctx, tid) / stage; trj = _track_dir(ctx, tid) / "Trajectories"
        if not sd.is_dir() or not (trj / "trajectories.npy").exists():
            continue
        Xc = np.load(trj / "trajectories.npy"); Fc = np.load(trj / "reference_faces.npy")
        Tc, nc = Xc.shape[0], Xc.shape[1]; nfc = len(Fc); st = ctx["start"][tid]
        vmap, fmap = {}, {}

        def vmaps(j):                                           # particles of this cell at local frame j -> track vertex
            if j not in vmap:
                k = st + j
                P = np.flatnonzero(CELL[k] == tid) if 0 <= k < T else np.zeros(0, np.int64)
                vmap[j] = (P, cKDTree(Xc[min(j, Tc - 1)]).query(XL[k, P])[1] if len(P) else np.zeros(0, np.int64))
            return vmap[j]

        def fmaps(j):
            if j not in fmap:
                k = st + j; Fs = scene_frame_faces(k, tid) if 0 <= k < T else np.zeros((0, 3), np.int64)
                Xt = Xc[min(j, Tc - 1)]
                fmap[j] = (Fs, cKDTree(Xt[Fc].mean(1)).query(XL[k][Fs].mean(1))[1] if len(Fs) else np.zeros(0, np.int64))
            return fmap[j]

        def transfer(arr, j, store, key):
            a = np.asarray(arr)
            # the vertex / face axis may be the first or the second one (e.g. Hodge components: (4, faces, 3))
            if a.ndim >= 2 and a.shape[0] not in (nc, nfc) and a.shape[1] in (nc, nfc) and np.issubdtype(a.dtype, np.number):
                sub = {}
                if transfer(np.moveaxis(a, 1, 0), j, sub, key):
                    for kk, v in sub.items():
                        store.setdefault("__axis1__", set()).add(kk)
                        if kk.endswith("__faces"):
                            store.setdefault(kk, []).extend(v)
                        else:
                            store[kk] = v
                    return True
                return False
            if a.ndim >= 1 and a.shape[0] == nc and np.issubdtype(a.dtype, np.number):
                P, vi = vmaps(j)
                if key not in store:
                    store[key] = np.full((N,) + a.shape[1:], -1 if np.issubdtype(a.dtype, np.integer) else np.nan,
                                         dtype=a.dtype if np.issubdtype(a.dtype, np.integer) else float)
                store[key][P] = a[vi]
                return True
            if a.ndim >= 1 and a.shape[0] == nfc and np.issubdtype(a.dtype, np.floating):
                Fs, fi = fmaps(j)
                fkey = key + "__faces"
                store.setdefault(fkey, []).append((Fs, a[fi]))
                return True
            return False
        def one_file(f):
            rel = f.relative_to(sd)
            if f.suffix == ".csv":
                tables.setdefault(str(rel), []).append((tid, f)); return
            if f.suffix not in (".npy", ".npz"):
                return
            m = _FRAME_RE.search(f.name)
            if m:                                               # per-frame file
                j = int(m.group(2)); k = st + j
                name = f.name[:m.start(2)] + f"{k:04d}" + f.name[m.end(2):]
                key_file = str(rel.parent / name)
                store = per_frame.setdefault(key_file, {"k": k})
                if f.suffix == ".npy":
                    arr_ = np.load(f, allow_pickle=True)
                    if not transfer(arr_, j, store, "__array__"):
                        store.setdefault("__cells__", {})[tid] = arr_
                        log.append(f"cell {tid} {rel}: per-frame array NOT transferred (shape {np.shape(arr_)}; tracked "
                                   f"mesh {nc} vertices / {nfc} faces) -> kept per cell")
                else:
                    z = np.load(f, allow_pickle=True)
                    for kk in z.files:
                        if kk in ("vertices", "V", "points"):
                            store[kk] = XL[k]
                        elif kk in ("faces", "F") and np.asarray(z[kk]).ndim == 2:
                            store[kk] = F[~cut[k]]
                        elif not transfer(z[kk], j, store, kk):
                            store.setdefault("__copy__", {})[kk] = z[kk]
                return
            a = np.load(f, allow_pickle=True) if f.suffix == ".npy" else None
            if a is not None and a.ndim >= 2 and a.shape[1] == nc and a.shape[0] in (Tc, Tc - 1) and np.issubdtype(a.dtype, np.number):
                S = stacked.setdefault(str(rel), np.full((T, N) + a.shape[2:], np.nan))
                D = None
                if rel.name == "nonrigid_trajectories.npy":      # the deformation displacement of the cell (since its birth)
                    D = stacked.setdefault(str(rel.parent / "nonrigid_displacement.npy"), np.full((T, N) + a.shape[2:], np.nan))
                for r in range(a.shape[0]):
                    k = st + r
                    if 0 <= k < T:
                        P, vi = vmaps(r)
                        S[k, P] = a[r][vi]
                        if D is not None:
                            D[k, P] = (a[r] - a[0])[vi]
                return
            dst = out_root / "cells" / f"cell_{tid:02d}" / rel             # static per-cell results
            ensure_dir(dst.parent); shutil.copy2(f, dst)
            log.append(f"cell {tid} {rel}: kept per cell (no per-frame vertex / face axis: shape "
                       f"{getattr(a, 'shape', '?')}, tracked mesh {nc} vertices / {nfc} faces)")

        for f in sorted(sd.rglob("*")):
            if f.suffix == ".png":                                # the cell's own plots
                dst = out_root / "cells" / f"cell_{tid:02d}" / f.relative_to(sd)
                ensure_dir(dst.parent); shutil.copy2(f, dst); continue
            try:
                one_file(f)
            except Exception as exc:  # noqa: BLE001 - one file never stops the stage; reported in the log
                log.append(f"cell {tid} {f.relative_to(sd)}: FAILED ({type(exc).__name__}: {exc})")
    # write the per-frame files
    for key_file, store in per_frame.items():
        k = store.pop("k"); cells_only = store.pop("__cells__", None); copy = store.pop("__copy__", {})
        axis1 = store.pop("__axis1__", set())
        dst = out_root / key_file; ensure_dir(dst.parent)
        faces_vals = {kk: v for kk, v in store.items() if kk.endswith("__faces")}
        for kk in faces_vals:
            store.pop(kk)
        for kk, parts in faces_vals.items():                    # face values on ALL the faces (cut ones: no value)
            base = kk[:-len("__faces")]
            arr = np.full((len(F),) + parts[0][1].shape[1:], np.nan)
            pos_ = {tuple(r): i for i, r in enumerate(F)}
            for Fs, vals in parts:
                for r, v in zip(Fs, vals):
                    i = pos_.get(tuple(r))
                    if i is not None:
                        arr[i] = v
            store[base] = arr
        for kk in list(store):                                 # back to the cells' layout (axis 1)
            if kk in axis1 or kk + "__faces" in axis1:
                store[kk] = np.moveaxis(store[kk], 0, 1)
        if dst.suffix == ".npy":
            if "__array__" in store:
                np.save(dst, store["__array__"])
            elif cells_only:
                for tid, a in cells_only.items():
                    d2 = out_root / "cells" / f"cell_{tid:02d}" / key_file; ensure_dir(d2.parent); np.save(d2, a)
        else:
            np.savez_compressed(dst, **{**copy, **store})
    # surgeries: transitions no cell defines get an EMPTY (NaN) file, so every series stays aligned with the frames
    series = {}
    for key_file in per_frame:
        m = _FRAME_RE.search(Path(key_file).name)
        pat = str(Path(key_file).parent / (Path(key_file).name[:m.start(2)] + "{:04d}" + Path(key_file).name[m.end(2):]))
        series.setdefault(pat, []).append(int(m.group(2)))
    for pat, ks in series.items():
        ref = out_root / pat.format(ks[0])
        if not ref.exists():
            continue
        for k in range(min(ks), max(ks) + 1):
            dst = out_root / pat.format(k)
            if dst.exists():
                continue
            if ref.suffix == ".npy":
                a = np.load(ref, allow_pickle=True)
                np.save(dst, np.full(a.shape, np.nan) if np.issubdtype(a.dtype, np.floating) else np.full(a.shape, -1, a.dtype))
            else:
                z = np.load(ref, allow_pickle=True); out_ = {}
                for kk in z.files:
                    a = z[kk]
                    if kk in ("vertices", "V", "points"):
                        out_[kk] = XL[k] if 0 <= k < T else a
                    elif kk in ("faces", "F") and np.asarray(a).ndim == 2:
                        out_[kk] = F[~cut[min(max(k, 0), T - 1)]]
                    elif np.issubdtype(np.asarray(a).dtype, np.floating):
                        out_[kk] = np.full(a.shape, np.nan)
                    else:
                        out_[kk] = a
                np.savez_compressed(dst, **out_)
    # continuity of the display (hold=True): elements without a value — the surgery frames that no cell defines, the
    # first frame of material created at an event or a re-seeding — take the value of the SAME particle / face at the
    # nearest frame that has one (the values that exist are unchanged)
    if hold:
        _hold_series(out_root, series, log, scene_traj)
    for rel, S in stacked.items():
        if hold:
            S = _exist_fill(_time_hold(S), list(range(S.shape[0])), scene_traj)
            if Path(rel).name.startswith(_CUMULATIVE):       # cumulative along the lineage: no reset at the events
                S = _lineage_accumulate(S, scene_traj)
                log.append(f"scene {rel}: accumulated along the lineage (no reset at the divisions / fusions)")
        dst = out_root / rel; ensure_dir(dst.parent); np.save(dst, S)
    # tables: per scene frame (mean over the cells) + concatenated per cell (every table isolated: one that fails is
    # reported and the others are written)
    for rel, items in tables.items():
      try:
          dfs = []
          for tid, f in items:
              df = _read(f)
              if not len(df):
                  continue
              df = df.copy(); st = ctx["start"][tid]
              # times: some stages use the cell's LOCAL clock (0 at its first frame) -> scene times, by interpolating
              # the cell's local frame times onto its scene frame times
              scene_t = ctx["t"][st:ctx["end"][tid] + 1]
              lt_f = _track_dir(ctx, tid) / "Trajectories" / "time.npy"
              local_t = np.load(lt_f) if lt_f.exists() else np.arange(len(scene_t), dtype=float)
              if len(local_t) != len(scene_t):
                  local_t = np.arange(len(scene_t), dtype=float)
              tcols = [c for c in df.columns if (c.lower() == "time" or c.lower().endswith("_time"))
                       and pd.api.types.is_numeric_dtype(df[c])]
              if tcols and len(scene_t) > 1 and not np.allclose(local_t, scene_t):
                  for c in tcols:
                      df[c] = np.interp(df[c].to_numpy(float), local_t, scene_t)
              fc = next((c for c in df.columns if c.lower() in _FRAME_COLS), None)
              if fc is not None and pd.api.types.is_numeric_dtype(df[fc]):
                  df[fc] = df[fc].astype(int) + st
              vcols = [c for c in df.columns if c.lower() in ("extremum", "saddle", "vertex") and pd.api.types.is_integer_dtype(df[c])]
              trj_c = _track_dir(ctx, tid) / "Trajectories" / "trajectories.npy"
              if vcols and trj_c.exists() and ("time" in df or fc is not None):
                  # vertex numbers of the cell's tracked mesh -> the nearest scene particle of that cell at that frame
                  Xc = np.load(trj_c)
                  ks = (np.abs(df["time"].to_numpy(float)[:, None] - ctx["t"][None, :]).argmin(1) if "time" in df
                        else df[fc].to_numpy(int))
                  ks = np.clip(ks, st, ctx["end"][tid])            # within the cell's own lifetime
                  for c in vcols:
                      vals = df[c].to_numpy(np.int64); new = np.full(len(vals), -1, np.int64)
                      for k in np.unique(ks):
                          r = np.flatnonzero((ks == k) & (vals >= 0) & (vals < Xc.shape[1]))
                          P = np.flatnonzero(CELL[min(k, T - 1)] == tid)
                          if len(r) and len(P):
                              jj = int(np.clip(k - st, 0, Xc.shape[0] - 1))
                              new[r] = P[cKDTree(XL[k, P]).query(Xc[jj][vals[r]])[1]]
                      df[c] = new
              for c in df.columns:                                 # ids unique per cell
                  if c.lower() in ("track", "track_id", "curr_track_id", "region_id") and pd.api.types.is_integer_dtype(df[c]):
                      df[c] = df[c] + tid * 100000
                  elif c.lower() in ("branch", "other") and df[c].dtype == object:
                      df[c] = f"c{tid}_" + df[c].astype(str)
              df.insert(0, "cell", tid); dfs.append(df)
          if not dfs:
              continue
          cat = pd.concat(dfs, ignore_index=True)
          dst = out_root / rel; ensure_dir(dst.parent)
          cat.to_csv(dst.parent / f"cells_{dst.name}", index=False)
          agg, how = _aggregate_cells(cat, Path(rel).name, _cell_areas(ctx), ctx["start"])
          (agg if agg is not None else cat).to_csv(dst, index=False)
          log.append(f"table {rel}: {how}")
      except Exception as exc:  # noqa: BLE001
        log.append(f"table {rel}: FAILED ({type(exc).__name__}: {exc})")

    # report: every per-frame series with its finite share (empty frames = surgeries)
    for key_file in sorted(per_frame):
        f = out_root / key_file
        if f.exists():
            try:
                if f.suffix == ".npy":
                    a = np.load(f, allow_pickle=True)
                    fin = float(np.isfinite(a).mean()) if np.issubdtype(a.dtype, np.floating) else 1.0
                else:
                    z = np.load(f, allow_pickle=True)
                    fl = [z[k] for k in z.files if np.issubdtype(np.asarray(z[k]).dtype, np.floating)]
                    fin = float(np.mean([np.isfinite(v).mean() for v in fl])) if fl else 1.0
                log.append(f"scene {key_file}: finite share {fin:.3f}")
            except Exception as exc:  # noqa: BLE001
                log.append(f"scene {key_file}: unreadable ({exc})")
    with open(out_root / "transfer_log.txt", "w", encoding="utf-8") as fh:
        fh.write(f"{stage}: cells' results transferred onto the scene particles ({N} particles, {T} frames)\n")
        fh.write("\n".join(log) + "\n")
    try:
        _plot_stage_tables(ctx, out_root, stage)
    except Exception as exc:  # noqa: BLE001
        print(f"[Scene {stage}] plots: {exc}")
    return {"per_frame_files": len(per_frame), "stacked": list(stacked), "tables": len(tables),
            "failed": sum("FAILED" in l for l in log)}


def _plot_stage_tables(ctx, out_root, stage):
    """
    Scene plots of a transferred stage: for every table of the cells (cells_<name>.csv) its main quantities over the
    scene time, divisions / fusions marked. ONE CURVE PER TRACKED OBJECT: the rows of a table are grouped by the cell
    and its key columns (region, branch, animation, mode, signal, track id ...), every group sorted in time and broken
    where the object is missing (never a line across a gap or between different objects); identifier / index columns
    (keys, *_node, *_id, step) are never plotted as quantities. Curves are coloured by cell; tables keyed only by frame
    also show the scene aggregate (dashed, with its rule).
    """
    from matplotlib.lines import Line2D
    d = ensure_dir(out_root / "plots")
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    for f in sorted(out_root.rglob("cells_*.csv")):
        df = _read(f)
        fc = next((c for c in df.columns if c.lower() in _FRAME_COLS), None)
        if not len(df) or (fc is None and "time" not in df):
            continue
        if fc is not None:
            df = df[(df[fc] >= 0) & (df[fc] < ctx["T"])]
            xcol = lambda g: ctx["t"][g[fc].to_numpy(int)]                  # noqa: E731
        else:                                                                # sub-frame samples: scene time axis
            fc = "time"; xcol = lambda g: g["time"].to_numpy(float)          # noqa: E731
        keys = [c for c in df.columns if c not in ("cell", fc, "time", "frame_start", "frame_end") and _is_key_column(df, c)]
        excluded = set(keys) | {"cell", fc, "time"} | {c for c in df.columns if c.lower() in _INDEX_COLUMNS}
        num = [c for c in df.columns if c not in excluded and not c.lower().endswith(("_node", "_id", "node"))
               and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 2 and np.nanstd(df[c].to_numpy(float)) > 0]
        num = sorted(num, key=lambda c: -np.nanstd(df[c].to_numpy(float)) / (abs(np.nanmean(df[c].to_numpy(float))) + 1e-12))[:6]
        if not num:
            continue
        groups = list(df.groupby(["cell"] + keys, sort=True)) if keys else list(df.groupby(["cell"], sort=True))
        many = len(groups) > 12
        combos = sorted({tuple(k[1:]) for k, _ in groups}) if keys else []
        styles = {cb: ls for cb, ls in zip(combos, ["-", "--", ":", "-."])} if 1 < len(combos) <= 4 else {}
        fig, axs = plt.subplots(len(num), 1, figsize=(12, 2.2 * len(num) + 0.6), sharex=True, squeeze=False)
        cells_seen = {}
        for a_, c in zip(axs[:, 0], num):
            for key, g in groups:
                tid = int(key[0] if isinstance(key, tuple) else key)
                x = xcol(g); y = g[c].to_numpy(float); o = np.argsort(x, kind="stable"); x, y = x[o], y[o]
                if len(x) > 2:                                   # break the curve where the object is missing
                    dx = np.diff(x); step = np.median(dx[dx > 0]) if (dx > 0).any() else 1.0
                    cut_at = np.flatnonzero(dx > 1.5 * step) + 1
                    x = np.insert(x.astype(float), cut_at, np.nan); y = np.insert(y, cut_at, np.nan)
                col = colors[tid % len(colors)]; cells_seen[tid] = col
                ls = styles.get(tuple(key[1:]), "-") if isinstance(key, tuple) else "-"
                a_.plot(x, y, ls, marker=None if many else "o", ms=2.0, lw=0.7 if many else 1.0, alpha=0.55 if many else 0.9,
                        color=col)
            _event_lines(a_, ctx); a_.set_ylabel(c, fontsize=7); a_.grid(alpha=0.3)
        name = f.stem[len("cells_"):]
        agg_f = f.parent / f"{name}.csv"                      # the scene aggregate (only for tables that combine)
        agg = _read(agg_f) if agg_f.exists() else pd.DataFrame()
        if keys or not ("n_cells" in agg.columns and fc in agg.columns):
            agg = None
        handles = [Line2D([0], [0], color=col, lw=1.5, label=f"cell {tid}") for tid, col in sorted(cells_seen.items())]
        for a_, c in zip(axs[:, 0], num):
            if agg is not None and c in agg.columns:
                kk = agg[fc].to_numpy(int); xa = ctx["t"][kk].astype(float); ya = agg[c].to_numpy(float)
                # broken where frames are missing AND at every division / fusion: the set of cells changes there, and
                # quantities measured from a cell's own first frame (strain, anisotropy, stretch) restart
                ev_k = set(int(r.frame) for r in ctx["tr"].itertuples()
                           if r.event in ("division", "fusion")) if len(ctx["tr"]) else set()
                gap = np.flatnonzero((np.diff(kk) > 1) | np.isin(kk[1:], list(ev_k))) + 1
                a_.plot(np.insert(xa, gap, np.nan), np.insert(ya, gap, np.nan), "k--", lw=1.4)
                rule = {"wmean": "area-weighted mean", "sum": "sum", "max": "max", "min": "min"}[_column_rule(c)]
                a_.set_ylabel(f"{c}\n(scene: {rule})", fontsize=7)
        if agg is not None:
            handles.append(Line2D([0], [0], color="k", ls="--", lw=1.4, label="scene (all cells)"))
        for cb, ls in styles.items():
            handles.append(Line2D([0], [0], color="gray", ls=ls, lw=1.2,
                                  label=", ".join(f"{k}={v}" for k, v in zip(keys, cb))))
        axs[0, 0].legend(handles=handles, fontsize=7, ncol=6)
        what = f"one curve per cell and {', '.join(keys)}" if keys else "one curve per cell"
        axs[0, 0].set_title(f"{stage}: {name} on the scene timeline ({what})")
        axs[-1, 0].set_xlabel("time")
        fig.tight_layout(); fig.savefig(d / f"scene_{name}.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------------------------------- #
#  parametrization of the scene: the cells' sphere maps and SPHARM reconstructions on the scene frames
# --------------------------------------------------------------------------------------------------------------- #
def _read_obj(path):
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.asarray(V, float), np.asarray(F, np.int64)


def _write_obj(path, V, F):
    with open(path, "w") as fh:
        fh.write("".join(f"v {x:.7g} {y:.7g} {z:.7g}\n" for x, y, z in V))
        fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F))


def write_parametrization(ctx):
    """
    Parametrization of the whole scene in the usual files: Spherical/sphere_T####.npy and theta_phi_T####.npy hold, on
    the vertices of the scene frame, the harmonic sphere map of every cell PLACED at its cell (the viewer draws a map at
    c0 + R0 U with c0, R0 the centre and mean radius of the frame, so U = (c_cell + R_cell U_cell - c0) / R0 shows every
    cell's sphere at its own position and size); SPHARM/reconstruction_T####_L##.obj join the cells' reconstructions.
    """
    frames, T = ctx["frames"], ctx["T"]
    out = ctx["target"] / "Parametrization"; ds = ensure_dir(out / "Spherical"); dh = ensure_dir(out / "SPHARM")
    written = 0
    for k in pbar(range(T), desc="Scene parametrization"):
        V = frames[k]["V"]; c0 = V.mean(axis=0); R0 = float(np.linalg.norm(V - c0, axis=1).mean()) or 1.0
        U = (V - c0) / R0; tp = np.zeros((len(V), 2)); any_ = False
        recon = {}
        for tid in _tracks_at(ctx, k):
            j = k - ctx["start"][tid]; td = _track_dir(ctx, tid) / "Parametrization"
            fs, ft = td / "Spherical" / f"sphere_T{j:04d}.npy", td / "Spherical" / f"theta_phi_T{j:04d}.npy"
            kp = ctx["keeps"][(k, tid)]
            if fs.exists():
                Uc = np.load(fs)
                if len(Uc) == len(kp):
                    Vc = V[kp]; cc = Vc.mean(axis=0); Rc = float(np.linalg.norm(Vc - cc, axis=1).mean())
                    U[kp] = (cc + Rc * Uc - c0) / R0; any_ = True
                    if ft.exists():
                        tc = np.load(ft)
                        if len(tc) == len(kp):
                            tp[kp] = tc
            for f in (td / "SPHARM").glob(f"reconstruction_T{j:04d}_L*.obj"):
                if f.stem.endswith("_mesh"):
                    continue
                L = f.stem.split("_L")[-1]
                recon.setdefault(L, []).append(_read_obj(f))
        if any_:
            np.save(ds / f"sphere_T{k:04d}.npy", U); np.save(ds / f"theta_phi_T{k:04d}.npy", tp); written += 1
        for L, parts in recon.items():
            Vs, Fs, off = [], [], 0
            for Vr, Fr in parts:
                Vs.append(Vr); Fs.append(Fr + off); off += len(Vr)
            _write_obj(dh / f"reconstruction_T{k:04d}_L{L}.obj", np.concatenate(Vs), np.concatenate(Fs))
    return written


# --------------------------------------------------------------------------------------------------------------- #
#  single-surface stages: the tables of the tracks on the scene timeline
# --------------------------------------------------------------------------------------------------------------- #
def write_tables(ctx, stage):
    """Per-frame tables of every track of a single-surface stage (Parametrization, ReebDynamics) on the scene timeline
    (scene_<table>.csv, with 'track', scene 'frame' and 'time') and a plot of their main columns."""
    out = ctx["target"] / stage; written = []
    names = set()
    for tid in ctx["start"]:
        sd = _track_dir(ctx, tid) / stage
        if sd.is_dir():
            names |= {p.name for p in sd.glob("*.csv")}
    for name in sorted(names):
        dfs = []
        for tid in ctx["start"]:
            df = _read(_track_dir(ctx, tid) / stage / name)
            if not len(df):
                continue
            n_fr = ctx["end"][tid] - ctx["start"][tid] + 1
            fc = next((c for c in df.columns if c.lower() in ("frame", "timestep", "t_index")), None)
            if fc is not None and pd.api.types.is_numeric_dtype(df[fc]):
                fr = df[fc].astype(int).to_numpy()
            elif "time" in df and pd.api.types.is_numeric_dtype(df["time"]):
                # sub-frame samples (Reeb dynamics): the tracks' times are scene times; frame = the nearest frame
                fr = np.abs(df["time"].to_numpy(float)[:, None] - ctx["t"][None, :]).argmin(1) - ctx["start"][tid]
            elif len(df) in (n_fr, n_fr - 1):
                fr = np.arange(len(df)) + (n_fr - len(df))
            else:
                continue                                       # not a per-frame table
            df = df.copy(); df["frame"] = fr + ctx["start"][tid]
            df = df[(df.frame >= 0) & (df.frame < ctx["T"])]
            df.insert(0, "cell", tid)
            if "time" not in df:
                df["time"] = ctx["t"][df.frame.to_numpy(int)]
            dfs.append(df)
        if not dfs:
            continue
        sc = pd.concat(dfs, ignore_index=True)
        ensure_dir(out); sc.to_csv(out / f"scene_{name}", index=False); written.append(name)
        num = [c for c in sc.columns if c not in ("cell", "track", "frame", "time") and pd.api.types.is_numeric_dtype(sc[c])
               and sc[c].notna().sum() > 2 and np.nanstd(sc[c].to_numpy(float)) > 0]
        num = sorted(num, key=lambda c: -np.nanstd(sc[c].to_numpy(float)) / (abs(np.nanmean(sc[c].to_numpy(float))) + 1e-12))[:6]
        if not num:
            continue
        fig, axs = plt.subplots(len(num), 1, figsize=(12, 2.2 * len(num) + 0.6), sharex=True, squeeze=False)
        for a, c in zip(axs[:, 0], num):
            for tid, g in sc.groupby("cell"):
                a.plot(g.time, g[c], "-o", ms=2.5, lw=1, label=f"cell {tid}")
            _event_lines(a, ctx); a.set_ylabel(c, fontsize=7); a.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7, ncol=6); axs[0, 0].set_title(f"{stage}: {name} of every cell on the scene timeline")
        axs[-1, 0].set_xlabel("time")
        ensure_dir(out / "plots")
        fig.tight_layout(); fig.savefig(out / "plots" / f"scene_{Path(name).stem}.png", dpi=200); plt.close(fig)
    return written


# --------------------------------------------------------------------------------------------------------------- #
#  topology plots: shape biomechanics and pinch singularity (TopologyEvents/plots)
# --------------------------------------------------------------------------------------------------------------- #
def topology_plots(ctx):
    te = ctx["te"]; d = ensure_dir(te / "plots")
    surf, necks = _read(te / "surfaces.csv"), _read(te / "necks.csv")
    if len(surf):
        tl = surf.merge(necks[["frame", "track", "neck_ratio", "conformal_neck_index"]], on=["frame", "track"], how="left") if len(necks) else surf
        keys = [k for k in ("reduced_volume", "bending_energy", "fiedler_norm", "solidity", "neck_ratio", "conformal_neck_index") if k in tl]
        titles = {"reduced_volume": "reduced volume (1 sphere, 0.71 two touching spheres)",
                  "bending_energy": "Helfrich bending energy / 16 pi (1 sphere, 2 two spheres)",
                  "fiedler_norm": "Fiedler value of the cotangent Laplacian (1 sphere, -> 0 at a bottleneck)",
                  "solidity": "solidity V / V(convex hull)", "neck_ratio": "neck ratio",
                  "conformal_neck_index": "conformal neck index (surface Ricci flow)"}
        fig, axs = plt.subplots(len(keys), 1, figsize=(12, 2.3 * len(keys) + 0.6), sharex=True, squeeze=False)
        for a, key in zip(axs[:, 0], keys):
            for tid, g in tl.groupby("track"):
                a.plot(g.time, g[key], "o-", ms=2.5, label=f"cell {tid}")
            _event_lines(a, ctx); a.set_title(titles[key], fontsize=9); a.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7, ncol=6); axs[-1, 0].set_xlabel("time")
        fig.tight_layout(); fig.savefig(d / "5_shape_biomechanics.png", dpi=200); plt.close(fig)
    pdf = _read(te / "pinch_singularity.csv")
    if len(pdf) and len(necks):
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in pdf.itertuples():
            g = necks[(necks.track == r.mother_track) & (necks.time < r.observed_split_time) & (necks.neck_ratio < 0.95)].tail(10)
            x = r.predicted_pinch_time - g.time.to_numpy(float)
            ax.loglog(x, g.neck_radius, "o", label=f"division {r.division}")
            xx = np.geomspace(max(x.min(), 1e-6), x.max(), 50)
            ax.loglog(xx, r.prefactor * xx ** r.alpha, "-", label=f"alpha = {r.alpha:.2f} ({r.closest_regime})")
        ax.set_xlabel("time to the pinch T - t"); ax.set_ylabel("neck radius"); ax.grid(alpha=0.3, which="both")
        ax.set_title("neck-pinch singularity: r = c (T - t)^alpha"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(d / "6_pinch_singularity.png", dpi=200); plt.close(fig)
    mat = _read(te / "Lineage_tracking" / "material_summary.csv")
    if len(mat):
        fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
        ax[0].plot(mat.time, mat.new_material_area_fraction, "o-", ms=3, color="tab:red"); ax[0].set_ylabel("new material")
        ax[1].plot(mat.time, mat.material_coherence, "o-", ms=3); ax[1].set_ylabel("material coherence (1 = ideal)")
        ax[0].set_title("material of the scene: area created since the first frame, coherence of the tracking")
        for a in ax:
            _event_lines(a, ctx); a.grid(alpha=0.3)
        ax[1].set_xlabel("time")
        fig.tight_layout(); fig.savefig(d / "7_material.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------------------------------------------- #
@logged_stage("TopologyEvents", (0, "target_folder"), "Scene assembly")
def assemble_scene(target_folder, mesh_folder, loader=None, params=None, maps="auto", verbose=True):
    """
    Builds the standard results of the whole scene in Results/<scene>/ from the lineage tracks (module doc).
    params: the pipeline parameters of the run (decide which stages were computed for the tracks and are assembled /
    computed for the scene, and give their parameters).
    """
    from PynamicMesh.utils.tools import load_aligned_mesh
    loader = loader or load_aligned_mesh
    p = dict(params or {})
    ctx = _context(target_folder, mesh_folder, loader)
    tp_opts = dict(p.get("topology_params") or {})
    ctx["opts"] = {k: tp_opts[k] for k in PARTICLE_OPTIONS + TRANSFER_OPTIONS if k in tp_opts}
    steps = ["correspondences", "functional maps", "diagonal analysis", "physical fields", "graphs", "Morse-Smale",
             "material and trajectories", "graph animation", "dynamic analysis", "motion analysis", "reeb dynamics",
             "tables and plots"]
    progress(total=len(steps), advance=0)
    done, status = {}, {}

    def run(name, fn):
        try:
            done[name] = fn(); status[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - one failing scene stage never stops the others
            status[name] = f"failed: {exc}"; print(f"\n[Warning] scene {name}: {exc}")
        progress(name)

    run("correspondences", lambda: scene_maps(ctx, maps))
    M, src = done.get("correspondences", ({}, pd.DataFrame()))
    run("functional maps", lambda: write_transform(ctx, M, src))
    fm = done.get("functional maps")
    if fm:
        try:
            write_displacements(ctx, fm[0])
        except Exception as exc:  # noqa: BLE001
            print(f"[Warning] scene displacements: {exc}")
    if p.get("diagonal_analysis", True) and fm:
        run("diagonal analysis", lambda: write_diagonal(ctx, fm[1]))
    else:
        progress("diagonal analysis")
    if p.get("compute_physic_fields", True) and fm:
        run("physical fields", lambda: write_physics(ctx, M))
    else:
        progress("physical fields")
    graphs = {}
    run("graphs", lambda: write_reeb(ctx)); graphs["reeb"] = done.get("graphs") or {}
    run("Morse-Smale", lambda: write_mscomplex(ctx)); graphs["mscomplex"] = done.get("Morse-Smale") or {}
    try:
        write_center_lineage(ctx)
    except Exception as exc:  # noqa: BLE001
        print(f"[Warning] centre lineage: {exc}")
    try:
        graph_surgery(ctx, M, graphs)
        write_graph_analysis(ctx, p)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[Warning] scene graph analysis: {exc}")
    tp_ = dict(p.get("traj_params") or {})
    run("material and trajectories", lambda: (material_tracking(ctx, M),
                                              write_trajectories(ctx, M, schemes=tp_.get("schemes", ("hermite_catmull_rom",)),
                                                                 substeps=int(tp_.get("substeps", 4))))[1])
    traj = done.get("material and trajectories")
    frames = ctx["frames"]
    if traj is not None and p.get("compute_graph_animation", False):
        from PynamicMesh.core.graph_animation import compute_graph_animation
        if (ctx["target"] / "GraphAnimation").is_dir():         # the scene's copy: no stale models of earlier runs
            shutil.rmtree(ctx["target"] / "GraphAnimation")
        run("graph animation", lambda: compute_graph_animation(ctx["target"], frames=frames, verbose=False,
                                                               **dict(p.get("graph_animation_params") or {})))
    else:
        progress("graph animation")
    # Lagrangian analyses (fixed connectivity): the cells' results transferred onto the scene particles
    scene_traj = None
    if traj is not None:
        try:
            tdir = ctx["target"] / "Trajectories"
            scene_traj = (np.load(tdir / "trajectories_lifecycle.npy"), np.load(tdir / "material_cell.npy"),
                          np.load(tdir / "reference_faces.npy"), np.load(tdir / "cut_faces.npy"))
        except Exception:  # noqa: BLE001
            scene_traj = None
    for flag, stages, label in (("compute_dynamic_analysis", ("DynamicAnalysis",), "dynamic analysis"),
                                ("compute_motion_analysis", ("MotionAnalysis", "Physical_fields/from_trajectories"),
                                 "motion analysis"),
                                ("compute_reeb_dynamics", ("ReebDynamics",), "reeb dynamics")):
        if scene_traj is not None and p.get(flag, False):
            run(label, lambda sts=stages: [transfer_stage(ctx, st_, scene_traj, hold=bool(ctx["opts"].get("hold", True)))
                                           for st_ in sts])
        else:
            progress(label)

    def tables():
        if p.get("compute_parametrization", False):
            write_parametrization(ctx)
        out = {s: write_tables(ctx, s) for s in ("Parametrization",)}
        topology_plots(ctx)
        return out
    run("tables and plots", tables)
    summary = {"frames": ctx["T"], "tracks": int(len(ctx["lin"])),
               "events": ctx["tr"].event.value_counts().to_dict() if len(ctx["tr"]) else {},
               "mapped_vertex_fraction": float(np.mean([np.mean(m >= 0) for m in M.values()])) if M else np.nan,
               "stages": status}
    with open(ctx["te"] / "scene_assembly.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if verbose:
        print(f"Scene assembly: {sum(v == 'ok' for v in status.values())}/{len(status)} scene stages written in {ctx['target']}")
    return summary
