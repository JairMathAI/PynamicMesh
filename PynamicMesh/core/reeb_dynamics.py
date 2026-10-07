"""
reeb_dynamics.py  —  time-varying (Lagrangian) Reeb graph: branches, critical-point paths and events
==================================================================================================

Edelsbrunner, Harer, Mascarenhas, Pascucci & Snoeyink (2008) connect the Reeb graphs of a time-varying function through
the Jacobi curve — the paths traced by the critical points in space-time — and show that the graph changes only at
  * birth–death events: two critical points with indices differing by one appear / annihilate together (index lemma);
    on a surface: maximum + saddle (a BRANCH sprouts / retracts: a protrusion) or minimum + saddle (a dent);
  * interchange events: two critical points of the same level-set component swap their function values (the order
    of branching along the graph changes).

The tracked mesh of the pipeline (Trajectories: one triangulation followed through the sequence) makes this exact
and cheap: the scalar field of the Reeb graphs is evaluated on the tracked mesh — with the SAME material source point
in every frame — and interpolated linearly in time (a piecewise-linear space-time function, the setting of the paper).
With a fixed triangulation the critical vertices depend only on the vertex values, so `substeps` samples per interval
resolve events between the observed frames. At every sample the branches are the persistence pairs of the field
(union-find: a maximum and the saddle where its branch joins the rest of the graph; same for minima): on a genus-0
surface these are exactly the branches of the Reeb graph. Branches are followed through time by the material
position of their extremum (Hungarian matching) — the discrete Jacobi curve — with hysteresis (born above the
persistence threshold, kept down to half of it) so that noise does not create flickering events.

Outputs (Results/<scene>/ReebDynamics/): samples.csv, paths.csv (critical-point paths), tracks.csv (one row per branch:
kind, birth / death time, lifetime, persistence), events.csv (birth, death, interchange, dominance change), summary.json,
plots/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PynamicMesh.core.dyn_common import logged_stage, progress, pbar, ensure_dir, find_frame_times, face_normals_areas

REEB_DYNAMICS_DEFAULTS = {"persistence": 0.08, "substeps": 4, "match_radius": 0.12, "k_spectral": 60, "plots": True,
                          "ms_regions": True, "contact_threshold": 0.02}


# --------------------------------------------------------------------------------------------------------------- #
#  fields on the tracked mesh
# --------------------------------------------------------------------------------------------------------------- #
def fields_on_tracked_mesh(X, F0, method, kwargs0, spectral=False, k_spectral=60):
    """The Reeb scalar field of every frame evaluated on the tracked mesh (same material source point)."""
    from PynamicMesh.core.reeb_graph import get_scalar_field
    T, n, _ = X.shape
    out = np.zeros((T, n))
    ident = np.arange(n)
    for k in pbar(range(T), desc="Reeb field on the tracked mesh"):
        tm = None
        if spectral:
            from pyFM.mesh import TriMesh
            tm = TriMesh(np.asarray(X[k], float), np.asarray(F0, int)); tm.process(k=int(k_spectral), verbose=False)
        out[k] = get_scalar_field(X[k], F0, method=method, prev_vertices=X[k - 1] if k else None,
                                  p2p=ident if k else None, trimesh_obj=tm, **dict(kwargs0 or {}))
    return out


# --------------------------------------------------------------------------------------------------------------- #
#  persistence pairs (0-dimensional, union-find)
# --------------------------------------------------------------------------------------------------------------- #
def _edges(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def branch_pairs(f, nbr_ptr, nbr_idx, tau, regions=False):
    """
    Persistence pairs of the minima of f (sweep upwards, union-find, elder rule): every minimum m whose component
    merges into an older one at the saddle s gives (m, s, f(s) − f(m), elder minimum). Pairs with persistence ≥ tau are
    returned, plus the global minimum as an essential pair (saddle −1). Ties broken by vertex index (simulation of
    simplicity). For the maxima call it with −f.
    regions=True also returns, for every vertex, the extremum of the component it joined (its region before any
    simplification: for −f these are the Morse–Smale cells of the maxima) and, for every merge, the extremum that
    absorbed the younger one: (pairs, ext_of_vertex, merged_into).
    """
    n = len(f)
    order = np.lexsort((np.arange(n), f))
    rank = np.empty(n, dtype=np.int64); rank[order] = np.arange(n)
    parent = np.full(n, -1, dtype=np.int64)
    birth = np.full(n, -1, dtype=np.int64)            # extremum (vertex) of the component a root represents

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r
    pairs = []
    ext_of = np.full(n, -1, dtype=np.int64) if regions else None
    merged_into = {}
    for v in order:
        nb = nbr_idx[nbr_ptr[v]:nbr_ptr[v + 1]]
        nb = nb[rank[nb] < rank[v]]
        parent[v] = v
        if not len(nb):
            birth[v] = v
            if regions:
                ext_of[v] = v
            continue
        roots = {find(w) for w in nb}
        if len(roots) == 1:
            r = roots.pop(); parent[v] = r
            if regions:
                ext_of[v] = birth[r]
            continue
        roots = sorted(roots, key=lambda r: rank[birth[r]])   # elder = lowest extremum
        elder = roots[0]
        for r in roots[1:]:
            p = f[v] - f[birth[r]]
            if p >= tau:
                pairs.append((int(birth[r]), int(v), float(p), int(birth[elder])))
            if regions:
                merged_into[int(birth[r])] = int(birth[elder])
            parent[r] = elder
        parent[v] = elder
        if regions:
            ext_of[v] = birth[elder]
    gmin = int(order[0])
    pairs.append((gmin, -1, float(f.max() - f.min()), -1))
    if regions:
        return pairs, ext_of.astype(np.int32), merged_into
    return pairs


def steepest_ascent_maxima(f, nbr_ptr, nbr_idx):
    """
    Morse–Smale (ascending) cells of the maxima: every vertex follows its steepest-ascent neighbour (the highest
    neighbour, if higher than itself) up to a local maximum; returns that maximum for every vertex. Ties are broken
    by the vertex index, consistently with the sweep of branch_pairs (simulation of simplicity). Vectorised: one
    pointer per vertex, then pointer jumping (about log2(n) rounds).
    """
    n = len(f)
    order = np.lexsort((np.arange(n), -f))                     # same total order as the sweep of -f
    rank = np.empty(n, dtype=np.int64); rank[order] = np.arange(n)   # smaller rank = higher
    src = np.repeat(np.arange(n), np.diff(nbr_ptr))
    r_nb = rank[nbr_idx]
    # highest neighbour of every vertex: minimum rank per source (sort by (src, rank), first of each block)
    o = np.lexsort((r_nb, src))
    first = np.r_[0, np.flatnonzero(np.diff(src[o])) + 1]
    best_src = src[o][first]; best_nb = nbr_idx[o][first]
    nxt = np.arange(n)
    up = rank[best_nb] < rank[best_src]
    nxt[best_src[up]] = best_nb[up]
    for _ in range(64):                                         # pointer jumping to the local maximum
        nn = nxt[nxt]
        if np.array_equal(nn, nxt):
            break
        nxt = nn
    return nxt.astype(np.int32)


def _sample_task(args):
    """Branches of one time sample (worker-friendly): maxima from −f, minima from f."""
    f, nbr_ptr, nbr_idx, tau_low = args[:4]
    with_regions = bool(args[4]) if len(args) > 4 else False
    fr = float(np.ptp(f)) or 1.0
    if with_regions:
        pmx, _, merged = branch_pairs(-f, nbr_ptr, nbr_idx, tau_low * fr, regions=True)
        ext_of = steepest_ascent_maxima(f, nbr_ptr, nbr_idx)          # Morse–Smale cells of the maxima
    else:
        pmx, ext_of, merged = branch_pairs(-f, nbr_ptr, nbr_idx, tau_low * fr), None, None
    mx = [(e, s, p / fr, par) for e, s, p, par in pmx]
    mn = [(e, s, p / fr, par) for e, s, p, par in branch_pairs(f, nbr_ptr, nbr_idx, tau_low * fr)]
    return mx, mn, ext_of, merged


# --------------------------------------------------------------------------------------------------------------- #
#  tracking (discrete Jacobi curve) and events
# --------------------------------------------------------------------------------------------------------------- #
def _track(samples, P0, size, tau, radius, kind):
    """Follows the branches of one kind through the samples; returns rows of the paths and the events."""
    from scipy.optimize import linear_sum_assignment
    next_id = [0]
    active = {}                                         # track id -> (extremum vertex, saddle, persistence, parent ext)
    rows, events = [], []
    names = {"max": ("protrusion born (maximum + saddle)", "protrusion retracted (maximum + saddle annihilated)"),
             "min": ("dent born (minimum + saddle)", "dent vanished (minimum + saddle annihilated)")}[kind]
    prev_dominant = None
    prev_saddles = {}
    confirmed = {}                                      # (a, b) -> +1 / -1: order of two junctions, confirmed by a margin
    for si, (t, items) in enumerate(samples):
        ids = list(active.keys())
        cand = list(range(len(items)))
        assign = {}
        if ids and cand:
            A = P0[[active[i][0] for i in ids]]; B = P0[[items[j][0] for j in cand]]
            # identity = the path of the extremum (Jacobi curve), whatever its role: the essential trunk can pass from
            # one extremum to another (dominance change) without any branch being born or retracting.
            # Stage 1: match by extremum position (keeps the identity of distinct branches).
            C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2) / max(size, 1e-300)
            r_, c_ = linear_sum_assignment(C)
            for a, b in zip(r_, c_):
                if C[a, b] <= radius:
                    assign[ids[a]] = cand[b]
            # Stage 2: a branch is the PAIR (extremum, junction saddle). When two nearly equal maxima of one branch
            # alternate, its extremum jumps but its junction stays: the branches left unmatched are matched by their
            # junction (same branch, no birth / death).
            ra = [k for k, i_ in enumerate(ids) if i_ not in assign and active[i_][1] >= 0]
            rb = [k for k, j_ in enumerate(cand) if j_ not in assign.values() and items[j_][1] >= 0]
            if ra and rb:
                SA = P0[[active[ids[k]][1] for k in ra]]; SB = P0[[items[cand[k]][1] for k in rb]]
                Cs = np.linalg.norm(SA[:, None, :] - SB[None, :, :], axis=2) / max(size, 1e-300)
                r2, c2 = linear_sum_assignment(Cs)
                for a, b in zip(r2, c2):
                    if Cs[a, b] <= radius:
                        assign[ids[ra[a]]] = cand[rb[b]]
        used = set(assign.values())
        new_active = {}
        for tid, j in assign.items():
            new_active[tid] = items[j]
        for tid in ids:
            if tid not in assign:
                if active[tid][1] >= 0:
                    events.append({"time": t, "sample": si, "event": names[1], "kind": kind, "track": tid, "other": ""})
        for j in cand:
            if j in used:
                continue
            e, s, p, par = items[j]
            if s >= 0 and p < tau:
                continue                                # hysteresis: a new branch must exceed the threshold
            tid = next_id[0]; next_id[0] += 1
            new_active[tid] = items[j]
            if si > 0 and s >= 0:
                events.append({"time": t, "sample": si, "event": names[0], "kind": kind, "track": tid, "other": ""})
        active = new_active
        ext_to_tid = {v[0]: k for k, v in active.items()}
        for tid, (e, s, p, par) in active.items():
            rows.append({"sample": si, "time": t, "kind": kind, "track": tid, "extremum": e, "saddle": s,
                         "persistence": p, "essential": s < 0, "parent_track": ext_to_tid.get(par, -1)})
        # interchange events: two related junctions (siblings on the same parent, or a branch and its parent)
        # swap their order along the graph between consecutive samples
        cur = {tid: (s, par) for tid, (e, s, p, par) in active.items() if s >= 0}
        margin = 0.25 * tau * float(np.ptp(_FIELD["f"][si]))  # hysteresis: a swap must be clear, not noise
        for a in cur:
            for b in cur:
                if a >= b:
                    continue
                pa, pb = ext_to_tid.get(cur[a][1], -1), ext_to_tid.get(cur[b][1], -1)
                related = (pa == pb and pa >= 0) or pa == b or pb == a
                if not related:
                    continue
                now = samples_value(cur[a][0], si) - samples_value(cur[b][0], si)
                if abs(now) <= margin:
                    continue
                sign = 1 if now > 0 else -1
                old_sign = confirmed.get((a, b))
                confirmed[(a, b)] = sign
                if old_sign is not None and old_sign != sign:
                    what = ("interchange: sibling branches swap their branching order" if pa == pb
                            else "interchange: a branch junction passes the junction of its parent branch")
                    events.append({"time": t, "sample": si, "event": what, "kind": kind, "track": a, "other": b})
        prev_saddles = {tid: (samples_value(s, si), par) for tid, (s, par) in cur.items()}
        dom = [tid for tid, (e, s, p, par) in active.items() if s < 0]
        if dom and prev_dominant is not None and dom[0] != prev_dominant:
            events.append({"time": t, "sample": si, "event": f"dominant {'protrusion' if kind == 'max' else 'dent'} changed",
                           "kind": kind, "track": dom[0], "other": prev_dominant})
        prev_dominant = dom[0] if dom else prev_dominant
    return rows, events


_FIELD = {"f": None}                                    # field of the current samples (for the interchange test)


def samples_value(vertex, si):
    return float(_FIELD["f"][si][vertex]) if vertex >= 0 else np.nan


# --------------------------------------------------------------------------------------------------------------- #
#  stage driver
# --------------------------------------------------------------------------------------------------------------- #
@logged_stage("ReebDynamics", (0, "target_folder"), "ReebDynamics")
def compute_reeb_dynamics(target_folder, reeb_scalar="geodesic", scalar_kwargs=None, selections=None, X=None, F0=None,
                          times=None, spectral=False, verbose=True, **params):
    """Stage driver (see the module docstring); `params` override REEB_DYNAMICS_DEFAULTS."""
    p = {**REEB_DYNAMICS_DEFAULTS, **params}
    target_folder = Path(target_folder); out = ensure_dir(target_folder / "ReebDynamics"); ensure_dir(out / "plots")
    if X is None:
        xf = target_folder / "Trajectories" / "trajectories.npy"
        if not xf.exists():
            print("Reeb dynamics skipped: no trajectories (compute_trajectories)."); return None
        X = np.load(xf); F0 = np.load(target_folder / "Trajectories" / "reference_faces.npy")
    X = np.asarray(X, float); F0 = np.asarray(F0, int); T, n, _ = X.shape
    tfile = target_folder / "Trajectories" / "time.npy"
    t = np.asarray(getattr(times, "t", times), float) if times is not None else (np.load(tfile) if tfile.exists() else np.arange(T, dtype=float))
    unit = (find_frame_times(target_folder).unit if find_frame_times(target_folder) is not None else "frame")
    progress(total=5, advance=0)
    # 1. the Reeb field on the tracked mesh, the source resolved once on the reference (= frame 0) vertices
    from PynamicMesh.utils.tools import resolve_scalar_args
    kw0 = resolve_scalar_args(reeb_scalar, dict(scalar_kwargs or {}), 0, n, selections)
    # automatic references chosen once on the reference frame: the tracked mesh keeps the same material vertices
    from PynamicMesh.core.reeb_graph import resolve_auto_references
    kw0, auto_state = resolve_auto_references(reeb_scalar, kw0, X[0], F0)
    if auto_state:
        with open(out / "auto_reference.json", "w") as fh:
            json.dump(auto_state, fh, indent=1)
    Fk = fields_on_tracked_mesh(X, F0, reeb_scalar, kw0, spectral, p["k_spectral"])
    np.save(out / "field_on_tracked_mesh.npy", Fk.astype(np.float32)); progress("field on the tracked mesh")
    # 2. samples in time: linear interpolation of the field inside every observed interval
    sub = max(1, int(p["substeps"]))
    st, sf = [], []
    for k in range(T - 1):
        for j in range(sub):
            a = j / sub; st.append((1 - a) * t[k] + a * t[k + 1]); sf.append((1 - a) * Fk[k] + a * Fk[k + 1])
    st.append(t[-1]); sf.append(Fk[-1])
    st = np.array(st)
    # 3. branches (persistence pairs) of every sample, in parallel
    E = _edges(F0)
    nb = np.concatenate([E, E[:, ::-1]]); nb = nb[np.argsort(nb[:, 0], kind="stable")]
    ptr = np.searchsorted(nb[:, 0], np.arange(n + 1)); idx = nb[:, 1].copy()
    tau = float(p["persistence"])
    from PynamicMesh.core import accel
    with_regions = bool(p.get("ms_regions", True))
    res = accel.parallel_map(_sample_task, [(f_, ptr, idx, 0.5 * tau, with_regions) for f_ in sf], desc="Branches per time sample")
    progress("branches per sample")
    # 4. tracking (discrete Jacobi curve) and events
    size = float(np.sqrt(face_normals_areas(X[0], F0)[1].sum()))
    _FIELD["f"] = sf
    rows, events = [], []
    for kind, pos in (("max", 0), ("min", 1)):
        r, e = _track([(st[i], res[i][pos]) for i in range(len(st))], X[0], size, tau, float(p["match_radius"]), kind)
        rows += r; events += e
    _FIELD["f"] = None
    paths = pd.DataFrame(rows)
    if len(paths):
        P = X[0][paths.extremum.to_numpy(int)]
        paths["x_ref"], paths["y_ref"], paths["z_ref"] = P[:, 0], P[:, 1], P[:, 2]
        # current position: tracked mesh interpolated in time
        kk = np.clip(np.searchsorted(t, paths.time.to_numpy(), side="right") - 1, 0, T - 2) if T > 1 else np.zeros(len(paths), int)
        a = (paths.time.to_numpy() - t[kk]) / np.maximum(t[np.minimum(kk + 1, T - 1)] - t[kk], 1e-300) if T > 1 else np.zeros(len(paths))
        v = paths.extremum.to_numpy(int)
        Q = (1 - a)[:, None] * X[kk, v] + a[:, None] * X[np.minimum(kk + 1, T - 1), v]
        paths["x"], paths["y"], paths["z"] = Q[:, 0], Q[:, 1], Q[:, 2]
    if len(paths):                                    # unique branch label: track numbers restart for every kind
        paths["branch"] = paths["kind"].astype(str) + "_" + paths["track"].astype(str)
    paths.to_csv(out / "paths.csv", index=False)
    reg_tab = adj_tab = None
    if with_regions and len(paths):
        # Morse–Smale regions of the protrusion branches on the tracked mesh, their adjacency (the MS graph) and its events
        reg_tab, adj_tab, reg_events = _ms_regions(out, st, res, paths, X, F0, t, E, float(p.get("contact_threshold", 0.02)))
        events += reg_events
    ev = pd.DataFrame(events, columns=["time", "sample", "event", "kind", "track", "other"]).sort_values(["time", "event"])
    if len(ev):
        ev["branch"] = ev["kind"].astype(str) + "_" + ev["track"].astype(str)
        is_reg = ev["kind"] == "region"                  # contacts between the regions of two protrusion branches
        if is_reg.any():
            # 'other' holds numbers (branch events) and branch names (region events): a general column, otherwise
            # pandas >= 3 refuses to write text into a column it stored as integers
            ev["other"] = ev["other"].astype(object)
            ev.loc[is_reg, "branch"] = ("max_" + ev.loc[is_reg, "track"].astype(int).astype(str)).to_numpy(object)
            ev.loc[is_reg, "other"] = ("max_" + ev.loc[is_reg, "other"].astype(int).astype(str)).to_numpy(object)
    ev.to_csv(out / "events.csv", index=False)
    samp = paths.groupby(["sample", "time", "kind"]).track.count().unstack(fill_value=0).reset_index() if len(paths) else pd.DataFrame()
    if len(samp):
        samp = samp.rename(columns={"max": "n_protrusion_branches", "min": "n_dent_branches"})
        samp.to_csv(out / "samples.csv", index=False)
    progress("critical-point paths and events")
    tracks = pd.DataFrame()
    if len(paths):
        g = paths.groupby(["kind", "track"])
        tracks = g.agg(birth_time=("time", "min"), death_time=("time", "max"), max_persistence=("persistence", "max"),
                       mean_persistence=("persistence", "mean"), essential=("essential", "max"), n_samples=("sample", "count")).reset_index()
        tracks["lifetime"] = tracks.death_time - tracks.birth_time
        tracks.insert(0, "branch", tracks["kind"].astype(str) + "_" + tracks["track"].astype(str))
        tracks["born_during"] = tracks.birth_time > st[0]; tracks["died_during"] = tracks.death_time < st[-1]
        tracks["path_length_ref"] = [float(np.sum(np.linalg.norm(np.diff(gg.sort_values("sample")[["x_ref", "y_ref", "z_ref"]].to_numpy(), axis=0), axis=1)))
                                     for _, gg in g]
        tracks.to_csv(out / "tracks.csv", index=False)
    summary = {"n_samples": int(len(st)), "substeps": sub, "persistence": tau, "time_unit": unit, "scalar": reeb_scalar,
               "n_protrusion_branches": int((tracks.kind == "max").sum()) if len(tracks) else 0,
               "n_dent_branches": int((tracks.kind == "min").sum()) if len(tracks) else 0,
               "events": ev.event.value_counts().to_dict() if len(ev) else {}}
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if p["plots"] and len(paths):
        _plots(out, paths, ev, samp, unit)
        _region_plots(out, reg_tab, adj_tab, unit)
    progress("tracks, summary and plots")
    if verbose:
        print(f"Reeb dynamics: {summary['n_protrusion_branches']} protrusion / {summary['n_dent_branches']} dent branches, "
              f"{len(ev)} events over {len(st)} samples")
    return summary


def _ms_regions(out, st, res, paths, X, F0, t, E, contact_threshold=0.02):
    """
    Morse–Smale regions of the protrusion branches on the tracked mesh. At every sample each vertex belongs to the
    maximum reached by steepest ascent (the ascending Morse–Smale cell of that maximum); a maximum that is not a tracked
    branch (below the persistence threshold) gives its cell to the branch it merges into in the sweep of −f — the
    persistence simplification of the Morse–Smale complex. Regions therefore carry the identity of their branch (max_<track>)
    through time, from the tracked mesh (no functional maps).
    Adjacency (the Morse–Smale graph): two regions touch when their common boundary (sum of the lengths of the mesh
    edges joining them, relative to sqrt(area)) exceeds contact_threshold; hysteresis: a contact starts above the
    threshold and ends below half of it. Events 'regions start touching' / 'regions stop touching' (kind 'region')
    between regions that both existed at the previous sample.
    Writes regions/labels.npz (labels per sample: track of every vertex, -1 = none), regions.csv (per sample and
    region: area, area fraction, centroid, number of neighbours) and adjacency.csv (touching pairs and their boundary).
    """
    T = X.shape[0]; n = X.shape[1]
    d = ensure_dir(out / "regions")
    mx = paths[paths.kind == "max"]
    by_sample = {int(si): dict(zip(g.extremum.astype(int), g.track.astype(int))) for si, g in mx.groupby("sample")}
    labels = np.full((len(st), n), -1, dtype=np.int32)
    reg_rows, adj_rows, events = [], [], []
    touching = {}                                       # (a, b) -> bool, with hysteresis
    prev_tracks = None
    for si, ts in enumerate(st):
        ext_of, merged = res[si][2], res[si][3]
        if ext_of is None:
            continue
        tracked = by_sample.get(si, {})
        lut = {}
        for e in np.unique(ext_of):
            e0 = int(e); seen = 0
            while e0 not in tracked and e0 in merged and seen < 100000:
                e0 = merged[e0]; seen += 1
            lut[int(e)] = tracked.get(e0, -1)
        lab = np.vectorize(lut.get, otypes=[np.int32])(ext_of)
        labels[si] = lab
        # geometry at the sample: tracked mesh interpolated in time
        k = int(np.clip(np.searchsorted(t, ts, side="right") - 1, 0, max(T - 2, 0)))
        a = 0.0 if T < 2 else float((ts - t[k]) / max(t[min(k + 1, T - 1)] - t[k], 1e-300))
        Xs = (1 - a) * X[k] + a * X[min(k + 1, T - 1)]
        fa = face_normals_areas(Xs, F0)[1]
        va = np.zeros(n); np.add.at(va, F0.ravel(), np.repeat(fa / 3.0, 3))
        size = float(np.sqrt(fa.sum())) or 1.0
        ids = sorted(int(x) for x in np.unique(lab) if x >= 0)
        # boundaries between regions
        la, lb = lab[E[:, 0]], lab[E[:, 1]]
        cross = (la != lb) & (la >= 0) & (lb >= 0)
        lens = np.linalg.norm(Xs[E[cross, 0]] - Xs[E[cross, 1]], axis=1) / size
        pa, pb = np.minimum(la[cross], lb[cross]), np.maximum(la[cross], lb[cross])
        contact = {}
        for x_, y_, l_ in zip(pa, pb, lens):
            contact[(int(x_), int(y_))] = contact.get((int(x_), int(y_)), 0.0) + float(l_)
        now = {}
        for pair in set(contact) | set(touching):
            c = contact.get(pair, 0.0); was = touching.get(pair, False)
            on = c >= contact_threshold if not was else c >= 0.5 * contact_threshold
            if on and pair[0] in ids and pair[1] in ids:
                now[pair] = True
                adj_rows.append({"sample": si, "time": float(ts), "branch_a": f"max_{pair[0]}", "branch_b": f"max_{pair[1]}",
                                 "track_a": pair[0], "track_b": pair[1], "contact": c})
            existed = prev_tracks is not None and pair[0] in prev_tracks and pair[1] in prev_tracks
            if existed and on != was and pair[0] in ids and pair[1] in ids:
                events.append({"time": float(ts), "sample": si, "event": "regions start touching" if on else "regions stop touching",
                               "kind": "region", "track": pair[0], "other": pair[1]})
        touching = now
        nbrs = {i: 0 for i in ids}
        for (x_, y_) in now:
            nbrs[x_] = nbrs.get(x_, 0) + 1; nbrs[y_] = nbrs.get(y_, 0) + 1
        tot = float(va.sum()) or 1.0
        for i in ids:
            m = lab == i; A = float(va[m].sum())
            c = (va[m, None] * Xs[m]).sum(0) / max(A, 1e-300)
            reg_rows.append({"sample": si, "time": float(ts), "branch": f"max_{i}", "track": i, "area": A,
                             "area_fraction": A / tot, "x": c[0], "y": c[1], "z": c[2], "n_neighbors": nbrs.get(i, 0)})
        prev_tracks = set(ids)
    np.savez_compressed(d / "labels.npz", labels=labels, sample_time=np.asarray(st, float))
    reg = pd.DataFrame(reg_rows); adj = pd.DataFrame(adj_rows)
    reg.to_csv(out / "regions.csv", index=False); adj.to_csv(out / "adjacency.csv", index=False)
    return reg, adj, events


def _region_plots(out, reg, adj, unit):
    if reg is None or not len(reg):
        return
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for b, g in reg.groupby("branch"):
        ax[0].plot(g.time, g.area_fraction, "-", lw=1.4, label=b)
        ax[1].step(g.time, g.n_neighbors, where="post", lw=1.2)
    ax[0].set_ylabel("area fraction of the surface"); ax[0].set_title("Morse–Smale regions of the protrusion branches (tracked mesh)")
    ax[0].legend(fontsize=7, ncol=4); ax[0].grid(alpha=0.3)
    ax[1].set_ylabel("neighbouring regions"); ax[1].set_xlabel(f"time ({unit})"); ax[1].grid(alpha=0.3)
    ax[1].set_title("number of neighbours of every region in the Morse–Smale graph")
    fig.tight_layout(); fig.savefig(Path(out) / "plots" / "4_ms_regions.png", dpi=200); plt.close(fig)
    if adj is not None and len(adj):
        pairs = sorted({(r.branch_a, r.branch_b) for r in adj.itertuples()})
        fig, ax = plt.subplots(figsize=(12, 1.6 + 0.32 * len(pairs)))
        times = np.sort(reg.time.unique()); dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        for i, pr in enumerate(pairs):
            g = adj[(adj.branch_a == pr[0]) & (adj.branch_b == pr[1])]
            ax.scatter(g.time, np.full(len(g), i), c=g.contact, cmap="viridis", s=18, marker="s",
                       vmin=0, vmax=float(adj.contact.quantile(0.95)) or 1.0)
        ax.set_yticks(range(len(pairs))); ax.set_yticklabels([f"{a} – {b}" for a, b in pairs], fontsize=7)
        ax.set_xlabel(f"time ({unit})"); ax.grid(alpha=0.3, axis="x")
        ax.set_title("contacts between regions (edges of the Morse–Smale graph; colour = length of the common boundary)")
        fig.tight_layout(); fig.savefig(Path(out) / "plots" / "5_ms_contacts.png", dpi=200); plt.close(fig)


def _plots(out, paths, ev, samp, unit):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for a, kind, ttl in ((axes[0], "max", "protrusion branches (maximum–saddle pairs)"), (axes[1], "min", "dent branches (minimum–saddle pairs)")):
        sub = paths[(paths.kind == kind) & (~paths.essential)]
        for tid, g in sub.groupby("track"):
            a.plot(g.time, g.persistence, "-", lw=1.2)
        e = ev[ev.kind == kind]
        for name, mk, col in (("born", "^", "tab:green"), ("retracted", "v", "tab:red"), ("vanished", "v", "tab:red")):
            sel = e[e.event.str.contains(name)]
            if len(sel):
                pv = [sub[(sub.track == r.track)].persistence.iloc[0] if (sub.track == r.track).any() else 0 for r in sel.itertuples()]
                a.scatter(sel.time, pv, marker=mk, color=col, s=40, zorder=3)
        a.set_ylabel("relative persistence"); a.set_title(ttl + " — ▲ birth, ▼ death"); a.grid(alpha=0.3)
    axes[1].set_xlabel(f"time ({unit})")
    fig.tight_layout(); fig.savefig(Path(out) / "plots" / "1_branch_persistence.png", dpi=200); plt.close(fig)
    if len(samp):
        fig, ax = plt.subplots(figsize=(12, 3.8))
        for c in ("n_protrusion_branches", "n_dent_branches"):
            if c in samp:
                ax.step(samp.time, samp[c], where="post", label=c.replace("n_", "").replace("_", " "))
        ax.set_xlabel(f"time ({unit})"); ax.set_ylabel("branches (incl. the trunk)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("number of branches of the Reeb graph over time (tracked mesh, between the frames)")
        fig.tight_layout(); fig.savefig(Path(out) / "plots" / "2_branch_count.png", dpi=200); plt.close(fig)
    if len(ev):
        kinds = list(dict.fromkeys(ev.event))
        fig, ax = plt.subplots(figsize=(13, 1.4 + 0.5 * len(kinds)))
        for i, k in enumerate(kinds):
            e = ev[ev.event == k]; ax.scatter(e.time, np.full(len(e), i), s=40)
        ax.set_yticks(range(len(kinds))); ax.set_yticklabels(kinds, fontsize=8); ax.set_xlabel(f"time ({unit})")
        ax.set_title("events of the time-varying Reeb graph"); ax.grid(alpha=0.3, axis="x")
        fig.tight_layout(); fig.savefig(Path(out) / "plots" / "3_reeb_events.png", dpi=200); plt.close(fig)
