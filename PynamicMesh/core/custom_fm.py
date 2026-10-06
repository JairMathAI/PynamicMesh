"""
Custom functional-map wrapper around pyFM.functional.FunctionalMapping.

Additions with respect to the stock class
-----------------------------------------
* Descriptor combinations with explicit energy weights:
      "WKS", "WKS+HKS", "0.7*WKS + 0.3*MKS", ["WKS", "XYZ"], {"WKS": 0.6, "HKS": 0.4}
  Every block is normalized so that its total descriptor energy equals its weight
  (weights are renormalized to sum to one); the global descriptor energy is then
  controlled only by `w_descr` in fit().
* Descriptors: WKS, HKS, MKS (Matérn kernel signature, closed form in the LB basis),
  XYZ (aligned coordinates, LB low-pass filtered), NRM (vertex normals, LB low-pass filtered).
* Landmarks: explicit (m,) / (m, 2) arrays or 'auto' (farthest-point sampling on the source,
  extrinsic/descriptor matching on the target, geodesic-consistency outlier rejection).
* Landmark-free symmetry handling: orientation-preserving operator term (w_orient) and/or
  extrinsic descriptor blocks, selected with `symmetry_mode`.

All descriptors are computed from the mesh eigendecomposition (mesh.eigenvalues /
mesh.eigenvectors) so the code does not depend on the exact pyFM version of the
signature helpers; only fit / refine of the base class are reused.
"""
import re
import inspect
import warnings
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from pyFM.functional import FunctionalMapping

try:
    import cupy as xp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    GPU_AVAILABLE = False


SPECTRAL_DESCRIPTORS = ("WKS", "HKS", "MKS")
EXTRINSIC_DESCRIPTORS = ("XYZ", "NRM")
DESCRIPTOR_ALIASES = {
    "MATERN": "MKS", "MATERN_KERNEL": "MKS",
    "COORDS": "XYZ", "COORDINATES": "XYZ", "POS": "XYZ",
    "NORMALS": "NRM", "NORMAL": "NRM",
}
SYMMETRY_MODES = ("none", "landmarks", "orientation", "extrinsic")



def _faces(mesh):
    f = np.asarray(mesh.faces)
    if f.ndim == 1:                                   
        f = f.reshape(-1, 4)[:, 1:]
    return f.astype(np.int64)


def vertex_areas(vertices, faces):
    """Lumped (barycentric) vertex areas."""
    v = np.asarray(vertices, dtype=np.float64)
    tri = v[faces]
    fa = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    va = np.zeros(v.shape[0])
    for k in range(3):
        np.add.at(va, faces[:, k], fa / 3.0)
    return np.maximum(va, 1e-16)


def vertex_normals(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def edge_adjacency(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = e[e[:, 0] != e[:, 1]]
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    n = v.shape[0]
    adj = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return adj.maximum(adj.T)


def geodesic_from(adj, sources):
    d = dijkstra(adj, directed=False, indices=np.atleast_1d(sources), min_only=True)
    if np.isinf(d).any():
        d[np.isinf(d)] = d[np.isfinite(d)].max() * 2.0 if np.isfinite(d).any() else 0.0
    return d


def farthest_point_sampling(vertices, faces, n_samples, adj=None, first=None):
    """Geodesic farthest-point sampling; returns vertex indices."""
    v = np.asarray(vertices, dtype=np.float64)
    adj = edge_adjacency(v, faces) if adj is None else adj
    if first is None:
        first = int(np.argmax(np.linalg.norm(v - v.mean(axis=0), axis=1)))
    samples = [first]
    dist = geodesic_from(adj, first)
    for _ in range(1, min(n_samples, v.shape[0])):
        nxt = int(np.argmax(dist))
        samples.append(nxt)
        dist = np.minimum(dist, geodesic_from(adj, nxt))
    return np.asarray(samples, dtype=np.int64)


def mass_matrix(mesh, faces):
    A = getattr(mesh, "A", None)
    if A is not None and hasattr(A, "shape") and A.shape[0] == np.asarray(mesh.vertices).shape[0]:
        return A
    return diags(vertex_areas(mesh.vertices, faces))


def l2_column_norms(func, A):
    """sqrt(<f, f>_A) for each column of func."""
    return np.sqrt(np.maximum(np.einsum("ij,ij->j", func, A @ func), 0.0))



def _canonical_name(name):
    name = str(name).strip().upper()
    name = DESCRIPTOR_ALIASES.get(name, name)
    if name not in SPECTRAL_DESCRIPTORS + EXTRINSIC_DESCRIPTORS:
        raise ValueError(f"Unsupported descriptor '{name}'. "
                         f"Available: {SPECTRAL_DESCRIPTORS + EXTRINSIC_DESCRIPTORS} (+ aliases {tuple(DESCRIPTOR_ALIASES)}).")
    return name


def parse_descriptor_spec(spec):
    """
    Returns an ordered list of (name, weight) with weights summing to one.

    Accepted: "WKS", "WKS+HKS", "0.7*WKS + 0.3*MKS", "WKS*0.7 + MKS*0.3",
              ["WKS", "HKS"], [("WKS", 0.7), ("MKS", 0.3)], {"WKS": 0.7, "MKS": 0.3}.
    Duplicated names are merged (weights added).
    """
    items = []
    if isinstance(spec, dict):
        items = [(k, float(w)) for k, w in spec.items()]
    elif isinstance(spec, (list, tuple)):
        for it in spec:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                items.append((it[0], float(it[1])))
            else:
                items.append((it, None))
    else:
        for term in str(spec).replace("-", "+").split("+"):
            term = term.strip()
            if not term:
                continue
            parts = [p.strip() for p in term.split("*")]
            if len(parts) == 1:
                items.append((parts[0], None))
            elif len(parts) == 2:
                num, name = (parts[0], parts[1]) if re.fullmatch(r"[\d.eE+-]+", parts[0]) else (parts[1], parts[0])
                items.append((name, float(num)))
            else:
                raise ValueError(f"Cannot parse descriptor term '{term}'.")
    if not items:
        raise ValueError("Empty descriptor specification.")

    merged = {}
    order = []
    n_unweighted = sum(1 for _, w in items if w is None)
    given = sum(w for _, w in items if w is not None)
    # Unweighted terms share what is left (or equal weights if nothing was given).
    default_w = max(1.0 - given, 0.0) / n_unweighted if n_unweighted and given < 1.0 else 1.0
    for name, w in items:
        name = _canonical_name(name)
        w = default_w if w is None else w
        if w < 0:
            raise ValueError("Descriptor weights must be non-negative.")
        if name not in merged:
            order.append(name)
            merged[name] = 0.0
        merged[name] += w
    total = sum(merged.values())
    if total <= 0:
        raise ValueError("Descriptor weights sum to zero.")
    return [(n, merged[n] / total) for n in order]



def _positive_evals(evals):
    ev = np.asarray(evals, dtype=np.float64)
    pos = ev[ev > 1e-10]
    if pos.size < 2:
        raise ValueError("At least two non-zero Laplace–Beltrami eigenvalues are required for spectral descriptors.")
    return ev, pos


def hks_filters(evals, n_descr):
    """Heat kernel filters exp(-t λ) for log-spaced t spanning the resolvable range (Sun et al. 2009)."""
    ev, pos = _positive_evals(evals)
    t = np.geomspace(4.0 * np.log(10) / pos[-1], 4.0 * np.log(10) / pos[0], n_descr)
    return np.exp(-ev[:, None] * t[None, :])                              # (K, n_descr)


def wks_filters(evals, n_descr):
    """Wave kernel filters (Aubry et al. 2011); the constant eigenfunction is excluded."""
    ev, pos = _positive_evals(evals)
    e_min, e_max = np.log(pos[0]), np.log(pos[-1])
    delta = (e_max - e_min) / n_descr
    sigma = 7.0 * delta
    energies = np.linspace(e_min + 2 * delta, e_max - 2 * delta, n_descr)
    log_ev = np.full(ev.shape, -np.inf)
    log_ev[ev > 1e-10] = np.log(ev[ev > 1e-10])
    g = np.exp(-np.square(energies[None, :] - log_ev[:, None]) / (2.0 * sigma ** 2))
    g[~(ev > 1e-10), :] = 0.0
    return g / np.maximum(g.sum(axis=0, keepdims=True), 1e-300)           # (K, n_descr)


def mks_filters(evals, n_descr, nu=1.5, min_l=None, max_l=None):
    """
    Matérn spectral filters (2ν/ℓ² + λ)^-(ν+1) (d = 2). The lengthscale range defaults to
    the mesh-adaptive interval [sqrt(2ν/λ_K), sqrt(2ν/λ_1)], where the filter transitions.
    """
    ev, pos = _positive_evals(evals)
    if min_l is None:
        min_l = np.sqrt(2.0 * nu / pos[-1])
    if max_l is None:
        max_l = np.sqrt(2.0 * nu / pos[0])
    ls = np.geomspace(min_l, max_l, n_descr)
    if np.isinf(nu):
        g = np.exp(-0.5 * (ls[None, :] ** 2) * ev[:, None])
    else:
        g = (2.0 * nu / ls[None, :] ** 2 + ev[:, None]) ** (-(nu + 1.0))
    return g / np.maximum(g.sum(axis=0, keepdims=True), 1e-300)           # (K, n_descr)


def spectral_signature(evecs, filters, landmarks=None):
    """
    Point signature  S(x) = Σ_k g(λ_k) φ_k(x)²                       -> (n, n_descr)
    Landmark signature  L_l(x) = Σ_k g(λ_k) φ_k(l) φ_k(x)  for each l   -> (n, n_descr * m)
    """
    phi = xp.asarray(evecs)
    g = xp.asarray(filters)
    if landmarks is None:
        out = (phi ** 2) @ g
    else:
        blocks = [(phi * phi[int(l), :][None, :]) @ g for l in np.atleast_1d(landmarks)]
        out = xp.hstack(blocks)
    return out.get() if GPU_AVAILABLE else np.asarray(out)


def lb_lowpass(func, evecs, A, k):
    """Projects func onto the first k LB eigenfunctions (A-orthonormal basis): Φ_k Φ_kᵀ A f."""
    if k is None or k <= 0 or k >= evecs.shape[1]:
        return func
    phi = evecs[:, :k]
    return phi @ (phi.T @ (A @ func))



def auto_landmarks(mesh1, mesh2, n_landmarks=12, match="hybrid", descr1=None, descr2=None,
                   max_rel_dist=0.15, max_distortion=0.25, min_landmarks=4, search_rel_radius=0.1,
                   first=None, verbose=False):
    """
    Robust automatic landmark selection for a (near-)isometric pair of *aligned* meshes.

    1. Farthest-point sampling (geodesic) on mesh1 → well spread candidates, extremities first
       (legs, head, tail…), which are exactly the points that disambiguate symmetries.
    2. Correspondence on mesh2:
         'identity'  : same vertex index (meshes share connectivity/tracking)
         'extrinsic' : nearest vertex in 3D (aligned frames, small motion)
         'hybrid'    : among mesh2 vertices within `search_rel_radius`·diag of the landmark,
                       the best match in descriptor space (falls back to 'extrinsic' without descriptors)
    3. Rejection of unreliable pairs: extrinsic distance > `max_rel_dist`·diag, duplicated
       targets, and pairs breaking the geodesic-distance consistency between landmarks by more
       than `max_distortion` (relative), removed iteratively worst-first.

    Returns an (m, 2) int array [idx_mesh1, idx_mesh2] or None if fewer than `min_landmarks` survive.
    """
    v1, v2 = np.asarray(mesh1.vertices, dtype=np.float64), np.asarray(mesh2.vertices, dtype=np.float64)
    f1, f2 = _faces(mesh1), _faces(mesh2)
    diag = np.linalg.norm(v1.max(axis=0) - v1.min(axis=0))

    adj1 = edge_adjacency(v1, f1)
    idx1 = farthest_point_sampling(v1, f1, n_landmarks, adj=adj1, first=first)

    if match == "identity":
        if v1.shape[0] != v2.shape[0]:
            raise ValueError("match='identity' requires the same number of vertices on both meshes.")
        idx2 = idx1.copy()
        ext_dist = np.linalg.norm(v1[idx1] - v2[idx2], axis=1)
    else:
        tree = cKDTree(v2)
        if match == "hybrid" and descr1 is not None and descr2 is not None:
            idx2 = np.empty_like(idx1)
            ext_dist = np.empty(len(idx1))
            for j, i in enumerate(idx1):
                cand = tree.query_ball_point(v1[i], r=search_rel_radius * diag)
                if len(cand) == 0:
                    _, c = tree.query(v1[i], k=1)
                    cand = [int(c)]
                cand = np.asarray(cand)
                cost = np.linalg.norm(descr2[cand] - descr1[i][None, :], axis=1)
                idx2[j] = cand[int(np.argmin(cost))]
                ext_dist[j] = np.linalg.norm(v1[i] - v2[idx2[j]])
        else:
            ext_dist, idx2 = tree.query(v1[idx1], k=1)
            idx2 = np.asarray(idx2, dtype=np.int64)

    keep = ext_dist <= max_rel_dist * diag
    if verbose and (~keep).any():
        print(f"[auto_landmarks] {int((~keep).sum())} pair(s) rejected: displacement > {max_rel_dist:.2f}·diag")
    idx1, idx2 = idx1[keep], idx2[keep]

    _, first_occ = np.unique(idx2, return_index=True)          # drop duplicated targets
    first_occ = np.sort(first_occ)
    idx1, idx2 = idx1[first_occ], idx2[first_occ]

    # Geodesic consistency (normalized by sqrt(total area) so scale changes are tolerated).
    if len(idx1) >= 3:
        adj2 = edge_adjacency(v2, f2)
        s1 = np.sqrt(vertex_areas(v1, f1).sum())
        s2 = np.sqrt(vertex_areas(v2, f2).sum())
        D1 = np.stack([geodesic_from(adj1, i)[idx1] for i in idx1]) / s1
        D2 = np.stack([geodesic_from(adj2, i)[idx2] for i in idx2]) / s2
        alive = np.ones(len(idx1), dtype=bool)
        while alive.sum() > min_landmarks:
            a = np.flatnonzero(alive)
            rel = np.abs(D1[np.ix_(a, a)] - D2[np.ix_(a, a)]) / (D1[np.ix_(a, a)] + 1e-9)
            np.fill_diagonal(rel, 0.0)
            score = np.median(rel, axis=1)
            worst = int(np.argmax(score))
            if score[worst] <= max_distortion:
                break
            if verbose:
                print(f"[auto_landmarks] pair {idx1[a[worst]]}->{idx2[a[worst]]} rejected: geodesic distortion {score[worst]:.2f}")
            alive[a[worst]] = False
        idx1, idx2 = idx1[alive], idx2[alive]

    if len(idx1) < min_landmarks:
        warnings.warn(f"auto_landmarks: only {len(idx1)} reliable landmark(s) found (< {min_landmarks}); no landmarks used.")
        return None
    return np.stack([idx1, idx2], axis=1)



def _as_scalar(x):
    """float from a scalar, 0-d/1-element array, or a (nested) 1-tuple of those."""
    while isinstance(x, (tuple, list)):
        x = x[0]
    return float(np.ravel(x)[0])


def p2p_from_FM(FM_12, evects1, evects2):
    """Point-to-point map mesh2 -> mesh1 from the functional map C (k2 × k1): nearest neighbour of Φ₂ in Φ₁Cᵀ."""
    k2, k1 = FM_12.shape
    emb1 = evects1[:, :k1] @ FM_12.T
    emb2 = evects2[:, :k2]
    from PynamicMesh.core.accel import knn               # exact; GPU(s) when available, cKDTree otherwise
    return knn(emb1, emb2, k=1)


class CustomFunctionalMapping(FunctionalMapping):
    """
    Functional map with weighted descriptor combinations, automatic landmarks and
    landmark-free symmetry handling. See module docstring.
    """

    def __init__(self, mesh1, mesh2):
        super().__init__(mesh1, mesh2)
        self.descriptor_blocks = []        # [(name, weight, n_columns)]
        self.landmarks = None              # (m, 2) array actually used, or None
        self.symmetry_modes = ("none",)
        self._fit_defaults = {}
        self._n_ev = (None, None)          # (k1, k2) requested map size, independent of how pyFM stores it

    def _set_map_size(self, k1, k2):
        """
        Stores the functional-map size in the way the installed pyFM expects: plain attributes k1/k2
        in old versions, read-only properties backed by _k1/_k2 in recent ones.
        """
        k1, k2 = int(k1), int(k2)
        self._n_ev = (k1, k2)
        for name, val in (("k1", k1), ("k2", k2)):
            try:
                setattr(self, name, val)
            except AttributeError:                       # read-only property -> set the private attribute
                setattr(self, "_" + name, val)
        try:
            ok = int(self.k1) == k1 and int(self.k2) == k2
        except Exception:                                # noqa: BLE001
            ok = False
        if not ok:
            warnings.warn("Could not set the functional-map size on this pyFM version (k1/k2 mismatch); "
                          "fit() may use a different basis size than requested.")
        return k1, k2



    def _set_descriptors(self, descr1, descr2):
        self.descr1 = np.ascontiguousarray(descr1, dtype=np.float64)
        self.descr2 = np.ascontiguousarray(descr2, dtype=np.float64)

    @staticmethod
    def _filters(name, evals, n_descr, descr_params):
        if name == "HKS":
            return hks_filters(evals, n_descr)
        if name == "WKS":
            return wks_filters(evals, n_descr)
        if name == "MKS":
            return mks_filters(evals, n_descr, nu=descr_params.get("nu", 1.5),
                               min_l=descr_params.get("min_l"), max_l=descr_params.get("max_l"))
        raise ValueError(name)

    def _extrinsic_block(self, mesh, faces, A, name, center, scale, k_smooth):
        v = np.asarray(mesh.vertices, dtype=np.float64)
        F = (v - center) / scale if name == "XYZ" else vertex_normals(v, faces)
        return lb_lowpass(F, np.asarray(mesh.eigenvectors), A, k_smooth)

    @staticmethod
    def _normalize_block(d, A, weight):
        """Unit L2(A)-norm per column, then scale so that the block's total energy is `weight`."""
        norms = l2_column_norms(d, A)
        d = d / np.maximum(norms, 1e-12)[None, :]
        return d * np.sqrt(weight / max(d.shape[1], 1))

    # ---- preprocessing ------------------------------------------------------ #

    def preprocess(self, n_ev=(30, 30), n_descr=100, descr_type="WKS", landmarks=None, subsample_step=1,
                   k_process=None, symmetry_mode="none", landmark_params=None, descr_params=None,
                   verbose=False, K=None, **kwargs):
        """
        Parameters
        ----------
        n_ev           : int or (k1, k2) – size of the functional map (alias: K).
        n_descr        : number of filters per spectral descriptor block.
        descr_type     : descriptor combination, see parse_descriptor_spec().
        landmarks      : None | 'auto' | (m,) | (m, 2) array of vertex indices [mesh1, mesh2].
        subsample_step : keep one column out of `subsample_step` in spectral point-signature blocks.
        k_process      : number of eigenpairs to compute on each mesh (>= max(n_ev)).
        symmetry_mode  : 'none' | 'landmarks' | 'orientation' | 'extrinsic', combinable with '+'
                         ('orientation+extrinsic'). 'landmarks' with landmarks=None triggers 'auto'.
        landmark_params: dict for auto_landmarks() (n_landmarks, match, max_rel_dist, max_distortion,
                         min_landmarks, search_rel_radius) plus 'weight' (energy share of the
                         landmark block, default 1.0 = same as the whole point-signature part) and
                         'descriptor' (which spectral filter to localize, default: first spectral block).
        descr_params   : dict: nu, min_l, max_l (MKS); k_smooth (LB low-pass for XYZ/NRM, default 30);
                         xyz_weight (energy of the XYZ block auto-added by 'extrinsic', default 0.3).
        """
        landmark_params = dict(landmark_params or {})
        descr_params = dict(descr_params or {})
        descr_params.update({k: v for k, v in kwargs.items() if k in ("nu", "min_l", "max_l", "k_smooth", "xyz_weight")})

        if K is not None:
            n_ev = K
        if np.isscalar(n_ev):
            n_ev = (int(n_ev), int(n_ev))
        k1, k2 = self._set_map_size(n_ev[0], n_ev[1])

        modes = tuple(m.strip().lower() for m in str(symmetry_mode).split("+") if m.strip())
        for m in modes:
            if m not in SYMMETRY_MODES:
                raise ValueError(f"Unknown symmetry_mode '{m}'. Options: {SYMMETRY_MODES} (combine with '+').")
        self.symmetry_modes = modes or ("none",)

        # --- spectral basis (never recompute with fewer eigenpairs than already available) ---
        k_process = int(k_process) if k_process else max(k1, k2, 100)
        k_needed = max(k_process, k1, k2)
        for mesh in (self.mesh1, self.mesh2):
            if getattr(mesh, "eigenvalues", None) is None or len(mesh.eigenvalues) < k_needed:
                mesh.process(k=k_needed)

        f1, f2 = _faces(self.mesh1), _faces(self.mesh2)
        A1, A2 = mass_matrix(self.mesh1, f1), mass_matrix(self.mesh2, f2)
        ev1, ev2 = np.asarray(self.mesh1.eigenvalues), np.asarray(self.mesh2.eigenvalues)
        phi1, phi2 = np.asarray(self.mesh1.eigenvectors), np.asarray(self.mesh2.eigenvectors)

        # --- descriptor plan ---
        blocks = parse_descriptor_spec(descr_type)
        if "extrinsic" in self.symmetry_modes and not any(n in EXTRINSIC_DESCRIPTORS for n, _ in blocks):
            w_xyz = float(descr_params.get("xyz_weight", 0.3))
            blocks = [(n, w * (1.0 - w_xyz)) for n, w in blocks] + [("XYZ", w_xyz)]
        spectral_names = [n for n, _ in blocks if n in SPECTRAL_DESCRIPTORS]

        # --- point-signature blocks (needed before auto-landmarks for descriptor matching) ---
        d1_blocks, d2_blocks, plan = [], [], []
        center = np.asarray(self.mesh1.vertices, dtype=np.float64).mean(axis=0)
        scale = np.linalg.norm(np.ptp(np.asarray(self.mesh1.vertices), axis=0)) or 1.0
        k_smooth = int(descr_params.get("k_smooth", 30))
        for name, w in blocks:
            if name in SPECTRAL_DESCRIPTORS:
                d1 = spectral_signature(phi1, self._filters(name, ev1, n_descr, descr_params))
                d2 = spectral_signature(phi2, self._filters(name, ev2, n_descr, descr_params))
                if subsample_step > 1:
                    d1, d2 = d1[:, ::subsample_step], d2[:, ::subsample_step]
            else:
                d1 = self._extrinsic_block(self.mesh1, f1, A1, name, center, scale, k_smooth)
                d2 = self._extrinsic_block(self.mesh2, f2, A2, name, center, scale, k_smooth)
            d1_blocks.append(self._normalize_block(d1, A1, w))
            d2_blocks.append(self._normalize_block(d2, A2, w))
            plan.append((name, w, d1.shape[1]))

        # --- landmarks ---
        lm = self._resolve_landmarks(landmarks, landmark_params, np.hstack(d1_blocks), np.hstack(d2_blocks), verbose)
        self.landmarks = lm
        if lm is not None and spectral_names:
            lm_name = _canonical_name(landmark_params.get("descriptor", spectral_names[0]))
            lm_w = float(landmark_params.get("weight", 1.0))
            g1 = self._filters(lm_name, ev1, n_descr, descr_params)
            g2 = self._filters(lm_name, ev2, n_descr, descr_params)
            l1 = spectral_signature(phi1, g1, landmarks=lm[:, 0])
            l2 = spectral_signature(phi2, g2, landmarks=lm[:, 1])
            if subsample_step > 1:
                l1, l2 = l1[:, ::subsample_step], l2[:, ::subsample_step]
            d1_blocks.append(self._normalize_block(l1, A1, lm_w))
            d2_blocks.append(self._normalize_block(l2, A2, lm_w))
            plan.append((f"LM-{lm_name}", lm_w, l1.shape[1]))
        elif lm is not None:
            warnings.warn("Landmarks were provided but no spectral descriptor is requested; landmark descriptors need WKS/HKS/MKS.")

        self.descriptor_blocks = plan
        self._set_descriptors(np.hstack(d1_blocks), np.hstack(d2_blocks))

        # Orientation term (Ren et al. 2018): reflections are orientation reversing, so this
        # penalizes the symmetric flip without any landmark.
        self._fit_defaults = {"w_orient": float(kwargs.get("w_orient", 1.0))} if "orientation" in self.symmetry_modes else {}

        if verbose:
            print(self.describe())
        return self

    def _resolve_landmarks(self, landmarks, landmark_params, descr1, descr2, verbose):
        want_auto = (isinstance(landmarks, str) and landmarks.lower() == "auto") or \
                    (landmarks is None and "landmarks" in self.symmetry_modes)
        if want_auto:
            params = {k: v for k, v in landmark_params.items() if k not in ("weight", "descriptor")}
            return auto_landmarks(self.mesh1, self.mesh2, descr1=descr1, descr2=descr2, verbose=verbose, **params)
        if landmarks is None or (hasattr(landmarks, "__len__") and len(landmarks) == 0):
            return None
        lm = np.asarray(landmarks, dtype=np.int64)
        if lm.ndim == 1:
            lm = np.stack([lm, lm], axis=1)
        if lm.ndim != 2 or lm.shape[1] != 2:
            raise ValueError("landmarks must be 'auto', an (m,) array or an (m, 2) array [idx_mesh1, idx_mesh2].")
        n1, n2 = np.asarray(self.mesh1.vertices).shape[0], np.asarray(self.mesh2.vertices).shape[0]
        if lm[:, 0].min() < 0 or lm[:, 0].max() >= n1 or lm[:, 1].min() < 0 or lm[:, 1].max() >= n2:
            raise IndexError("Landmark indices out of range.")
        return lm

    # ---- fitting ------------------------------------------------------------ #

    def fit(self, **fit_kwargs):
        """
        Fits the functional map. Accepts w_descr, w_lap, w_dcomm, w_orient, orient_reversing, optinit,
        verbose. w_orient defaults to 1.0 when symmetry_mode includes 'orientation'.

        When the orientation term is active the optimization is run here with pyFM's own energy
        functions (identical objective) instead of FunctionalMapping.fit: some pyFM releases crash in
        the orientation-weight rescaling of fit() ("cannot unpack non-iterable numpy.float64").
        """
        params = dict(self._fit_defaults)
        params.update(fit_kwargs)
        w_orient = float(params.get("w_orient", 0.0) or 0.0)

        if w_orient > 0:
            try:
                return self._fit_with_orientation(**params)
            except Exception as exc:                     # noqa: BLE001
                warnings.warn(f"Custom orientation-aware fit failed ({exc}); falling back to pyFM.fit.")

        supported = inspect.signature(FunctionalMapping.fit).parameters
        dropped = [k for k in params if k not in supported]
        if dropped:
            warnings.warn(f"Installed pyFM.fit does not support {dropped}; ignored.")
        return super().fit(**{k: v for k, v in params.items() if k in supported})

    def _call_variant(self, names, *args, **kwargs):
        for name in names:
            fn = getattr(self, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        raise AttributeError(f"pyFM.FunctionalMapping has none of {names}")

    def _store_FM(self, C):
        try:
            self.FM = C
        except AttributeError:                           # very old pyFM: FM is a read-only property
            self._FM_base = C
            if hasattr(self, "change_FM_type"):
                self.change_FM_type("classic")

    def _fit_with_orientation(self, w_descr=1e-1, w_lap=1e-3, w_dcomm=1.0, w_orient=1.0,
                              orient_reversing=False, optinit="zeros", verbose=False, **_ignored):
        """
        Minimizes  w_descr ||C A - B||² + w_lap ||C Λ1 - Λ2 C||² + w_dcomm Σ||C D_Ai - D_Bi C||²
                   + w_orient Σ||C G_Ai - G_Bi C||²
        (pyFM's objective, Ren et al. 2018 for the orientation operators G).
        """
        from scipy.optimize import fmin_l_bfgs_b
        try:
            from pyFM.optimize import base_functions as opt_func
        except ImportError:                              # older layout
            import pyFM.optimize as opt_func

        if optinit not in ("zeros", "identity", "random"):
            raise ValueError('optinit must be "zeros", "identity" or "random"')
        if self.descr1 is None or self.descr2 is None:
            raise ValueError("Call preprocess() before fit().")
        k1, k2 = self._n_ev

        descr1_red = self.project(self.descr1, mesh_ind=1)          # (k1, p)
        descr2_red = self.project(self.descr2, mesh_ind=2)          # (k2, p)
        list_descr = self._call_variant(("_compute_descr_op", "compute_descr_op")) if w_dcomm > 0 else []
        orient_op = self._call_variant(("_compute_orientation_op", "compute_orientation_op"), reversing=orient_reversing)

        ev1 = np.asarray(self.mesh1.eigenvalues)[:k1]
        ev2 = np.asarray(self.mesh2.eigenvalues)[:k2]
        ev_sqdiff = np.square(ev1[None, :] - ev2[:, None])          # (k2, k1)
        ev_sqdiff /= max(ev_sqdiff.sum(), 1e-300)

        # Rescale the orientation weight relative to the other terms, tolerant to the return type.
        C_eye = np.eye(k2, k1)
        e_native = opt_func.energy_func_std(C_eye, w_descr, w_lap, w_dcomm, 0.0,
                                            descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff)
        e_native = _as_scalar(e_native)
        e_orient = _as_scalar(opt_func.oplist_commutation(C_eye, orient_op))
        if e_orient > 0:
            w_orient = w_orient * e_native / e_orient
            if verbose:
                print(f"\t[custom fit] orientation weight rescaled to {w_orient:.3e}")
        elif verbose:
            print("\t[custom fit] orientation operators have zero energy at identity; no rescaling")

        args = (w_descr, w_lap, w_dcomm, w_orient, descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff)
        x0 = self._call_variant(("_get_x0", "get_x0"), optinit)

        if hasattr(opt_func, "energy_and_grad_std"):
            res = fmin_l_bfgs_b(opt_func.energy_and_grad_std, np.ravel(x0), args=args)
        else:
            res = fmin_l_bfgs_b(opt_func.energy_func_std, np.ravel(x0), fprime=opt_func.grad_energy_std, args=args)

        if verbose:
            info = res[2]
            print(f"\t[custom fit] {self.descr1.shape[1]} descriptors, k=({k2}x{k1}), "
                  f"funcalls={info.get('funcalls')}, warnflag={info.get('warnflag')}")
        self._store_FM(res[0].reshape((k2, k1)))
        return self

    def get_p2p_from_FM(self, FM=None):
        """Point-to-point map mesh2 -> mesh1 for an arbitrary (possibly refined/zoomout) FM."""
        FM = self.FM if FM is None else np.asarray(FM)
        return p2p_from_FM(FM, np.asarray(self.mesh1.eigenvectors), np.asarray(self.mesh2.eigenvectors))

    def describe(self):
        parts = [f"{n}: weight={w:.3f}, {c} columns" for n, w, c in self.descriptor_blocks]
        lm = "none" if self.landmarks is None else f"{len(self.landmarks)} pairs"
        return (f"CustomFunctionalMapping | FM size ({self._n_ev[1]} x {self._n_ev[0]}) | symmetry={'+'.join(self.symmetry_modes)} | "
                f"landmarks={lm} | blocks: " + "; ".join(parts))