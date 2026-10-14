"""
ricci_flow.py  —  discrete SURFACE Ricci flow on triangle meshes, vectorized (GPU when available)
===================================================================================================

Discrete surface Ricci flow (Hamilton; Thurston circle packing; Luo's discrete conformal / vertex-scaling flow)
deforms the metric of a mesh conformally until every vertex reaches a prescribed Gaussian curvature. With per-vertex
log-scale factors u, the edge lengths are l_ij = l0_ij * exp((u_i + u_j) / 2); the flow du/dt = K_target - K is the
negative gradient of a convex energy whose Hessian is the cotangent Laplacian / 2, so Newton's method converges in a
few steps (Luo 2004; Gu & Yau; Jin, Kim & Gu 2008).

Following Wei et al. (2025, "GPU-accelerated optimization of discrete Ricci flow for high-resolution triangular
meshes") every step is written as array operations on the whole mesh — edge lengths, corner angles (law of cosines),
angle-deficit curvatures and the sparse Hessian — instead of half-edge traversals, so it runs on the GPU (CuPy) and
scales to high-resolution meshes; the sparse solve uses conjugate gradients on the GPU, or a direct solve on the CPU.

The result used by the pipeline is the CONFORMAL FACTOR u of the uniformization: the log of the local scaling that
makes the surface uniformly curved (target K_i = 2 pi chi A_i / A). It is an intrinsic shape descriptor (Ben-Chen &
Gotsman 2008): thin parts — necks, stems of protrusions — must be stretched a lot and get a large u; at a neck that
closes, u grows like -2 log(neck radius / size). It is used as a scalar field ('ricci_conformal') and as the
conformal neck index of the topology events.
"""
import numpy as np


def _xp(device="auto"):
    if device in ("auto", "gpu", "cuda"):
        try:
            from PynamicMesh.core.dyn_common import GPU_AVAILABLE
            if GPU_AVAILABLE:
                import cupy as cp
                return cp
        except Exception:  # noqa: BLE001
            pass
    return np


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def mesh_edges(F):
    """Unique edges (ne, 2) and, for every face, the index of the edge OPPOSITE each of its corners (nf, 3)."""
    F = np.asarray(F, dtype=np.int64)
    he = np.stack([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]], axis=1)           # edge opposite corner 0, 1, 2
    key = np.sort(he.reshape(-1, 2), axis=1)
    E, inv = np.unique(key, axis=0, return_inverse=True)
    return E, inv.reshape(-1, 3)


def vertex_areas(V, F):
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    fa = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    a = np.zeros(len(V)); np.add.at(a, F.ravel(), np.repeat(fa / 3.0, 3))
    return a


def _angles(lf, xp):
    """Corner angles of every face from its three edge lengths (law of cosines, vectorized); a degenerate triangle
    (triangle inequality violated) gets angles 0, 0, pi (Bobenko–Pinkall–Springborn extension)."""
    a, b, c = lf[:, 0], lf[:, 1], lf[:, 2]                                       # opposite corners 0, 1, 2
    def ang(x, y, z):                                                            # angle opposite x
        return xp.arccos(xp.clip((y * y + z * z - x * x) / xp.maximum(2 * y * z, 1e-300), -1.0, 1.0))
    return xp.stack([ang(a, b, c), ang(b, c, a), ang(c, a, b)], axis=1)


def ricci_flow(V, F, target="uniform", max_iter=60, tol=1e-6, device="auto", verbose=False, linear=False):
    """
    Discrete surface Ricci flow (vertex scaling), Newton's method, vectorized.
    target: 'uniform' (K_i = 2 pi chi A_i / A: the uniformization metric), a scalar (same curvature at every interior
    vertex, rescaled to satisfy Gauss–Bonnet) or a per-vertex array (must sum to 2 pi chi).
    Returns dict(u: log scale factor per vertex (area-weighted mean 0), K: final curvature, K_target, error
    history (max |K_target - K|), iterations, degenerate: number of degenerate triangles at the end).
    """
    xp = _xp(device)
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    n = len(V)
    E, fe = mesh_edges(F)
    l0 = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    # boundary vertices (curvature pi - sum of angles)
    cnt = np.bincount(fe.ravel(), minlength=len(E))
    bnd = np.zeros(n, bool); bnd[E[cnt == 1].ravel()] = True
    chi = n - len(E) + len(F)
    A = vertex_areas(V, F)
    if isinstance(target, str) and target == "uniform":
        Kt = 2 * np.pi * chi * A / A.sum()
        Kt[bnd] = 0.0 if bnd.any() else Kt[bnd]
        if bnd.any():                                                           # boundary: geodesic-curvature free
            Kt[~bnd] = 2 * np.pi * chi * A[~bnd] / A[~bnd].sum()
    elif np.isscalar(target):
        Kt = np.full(n, float(target)); Kt *= 2 * np.pi * chi / max(Kt.sum(), 1e-300) if Kt.sum() else 1.0
    else:
        Kt = np.asarray(target, float)
    used = np.zeros(n, bool); used[F.ravel()] = True
    l0g, Eg, feg, Fg = xp.asarray(l0), xp.asarray(E), xp.asarray(fe), xp.asarray(F)
    Ktg, bndg = xp.asarray(Kt), xp.asarray(bnd)
    u = xp.zeros(n)

    def state(u):
        l = l0g * xp.exp(0.5 * (u[Eg[:, 0]] + u[Eg[:, 1]]))
        lf = l[feg]
        ang = _angles(lf, xp)
        s = xp.zeros(n); 
        if xp is np:
            np.add.at(s, F.ravel(), ang.ravel())
        else:
            s = xp.bincount(Fg.ravel(), weights=ang.ravel(), minlength=n)
        K = xp.where(bndg, np.pi - s, 2 * np.pi - s)
        lfa, lfb, lfc = lf[:, 0], lf[:, 1], lf[:, 2]
        degen = (lfa >= lfb + lfc) | (lfb >= lfa + lfc) | (lfc >= lfa + lfb)
        return K, ang, int(_host(degen.sum()))

    hist = []
    K, ang, ndeg = state(u)
    for it in range(int(max_iter)):
        g = Ktg - K
        g = xp.where(xp.asarray(used), g, 0.0)
        err = float(_host(xp.max(xp.abs(g))))
        hist.append(err)
        if verbose:
            print(f"  Ricci flow {it}: max |K_target - K| = {err:.3e}")
        if err < tol:
            break
        # Hessian = (cotangent Laplacian) / 2 with the CURRENT angles: dK_i/du_j = -(cot a + cot b)/2 on edge ij
        cot = 1.0 / xp.tan(xp.clip(ang, 1e-6, np.pi - 1e-6))
        cot = xp.where((ang < 1e-6) | (ang > np.pi - 1e-6), 0.0, cot)
        w = xp.zeros(len(E))
        if xp is np:
            np.add.at(w, fe.ravel(), 0.5 * cot.ravel())
        else:
            w = xp.bincount(feg.ravel(), weights=0.5 * cot.ravel(), minlength=len(E))
        w = _host(w)
        from scipy.sparse import coo_matrix, diags
        W = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
        H = 0.5 * (diags(np.asarray(W.sum(1)).ravel()) - W) + diags(np.full(n, 1e-9))   # PSD + tiny shift
        gh = _host(g)
        du = None
        if xp is not np:
            try:
                import cupyx.scipy.sparse as csp
                import cupyx.scipy.sparse.linalg as cspl
                du, _ = cspl.cg(csp.csr_matrix(H), xp.asarray(gh), tol=1e-10, maxiter=2000)
            except Exception:  # noqa: BLE001
                du = None
        if du is None:
            from scipy.sparse.linalg import spsolve
            du = xp.asarray(spsolve(H.tocsc(), gh))
        du = du - xp.sum(xp.asarray(A) * du) / A.sum()                      # gauge: area-weighted mean 0
        # damped Newton on the EXTENDED energy (Bobenko–Pinkall–Springborn: degenerate triangles keep angles 0, 0,
        # pi; the energy stays convex), so steps through degeneracies are allowed; halve until the error decreases
        step = 1.0
        e2 = float(_host(xp.sqrt(xp.sum(g * g))))
        for _ in range(30):
            un = u + step * du
            Kn, angn, ndn = state(un)
            gn = xp.where(xp.asarray(used), Ktg - Kn, 0.0)
            if float(_host(xp.sqrt(xp.sum(gn * gn)))) < e2:
                break
            step *= 0.5
        if linear:                                       # Ben-Chen & Gotsman: the linearized problem, one solve
            u = u + du; K, ang, ndeg = state(u); hist.append(float(_host(xp.max(xp.abs(xp.where(xp.asarray(used), Ktg - K, 0.0))))))
            break
        u, K, ang, ndeg = un, Kn, angn, ndn
    uh = _host(u)
    uh = uh - np.sum(A * uh) / A.sum()
    return {"u": uh, "K": _host(K), "K_target": Kt, "history": hist, "iterations": len(hist), "degenerate": ndeg}


def manifold_core(V, F, min_fraction=0.01):
    """
    A manifold copy of a mesh FOR THE FLOW ONLY: vertices shared by separate fans of triangles (pinch vertices) and
    non-manifold edges (shared by more than two triangles, e.g. small folded pockets left by an exporter) are split
    by fans (every fan gets its own copy of the vertex), then the pieces holding at least `min_fraction` of the area
    are kept (tiny pockets dropped). Returns (vertices, faces, original vertex of every new vertex, piece of every new
    face). The input mesh is not modified.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    V = np.asarray(V, float); F = np.asarray(F, np.int64); nf = len(F)
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(nf), 3)
    key = np.sort(he, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); inv = inv.ravel()
    reg = np.flatnonzero(cnt[inv] == 2)                          # sides of regular edges, paired
    order = reg[np.argsort(inv[reg], kind="stable")]
    a_s, b_s = order[0::2], order[1::2]
    fa, fb, ea = fid[a_s], fid[b_s], key[a_s]
    corner = np.arange(3 * nf).reshape(nf, 3)

    def corner_of(faces_, verts_):
        return corner[faces_, np.argmax(F[faces_] == verts_[:, None], axis=1)]
    rows = np.concatenate([corner_of(fa, ea[:, 0]), corner_of(fa, ea[:, 1])])
    cols = np.concatenate([corner_of(fb, ea[:, 0]), corner_of(fb, ea[:, 1])])
    _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3 * nf, 3 * nf)), directed=False)
    fan_ids, first = np.unique(lab, return_index=True)
    orig = F.ravel()[first]                                      # original vertex of every fan (= new vertex)
    lut = np.empty(lab.max() + 1, dtype=np.int64); lut[fan_ids] = np.arange(len(fan_ids))
    F2 = lut[lab].reshape(nf, 3)
    # largest piece through regular edges (face adjacency)
    fadj = coo_matrix((np.ones(len(fa)), (fa, fb)), shape=(nf, nf))
    _, fl = connected_components(fadj, directed=False)
    fa_area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    areas = np.bincount(fl, weights=fa_area)
    good = np.flatnonzero(areas >= min_fraction * areas.sum())
    fsel = np.isin(fl, good)
    Fk = F2[fsel]
    keep = np.unique(Fk); idx = -np.ones(len(orig), np.int64); idx[keep] = np.arange(len(keep))
    return V[orig[keep]], idx[Fk], orig[keep], fl[fsel]


def conformal_factor(V, F, device="auto", max_iter=60, return_info=False, method="auto"):
    """
    The conformal factor of the uniformization (target 'uniform') per vertex: large on thin parts (necks, stems).
    Computed on the manifold core of the mesh (pinch vertices and non-manifold edges split for the flow only);
    vertices outside it take the value of the nearest core vertex.
    """
    V = np.asarray(V, float)
    Vc, Fc, orig, piece = manifold_core(V, F)
    uc = np.zeros(len(Vc)); used, its, errs, degs = set(), 0, [], 0
    for pc in np.unique(piece):                         # every piece separately (Gauss–Bonnet per piece)
        Fp = Fc[piece == pc]; vp = np.unique(Fp); ip = -np.ones(len(Vc), np.int64); ip[vp] = np.arange(len(vp))
        Vp, Fp = Vc[vp], ip[Fp]
        if method == "linear":
            r = ricci_flow(Vp, Fp, "uniform", max_iter=1, device=device, linear=True); used.add("linear")
        else:
            r = ricci_flow(Vp, Fp, "uniform", max_iter=max_iter, device=device)
            if method == "auto" and not (r["history"] and r["history"][-1] < 1e-4):
                r = ricci_flow(Vp, Fp, "uniform", max_iter=1, device=device, linear=True)
                used.add("linear (flow did not converge)")
            else:
                used.add("ricci_flow")
        uc[vp] = r["u"]; its = max(its, r["iterations"]); errs.append(r["history"][-1] if r["history"] else np.nan)
        degs += r["degenerate"]
    r = {"iterations": its, "history": [max(errs)] if errs else [], "degenerate": degs}
    used = ", ".join(sorted(used))
    u = np.full(len(V), np.nan)
    s = np.zeros(len(V)); c = np.zeros(len(V))
    np.add.at(s, orig, uc); np.add.at(c, orig, 1)
    ok = c > 0; u[ok] = s[ok] / c[ok]
    if (~ok).any():
        from scipy.spatial import cKDTree
        _, j = cKDTree(V[ok]).query(V[~ok]); u[~ok] = u[ok][j]
    info = {"method": used, "iterations": r["iterations"], "error": r["history"][-1] if r["history"] else np.nan,
            "converged": bool(r["history"] and r["history"][-1] < 1e-4) or "linear" in used, "degenerate": r["degenerate"],
            "core_vertices": int(len(Vc)), "split_vertices": int(len(Vc) - len(np.unique(orig)))}
    return (u, info) if return_info else u
