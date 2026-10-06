"""
dyn_common.py
=============
Shared numerical infrastructure for the continuous-description modules of PynamicMesh:

    parametrization.py    explicit surface parametrizations (spherical / SPHARM / manifold harmonics / ARAP-planar)
    trajectories.py       consistent vertex trajectories, Hermite / spline / as-rigid-as-possible interpolation
    dynamic_analysis.py   Hodge decomposition, FTLE, shape space, reduced coordinates, change points, topology

Everything here works on plain (vertices, faces) arrays so that the modules do not depend on pyFM
internals; a pyFM TriMesh (or any object with .vertices / .faces) is accepted through `vf()`.

GPU policy: as in reeb_graph.py, CuPy is used when it is importable (dense linear algebra, SVDs,
basis evaluations); sparse factorizations stay on the CPU (SuperLU) because they are cheap and
CuPy's sparse direct solvers are not always available. `to_gpu` / `to_cpu` move data across.
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.sparse.csgraph import connected_components
from PynamicMesh.utils.plot_style import install as _install_plot_style; _install_plot_style()  # readable plots

try:  # pragma: no cover - depends on the machine
    import cupy as xp
    GPU_AVAILABLE = True
    if __import__("multiprocessing").current_process().name == "MainProcess":            # worker processes import this module again: say it once
        print("[INFO] CuPy detected. Parametrization / trajectory / analysis modules will use the GPU for dense algebra.")
except ImportError:  # pragma: no cover
    xp = np
    GPU_AVAILABLE = False

def to_gpu(arr):
    if GPU_AVAILABLE and isinstance(arr, np.ndarray):
        return xp.asarray(arr)
    return arr


def to_cpu(arr):
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return arr


def device_report():
    return {"cupy": GPU_AVAILABLE}


# ----------------------------------------------------------------------------- #
#  Mesh accessors and I/O
# ----------------------------------------------------------------------------- #

def as_faces(faces):
    """(F,3) int64 array from (F,3) or pyvista flat [3,i,j,k,...] arrays."""
    f = np.asarray(to_cpu(faces))
    if f.ndim == 1:
        if f.size % 4 != 0 or not np.all(f[::4] == 3):
            raise ValueError("Flat face array must be in pyvista [3, i, j, k, ...] format.")
        f = f.reshape(-1, 4)[:, 1:]
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3); got {f.shape}.")
    return f.astype(np.int64)


def vf(mesh):
    """(vertices, faces) as float64/int64 numpy arrays from a pyFM TriMesh, pyvista mesh or a (V, F) tuple."""
    if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        return np.asarray(mesh[0], dtype=np.float64), as_faces(mesh[1])
    for va, fa in (("vertices", "faces"), ("vertlist", "facelist"), ("points", "faces")):
        if hasattr(mesh, va) and hasattr(mesh, fa):
            return np.asarray(getattr(mesh, va), dtype=np.float64), as_faces(getattr(mesh, fa))
    raise TypeError("Unsupported mesh object; expected (V, F), pyFM TriMesh or pyvista PolyData.")


def natural_sort_key(path):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", Path(path).name)]


def frame_number(path):
    m = re.search(r"T(\d+)", Path(path).stem)
    return int(m.group(1)) if m else float("inf")


def list_mesh_files(folder):
    folder = Path(folder)
    return sorted([f for f in folder.iterdir() if f.is_file() and f.suffix in (".obj", ".mat")], key=natural_sort_key)


def default_loader():
    """
    The mesh loader of the project (utils.tools.load_aligned_mesh: .obj / .mat + the rigid alignment used by
    the whole pipeline) when it can be imported, otherwise the generic reader below.  The viewers and the
    stand-alone drivers use it so that they read exactly the frames the pipeline analysed.
    """
    try:
        from PynamicMesh.utils.tools import load_aligned_mesh
        return load_aligned_mesh
    except Exception:  # noqa: BLE001 - pyFM / tools not importable (e.g. tests)
        return read_mesh


def load_frames(folder, loader=None, max_frames=None):
    """
    Loads the ordered mesh sequence of a scene.  `loader(path)` should return an object accepted by
    `vf()`; None -> default_loader() (the pipeline's load_aligned_mesh when available).
    Returns list of dicts {name, V, F, mesh}.
    """
    files = list_mesh_files(folder)
    if max_frames is not None:
        files = files[:max_frames]
    loader = loader or default_loader()
    frames = []
    for fpath in files:
        mesh = loader(fpath)
        V, F = vf(mesh)
        frames.append({"name": fpath.stem, "path": str(fpath), "V": V, "F": F, "mesh": mesh})
    return frames


def read_mesh(path):
    """Generic reader: ASCII .obj, MATLAB .mat (vertex (n,3) + face (m,3) arrays, 1- or 0-based), or anything
    pyvista can read (.ply/.stl/.vtp/...).  Returns (V, F)."""
    path = Path(path)
    if path.suffix.lower() == ".obj":
        try:
            V, F = read_obj(path)
            if len(V) and len(F):
                return V, F
        except ValueError:
            pass                                          # binary / non-standard obj -> pyvista below
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        d = loadmat(str(path))
        arrays = {k: np.asarray(v) for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray)}
        V = F = None
        for k, a in arrays.items():
            a = a.T if (a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3) else a
            if a.ndim == 2 and a.shape[1] == 3:
                if np.issubdtype(a.dtype, np.integer) or (np.all(np.mod(a, 1) == 0) and a.min() >= 0 and k.lower().startswith(("f", "tri", "face"))):
                    F = a.astype(np.int64)
                elif V is None or k.lower().startswith(("v", "x", "vert", "pos")):
                    V = a.astype(np.float64)
        if V is None or F is None:
            raise ValueError(f"{path}: no (n,3) vertex and (m,3) face arrays found in the .mat file")
        if F.min() == 1:
            F = F - 1
        return V, F
    import pyvista as pv
    m = pv.read(str(path)).triangulate()
    return np.asarray(m.points, dtype=np.float64), np.asarray(m.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def read_obj(path):
    V, F = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):          # fan-triangulate polygons
                    F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int64)


# --------------------------------------------------------------------------------------------------------
#  output logging: every stage writes its console output to <result folder>/output_log.txt
# --------------------------------------------------------------------------------------------------------
_LOG_STACK = []


def progress(msg=None, advance=1, total=None):
    """
    Progress hook for the stages: advances the console bar of the active OutputLog by `advance`, optionally
    (re)defines its total and shows `msg` as the bar postfix; `msg` is also written to the log file.  Outside
    an OutputLog context it is a no-op, so the core functions can be called stand-alone.
    """
    if _LOG_STACK:
        _LOG_STACK[-1].progress(msg, advance, total)


class OutputLog:
    """
    Context manager redirecting stdout + stderr (prints, warnings, progress bars of the libraries) of a stage
    into <folder>/output_log.txt.  On the console the stage is represented by ONE tqdm bar (aligned with the
    pipeline's own "processing folder" bar) that advances every time the stage calls progress(); no other line
    is printed.  console=False shows nothing at all.  Nested stages each write their own file and have their
    own bar; the redirection is restored afterwards, also on exceptions.
    """

    def __init__(self, folder, stage, console=True, total=None, show_bar=True):
        self.folder = Path(folder); self.stage = stage; self.console_on = console; self.total = total
        self.show_bar = show_bar                         # the stage bar is shown even when the messages are muted
        self.path = self.folder / "output_log.txt"; self.bar = None

    def __enter__(self):
        import datetime, os, sys
        self.folder.mkdir(parents=True, exist_ok=True)
        self._out, self._err = sys.stdout, sys.stderr
        # the console bar is created BEFORE tqdm is silenced and writes to the real stderr like the pipeline bar
        # (not in a scene-parallel worker: PYNAMIC_NO_BARS, the main process shows one bar per scene)
        if self.show_bar and not os.environ.get("PYNAMIC_NO_BARS"):
            try:
                from tqdm import tqdm as _bar
                self.bar = _bar(total=self.total, desc=f"  {self.stage}", file=self._err, leave=False, dynamic_ncols=True,
                                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}")
            except Exception:  # noqa: BLE001
                self.bar = None
        self._tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:                                              # library progress bars are meaningless in a file: disable them
            import tqdm as _tqdm
            self._tqdm_init = _tqdm.tqdm.__init__

            def _quiet_init(inner, *a, **kw):
                kw["disable"] = True; self._tqdm_init(inner, *a, **kw)
            _tqdm.tqdm.__init__ = _quiet_init
        except Exception:  # noqa: BLE001
            self._tqdm_init = None
        self.fh = open(self.path, "a", encoding="utf-8")
        self.fh.write(f"\n===== {self.stage}  {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        sys.stdout = sys.stderr = self.fh
        _LOG_STACK.append(self)
        return self

    def progress(self, msg=None, advance=1, total=None):
        if msg:
            self.fh.write(f"--- {msg}\n"); self.fh.flush()
        if self.bar is not None:
            if total is not None:
                self.bar.total = total
            if msg:
                self.bar.set_postfix_str(str(msg)[:40], refresh=False)
            self.bar.update(advance)

    def console(self, msg):
        """One line to the real console (through tqdm.write so bars are not broken) and to the log."""
        self.fh.write(msg + "\n")
        if self.console_on:
            try:
                from tqdm import tqdm as _bar
                _bar.write(msg, file=self._err)
            except Exception:  # noqa: BLE001
                self._err.write(msg + "\n")

    def __exit__(self, exc_type, exc, tb):
        import os, sys, traceback
        if exc_type is not None:
            self.fh.write("".join(traceback.format_exception(exc_type, exc, tb)))
        sys.stdout, sys.stderr = self._out, self._err
        if _LOG_STACK and _LOG_STACK[-1] is self:
            _LOG_STACK.pop()
        if getattr(self, "_tqdm_init", None) is not None:
            import tqdm as _tqdm
            _tqdm.tqdm.__init__ = self._tqdm_init
        if self._tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = self._tqdm
        self.fh.close()
        if self.bar is not None:
            if exc_type is None:
                if self.bar.total is not None and self.bar.n < self.bar.total:
                    self.bar.update(self.bar.total - self.bar.n)
                self.bar.set_postfix_str(f"done -> {self.path.name}", refresh=True)
            else:
                self.bar.set_postfix_str(f"FAILED ({exc_type.__name__}) see {self.path}", refresh=True)
            self.bar.close()
        return False


def pbar(iterable=None, total=None, desc=None, **kw):
    """
    Nested progress bar of the pipeline (leave=False, like every inner bar). Inside a logged stage (OutputLog, where
    library bars are muted and the output goes to the log file) it is created with the ORIGINAL tqdm and drawn on the
    real console under the stage bar; elsewhere it is a normal tqdm bar.
    """
    import os
    import tqdm as _tqdm_mod
    kw.setdefault("leave", False); kw.setdefault("dynamic_ncols", True)
    if os.environ.get("PYNAMIC_NO_BARS"):                 # scene-parallel worker: bars off
        return _tqdm_mod.tqdm(iterable, total=total, disable=True, **{k: v for k, v in kw.items() if k != "disable"})
    if desc is not None:
        kw["desc"] = f"    {desc}"
    log = _LOG_STACK[-1] if _LOG_STACK else None
    if log is None:
        return _tqdm_mod.tqdm(iterable, total=total, **kw)
    bar = _tqdm_mod.tqdm.__new__(_tqdm_mod.tqdm)
    init = getattr(log, "_tqdm_init", None) or _tqdm_mod.tqdm.__init__
    init(bar, iterable, total=total, file=log._err, disable=False, **kw)
    return bar


def logged_stage(subfolder, target_arg, stage=None, total=None):
    """
    Decorator: runs the wrapped driver inside OutputLog(<target>/<subfolder>).  `target_arg` is the
    (position, name) of the result-folder argument in the wrapped signature; `total` the number of
    progress() calls the stage makes (the stage may redefine it with progress(total=...)).
    """
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            pos, name = target_arg
            target = kwargs.get(name, args[pos] if len(args) > pos else None)
            if target is None:
                return fn(*args, **kwargs)
            folder = Path(target) / subfolder if subfolder else Path(target)
            # verbose only mutes the console MESSAGES (they always go to the log); the stage bar is always shown, so
            # the user sees which stage is running inside the pipeline's "processing folder" bar
            with OutputLog(folder, stage or fn.__name__, console=kwargs.get("verbose", True) is not False, total=total,
                           show_bar=True):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def write_obj(path, V, F):
    V = np.asarray(to_cpu(V)); F = as_faces(F)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# PynamicMesh export\n")
        for p in V:
            fh.write(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        for t in F + 1:
            fh.write(f"f {t[0]} {t[1]} {t[2]}\n")


def to_pyvista(V, F, point_data=None):
    """pyvista PolyData (lazy import: the core modules must run head-less without pyvista)."""
    import pyvista as pv
    F = as_faces(F)
    faces = np.hstack([np.full((F.shape[0], 1), 3, dtype=np.int64), F]).ravel()
    poly = pv.PolyData(np.asarray(to_cpu(V), dtype=np.float64), faces)
    for k, val in (point_data or {}).items():
        poly.point_data[k] = np.asarray(to_cpu(val))
    return poly


def save_vtp(path, V, F, point_data=None):
    try:
        to_pyvista(V, F, point_data).save(str(path))
        return True
    except Exception as exc:  # noqa: BLE001  (pyvista missing or write failure)
        warnings.warn(f"Could not write {path}: {exc}")
        return False


def load_p2p_chain(matrix_folder, n_frames):
    """
    p2p maps of the functional-map stage: p2p[i][j] = vertex of frame i-1 matched to vertex j of frame i
    (FMV_T{i:04d}_T{i-1:04d}.npy, legacy FMV_{i}{i-1}.npy).  Returns a list with None at index 0.
    Also returns the functional maps C_{i-1 -> i} when available.
    """
    matrix_folder = Path(matrix_folder)
    p2p, fms = [None], [None]
    for i in range(1, n_frames):
        cands = [matrix_folder / f"FMV_T{i:04d}_T{i-1:04d}.npy", matrix_folder / f"FMV_{i}{i-1}.npy"]
        arr = next((np.load(c) for c in cands if c.exists()), None)
        p2p.append(None if arr is None else np.asarray(arr, dtype=np.int64).ravel())
        cands = [matrix_folder / f"FMC_T{i-1:04d}_T{i:04d}.npy", matrix_folder / f"FMC_{i-1}{i}.npy"]
        arr = next((np.load(c) for c in cands if c.exists()), None)
        fms.append(arr)
    return p2p, fms


def ensure_dir(path):
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------------------------- #
#  Discrete differential geometry
# ----------------------------------------------------------------------------- #

def face_normals_areas(V, F):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    dbl = np.linalg.norm(n, axis=1)
    area = 0.5 * dbl
    n = n / np.maximum(dbl, 1e-300)[:, None]
    return n, area


def vertex_areas(V, F):
    """Barycentric (lumped) vertex areas: one third of the incident triangle areas."""
    _, area = face_normals_areas(V, F)
    A = np.zeros(V.shape[0])
    for k in range(3):
        np.add.at(A, F[:, k], area / 3.0)
    return A


def vertex_normals(V, F):
    n, area = face_normals_areas(V, F)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n * area[:, None])
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]


def cotan_weights(V, F, clip_negative=False):
    """
    Edge weights w_ij = ½ (cot α_ij + cot β_ij) (Pinkall–Polthier / Meyer et al.), the weights used in
    the ARAP energy (Sorkine & Alexa 2007, Sec. 2.2).  Returns a symmetric sparse (n, n) matrix.
    """
    n = V.shape[0]
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u = V[i] - V[l]; w = V[j] - V[l]
        cross = np.linalg.norm(np.cross(u, w), axis=1)
        cot = np.einsum("ij,ij->i", u, w) / np.maximum(cross, 1e-300)
        rows.append(i); cols.append(j); vals.append(0.5 * cot)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    Wij = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    Wij = Wij + Wij.T
    if clip_negative:
        Wij.data = np.maximum(Wij.data, 0.0)
    Wij.eliminate_zeros()
    return Wij


def cotan_laplacian(V, F, clip_negative=False):
    """
    Stiffness matrix  W = D - Wij  (symmetric positive semi-definite, the pyFM `mesh.W` convention:
    the discrete Laplace–Beltrami operator is Δ = -M⁻¹ W) and the lumped mass matrix M.
    """
    Wij = cotan_weights(V, F, clip_negative)
    d = np.asarray(Wij.sum(axis=1)).ravel()
    W = sparse.diags(d) - Wij
    M = sparse.diags(vertex_areas(V, F))
    return W.tocsr(), M.tocsr(), Wij


def mesh_edges(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def mesh_topology(V, F):
    """Euler characteristic, genus, boundary loops and connected components of a triangle mesh."""
    E = mesh_edges(F)
    e_all = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e_all, axis=0, return_counts=True)
    n_boundary_edges = int(np.sum(counts == 1))
    n_nonmanifold = int(np.sum(counts > 2))
    adj = sparse.coo_matrix((np.ones(E.shape[0]), (E[:, 0], E[:, 1])), shape=(V.shape[0],) * 2)
    referenced = np.unique(F)                                  # unreferenced (isolated) vertices do not count
    n_comp = int(connected_components(adj, directed=False)[0] - (V.shape[0] - referenced.size))
    chi = referenced.size - E.shape[0] + F.shape[0]
    # boundary loops
    n_loops = 0
    if n_boundary_edges:
        bE = np.unique(e_all, axis=0)[counts == 1]
        badj = sparse.coo_matrix((np.ones(bE.shape[0]), (bE[:, 0], bE[:, 1])), shape=(V.shape[0],) * 2)
        n_loops_all, lab = connected_components(badj, directed=False)
        used = np.unique(bE.ravel())
        n_loops = len(np.unique(lab[used]))
    genus = (2 * n_comp - chi - n_loops) / 2.0
    return {"n_vertices": int(V.shape[0]), "n_faces": int(F.shape[0]), "n_edges": int(E.shape[0]),
            "n_unreferenced_vertices": int(V.shape[0] - referenced.size),
            "euler_characteristic": int(chi), "n_components": int(n_comp), "n_boundary_loops": int(n_loops),
            "n_nonmanifold_edges": n_nonmanifold, "genus": float(genus),
            "closed": n_boundary_edges == 0,
            "topology": ("sphere" if (n_boundary_edges == 0 and chi == 2 and n_comp == 1) else
                         "disk" if (n_loops == 1 and chi == 1 and n_comp == 1) else
                         f"genus-{genus:g}" if n_boundary_edges == 0 else "open")}


def laplacian_eigenbasis(W, M, k, trimesh_obj=None):
    """
    First k Laplace–Beltrami eigenpairs  W φ = λ M φ, M-orthonormal.  Reuses the pyFM basis when a
    processed TriMesh is given (avoids a second eigendecomposition), otherwise shift-invert ARPACK.
    """
    if trimesh_obj is not None and getattr(trimesh_obj, "eigenvalues", None) is not None \
            and len(trimesh_obj.eigenvalues) >= k:
        return (np.asarray(trimesh_obj.eigenvalues[:k], dtype=np.float64),
                np.asarray(trimesh_obj.eigenvectors[:, :k], dtype=np.float64))
    n = W.shape[0]
    k = int(min(k, n - 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evals, evecs = splinalg.eigsh(W.tocsc(), k=k, M=M.tocsc(), sigma=-1e-8, which="LM")
    order = np.argsort(evals)
    evals, evecs = np.maximum(evals[order], 0.0), evecs[:, order]
    # enforce M-orthonormality (ARPACK does it up to round-off)
    nrm = np.sqrt(np.einsum("ij,ij->j", evecs, M @ evecs))
    return evals, evecs / nrm


def face_gradient_operator(V, F):
    """
    Sparse (3F, n) operator G such that (G f).reshape(F,3) is the per-face gradient of a vertex
    function f (constant on each triangle, tangent to it).  Standard formula
        ∇f|_t = 1/(2A_t) Σ_i f_i (n_t × e_i),  e_i = edge opposite to vertex i.
    """
    n, area = face_normals_areas(V, F)
    nf = F.shape[0]
    rows, cols, vals = [], [], []
    for k in range(3):
        i = F[:, k]
        e_opp = V[F[:, (k + 2) % 3]] - V[F[:, (k + 1) % 3]]
        g = np.cross(n, e_opp) / (2.0 * np.maximum(area, 1e-300))[:, None]     # (F,3)
        for c in range(3):
            rows.append(3 * np.arange(nf) + c); cols.append(i); vals.append(g[:, c])
    G = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(3 * nf, V.shape[0])).tocsr()
    return G, n, area


def mean_curvature(V, F, W=None, M=None):
    """Signed mean curvature H (κ1+κ2)/2 from the cotangent formula  H n = ½ M⁻¹ W x."""
    if W is None or M is None:
        W, M, _ = cotan_laplacian(V, F)
    Hn = 0.5 * (W @ V) / np.maximum(M.diagonal(), 1e-300)[:, None]
    N = vertex_normals(V, F)
    return np.einsum("ij,ij->i", Hn, N)


def gaussian_curvature(V, F, boundary_free=True):
    """Angle-defect Gaussian curvature K = (2π − Σθ)/A_i."""
    n = V.shape[0]
    defect = np.full(n, 2.0 * np.pi)
    for k in range(3):
        i, j, l = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u = V[j] - V[i]; w = V[l] - V[i]
        cosang = np.einsum("ij,ij->i", u, w) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300)
        np.add.at(defect, i, -np.arccos(np.clip(cosang, -1.0, 1.0)))
    return defect / np.maximum(vertex_areas(V, F), 1e-300)


def harmonic_inpaint(W, values, known_mask):
    """
    Fills unknown vertex values (rows of `values`, (n,d)) with the discrete harmonic extension of the
    known ones: W_UU x_U = -W_UK x_K.  Used to complete trajectories where the p2p map has gaps.
    """
    values = np.array(values, dtype=np.float64, copy=True)
    unknown = np.flatnonzero(~known_mask)
    known = np.flatnonzero(known_mask)
    if unknown.size == 0:
        return values
    if known.size == 0:
        raise ValueError("harmonic_inpaint: no known values.")
    W = W.tocsr()
    W_UU = W[unknown][:, unknown].tocsc()
    W_UK = W[unknown][:, known]
    rhs = -W_UK @ values[known]
    eps = 1e-9 * (abs(W_UU.diagonal()).mean() + 1e-12)
    sol = splinalg.spsolve(W_UU + eps * sparse.identity(W_UU.shape[0], format="csc"), rhs)
    values[unknown] = np.asarray(sol).reshape(len(unknown), -1)
    return values


def kabsch_rotation(P, Q, weights=None):
    """Rotation R (det=+1) minimising Σ w ||R p − q||² (Horn 1987 / Kabsch), inputs (n,3) already centred."""
    w = np.ones(P.shape[0]) if weights is None else np.asarray(weights)
    H = (P * w[:, None]).T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    return Vt.T @ D @ U.T


def icosphere(subdivisions=4):
    """Unit icosphere (template for spherical-harmonic synthesis)."""
    t = (1.0 + 5 ** 0.5) / 2.0
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    V /= np.linalg.norm(V, axis=1)[:, None]
    for _ in range(subdivisions):
        cache = {}
        newF = []
        Vl = V.tolist()

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = np.array(Vl[a]) + np.array(Vl[b]); p /= np.linalg.norm(p)
                Vl.append(p.tolist()); cache[key] = len(Vl) - 1
            return cache[key]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = np.asarray(Vl); F = np.asarray(newF, dtype=np.int64)
    return V, F


# ----------------------------------------------------------------------------- #
#  Small numerical helpers shared by the time-series modules
# ----------------------------------------------------------------------------- #

def smooth_derivative(Z, t, window=None, polyorder=3):
    """
    Time derivative of a (T, d) series on a uniform grid with a Savitzky–Golay filter (falls back to
    central differences for very short series).  Robust to the frame-to-frame noise of the p2p maps.
    """
    Z = np.asarray(Z, dtype=np.float64); T = Z.shape[0]
    t = np.asarray(t, dtype=np.float64)
    dt = float(np.mean(np.diff(t)))
    steps = np.diff(t)
    if T >= 3 and np.ptp(steps) > 1e-6 * max(abs(dt), 1e-300):    # irregular frame times: the filter needs a
        return np.gradient(Z, t, axis=0)                           # uniform grid, central differences do not
    if T < 5:
        return np.gradient(Z, dt, axis=0)
    from scipy.signal import savgol_filter
    if window is None:
        window = min(T if T % 2 == 1 else T - 1, 7)
    window = max(window, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
    window = min(window, T if T % 2 == 1 else T - 1)
    return savgol_filter(Z, window_length=window, polyorder=min(polyorder, window - 1), deriv=1, delta=dt, axis=0)


# --------------------------------------------------------------------------- #
#  Frame times (time between the frames of the time-lapse)
# --------------------------------------------------------------------------- #

FRAME_TIMES_FILE = "frame_times.csv"
TIME_FILE_CANDIDATES = ("frame_times.csv", "frame_times.txt", "times.csv", "times.txt", "frame_times.json")


class FrameTimes:
    """
    Acquisition times of the frames of a sequence.
      t       (n,) time of every frame (starts at 0 unless absolute times were given)
      dt      (n-1,) interval of every transition T(i-1) -> T(i)
      known   True when the times were given (interval, per-frame times or a times file); False = the old
              behaviour (t = k * legacy_dt, unit 'frame')
      unit    time unit of t (e.g. 's', 'min')
      uniform True when all intervals are equal
      source  where the times came from
    """

    def __init__(self, t, known, unit="frame", source="frame index"):
        self.t = np.asarray(t, dtype=np.float64)
        self.dt = np.diff(self.t)
        self.known = bool(known); self.unit = str(unit); self.source = str(source)
        span = float(np.ptp(self.dt)) if self.dt.size else 0.0
        self.uniform = bool(self.dt.size == 0 or span <= 1e-9 * max(float(np.abs(self.dt).max()), 1e-300))

    def __len__(self):
        return int(self.t.size)

    @property
    def mean_dt(self):
        return float(np.mean(self.dt)) if self.dt.size else 1.0

    def interval(self, i):
        """Interval of the transition T(i-1) -> T(i) (i >= 1)."""
        return float(self.dt[i - 1])

    def label(self, k):
        return f"t = {self.t[k]:.4g} {self.unit}"

    def to_frame(self, names=None):
        import pandas as pd
        return pd.DataFrame({"frame": np.arange(len(self)), "file": list(names) if names is not None else [""] * len(self),
                             "time": self.t, "dt_from_previous": np.r_[np.nan, self.dt], "unit": self.unit,
                             "source": self.source})

    def save(self, folder, names=None):
        """Writes <folder>/frame_times.csv (read back by every stage and viewer through find_frame_times)."""
        path = Path(folder) / FRAME_TIMES_FILE
        self.to_frame(names).to_csv(path, index=False)
        return path


def _read_time_file(path, names=None):
    """Times from a csv (column 'time', optional 'file' to match the mesh names, optional 'unit'), a txt file with one
    number per line (or 'name time' pairs) or a json list. Returns (times, unit or None)."""
    import json as _json
    import pandas as pd
    path = Path(path)
    unit = None
    if path.suffix.lower() == ".json":
        data = _json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            unit = data.get("unit"); data = data.get("times", data.get("time"))
        return np.asarray(data, dtype=np.float64), unit
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}
        tcol = cols.get("time") or cols.get("t") or cols.get("seconds") or df.columns[-1]
        if "unit" in cols and df[cols["unit"]].notna().any():
            unit = str(df[cols["unit"]].dropna().iloc[0])
        if names is not None and "file" in cols and df[cols["file"]].notna().all() and set(map(str, names)) <= set(map(str, df[cols["file"]])):
            lut = dict(zip(df[cols["file"]].astype(str), df[tcol].astype(float)))
            return np.array([lut[str(n)] for n in names], dtype=np.float64), unit
        return df[tcol].to_numpy(dtype=np.float64), unit
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            vals.append(float(line.replace(",", " ").split()[-1]))
    return np.asarray(vals, dtype=np.float64), unit


def resolve_frame_times(n_frames, frame_interval=None, frame_times=None, time_unit="s", scene_folder=None,
                        auto_detect=True, names=None, legacy_dt=1.0, verbose=True):
    """
    Times of the n_frames frames of a scene. Priority:
      1. frame_times  : per-frame acquisition times (list / array) or a file (csv / txt / json, relative paths are
                        looked up in the scene folder) - irregular intervals are allowed;
      2. a times file found in the scene folder (frame_times.csv, frame_times.txt, times.csv, times.txt,
         frame_times.json) when auto_detect;
      3. frame_interval: one interval between consecutive frames (uniform time-lapse);
      4. unknown      : t = k * legacy_dt (the previous behaviour, unit 'frame'), FrameTimes.known = False.
    Times must be strictly increasing and one per frame; otherwise they are ignored with a warning.
    """
    n = int(n_frames)

    def _check(t, src):
        t = np.asarray(t, dtype=np.float64).ravel()
        if t.size != n:
            if verbose:
                print(f"[Frame times] {src}: {t.size} times for {n} frames - ignored.")
            return None
        if n > 1 and not np.all(np.diff(t) > 0):
            if verbose:
                print(f"[Frame times] {src}: the times are not strictly increasing - ignored.")
            return None
        return t
    candidates = []
    if frame_times is not None:
        if isinstance(frame_times, (str, Path)):
            fp = Path(frame_times)
            if not fp.is_absolute() and scene_folder is not None and (Path(scene_folder) / fp).exists():
                fp = Path(scene_folder) / fp
            if fp.exists():
                candidates.append(("file " + str(fp), fp))
            elif verbose:
                print(f"[Frame times] file {frame_times} not found - ignored.")
        else:
            candidates.append(("frame_times", np.asarray(frame_times, dtype=np.float64)))
    if auto_detect and scene_folder is not None:
        for name in TIME_FILE_CANDIDATES:
            fp = Path(scene_folder) / name
            if fp.exists():
                candidates.append(("file " + str(fp), fp)); break
    for src, val in candidates:
        unit = time_unit
        if isinstance(val, Path):
            try:
                val, u = _read_time_file(val, names)
                unit = u or time_unit
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[Frame times] could not read {src}: {exc} - ignored.")
                continue
        t = _check(val, src)
        if t is not None:
            return FrameTimes(t - t[0], True, unit, src)
    if frame_interval is not None and float(frame_interval) > 0:
        return FrameTimes(np.arange(n, dtype=np.float64) * float(frame_interval), True, time_unit,
                          f"frame_interval = {float(frame_interval):g} {time_unit}")
    return FrameTimes(np.arange(n, dtype=np.float64) * float(legacy_dt), False, "frame", "frame index (times unknown)")


def find_frame_times(path, levels=4):
    """FrameTimes from the frame_times.csv written by the pipeline in Results/<scene>/ (searched in `path` and up to
    `levels` parent folders), or None when the times of the scene are unknown."""
    import pandas as pd
    p = Path(path)
    for _ in range(levels + 1):
        f = p / FRAME_TIMES_FILE
        if f.exists():
            try:
                df = pd.read_csv(f)
                unit = str(df["unit"].iloc[0]) if "unit" in df else "s"
                return FrameTimes(df["time"].to_numpy(dtype=np.float64), True, unit,
                                  str(df["source"].iloc[0]) if "source" in df else str(f))
            except Exception:  # noqa: BLE001
                return None
        if p.parent == p:
            break
        p = p.parent
    return None


def add_time_columns(df, folder, frame_col="Frame", prev_col=None, rate_cols=(), rate_suffix="_rate"):
    """
    Adds Time (time of `frame_col`), Dt (interval since `prev_col`, default frame - 1), Time_Unit and the rates
    <col>_rate = col / Dt of `rate_cols` to a per-frame / per-transition table - ONLY when the frame times of the scene
    are known (frame_times.csv found from `folder`); otherwise the table is returned unchanged.
    """
    ft = find_frame_times(folder)
    if ft is None or df is None or len(df) == 0 or frame_col not in df:
        return df
    fr = df[frame_col].to_numpy(dtype=np.int64)
    if fr.min() < 0 or fr.max() >= len(ft):
        return df
    df = df.copy()
    df["Time"] = ft.t[fr]
    if prev_col is not None or rate_cols:
        prev = df[prev_col].to_numpy(dtype=np.int64) if prev_col is not None and prev_col in df else fr - 1
        ok = prev >= 0
        dtv = np.full(len(df), np.nan); dtv[ok] = ft.t[fr[ok]] - ft.t[prev[ok]]
        df["Dt"] = dtv
        for c in rate_cols:
            if c in df:
                df[f"{c}{rate_suffix}"] = df[c].astype(float) / np.where(dtv > 0, dtv, np.nan)
    df["Time_Unit"] = ft.unit
    return df


def time_axis(df, default_col, default_label):
    """(x values, axis label): the acquisition time when the table has a Time column, else the default column."""
    if df is not None and "Time" in df:
        unit = df["Time_Unit"].iloc[0] if "Time_Unit" in df else "s"
        return df["Time"], f"time ({unit})"
    return df[default_col], default_label
