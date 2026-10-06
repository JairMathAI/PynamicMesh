"""
accel.py  —  compute devices, multi-GPU nearest neighbours and parallel workers
================================================================================

Where the time of the pipeline goes (profiled) and what this module does about it:

  * spectral nearest neighbours (functional map -> point-to-point map, pyFM ICP / ZoomOut refinement, Morse–Smale
    region transport): brute force O(n1 n2 k), CPU, single threaded  ->  knn(): exact k-NN on the GPU(s), queries split
    over ALL selected GPUs (one thread per device). Candidates are scored with float32 matrix products and the best
    candidates re-ranked in float64, so the matches are those of the exact float64 search. CPU fallback (cKDTree /
    sklearn) without GPU.
  * Laplace–Beltrami eigen-decompositions (one per mesh, scipy ARPACK shift-invert, CPU only - the GPU eigensolvers
    have no shift-invert) and the graph edit distances (combinatorial, CPU only): independent per frame / transition
    ->  prefetch() / parallel_map(): process pools, the results consumed in order. Workers are pinned round-robin to
    the selected GPUs, so per-frame GPU work of the workers also spreads over several devices.

Configuration (run_pipeline(compute_devices=..., n_workers=...) or configure()):
    compute_devices : 'all' (default: every visible CUDA device) | 'cpu' | a list / comma string of device ids
    n_workers       : processes for the per-frame work (default: CPU cores - 1, at most 8; 1 = sequential)
The environment variables PYNAMIC_DEVICES / PYNAMIC_WORKERS set the same options.
"""
import contextlib
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

_STATE = {"devices": None, "workers": None, "pyfm_backend": False}
KNN_GPU_MIN_WORK = 2e6           # n_keys x n_queries below which the CPU search is faster than the transfer
KNN_SPLIT_MIN_WORK = 2e9         # n_keys x n_queries per GPU below which ONE GPU is used: splitting a small search over
                                 # several GPUs only creates CUDA contexts / memory pools on the idle devices


def _cupy():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:  # noqa: BLE001
        return None


def available_devices():
    """Ids of the visible CUDA devices ([] without CuPy / GPU)."""
    cp = _cupy()
    if cp is None:
        return []
    try:
        return list(range(cp.cuda.runtime.getDeviceCount()))
    except Exception:  # noqa: BLE001
        return []


def configure(compute_devices=None, n_workers=None, verbose=False):
    """Selects the GPUs and the number of worker processes (see module docstring)."""
    req = os.environ.get("PYNAMIC_DEVICES") if os.environ.get("PYNAMIC_NO_BARS") and os.environ.get("PYNAMIC_DEVICES") \
        else (compute_devices if compute_devices is not None else os.environ.get("PYNAMIC_DEVICES", "all"))
    avail = available_devices()
    if isinstance(req, str):
        r = req.strip().lower()
        if r in ("cpu", "none", ""):
            devs = []
        elif r in ("all", "auto", "gpu"):
            devs = avail
        else:
            devs = [int(x) for x in r.replace(";", ",").split(",") if x.strip() != ""]
    elif isinstance(req, int):
        devs = [req]
    else:
        devs = [int(x) for x in req]
    devs = [d for d in devs if d in avail]
    _STATE["devices"] = devs
    w = os.environ.get("PYNAMIC_WORKERS") if os.environ.get("PYNAMIC_NO_BARS") and os.environ.get("PYNAMIC_WORKERS") \
        else (n_workers if n_workers is not None else os.environ.get("PYNAMIC_WORKERS"))
    if w is None:
        w = max(1, min(8, (os.cpu_count() or 2) - 1))
    _STATE["workers"] = max(1, int(w))
    if devs:
        install_pyfm_backend()
    if verbose:
        print(f"[Compute] GPUs: {devs if devs else 'none (CPU)'} | worker processes: {_STATE['workers']}")
    return list(devs), _STATE["workers"]


def devices():
    if _STATE["devices"] is None:
        configure()
    return list(_STATE["devices"])


def n_workers():
    if _STATE["workers"] is None:
        configure()
    return _STATE["workers"]


# --------------------------------------------------------------------------------------------------------------- #
#  Exact k nearest neighbours on one or several GPUs
# --------------------------------------------------------------------------------------------------------------- #
def _knn_on_device(dev, X, Y, k, m, out_idx, out_dst, sl):
    import cupy as cp
    with cp.cuda.Device(dev):
        Xg = cp.asarray(X, dtype=cp.float32); Xg64 = cp.asarray(X, dtype=cp.float64)
        xx = cp.sum(Xg * Xg, axis=1)
        Yc = Y[sl]
        free = cp.cuda.runtime.memGetInfo()[0]
        rows = int(max(256, min(len(Yc), 0.25 * free / max(4 * X.shape[0], 1))))   # chunk rows that fit in memory
        for s in range(0, len(Yc), rows):
            yb64 = cp.asarray(Yc[s:s + rows], dtype=cp.float64); yb = yb64.astype(cp.float32)
            d = xx[None, :] - 2.0 * (yb @ Xg.T)                                    # |x|^2 - 2 x.y  (|y|^2 constant)
            mm = min(m, X.shape[0])
            cand = cp.argpartition(d, mm - 1, axis=1)[:, :mm] if mm < X.shape[0] else cp.tile(cp.arange(X.shape[0]), (len(yb), 1))
            diff = Xg64[cand] - yb64[:, None, :]                                    # exact float64 re-ranking
            d64 = cp.sum(diff * diff, axis=2)
            order = cp.argsort(d64, axis=1)[:, :k]
            idx = cp.take_along_axis(cand, order, axis=1)
            dst = cp.sqrt(cp.take_along_axis(d64, order, axis=1))
            a, b = sl.start + s, sl.start + s + len(yb)
            out_idx[a:b] = cp.asnumpy(idx); out_dst[a:b] = cp.asnumpy(dst)


def knn(X, Y, k=1, return_distance=False, force_cpu=False):
    """
    Exact k nearest neighbours in X of every row of Y (Euclidean). GPU(s) when available and the problem is large
    enough, else scipy cKDTree. Returns indices (n2,) for k = 1 or (n2, k), and the distances when requested.
    """
    X = np.ascontiguousarray(X, dtype=np.float64); Y = np.ascontiguousarray(Y, dtype=np.float64)
    devs = [] if force_cpu else devices()
    if not devs or X.shape[0] * Y.shape[0] < KNN_GPU_MIN_WORK:
        from scipy.spatial import cKDTree
        dst, idx = cKDTree(X).query(Y, k=k)
    else:
        m = k + 16                                   # float32 candidates kept for the exact float64 re-ranking
        idx = np.empty((len(Y), k), dtype=np.int64); dst = np.empty((len(Y), k), dtype=np.float64)
        n_use = int(max(1, min(len(devs), (X.shape[0] * Y.shape[0]) // KNN_SPLIT_MIN_WORK)))
        devs = devs[:n_use]
        parts = np.array_split(np.arange(len(Y)), len(devs))
        sls = [slice(int(p[0]), int(p[-1]) + 1) for p in parts if len(p)]
        if len(sls) == 1:
            _knn_on_device(devs[0], X, Y, k, m, idx, dst, sls[0])
        else:                                        # one thread per GPU (CuPy releases the GIL in its kernels)
            with ThreadPoolExecutor(len(sls)) as ex:
                list(ex.map(lambda a: _knn_on_device(a[0], X, Y, k, m, idx, dst, a[1]), zip(devs, sls)))
            _release_pools(devs[1:])                 # no cached memory left on the secondary GPUs
        if k == 1:
            idx, dst = idx[:, 0], dst[:, 0]
    idx = np.asarray(idx, dtype=np.int64)
    return (np.asarray(dst), idx) if return_distance else idx


def _release_pools(devs):
    """Frees the CuPy memory pools of the given devices (the cached blocks otherwise stay reserved)."""
    cp = _cupy()
    if cp is None:
        return
    for d in devs:
        try:
            with cp.cuda.Device(d):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001
            pass


def install_pyfm_backend():
    """pyFM's brute-force nearest neighbours (ICP / ZoomOut refinement, FM -> p2p) on the GPU(s) via knn()."""
    if _STATE["pyfm_backend"]:
        return True
    try:
        import pyFM.spectral.nearest_neighbor.dispatch as D
        cpu_brute = D._BACKENDS["brute"]

        def gpu_brute(X, Y, k=1, return_distance=False, n_jobs=None, working_memory=None, **kw):
            if not devices() or np.shape(X)[0] * np.shape(Y)[0] < KNN_GPU_MIN_WORK:
                return cpu_brute(X, Y, k=k, return_distance=return_distance, n_jobs=n_jobs,
                                 working_memory=working_memory, **kw)
            return knn(X, Y, k=k, return_distance=return_distance)
        D._BACKENDS["brute"] = gpu_brute
        _STATE["pyfm_backend"] = True
        return True
    except Exception:  # noqa: BLE001 - other pyFM versions: the CPU search is kept
        return False


# --------------------------------------------------------------------------------------------------------------- #
#  Worker processes (pinned round-robin to the GPUs)
# --------------------------------------------------------------------------------------------------------------- #
_counter = None


def _physical(logical):
    """Physical id of a logical CUDA device: inside a process that already sees only some GPUs (CUDA_VISIBLE_DEVICES,
    e.g. a scene worker pinned to GPU 5), logical device 0 is that GPU and NOT the physical GPU 0."""
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip() != ""]
    return vis[int(logical)] if vis and int(logical) < len(vis) else str(logical)


def init_scene_worker(devs, counter, workers_per_scene):
    """Scene-parallel mode: one process per scene, pinned round-robin to ONE GPU (all its GPU work then runs on that
    device), with its share of the CPU worker processes; nested progress bars off (one log file per scene)."""
    with counter.get_lock():
        slot = counter.value; counter.value += 1
    if devs:
        os.environ["CUDA_VISIBLE_DEVICES"] = _physical(devs[slot % len(devs)])
        os.environ["PYNAMIC_DEVICES"] = "0"
    else:
        os.environ["PYNAMIC_DEVICES"] = "cpu"
    os.environ["PYNAMIC_WORKERS"] = str(max(1, int(workers_per_scene)))
    os.environ["PYNAMIC_NO_BARS"] = "1"


def _init_worker(devs, counter):
    """Pins the worker to one GPU (round-robin) BEFORE CuPy is imported in it; keeps BLAS single threaded so that the
    workers do not oversubscribe the cores."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    os.environ["PYNAMIC_QUIET"] = "1"
    if devs:
        with counter.get_lock():
            slot = counter.value; counter.value += 1
        os.environ["CUDA_VISIBLE_DEVICES"] = _physical(devs[slot % len(devs)])
        os.environ["PYNAMIC_DEVICES"] = "0"           # the pinned device is device 0 inside the worker
    else:
        os.environ["PYNAMIC_DEVICES"] = "cpu"


@contextlib.contextmanager
def _no_main_reimport():
    """
    'spawn' workers normally re-import the main script; execution scripts that call run_pipeline at top level (no
    `if __name__ == "__main__":` guard, e.g. execution_pipeline.py) would then start the pipeline again in every worker.
    The worker tasks live in importable modules (this one), so the main module is not needed: it is dropped from the
    spawn preparation data while the pool is alive.
    """
    import multiprocessing.spawn as sp
    orig = sp.get_preparation_data

    def prep(name):
        d = orig(name)
        d.pop("init_main_from_path", None); d.pop("init_main_from_name", None)
        return d
    sp.get_preparation_data = prep
    try:
        yield
    finally:
        sp.get_preparation_data = orig


_LIVE_POOLS = set()


def _shutdown_pool(ex, cancel=False):
    """Stops a pool and its worker processes (idempotent)."""
    try:
        ex.shutdown(wait=True, cancel_futures=cancel)
    except Exception:  # noqa: BLE001
        pass
    _LIVE_POOLS.discard(ex)


@__import__("atexit").register
def _shutdown_all_pools():
    """Safety net at interpreter exit: pools still alive (e.g. after an error) are cancelled and their worker
    processes terminated, so the program can never hang waiting for idle workers."""
    for ex in list(_LIVE_POOLS):
        try:
            ex.shutdown(wait=False, cancel_futures=True)
            for proc in list(getattr(ex, "_processes", {}).values()):
                if proc.is_alive():
                    proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        _LIVE_POOLS.discard(ex)


def scene_pool(n_slots, workers_per_scene):
    """Process pool of the scene-parallel mode (spawn; see init_scene_worker)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    ex = ProcessPoolExecutor(max_workers=n_slots, mp_context=ctx, initializer=init_scene_worker,
                             initargs=(devices(), counter, workers_per_scene))
    _LIVE_POOLS.add(ex)
    return ex


def _pool(workers):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")                     # CUDA-safe on every platform (fork + CUDA is not)
    counter = ctx.Value("i", 0)
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(devices(), counter))
    _LIVE_POOLS.add(ex)
    return ex


def _bar(total, desc):
    """Nested progress bar (leave=False, like the other bars of the pipeline); None without a description."""
    if not desc or os.environ.get("PYNAMIC_NO_BARS"):
        return None
    from tqdm.auto import tqdm
    return tqdm(total=total, desc=desc, leave=False)


def parallel_map(fn, items, workers=None, desc=None):
    """[fn(x) for x in items] in worker processes (in order); sequential when 1 worker or few items. `desc` shows a
    progress bar (results counted as they arrive; the first ones also wait for the workers to start)."""
    items = list(items)
    w = min(workers or n_workers(), len(items))
    bar = _bar(len(items), desc)
    try:
        if w <= 1:
            out = []
            for x in items:
                out.append(fn(x))
                if bar is not None:
                    bar.update(1)
            return out
        with _no_main_reimport():
            ex = _pool(w)
            try:
                out = []
                for r in ex.map(fn, items):
                    out.append(r)
                    if bar is not None:
                        bar.update(1)
                return out
            finally:
                _shutdown_pool(ex)
    finally:
        if bar is not None:
            bar.close()


def prefetch(fn, items, workers=None, ahead=None):
    """Generator of fn(x) for x in items, IN ORDER, computed ahead by worker processes (at most `ahead` results in
    flight, so memory stays bounded for long sequences); sequential when 1 worker."""
    items = list(items)
    w = min(workers or n_workers(), len(items))
    if w <= 1:
        for x in items:
            yield fn(x)
        return
    ahead = ahead or 2 * w
    # The pool is shut down BEFORE the last result is handed over: a consumer that takes exactly len(items) results
    # never resumes the generator, and a pool left open keeps its idle workers alive (the program then never ends).
    # Closing the generator early (error in the consumer, stream.close()) cancels the pending work.
    patch = _no_main_reimport(); patch.__enter__()
    ex = _pool(w)
    done = False
    try:
        futs = {}
        nxt = 0
        for i in range(len(items)):
            while nxt < len(items) and nxt < i + ahead:
                futs[nxt] = ex.submit(fn, items[nxt]); nxt += 1
            res = futs.pop(i).result()
            if i == len(items) - 1:
                _shutdown_pool(ex); patch.__exit__(None, None, None); done = True
            yield res
    finally:
        if not done:
            _shutdown_pool(ex, cancel=True); patch.__exit__(None, None, None)


# --------------------------------------------------------------------------------------------------------------- #
#  Tasks executed by the workers (top level: picklable)
# --------------------------------------------------------------------------------------------------------------- #
def load_and_process_mesh(args):
    """(path, k, needs_spectrum) -> pyFM TriMesh loaded with tools.load_aligned_mesh and processed (k eigenpairs)."""
    path, k, needs = args
    from PynamicMesh.utils.tools import load_aligned_mesh
    m = load_aligned_mesh(Path(path))
    if needs:
        m.process(k=k)
    return m


def graph_edit_distance_task(args):
    """(prev_pickle, curr_pickle, timeout) -> networkx graph edit distance (best path found within the timeout)."""
    import pickle
    import networkx as nx
    a, b, timeout = args
    with open(a, "rb") as fa, open(b, "rb") as fb:
        G1, G2 = pickle.load(fa), pickle.load(fb)
    return nx.graph_edit_distance(G1, G2, timeout=timeout)


# --------------------------------------------------------------------------------------------------------------- #
#  Memory: what is available, and releasing it between frames
# --------------------------------------------------------------------------------------------------------------- #
def available_memory():
    """Free memory in bytes: {'cpu': available RAM, 'gpu': free memory of the current GPU (+ blocks cached by CuPy's
    pool, which are reusable) or None without GPU}."""
    cpu = None
    try:
        import psutil
        cpu = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        cpu = int(line.split()[1]) * 1024
                        break
        except Exception:  # noqa: BLE001
            cpu = None
    if cpu is None:
        try:
            cpu = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except Exception:  # noqa: BLE001
            cpu = 4 * 1024 ** 3                          # unknown: assume 4 GB
    gpu = None
    if devices():
        cp = _cupy()
        try:
            free, _ = cp.cuda.runtime.memGetInfo()
            gpu = int(free + cp.get_default_memory_pool().free_bytes())
        except Exception:  # noqa: BLE001
            gpu = None
    return {"cpu": cpu, "gpu": gpu}


def free_memory():
    """Releases the memory of the previous computation: Python garbage and the blocks cached by CuPy's memory pools
    (otherwise kept by the pool after the arrays are deleted, so the next frame would start with less free memory)."""
    import gc
    gc.collect()
    cp = _cupy()
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001
            pass
