"""
synthetic_division.py — a synthetic cell that divides and fuses again, to try the Topology Events stage
=======================================================================================================

Sequence (one OBJ per frame, every frame re-meshed with its own triangulation, like segmented data):

  1. a round cell with a rippled surface and a few protrusions,
  2. that elongates into an ellipsoid,
  3. constricts: a neck forms and thins slowly over many frames,
  4. pinches and splits into two daughter cells,
  5. the daughters drift apart and come back,
  6. touch, and fuse again through a bridge that widens over several frames,
  7. and round up into one cell.

The shape is an implicit surface: a smooth union of two lobes (smooth minimum of their distance functions), whose
centre distance and blending width vary smoothly in time — so the neck thins and re-forms gradually, never as a jump.
A smooth time-varying ripple deforms the surface and localized protrusions, attached to each lobe, grow and retract.
Every frame is extracted with marching cubes, lightly smoothed (Taubin) and cleaned of tiny fragments.

Usage:
    python synthetic_division.py                      # writes ./synthetic_division/cell/cell_000.obj ...
    python synthetic_division.py --out data --frames 80 --seed 3 --noise 0.06

Requirements: numpy, scikit-image, trimesh (all installed with PynamicMesh).
"""
import argparse
from pathlib import Path

import numpy as np
from skimage import measure
import trimesh


def smooth_min(a, b, k):
    """Polynomial smooth minimum (k = blending width; k -> 0 is the plain union)."""
    k = max(float(k), 1e-4)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def schedule(n_frames):
    """Per frame: half distance between the lobe centres s, blending width k, lobe radii (r1, r2), phase name."""
    # fractions of the sequence for every phase (the split and the fusion get many frames)
    phases = [("round", 0.08), ("elongation", 0.12), ("constriction", 0.22), ("separated", 0.20),
              ("fusion", 0.22), ("rounding", 0.16)]
    counts = np.maximum(2, np.round(np.array([p[1] for p in phases]) * n_frames).astype(int))
    counts[-1] += n_frames - counts.sum()
    rows = []
    # key values at the phase boundaries: (s, k)
    keys = [(0.00, 1.10), (0.00, 1.10), (0.55, 1.00), (1.12, 0.06), (1.12, 0.06), (0.55, 0.90), (0.00, 1.10)]
    sep_far = 1.45                                   # largest separation while the daughters are apart
    for pi, ((name, _), c) in enumerate(zip(phases, counts)):
        (s0, k0), (s1, k1) = keys[pi], keys[pi + 1]
        for j in range(c):
            a = j / max(c - 1, 1) if pi < len(phases) - 1 else j / max(c - 1, 1)
            e = 0.5 - 0.5 * np.cos(np.pi * a)        # ease in / out: no jumps between phases
            if name == "separated":                  # drift apart and come back
                s = keys[3][0] + (sep_far - keys[3][0]) * np.sin(np.pi * a)
                k = keys[3][1]
            else:
                s = s0 + (s1 - s0) * e
                k = k0 + (k1 - k0) * e
            rows.append((s, k, name))
    return rows[:n_frames]


def make_noise(rng, n_waves=6, n_bumps=5):
    """Random smooth ripple (sum of 3D waves) and protrusions (Gaussian bumps on directions of each lobe)."""
    waves = [(rng.normal(size=3) * rng.uniform(2.5, 4.5), rng.uniform(0, 2 * np.pi), rng.uniform(0.2, 0.6))
             for _ in range(n_waves)]
    bumps = []
    for lobe in (0, 1):
        for _ in range(n_bumps):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            bumps.append({"lobe": lobe, "dir": d, "height": rng.uniform(0.18, 0.38), "width": rng.uniform(0.18, 0.28),
                          "period": rng.uniform(14, 30), "phase": rng.uniform(0, 2 * np.pi)})
    return waves, bumps


def field(X, Y, Z, t, s, k, r, waves, bumps, noise_amp):
    """Implicit function of frame t (negative inside)."""
    c1 = np.array([-s, 0.0, 0.0]); c2 = np.array([s * 0.95, 0.0, 0.0])
    d1 = np.sqrt((X - c1[0]) ** 2 + Y ** 2 + Z ** 2) - r[0]
    d2 = np.sqrt((X - c2[0]) ** 2 + Y ** 2 + Z ** 2) - r[1]
    f = smooth_min(d1, d2, k)
    # ripple: smooth deformation of the whole surface, slowly changing in time
    rip = np.zeros_like(X)
    for w, ph, speed in waves:
        rip += np.sin(w[0] * X + w[1] * Y + w[2] * Z + ph + speed * t)
    f += noise_amp * rip / np.sqrt(len(waves))
    # protrusions: bumps attached to each lobe, growing and retracting
    for b in bumps:
        c = c1 if b["lobe"] == 0 else c2
        rr = r[b["lobe"]]
        p = c + b["dir"] * rr
        h = b["height"] * max(0.0, np.sin(2 * np.pi * t / b["period"] + b["phase"])) ** 2
        if h > 1e-3:
            f -= h * np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2) / b["width"] ** 2)
    return f


def frame_mesh(f, grid, spacing, min_fraction=0.01, smooth_iter=6):
    V, F, _, _ = measure.marching_cubes(f, 0.0, spacing=(spacing,) * 3)
    V += np.array([g[0] for g in grid])
    m = trimesh.Trimesh(V, F[:, ::-1], process=True)          # outward normals
    parts = m.split(only_watertight=False)
    if len(parts) > 1:                                        # drop tiny fragments created by the noise
        areas = np.array([p.area for p in parts])
        m = trimesh.util.concatenate([p for p, a in zip(parts, areas) if a >= min_fraction * areas.sum()])
    trimesh.smoothing.filter_taubin(m, iterations=smooth_iter)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synthetic_division", help="output folder (contains the scene folder 'cell')")
    ap.add_argument("--frames", type=int, default=60, help="number of frames")
    ap.add_argument("--seed", type=int, default=7, help="random seed of the ripple and protrusions")
    ap.add_argument("--noise", type=float, default=0.05, help="amplitude of the surface ripple")
    ap.add_argument("--spacing", type=float, default=0.055, help="marching-cubes voxel size (smaller = finer mesh)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    waves, bumps = make_noise(rng)
    r = (1.0, 0.85)                                           # slightly asymmetric division
    xs = np.arange(-3.4, 3.4 + 1e-9, args.spacing)
    yz = np.arange(-1.9, 1.9 + 1e-9, args.spacing)
    X, Y, Z = np.meshgrid(xs, yz, yz, indexing="ij")
    scene = Path(args.out) / "cell"; scene.mkdir(parents=True, exist_ok=True)
    rows = schedule(args.frames)
    log = []
    for t, (s, k, phase) in enumerate(rows):
        f = field(X, Y, Z, t, s, k, r, waves, bumps, args.noise)
        m = frame_mesh(f, (xs, yz, yz), args.spacing)
        m.export(scene / f"cell_{t:03d}.obj")
        n_pieces = len(m.split(only_watertight=False))
        log.append((t, phase, s, k, len(m.vertices), n_pieces))
        print(f"frame {t:3d}  {phase:12s}  s={s:.2f}  k={k:.2f}  vertices={len(m.vertices):6d}  pieces={n_pieces}")
    with open(Path(args.out) / "schedule.csv", "w") as fh:
        fh.write("frame,phase,half_distance,blend_width,n_vertices,pieces\n")
        fh.writelines(f"{a},{b},{c:.4f},{d:.4f},{e},{g}\n" for a, b, c, d, e, g in log)
    print(f"\n{len(rows)} frames written to {scene}")


if __name__ == "__main__":
    main()
