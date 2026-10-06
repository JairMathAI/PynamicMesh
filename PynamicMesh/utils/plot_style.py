"""
plot_style.py  —  readable plots: an automatic layout pass applied to every figure the pipeline saves
=====================================================================================================

Plots of long sequences (many frames, tracks, events) easily become unreadable: legends cover the curves, tick labels
of transitions collide, long titles run over the neighbouring panel. tidy_figure(fig) fixes this right before a figure
is saved:

  1. legends covering data (line points, markers, bars, filled areas) are moved to the inside position with the least
     data under them; when every position is covered, they are placed outside the axes (right or below, where there
     is room) — the legend is never on top of the data;
  2. crowded tick labels: x labels that overlap are rotated; categorical labels that still collide are thinned;
  3. titles wider than their axes are wrapped;
  4. the layout is recomputed (tight_layout) so that nothing is cut or overlapping.

install() wraps matplotlib's Figure.savefig so that every plot of the pipeline gets the pass (PYNAMIC_TIDY_PLOTS=0
disables it); the plots of each module do not need to call it explicitly.
"""
import os
import textwrap
import warnings

import numpy as np

_STATE = {"installed": False, "orig": None}
_LOCS = [1, 2, 3, 4, 9, 8, 6, 7, 10, 5]          # upper right, upper left, lower left, lower right, upper/lower center,…


def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _data_points(ax):
    """Display coordinates of the data drawn in an axes (lines, markers, bars, filled areas), images excluded."""
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection, PolyCollection, LineCollection
    from matplotlib.patches import Rectangle
    pts = []
    tr = ax.transData
    for ln in ax.get_lines():
        if not isinstance(ln, Line2D) or not ln.get_visible():
            continue
        xy = np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) > 1:                          # densify long segments so a line crossing the legend is seen
            seg = [np.linspace(xy[i], xy[i + 1], 6, endpoint=False) for i in range(len(xy) - 1)]
            xy = np.vstack(seg + [xy[-1:]])
        if len(xy):
            pts.append(tr.transform(xy))
    for c in ax.collections:
        if not c.get_visible():
            continue
        if isinstance(c, PathCollection):
            off = c.get_offsets()
            if len(off):
                pts.append(c.get_offset_transform().transform(np.asarray(off, float)))
        elif isinstance(c, (PolyCollection, LineCollection)):
            for p in c.get_paths():
                v = p.vertices
                v = v[np.all(np.isfinite(v), axis=1)]
                if len(v):
                    pts.append(c.get_transform().transform(v))
    for p in ax.patches:
        if isinstance(p, Rectangle) and p.get_visible() and p.get_width() != 0 and p.get_height() != 0:
            x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            pts.append(tr.transform(np.array([[x + w * a, y + h * b] for a in (0.1, 0.5, 0.9) for b in (0.1, 0.5, 0.9)])))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _covered(leg, P, renderer, obstacles=()):
    """Data points under the legend (+ a large penalty for every other legend it overlaps)."""
    bb = leg.get_window_extent(renderer)
    n = 0
    if len(P):
        n = int(np.sum((P[:, 0] >= bb.x0) & (P[:, 0] <= bb.x1) & (P[:, 1] >= bb.y0) & (P[:, 1] <= bb.y1)))
    for ob in obstacles:
        if bb.x0 < ob.x1 and bb.x1 > ob.x0 and bb.y0 < ob.y1 and bb.y1 > ob.y0:
            n += 10000
    return n


def _set_loc(leg, loc, anchor=None, transform=None):
    if anchor is not None:
        leg.set_bbox_to_anchor(anchor, transform=transform)
    try:
        leg.set_loc(loc)
    except Exception:  # noqa: BLE001 - matplotlib < 3.8
        leg._loc = loc


def _relegend(ax, leg, **kw):
    """Re-creates a legend with the same entries (the column layout of a legend cannot change after creation)."""
    handles = list(getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", []))
    labels = [t.get_text() for t in leg.get_texts()]
    fs = leg.get_texts()[0].get_fontsize() if labels else None
    title = leg.get_title().get_text() or None
    leg.remove()
    return ax.legend(handles, labels, fontsize=fs, title=title, **kw)


def _below(ax, leg, renderer, placed):
    """Places the legend under the x tick labels / x label of its axes (and under legends already placed there)."""
    axbb = ax.get_window_extent(renderer)
    bottom = axbb.y0
    for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
        if t.get_visible() and t.get_text().strip():
            bottom = min(bottom, t.get_window_extent(renderer).y0)
    for ob in placed:                                  # a legend of a twin axis already below this axes
        if ob.x1 > axbb.x0 and ob.x0 < axbb.x1 and ob.y1 <= axbb.y0 + 2:
            bottom = min(bottom, ob.y0)
    n = len(leg.get_texts())
    ncol0 = max(1, int(getattr(leg, "_ncols", getattr(leg, "_ncol", 1)) or 1))
    w_entry = leg.get_window_extent(renderer).width / ncol0 * 0.95   # width of one column of the current legend
    ncol = int(max(1, min(n, np.floor(axbb.width / max(w_entry, 1.0)))))
    y = (bottom - axbb.y0) / max(axbb.height, 1e-9) - 0.03
    new = _relegend(ax, leg, loc="upper center", bbox_to_anchor=(0.5, y), bbox_transform=ax.transAxes, ncol=ncol,
                    frameon=True, framealpha=0.9)
    return new


def _fix_legends(fig, renderer):
    moved = False
    axes = [a for a in fig.get_axes() if a.get_legend() is not None]
    placed = []                                        # legends already placed: obstacles for the next ones
    for ax in axes:
        leg = ax.get_legend()
        if not leg.get_visible():
            continue
        others = [b for b in placed]
        P = _data_points(ax)
        # twin axes share the area: their data counts too
        for other in fig.get_axes():
            if other is not ax and np.allclose(other.get_position().bounds, ax.get_position().bounds):
                P = np.vstack([P, _data_points(other)]) if len(P) else _data_points(other)
        if _covered(leg, P, renderer, others) == 0:
            placed.append(leg.get_window_extent(renderer))
            continue
        best = None
        for loc in _LOCS:
            _set_loc(leg, loc, anchor=(0, 0, 1, 1), transform=ax.transAxes)
            n = _covered(leg, P, renderer, others)
            if best is None or n < best[0]:
                best = (n, loc)
            if n == 0:
                break
        if best[0] == 0:
            _set_loc(leg, best[1], anchor=(0, 0, 1, 1), transform=ax.transAxes); moved = True
            placed.append(leg.get_window_extent(renderer))
            continue
        # every inside position covers data: outside, BELOW the axes and horizontal (keeps the width of the axes,
        # never collides with a twin y axis on the right); legends of twin axes are stacked
        leg = _below(ax, leg, renderer, placed)
        placed.append(leg.get_window_extent(renderer))
        moved = True
    return moved


def _fix_ticks(fig, renderer):
    changed = False
    for ax in fig.get_axes():
        labels = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text().strip()]
        if len(labels) < 2:
            continue
        bbs = [t.get_window_extent(renderer) for t in labels]
        overl = any(bbs[i].x1 > bbs[i + 1].x0 + 1 for i in range(len(bbs) - 1)) if all(b.width > 0 for b in bbs) else False
        if not overl:
            continue
        for t in labels:
            t.set_rotation(35); t.set_ha("right"); t.set_rotation_mode("anchor")
        changed = True
        fig.canvas.draw()
        bbs = [t.get_window_extent(renderer) for t in labels]
        widths = np.array([b.width for b in bbs]); gaps = np.diff([b.x0 for b in bbs])
        if len(gaps) and np.median(gaps) > 0 and np.median(widths) * 0.6 > np.median(gaps):
            k = int(np.ceil(np.median(widths) * 0.6 / np.median(gaps)))
            for i, t in enumerate(labels):
                t.set_visible(i % k == 0)
    return changed


def _wrap_titles(fig, renderer):
    changed = False
    for ax in fig.get_axes():
        t = ax.title
        txt = t.get_text()
        if not txt:
            continue
        w_ax = ax.get_window_extent(renderer).width; w_t = t.get_window_extent(renderer).width
        if w_t > 1.05 * w_ax and w_ax > 0:
            n = max(10, int(len(txt) * w_ax / w_t * 0.95))
            t.set_text("\n".join(textwrap.wrap(txt, n))); changed = True
    return changed


def tidy_figure(fig):
    """The layout pass (see the module docstring). Never raises."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            renderer = _renderer(fig)
            changed = _wrap_titles(fig, renderer)
            if changed:
                fig.tight_layout(); renderer = _renderer(fig)
            changed = _fix_ticks(fig, renderer) or changed
            if changed:
                fig.tight_layout(); renderer = _renderer(fig)
            if _fix_legends(fig, renderer):
                fig.tight_layout()
                renderer = _renderer(fig)
                _fix_legends(fig, renderer)          # positions after the new layout
    except Exception:  # noqa: BLE001 - a layout problem must never break the pipeline
        pass
    return fig


def install():
    """Every Figure.savefig gets the layout pass (bbox_inches='tight' so outside legends are kept)."""
    if _STATE["installed"] or os.environ.get("PYNAMIC_TIDY_PLOTS", "1") == "0":
        return
    import matplotlib.figure as mfig
    orig = mfig.Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        tidy_figure(self)
        kwargs.setdefault("bbox_inches", "tight")
        return orig(self, fname, *args, **kwargs)
    mfig.Figure.savefig = savefig
    _STATE.update(installed=True, orig=orig)
