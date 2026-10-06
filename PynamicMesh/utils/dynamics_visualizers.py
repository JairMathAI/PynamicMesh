"""
dynamics_visualizers.py
=======================
PyVista viewers for the results of parametrization.py , trajectories.py  and dynamic_analysis.py :

  visualize_parametrization        Parametrization/        harmonic sphere map + SPHARM reconstruction
  visualize_trajectories           Trajectories/           mesh-to-mesh displacements, vertex trajectories, ghost stack
                                                           (h / f meshes, p / l and w / d opacity or line width, t dots)
  visualize_dynamic_trajectories   Trajectories/           the motion as a moving particle field
                                                           (p / l surface opacity, t particles, k / j line width)
  visualize_interpolation          Trajectories/interpolation   dense interpolated surface trajectory
  visualize_hodge                  DynamicAnalysis/Hodge   Helmholtz–Hodge components of the surface velocity
  visualize_ftle                   DynamicAnalysis/FTLE    finite-time Lyapunov exponents
  visualize_shape_modes            DynamicAnalysis/ShapeSpace   principal deformation modes

Every viewer:
  * shows "Frame k/N : <mesh name>" in the upper-left corner (same convention as utils/visualizers.py),
    the current modality on the line below and a one-line key reminder at the bottom left;
  * opens an instruction window with 'i', resets the camera with 'r', closes with 'q';
  * has VTK's single-key defaults disabled, so 'e', 's', 'w', 'p', 'f', '3' … never close the window or
    change the render mode behind the viewer's back (utils.visualizers.viewer_ui);
  * keeps scalar bars at a fixed position (they do not climb / vanish when frames or modes change);
  * Right / Left step the frames, Enter plays / pauses, space records a GIF of the current view, 'm' cycles the modality, 's' saves a screenshot
    next to the results as <name>_T<frame>_<n>.png (n counts the screenshots of that frame, nothing is
    overwritten; a small self-closing window shows the path, nothing is printed on the console),
    'h' hides / shows the cell mesh where that makes sense; where a vector field is
    shown, 'v' shows only the vectors and 'x' / 'y' / 'z' project them on the plane perpendicular to that axis.

Every function takes the *scene* Results folder (Results/<scene>) and reads what it needs from the
sub-folders written by the core modules.  pyvista is imported lazily.
"""
import re
from pathlib import Path

import contextlib
import numpy as np

from PynamicMesh.core.dyn_common import read_obj, list_mesh_files, load_frames, frame_number, vertex_normals

try:                                                        # shared UI helpers of the project viewers
    from PynamicMesh.utils.visualizers import viewer_ui, scalar_bar_args, clear_scalar_bars, show_instructions_window, renderer_info, configure_rendering
    from PynamicMesh.utils.visualizers import save_screenshot, screenshot_path, show_notification
except Exception:  # noqa: BLE001 - local copies when utils.visualizers cannot be imported (pyFM missing)
    HELP_FOOTER = "\n\nCommon keys:\n  i : instructions window\n  r : reset the camera\n  q : close the window"

    
    def show_instructions_window(title, text):
        """
        Pops up a small, comfortable read-only window (tkinter) with the instructions of a viewer: monospace
        text, scrollbar, Close button, Esc closes.  The 3-D window is paused while it is open.  Returns False
        when tkinter is not available (the caller then falls back to an on-screen text panel).
        """
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception:  # noqa: BLE001
            return False
        try:
            root = tk.Tk(); root.withdraw()
            win = tk.Toplevel(root); win.title(f"Instructions - {title}")
            try:
                win.attributes('-topmost', True)
            except Exception:  # noqa: BLE001
                pass
            frame = ttk.Frame(win, padding=10); frame.pack(fill='both', expand=True)
            n_lines = text.count('\n') + 2
            txt = tk.Text(frame, wrap='word', font=('Consolas', 11), width=96, height=min(34, max(8, n_lines)),
                          padx=14, pady=10, bg='#fffbe6', relief='flat')
            sb = ttk.Scrollbar(frame, orient='vertical', command=txt.yview); txt.configure(yscrollcommand=sb.set)
            txt.insert('1.0', text); txt.configure(state='disabled')
            txt.grid(row=0, column=0, sticky='nsew'); sb.grid(row=0, column=1, sticky='ns')
            frame.rowconfigure(0, weight=1); frame.columnconfigure(0, weight=1)
            ttk.Button(frame, text='Close  (Esc)', command=win.destroy).grid(row=1, column=0, columnspan=2, pady=(8, 0))
            win.bind('<Escape>', lambda e: win.destroy()); win.protocol('WM_DELETE_WINDOW', win.destroy)
            win.update_idletasks()
            w, h = win.winfo_reqwidth(), win.winfo_reqheight()
            x = (win.winfo_screenwidth() - w) // 2; y = (win.winfo_screenheight() - h) // 3
            win.geometry(f"+{x}+{y}"); win.lift(); win.focus_force()
            root.wait_window(win); root.destroy()
            return True
        except Exception:  # noqa: BLE001
            return False


    _RENDERER_INFO = None
    _SOFTWARE_HINTS = ("llvmpipe", "softpipe", "swiftshader", "mesa offscreen", "microsoft basic render", "gdi generic",
                       "software rasterizer", "osmesa", "warp")
    
    
    def renderer_info(refresh=False):
        """
        Detects the OpenGL device VTK renders with: {'gpu': bool, 'renderer': str, 'vendor': str}.  VTK / pyvista
        always render through OpenGL, so a discrete or integrated GPU is used automatically whenever the driver
        exposes it; this probe only tells whether that happened or whether a software rasterizer (llvmpipe,
        Microsoft Basic Render Driver, ...) is in use.  The result is cached (one hidden render window).
        """
        global _RENDERER_INFO
        if _RENDERER_INFO is not None and not refresh:
            return _RENDERER_INFO
        info = {'gpu': False, 'renderer': 'unknown', 'vendor': 'unknown'}
        try:
            import vtk
            rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1); rw.SetSize(8, 8); rw.Render()
            caps = rw.ReportCapabilities() or ""
            rw.Finalize()
            for line in caps.splitlines():
                low = line.lower()
                if low.startswith("opengl renderer string"):
                    info['renderer'] = line.split(":", 1)[1].strip()
                elif low.startswith("opengl vendor string"):
                    info['vendor'] = line.split(":", 1)[1].strip()
            info['gpu'] = info['renderer'] != 'unknown' and not any(h in info['renderer'].lower() for h in _SOFTWARE_HINTS)
        except Exception:  # noqa: BLE001
            pass
        _RENDERER_INFO = info
        return info
    
    
    def configure_rendering(pl):
        """
        Tunes a plotter for the detected device.  On a GPU: FXAA anti-aliasing and depth peeling (correct,
        smooth transparency of the translucent meshes) - both cheap on hardware and costly in software.  On a
        software rasterizer they are disabled and point sprites replace sphere impostors so that frame changes
        stay interactive.  Returns the renderer_info() dict.
        """
        info = renderer_info()
        try:
            import pyvista as pv
            if info['gpu']:
                pl.enable_anti_aliasing('fxaa')
                try:
                    pl.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
                except Exception:  # noqa: BLE001
                    pass
                pv.global_theme.render_points_as_spheres = True
            else:
                pl.disable_anti_aliasing()
                pv.global_theme.render_points_as_spheres = False
                pv.global_theme.smooth_shading = False
        except Exception:  # noqa: BLE001
            pass
        return info

    def viewer_ui(pl, help_text, title="Instructions", subplot=None):
        try:
            pl.iren.interactor.RemoveObservers('CharEvent')
        except Exception:  # noqa: BLE001
            pass
        dev = configure_rendering(pl)
        state = {'help': False}
        full_text = f"{help_text}{HELP_FOOTER}\n\nRendering device: {dev['renderer']} ({'GPU' if dev['gpu'] else 'software rasterizer'})"

        def _sub():
            if subplot is not None:
                try:
                    pl.subplot(*subplot)
                except Exception:  # noqa: BLE001
                    pass

        def show_help():
            if show_instructions_window(title, full_text):
                return
            _sub()
            if state['help']:
                pl.remove_actor('help_panel', render=False)
            else:
                pl.add_text(f"{title}\n{'-' * len(title)}\n{full_text}", name='help_panel', position='upper_right',
                            font_size=10, color='white', shadow=True)
            state['help'] = not state['help']; pl.render()

        pl.add_key_event('i', show_help)
        pl.add_key_event('r', lambda: (pl.reset_camera(), pl.render()))
        pl.add_key_event('q', pl.close)
        _sub(); pl.add_text("press 'i' for the instructions", name='help_hint', position='lower_right', font_size=6, color='gray')

    def scalar_bar_args(title, n_labels=3, **extra):
        a = {'title': title, 'n_labels': n_labels, 'vertical': True, 'position_x': 0.90, 'position_y': 0.20, 'width': 0.07,
             'height': 0.55, 'title_font_size': 12, 'label_font_size': 10}
        a.update(extra); return a

    def clear_scalar_bars(pl):
        try:
            for t in list(pl.scalar_bars.keys()):
                pl.remove_scalar_bar(t, render=False)
        except Exception:  # noqa: BLE001
            pass

    def screenshot_path(folder, name, frame=None):
        """Next free <folder>/<name>_T<frame>_<n>.png (never overwrites an older screenshot)."""
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        stem = f"{name}_T{int(frame):04d}" if frame is not None else str(name)
        n = 1
        while (folder / f"{stem}_{n:02d}.png").exists():
            n += 1
        return folder / f"{stem}_{n:02d}.png"

    def show_notification(pl, text, title="Screenshot saved", duration_ms=1800):
        """Self-removing message on the 3-D window (the full version with a small window lives in utils.visualizers)."""
        import time as _time
        t_end = _time.time() + duration_ms / 1000.0; n_steps = max(int(duration_ms / 100) + 3, 3)
        try:
            pl.add_text(f"{title}\n{text}", name='screenshot_note', position='upper_edge', font_size=9, color='darkgreen'); pl.render()

            def remove(step):
                if step >= n_steps - 1 or _time.time() >= t_end:
                    try:
                        pl.remove_actor('screenshot_note', render=True)
                    except Exception:  # noqa: BLE001
                        pass
            pl.add_timer_event(max_steps=n_steps, duration=100, callback=remove)
        except Exception:  # noqa: BLE001
            pass

    def save_screenshot(pl, folder, name, frame=None, notify=True):
        out = screenshot_path(folder, name, frame)
        try:
            pl.screenshot(str(out))
        except Exception as exc:  # noqa: BLE001
            if notify:
                show_notification(pl, f"could not save the screenshot:\n{exc}", title="Screenshot failed")
            return None
        if notify:
            show_notification(pl, f"{out}")
        return out


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _pv():
    import pyvista as pv
    return pv


def _poly(V, F):
    pv = _pv(); F = np.asarray(F)
    return pv.PolyData(np.asarray(V, dtype=np.float64), np.hstack([np.full((F.shape[0], 1), 3), F]).ravel())


def _sorted(folder, pattern):
    return sorted(Path(folder).glob(pattern), key=frame_number)


def _frame_names(results_path, mesh_path=None, n=None):
    """Mesh names for the frame label (+ the acquisition time when known): from the mesh folder, else
    Trajectories/frame_names.txt, else indices."""
    return _with_times(_frame_names_raw(results_path, mesh_path, n), results_path)


def _frame_names_raw(results_path, mesh_path=None, n=None):
    if mesh_path is not None:
        try:
            names = [f.name for f in list_mesh_files(mesh_path)]
            if names:
                return names
        except Exception:  # noqa: BLE001
            pass
    f = Path(results_path) / "Trajectories" / "frame_names.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").splitlines()
    return [f"T{k:04d}" for k in range(n or 0)]


def _with_times(names, results_path):
    """Appends 't = ... unit' to the frame names when the acquisition times of the scene are known."""
    from PynamicMesh.core.dyn_common import find_frame_times
    ft = find_frame_times(results_path)
    if ft is None or len(ft) != len(names):
        return names
    return [f"{nm}  ({ft.label(k)})" for k, nm in enumerate(names)]


def _observed_times(results_path, n):
    """Times of the n observed frames: frame_times.csv, else Trajectories/time.npy, else None (uniform)."""
    from PynamicMesh.core.dyn_common import find_frame_times
    ft = find_frame_times(results_path)
    if ft is not None and len(ft) == n:
        return ft.t
    f = Path(results_path) / "Trajectories" / "time.npy"
    if f.exists():
        t = np.load(f)
        if len(t) == n:
            return t
    return None


def _arrows(points, vectors, scale, shaft=0.01, tip=0.035, tip_len=0.2):
    """Thin arrow glyphs (shaft radius 1 % of the arrow length) - much lighter than pyvista's default arrows."""
    pv = _pv()
    pd_ = pv.PolyData(np.asarray(points, float)); pd_["vec"] = np.asarray(vectors, float)
    pd_["mag"] = np.linalg.norm(vectors, axis=1)
    geom = pv.Arrow(shaft_radius=shaft, tip_radius=tip, tip_length=tip_len)
    return pd_.glyph(orient="vec", scale="mag", factor=scale, geom=geom)


def _screenshot(plotter, folder, name, frame=None):
    """
    Screenshot of the current view as <folder>/<name>_T<frame>_<n>.png: n numbers the screenshots of the same frame
    (different angles, later sessions), so none overwrites another. The saved path is shown in a small message
    window that closes by itself (nothing is printed on the console).
    """
    return save_screenshot(plotter, folder, name, frame=frame)


_GIF_REQUEST = None          # set by record_gifs(): show() then exports GIFs instead of opening a window


def _screen_text(text):
    """On-screen text safe for VTK: strings with non-ASCII characters (arrows, x, sigma ...) are drawn by matplotlib's
    mathtext, which takes '|' as a table-cell separator ('Failed to compute rows and cols for cell' warnings on every
    frame); there the separators become middle dots."""
    text = str(text)
    return text.replace("|", "\u00b7") if any(ord(c) > 127 for c in text) else text


def _gif_dir(screenshot_dir):
    """gif/ folder of the stage of a viewer: <stage>/plots or <stage>/screenshots -> <stage>/gif."""
    d = Path(screenshot_dir) if screenshot_dir is not None else Path.cwd() / "Screenshots"
    return (d.parent if d.name in ("plots", "screenshots") else d) / "gif"


def _next_gif_path(folder):
    """<folder>/screenshot_<n>.gif with the first n not used yet (earlier captures and pipeline GIFs are kept)."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while (folder / f"screenshot_{n}.gif").exists():
        n += 1
    return folder / f"screenshot_{n}.gif"


@contextlib.contextmanager
def record_gifs(out_dir, prefix=None, **params):
    """
    Inside this block every viewer of this module writes GIF animations instead of opening its window: one GIF per
    modality (and '<modality>_vectors' for the modalities with a vector field), rendered off-screen with the viewer's
    own drawing code into out_dir (gif_export.GIF_DEFAULTS: fps, real_time, max_frames, panel_width, height, ...).
    Yields the request dict; request['written'] lists the files.
    """
    global _GIF_REQUEST
    from PynamicMesh.utils.gif_export import GIF_DEFAULTS
    prev = _GIF_REQUEST
    _GIF_REQUEST = {"out_dir": Path(out_dir), "prefix": prefix, **GIF_DEFAULTS, **params, "written": []}
    try:
        yield _GIF_REQUEST
    finally:
        _GIF_REQUEST = prev


def _launch(viewer, mode_names=None, times=None, animate=True):
    """Shows a viewer (or exports its GIFs inside record_gifs); mode_names / times label the GIF files / timing."""
    viewer.mode_names = list(mode_names) if mode_names is not None else None
    viewer.gif_times = None if times is None else np.asarray(times, dtype=float)
    return viewer.show(animate=animate)


class _FrameViewer:
    """
    Common state machine of the viewers: frame index k, modality index, play/pause, optional alternative
    layout.  Subclasses implement draw(); they add named actors and may call self.scalar(...) for a fixed
    scalar bar.  show() (re)opens the window; a subclass sets self.relaunch = True and changes self.shape to
    switch layouts (the loop re-creates the plotter keeping k / mode).
    """
    title = "viewer"; help_text = ""; keys_line = "← → frames | Enter play | space GIF | m modality | i help | q quit"

    def __init__(self, n_frames, n_modes=1, shape=(1, 1), names=None, screenshot_dir=None, screenshot_name="view"):
        self.pv = _pv()
        self.k, self.mode, self.n, self.n_modes = 0, 0, int(n_frames), max(int(n_modes), 1)
        self.shape = shape; self.base_shape = shape; self.names = names or [f"T{k:04d}" for k in range(self.n)]
        self.playing = False; self.relaunch = False; self.show_mesh = True; self.field_layout = False
        self.projection = None; self._camera_dirty = False
        self.screenshot_dir, self.screenshot_name = screenshot_dir, screenshot_name
        self.plotter = None

    # ---- key actions ----------------------------------------------------------------------------------
    def step(self, d):
        self.k = (self.k + d) % self.n; self.redraw()

    def next_mode(self):
        self.mode = (self.mode + 1) % self.n_modes
        if self.field_layout and self.vector_field() is None:      # new modality has no vectors: back to the surface view
            self.field_layout = False; self.projection = None; self._apply_layout(); return
        self.redraw()

    def toggle_play(self):
        self.playing = not self.playing

    def toggle_mesh(self):
        self.show_mesh = not self.show_mesh; self.redraw()

    # ---- generic vector-field view ('v', then 'x' / 'y' / 'z') -------------------------------------------
    def vector_field(self):
        """
        Subclasses that display a vector field return (points, vectors, label, scale) for the current frame /
        modality, or None when the current modality has no vectors.  When available:
          v        : toggles the VECTOR-FIELD view - only the vectors, no surface, in 3-D;
          x / y / z: in the vector-field view, project the field on the plane perpendicular to that axis
                     (view along the axis, parallel projection, grid); the same key again returns to 3-D.
        One picture at a time, so each view is easy to read.
        """
        return None

    def switch_layout(self):
        if not self.field_layout and self.vector_field() is None:
            return
        self.field_layout = not self.field_layout; self.projection = None
        self._apply_layout()

    def set_projection(self, axis):
        if self.vector_field() is None:
            return
        if not self.field_layout:
            self.field_layout = True; self.projection = axis
        else:
            self.projection = None if self.projection == axis else axis
        self._apply_layout()

    def _apply_layout(self):
        wanted = (1, 1) if self.field_layout else self.base_shape
        if wanted != self.shape:                          # multi-panel viewers: re-open with a single panel
            self.shape = wanted; self.relaunch = True; self.plotter.close(); return
        self._camera_dirty = True
        self.redraw()

    def draw_field_layout(self, points, vectors, label, scale):
        p = self.plotter; p.subplot(0, 0)
        P = np.asarray(points, float).copy(); Fv = np.asarray(vectors, float).copy()
        axis = {"x": 0, "y": 1, "z": 2}.get(self.projection)
        planes = {"x": "YZ", "y": "XZ", "z": "XY"}
        if axis is not None:
            P[:, axis] = 0.0; Fv[:, axis] = 0.0
        mag = np.linalg.norm(Fv, axis=1)
        p.add_mesh(_arrows(P, Fv, scale), scalars="mag", cmap="plasma", clim=[0.0, float(mag.max()) + 1e-12],
                   scalar_bar_args=self.scalar("|v|"))
        p.add_mesh(self.pv.PolyData(P), color="gray", point_size=2, render_points_as_spheres=True)
        if axis is not None:
            p.show_grid(font_size=7)
            if getattr(self, "_camera_dirty", True) or not getattr(p, "_field_cam_set", False):
                p.enable_parallel_projection(); p.camera_position = {"x": "yz", "y": "xz", "z": "xy"}[self.projection]; p.reset_camera()
                p._field_cam_set = True
            view = f"projection on the {planes[self.projection]} plane (view along {self.projection})"
        else:
            if not getattr(p, "_orientation_axes_added", False):
                p.add_axes(); p._orientation_axes_added = True   # the orientation widget survives clear(): add it once
            if getattr(self, "_camera_dirty", True) or not getattr(p, "_field_cam_set", False):
                p.disable_parallel_projection(); p.camera_position = "iso"; p.reset_camera()
                p._field_cam_set = True
            view = "3-D vector field"
        self._camera_dirty = False
        self.label(f"{view} - {label}\n[vector-field view: x / y / z project on a plane (again = back to 3-D), v returns to the surface view]")

    def _draw(self):
        if self.field_layout:
            vf = self.vector_field()
            if vf is not None:
                self.clear(); self.draw_field_layout(*vf); return
            self.field_layout = False; self.projection = None
        self.draw()
        if getattr(self, "_camera_dirty", False) and self.plotter is not None:   # back from a projection: perspective again,
            try:                                                                    # AFTER the actors exist (else the reset sees an empty scene)
                self.plotter.disable_parallel_projection(); self.plotter.camera_position = "iso"; self.plotter.reset_camera()
            except Exception:  # noqa: BLE001
                pass
            self._camera_dirty = False

    def _tick(self, _step):
        if self.playing:
            self.step(+1)

    # ---- drawing helpers -----------------------------------------------------------------------------
    def clear(self):
        p = self.plotter
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                p.subplot(i, j)
                try:
                    p.remove_bounds_axes()                # grid axes are managed by the renderer: drop them explicitly
                except Exception:  # noqa: BLE001
                    pass
                p.clear_actors()
        clear_scalar_bars(p)
        p.subplot(0, 0)

    def scalar(self, title):
        return scalar_bar_args(title)

    def label(self, mode_text=""):
        """Upper-left frame label (subplot 0,0) + bottom-left key reminder; both replace their previous text."""
        p = self.plotter; p.subplot(0, 0)
        name = self.names[self.k] if self.k < len(self.names) else ""
        p.add_text(_screen_text(f"Frame {self.k + 1}/{self.n} : {name}" + (f"\n{mode_text}" if mode_text else "")),
                   name="ui_text", position="upper_left", font_size=9)
        p.add_text(_screen_text(self.keys_line), name="keys_line", position="lower_left", font_size=6, color="black")

    def redraw(self):
        """Rebuilds the current frame with rendering suppressed (pyvista renders after every add_mesh otherwise) and
        renders once at the end."""
        p = self.plotter
        if p is not None and hasattr(p, "suppress_rendering"):
            outer = bool(p.suppress_rendering)            # already inside a single-render key callback
            p.suppress_rendering = True
            try:
                self._redraw_body()
            finally:
                p.suppress_rendering = outer
            if not outer:
                p.render()
            return
        self._redraw_body()

    def _redraw_body(self):
        if self.plotter is None:
            return
        self._draw(); self.plotter.render()

    def draw(self):  # pragma: no cover
        raise NotImplementedError

    # ---- window --------------------------------------------------------------------------------------
    def _make_plotter(self):
        p = self.pv.Plotter(shape=self.shape, title=self.title)
        p.add_key_event("Right", lambda: self.step(+1)); p.add_key_event("Left", lambda: self.step(-1))
        p.add_key_event("Return", self.toggle_play); p.add_key_event("space", self.capture_gif)
        p.add_key_event("m", self.next_mode)
        p.add_key_event("h", self.toggle_mesh); p.add_key_event("v", self.switch_layout)
        for ax in ("x", "y", "z"):
            p.add_key_event(ax, lambda ax=ax: self.set_projection(ax))
        if self.screenshot_dir is not None:
            p.add_key_event("s", lambda: _screenshot(p, self.screenshot_dir, self.screenshot_name, frame=self.k))
        viewer_ui(p, self.help_text + "\n\nEnter : play / pause\n"
                  f"space : GIF of the sequence with the CURRENT view and settings (camera, modality, selection, ...)\n"
                  f"        -> {_gif_dir(self.screenshot_dir)}/screenshot_<n>.gif\n"
                  f"s : screenshot -> {self.screenshot_dir}/{self.screenshot_name}_T<frame>_<n>.png\n"
                  "    (n counts the screenshots of that frame, nothing is overwritten; a small message confirms it)",
                  title=self.title, subplot=(0, self.shape[1] - 1))
        return p

    # ---- GIF of the current view (space) -------------------------------------------------------------------
    def capture_gif(self):
        """
        GIF of the whole sequence as it is shown NOW: the camera of every panel, the modality, the layout (vectors /
        projection) and every option chosen in the viewer are kept while the frames are stepped one by one; the
        viewer then returns to the frame it was showing. Saved as <stage>/gif/screenshot_<n>.gif (n never reused).
        """
        from PynamicMesh.utils.gif_export import write_gif, frame_durations, GIF_DEFAULTS
        p = self.plotter
        if p is None:
            return
        was_playing, self.playing = self.playing, False
        k0 = self.k
        cams = []
        for r in p.renderers:
            c = r.GetActiveCamera()
            cams.append((c.GetPosition(), c.GetFocalPoint(), c.GetViewUp(), c.GetViewAngle(), c.GetParallelScale()))

        def restore_cameras():
            for r, (pos, foc, up, ang, ps) in zip(p.renderers, cams):
                c = r.GetActiveCamera(); c.SetPosition(pos); c.SetFocalPoint(foc); c.SetViewUp(up)
                c.SetViewAngle(ang); c.SetParallelScale(ps); r.ResetCameraClippingRange()
        idx = np.unique(np.linspace(0, self.n - 1, min(self.n, GIF_DEFAULTS["max_frames"])).round().astype(int))
        imgs = []
        try:
            for k in idx:
                self.k = int(k); self._camera_dirty = False
                p.suppress_rendering = True
                try:
                    self._draw()
                finally:
                    p.suppress_rendering = False
                restore_cameras(); p.render()
                imgs.append(p.screenshot(return_img=True))
        finally:
            self.k = k0; self.redraw(); restore_cameras(); p.render()
            self.playing = was_playing
        times = getattr(self, "gif_times", None)
        tsel = np.asarray(times)[idx] if times is not None and len(times) == self.n else None
        path = _next_gif_path(_gif_dir(self.screenshot_dir))
        out = write_gif(imgs, path, frame_durations(tsel, len(idx), GIF_DEFAULTS["fps"], GIF_DEFAULTS["real_time"]),
                        GIF_DEFAULTS["max_width"], GIF_DEFAULTS["loop"])
        if out:
            show_notification(p, f"{out}  ({len(imgs)} frames)", title="GIF saved")

    # ---- GIF export (record_gifs) ------------------------------------------------------------------------
    def gif_variants(self):
        """[(label, setup)] datasets shown by the viewer (one GIF set each); the default is the viewer itself."""
        return [("", lambda: None)]

    def export_gifs(self, req):
        from PynamicMesh.utils.gif_export import slug
        names = list(getattr(self, "mode_names", None) or [f"mode{m + 1}" for m in range(self.n_modes)])
        prefix = req.get("prefix") if req.get("prefix") is not None else self.screenshot_name
        for vlabel, setup in self.gif_variants():
            setup()
            conds = []; seen_fields = set()
            per_mode = req.get("vectors", True) and self.n_modes > 1 and self._vec_per_mode()
            for m in range(self.n_modes):
                mlabel = slug(names[m]) if self.n_modes > 1 else ""
                conds.append((mlabel, m, False))
                if req.get("vectors", True):
                    self.mode, self.k, self.field_layout = m, min(1, self.n - 1), False
                    try:
                        vf = self.vector_field()
                    except Exception:  # noqa: BLE001
                        vf = None
                    if vf is not None:
                        # the same field in several modalities (e.g. particle / node velocities) -> one GIF only
                        sig = (np.asarray(vf[1]).shape, round(float(np.nansum(np.abs(vf[1]))), 6))
                        if sig not in seen_fields:
                            seen_fields.add(sig)
                            conds.append(((mlabel + "_" if mlabel and per_mode else "") + "vectors", m, True))
            for clabel, m, fl in conds:
                stem = "_".join(x for x in (prefix, vlabel, clabel) if x)
                try:
                    path = self._record_condition(req, m, fl, Path(req["out_dir"]) / f"{stem}.gif")
                    if path:
                        req["written"].append(path)
                except Exception as exc:  # noqa: BLE001 - one failing condition must not stop the others
                    req["written"].append(f"{stem}: failed ({exc})")
        self.plotter = None

    def _vec_per_mode(self):
        """True when the vector field differs between modalities (then every '_vectors' GIF carries its modality)."""
        sigs = set()
        for m in range(self.n_modes):
            self.mode = m
            try:
                vf = self.vector_field()
            except Exception:  # noqa: BLE001
                vf = None
            if vf is not None:
                sigs.add(round(float(np.nansum(np.abs(vf[1]))), 6))
        return len(sigs) > 1

    def _record_condition(self, req, mode, field_layout, path):
        from PynamicMesh.utils.gif_export import write_gif, frame_durations, _bounds_union
        shape = (1, 1) if field_layout else self.base_shape
        W = min(req["panel_width"] * shape[1], 3 * req["panel_width"]); H = req["height"] * shape[0]
        p = self.pv.Plotter(off_screen=True, shape=shape, window_size=(W, H))
        self.plotter = p; self.shape = shape; self.mode = mode
        self.field_layout = field_layout; self.projection = None; self._camera_dirty = False
        idx = np.unique(np.linspace(0, self.n - 1, min(self.n, req["max_frames"])).round().astype(int))
        # camera: union of the bounds of a few frames in every panel (moving parts stay in the picture)
        bnds = {(i, j): [] for i in range(shape[0]) for j in range(shape[1])}
        for k in np.unique(np.linspace(0, self.n - 1, min(self.n, 6)).round().astype(int)):
            self.k = int(k); self._draw()
            for (i, j) in bnds:
                p.subplot(i, j); bnds[(i, j)].append(p.renderer.ComputeVisiblePropBounds())
        if shape != (1, 1):
            p.link_views()
        for (i, j), bs in bnds.items():
            ub = _bounds_union(bs)
            p.subplot(i, j)
            if ub is not None:
                p.reset_camera(bounds=ub)
        p.subplot(0, 0)
        imgs = []
        from tqdm.auto import tqdm
        for k in tqdm(idx, desc=f"GIF {Path(path).stem}", leave=False):
            self.k = int(k); self._draw(); p.render()
            imgs.append(p.screenshot(return_img=True))
        p.close()
        times = getattr(self, "gif_times", None)
        tsel = times[idx] if times is not None and len(times) == self.n else None
        return write_gif(imgs, path, frame_durations(tsel, len(idx), req["fps"], req["real_time"]),
                         req["max_width"], req["loop"])

    def show(self, animate=True):
        if _GIF_REQUEST is not None:                      # record_gifs(): GIFs instead of the window
            return self.export_gifs(_GIF_REQUEST)
        while True:
            self.relaunch = False
            self.plotter = self._make_plotter()
            self._draw()
            if self.shape != (1, 1):
                self.plotter.link_views()
            # frame every panel on its content for the first render (pyvista skips the auto-reset once a camera was touched)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.plotter.subplot(i, j); self.plotter.reset_camera()
            self.plotter.subplot(0, 0)
            if animate:
                self.plotter.add_timer_event(max_steps=10 ** 7, duration=150, callback=self._tick)
            self.plotter.show(full_screen=True)
            if not self.relaunch:
                break


# --------------------------------------------------------------------------- #
#  (a) parametrization
# --------------------------------------------------------------------------- #

_FrameViewer.capture_gif._no_batch = True              # renders every frame itself (no single-render wrapping)

def visualize_parametrization(mesh_path, results_path, loader=None):
    """
    Left: mesh coloured by the spherical coordinates (θ, φ) of the harmonic map (iso-lines); centre: the
    sphere map; right: the SPHARM reconstruction ('m' cycles the exported degrees).
    """
    res = Path(results_path) / "Parametrization"
    frames = load_frames(mesh_path, loader)
    sph = _sorted(res / "Spherical", "sphere_T*.npy")
    if not sph:
        print("No spherical parametrization found (non genus-0 sequence?). Only the manifold-harmonics results exist.")
        return
    # keep only the sphere maps that match the meshes currently in mesh_path (stale results of another mesh set,
    # or frames beyond the current sequence, are reported and skipped)
    usable, skipped = [], []
    for f in sph:
        i = frame_number(f)
        if i >= len(frames):
            skipped.append(f"{f.name}: frame {i} but only {len(frames)} meshes in {mesh_path}"); continue
        n_map = np.load(f, mmap_mode="r").shape[0]
        if n_map != frames[i]["V"].shape[0]:
            skipped.append(f"{f.name}: {n_map} vertices in the map vs {frames[i]['V'].shape[0]} in {frames[i]['name']}"); continue
        usable.append(f)
    if skipped:
        print("[visualize_parametrization] results that do not match the meshes in mesh_path (computed for another mesh set? "
              "re-run compute_parametrization):\n  " + "\n  ".join(skipped))
    if not usable:
        print("No sphere map matches the current meshes; nothing to show.")
        return
    sph = usable
    idx = [frame_number(f) for f in sph]
    degrees = sorted({int(re.search(r"_L(\d+)", f.stem).group(1)) for f in (res / "SPHARM").glob(f"reconstruction_T{idx[0]:04d}_L*.obj")})
    names = [frames[i]["name"] for i in idx]

    class Viewer(_FrameViewer):
        title = "Parametrization: harmonic sphere map + SPHARM series"
        help_text = ("Left : surface with iso-lines of (θ, φ) of the harmonic map\nCentre : the sphere map itself (coloured by θ)\n"
                     "Right : surface synthesised from the spherical-harmonic series\n\n← → : frames | m : SPHARM degree | views are linked")
        keys_line = "← → frames | Enter play | space GIF | m degree | s screenshot | i help | q quit"

        def draw(self):
            self.clear(); p = self.plotter; i = idx[self.k]
            V, F = frames[i]["V"], frames[i]["F"]
            U = np.load(sph[self.k]); tp = np.load(res / "Spherical" / f"theta_phi_T{i:04d}.npy")
            iso = np.sin(8 * tp[:, 0]) * np.sin(8 * tp[:, 1])
            p.subplot(0, 0); m = _poly(V, F); m["iso"] = iso
            p.add_mesh(m, scalars="iso", cmap="coolwarm", show_scalar_bar=False, smooth_shading=True)
            # the sphere map is drawn at the size and position of the cell so that the linked cameras frame both
            c0 = V.mean(axis=0); R0 = np.linalg.norm(V - c0, axis=1).mean()
            p.subplot(0, 1); sm = _poly(c0 + R0 * U, F); sm["iso"] = iso
            p.add_mesh(sm, scalars="iso", cmap="coolwarm", show_scalar_bar=False, show_edges=V.shape[0] < 20000, edge_color="gray", line_width=0.3)
            p.add_text(f"sphere map (drawn at R = {R0:.3g})", name="t1", position="upper_edge", font_size=7)
            p.subplot(0, 2)
            deg_txt = ""
            if degrees:
                L = degrees[self.mode % len(degrees)]; deg_txt = f"SPHARM degree L = {L}"
                f = res / "SPHARM" / f"reconstruction_T{i:04d}_L{L:02d}.obj"
                if f.exists():
                    Vr, Fr = read_obj(f); p.add_mesh(_poly(Vr, Fr), color="wheat", smooth_shading=True)
                p.add_text(f"SPHARM reconstruction  L={L}", name="t2", position="upper_edge", font_size=7)
            self.label(f"left: surface with (θ,φ) iso-lines | centre: harmonic map to S² | right: {deg_txt or 'SPHARM'}")

    from PynamicMesh.core.dyn_common import find_frame_times
    ft = find_frame_times(results_path)
    p_times = ft.t[np.asarray(idx)] if ft is not None and len(idx) and max(idx) < len(ft) else None
    if ft is not None and len(idx) and max(idx) < len(ft):
        names = [f"{nm}  ({ft.label(i)})" for nm, i in zip(names, idx)]
    _launch(Viewer(len(sph), max(len(degrees), 1), shape=(1, 3), names=names, screenshot_dir=res / "plots",
                   screenshot_name="parametrization"), [f"L{d}" for d in degrees] or None, p_times, animate=False)


# --------------------------------------------------------------------------- #
#  (c) trajectories and interpolation
# --------------------------------------------------------------------------- #

class _DotSelection:
    """
    Selection of the dots (particles / vertices) of a trajectory view:
      'a'          all selected <-> none selected (any selection made by clicking is replaced)
      left click   on a dot: select / unselect that dot (a click = press + release without moving; dragging still rotates)
    Selected dots keep their colour and their trajectory is drawn; unselected dots are grey and have no trajectory.
    The host viewer provides `_selection_points()` (positions of the displayed dots, or None when the current mode has
    no selectable dots) and calls `_install_selection(plotter)` in `_make_plotter`.
    """
    SEL_KEY = "a"
    CLICK_PIXELS = 10          # a click selects the nearest dot within this many pixels
    DRAG_PIXELS = 4            # press-release farther apart than this is a camera drag, not a click

    def _init_selection(self, n_dots):
        self.sel = np.ones(int(n_dots), dtype=bool)
        self._press_pos = None

    def _install_selection(self, p):
        p.add_key_event(self.SEL_KEY, self._toggle_all)

        def _event_pos(obj):
            # pyvista attaches the observers to the interactor STYLE: the position lives on its interactor
            it = obj.GetInteractor() if hasattr(obj, "GetInteractor") else obj
            return tuple(it.GetEventPosition())

        def press(obj, _ev):
            self._press_pos = _event_pos(obj)

        def release(obj, _ev):
            if self._press_pos is None:
                return
            (x0, y0), (x1, y1) = self._press_pos, _event_pos(obj)
            self._press_pos = None
            if (x1 - x0) ** 2 + (y1 - y0) ** 2 <= self.DRAG_PIXELS ** 2:
                self._click_select(x1, y1)
        try:
            p.iren.add_observer("LeftButtonPressEvent", press)
            p.iren.add_observer("LeftButtonReleaseEvent", release)
        except Exception:  # noqa: BLE001 - no interactor (off-screen): keys only
            pass

    def _toggle_all(self):
        if self._selection_points() is None:
            return
        self.sel[:] = not self.sel.all()          # all -> none; none or partial -> all
        self.redraw()

    def _click_select(self, x, y):
        """Toggles the displayed dot nearest to the display position (x, y), in front if several overlap."""
        pts = self._selection_points()
        if pts is None or self.field_layout or not len(pts):
            return None
        p = self.plotter; p.subplot(0, 0); ren = p.renderer
        w, h = ren.GetSize(); ox, oy = ren.GetOrigin()
        if w <= 0 or h <= 0:
            return None
        M = ren.GetActiveCamera().GetCompositeProjectionTransformMatrix(w / h, -1, 1)
        M = np.array([[M.GetElement(i, j) for j in range(4)] for i in range(4)])
        ph = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ M.T
        ndc = ph[:, :3] / np.where(np.abs(ph[:, 3:4]) > 1e-12, ph[:, 3:4], 1e-12)
        xd = ox + (ndc[:, 0] + 1) * 0.5 * w; yd = oy + (ndc[:, 1] + 1) * 0.5 * h
        d2 = (xd - x) ** 2 + (yd - y) ** 2
        near = np.where((d2 <= self.CLICK_PIXELS ** 2) & (np.abs(ndc[:, 2]) <= 1))[0]
        if not near.size:
            return None
        i = int(near[np.lexsort((d2[near], ndc[near, 2]))][0])     # front-most, then closest to the cursor
        self.sel[i] = not self.sel[i]
        self.redraw()
        return i

    def _selection_text(self):
        n, m = int(self.sel.sum()), self.sel.size
        what = "all" if n == m else ("none" if n == 0 else f"{n} of {m}")
        return f"selected {what} ({self.SEL_KEY}: all <-> none, click a dot: select / unselect)"


def visualize_trajectories(results_path, mesh_path=None, loader=None, stride=1, scheme=None):
    """
    Vertex-to-vertex trajectories BETWEEN THE MESHES (functional-map assignment), for all vertices.
    Modalities ('m'):
      1. displacement field mesh k → mesh k+1 : a thin arrow from every vertex of mesh k to its matched vertex
         of mesh k+1 (Trajectories/displacements/forward_*.npy); mesh k translucent, mesh k+1 wireframe.
      2. vertex trajectories up to frame k     : the polyline followed by every reference vertex through the
         p2p chain (trajectories.npy), coloured by time; the current mesh is translucent.
      3. smooth trajectory approximation       : the same curves from the dense interpolation of `scheme`
         (interpolation/<scheme>.npy, default = best validated scheme) - the continuous surface trajectory.
      4. surface trajectory (ghost stack)      : the meshes visited so far drawn with increasing opacity plus
         the trajectories of the last step - how the whole surface moved.
    Mode 1: 'h' hides / shows mesh k (the translucent mesh k+1 stays), 'f' the translucent mesh k+1, 'p' / 'l' make
    mesh k more opaque / transparent, 'w' / 'd' the translucent mesh.  Trajectory modes: 't' hides / shows the dots
    (the lines stay), 'p' / 'l' thicker / thinner lines, 'w' / 'd' opacity of the mesh, 'a' selects all / none of the
    dots and a left click on a dot selects / unselects it (only the trajectories of the selected dots are drawn,
    unselected dots are grey).  `stride` subsamples the vertices (1 = every vertex).
    """
    res = Path(results_path) / "Trajectories"
    X = np.load(res / "trajectories.npy"); F0 = np.load(res / "reference_faces.npy")
    T, n0, _ = X.shape
    names = _frame_names(results_path, mesh_path, T)
    frames = load_frames(mesh_path, loader) if mesh_path is not None else None
    disp_dir = res / "displacements"
    if scheme is None:                                       # best validated scheme that was actually exported
        available = [f.stem for f in (res / "interpolation").glob("*.npy") if f.stem not in ("query_times",)]
        try:
            import pandas as pd
            v = pd.read_csv(res / "interpolation" / "validation.csv"); col = [c for c in v.columns if "error" in c.lower()][0]
            ranked = [str(x) for x in v.sort_values(col)[v.columns[0]]]
            scheme = next((r for r in ranked if r in available), available[0] if available else "hermite_catmull_rom")
        except Exception:  # noqa: BLE001
            scheme = available[0] if available else "hermite_catmull_rom"
    dense = np.load(res / "interpolation" / f"{scheme}.npy") if (res / "interpolation" / f"{scheme}.npy").exists() else None
    tq = np.load(res / "interpolation" / "query_times.npy") if dense is not None else None
    t_obs = np.load(res / "time.npy") if (res / "time.npy").exists() else np.arange(T, dtype=float)
    idx = np.arange(0, n0, max(1, stride))
    diam = np.linalg.norm(X[0].max(0) - X[0].min(0))
    modes = ["displacement field k → k+1 (all vertices)", "vertex trajectories up to frame k",
             f"smooth trajectory approximation ({scheme})", "surface trajectory (ghost stack)"]

    def polylines(P):
        """P (S, m, 3) → PolyData of m polylines with S points, coloured by the time index."""
        pv = _pv(); S, m, _ = P.shape
        pts = P.transpose(1, 0, 2).reshape(-1, 3)
        lines = np.hstack([np.concatenate([[S], np.arange(v * S, (v + 1) * S)]) for v in range(m)])
        pd_ = pv.PolyData(pts, lines=lines); pd_["time step"] = np.tile(np.arange(S, dtype=float), m)
        return pd_

    OP_STEP, LW_STEP, LW_RANGE = 0.05, 1.25, (0.2, 12.0)

    class Viewer(_DotSelection, _FrameViewer):
        title = "Trajectories: vertex-to-vertex motion between the meshes"
        help_text = ("m : modality\n  1 displacement field of every vertex of mesh k towards its match in mesh k+1 (FM assignment)\n"
                     "  2 trajectories of all reference vertices through the sequence (p2p chain), coloured by time\n"
                     "  3 smooth (interpolated) trajectories = continuous surface trajectory\n"
                     "  4 ghost stack of the meshes visited so far\n"
                     "v : (mode 1) vector-field view of the displacements, no surface; x / y / z : projection on a plane (again = 3-D)\n"
                     "← → : frames\n\n"
                     "Mode 1 (displacement field): mesh k = grey surface (source of the arrows), mesh k+1 = translucent blue wireframe\n"
                     "  h : hide / show mesh k (the translucent mesh k+1 stays)\n"
                     "  f : hide / show the translucent mesh k+1\n"
                     "  p / l : mesh k more opaque / more transparent\n"
                     "  w / d : translucent mesh k+1 more opaque / more transparent\n"
                     "Modes 2 - 3 (trajectories):\n"
                     "  h : hide / show the current mesh | w / d : current mesh more opaque / more transparent\n"
                     "  t : hide / show the dots (vertex positions), the trajectory lines stay\n"
                     "  p / l : thicker / thinner trajectory lines\n"
                     "  a : select all <-> none of the dots | left click on a dot : select / unselect it\n"
                     "      (selected: original colour + trajectory; unselected: grey, no trajectory)\n"
                     "Mode 4 (ghost stack): h fades the meshes | w / d ghost meshes more opaque / more transparent | p / l line thickness\n"
                     "(opacity steps of 0.05, line width x1.25 per press; values are kept between frames and modes and shown in the label)")
        keys_line = ("← → frames | Enter play | space GIF | m modality | h mesh | f translucent mesh (1) | p/l opacity (1) or line width (2-4) | "
                     "w/d translucent opacity | t dots (2-3) | a/click select (2-3) | v vectors (1) | x/y/z projection | s screenshot | i help | q quit")

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.show_target = True        # 'f': translucent mesh k+1 (mode 1)
            self.op_mesh = 0.30            # p / l (mode 1): mesh k
            self.op_target = 0.12          # w / d (mode 1): translucent mesh k+1
            self.op_traj_mesh = 0.30       # w / d (modes 2-3): current mesh
            self.ghost_gain = 1.0          # w / d (mode 4): multiplier of the ghost opacities
            self.line_scale = 1.0          # p / l (modes 2-4): trajectory line width multiplier
            self.show_dots = True          # 't' (modes 2-3)
            self._init_selection(len(idx)) # 'a' / click (modes 2-3)

        def _selection_points(self):
            """Dots of modes 2 - 3 (vertex positions at frame k); None elsewhere (no selectable dots)."""
            return X[self.k, idx] if (self.mode % len(modes) in (1, 2) and self.show_dots) else None

        def _make_plotter(self):
            p = super()._make_plotter()
            self._install_selection(p)
            p.add_key_event("f", self.toggle_target)
            p.add_key_event("t", self.toggle_dots)
            p.add_key_event("p", lambda: self._pl_key(+1)); p.add_key_event("l", lambda: self._pl_key(-1))
            p.add_key_event("w", lambda: self._wd_key(+1)); p.add_key_event("d", lambda: self._wd_key(-1))
            return p

        # ---- new key actions ------------------------------------------------------------------------------
        def toggle_target(self):
            if self.mode % len(modes) == 0:
                self.show_target = not self.show_target; self.redraw()

        def toggle_dots(self):
            if self.mode % len(modes) in (1, 2):
                self.show_dots = not self.show_dots; self.redraw()

        def _pl_key(self, d):
            """p / l: opacity of mesh k in mode 1, trajectory line width in the trajectory modes."""
            if self.mode % len(modes) == 0:
                self.op_mesh = float(np.clip(round(self.op_mesh + d * OP_STEP, 3), 0.0, 1.0))
            else:
                self.line_scale = float(np.clip(self.line_scale * (LW_STEP if d > 0 else 1.0 / LW_STEP), *LW_RANGE))
            self.redraw()

        def _wd_key(self, d):
            """w / d: opacity of the translucent mesh (mode 1: mesh k+1; modes 2-3: current mesh; mode 4: ghosts)."""
            mode = self.mode % len(modes)
            if mode == 0:
                self.op_target = float(np.clip(round(self.op_target + d * OP_STEP, 3), 0.0, 1.0))
            elif mode in (1, 2):
                self.op_traj_mesh = float(np.clip(round(self.op_traj_mesh + d * OP_STEP, 3), 0.0, 1.0))
            else:
                self.ghost_gain = float(np.clip(self.ghost_gain * (1.25 if d > 0 else 0.8), 0.1, 10.0))
            self.redraw()

        def _displacement(self, k):
            """(source vertices, displacement to the matched vertices of mesh k+1) or None on the last frame."""
            if frames is not None and k + 1 < len(frames):
                f = disp_dir / f"forward_T{k:04d}_T{k + 1:04d}.npy"
                return (frames[k]["V"], np.load(f)) if f.exists() else (frames[k]["V"], None)
            if k + 1 < T:
                return X[k], X[k + 1] - X[k]
            return X[k], None

        def vector_field(self):
            if self.mode % len(modes) != 0:
                return None
            Vk, d = self._displacement(self.k)
            if d is None:
                return None
            sel = np.arange(0, Vk.shape[0], max(1, stride)); mag = np.linalg.norm(d, axis=1)
            return Vk[sel], d[sel], f"{modes[0]}  (arrows ×{0.08 * diam / (mag.max() + 1e-12):.3g})", 0.08 * diam / (mag.max() + 1e-12)

        def draw(self):
            self.clear(); p = self.plotter; k = self.k; mode = self.mode % len(modes)
            onoff = lambda b: "on" if b else "off"
            if mode == 0:
                if frames is not None and k + 1 < len(frames):
                    Vk, Fk = frames[k]["V"], frames[k]["F"]; Vn, Fn = frames[k + 1]["V"], frames[k + 1]["F"]
                    f = disp_dir / f"forward_T{k:04d}_T{k + 1:04d}.npy"
                    d = np.load(f) if f.exists() else None
                elif k + 1 < T:
                    Vk, Fk = X[k], F0; Vn, Fn = X[k + 1], F0; d = X[k + 1] - X[k]
                else:
                    Vk, Fk = X[k], F0; Vn, Fn = X[k], F0; d = None
                state = (f"mesh k {onoff(self.show_mesh)} (opacity {self.op_mesh:.2f}, p/l) | "
                         f"translucent mesh k+1 {onoff(self.show_target)} (opacity {self.op_target:.2f}, w/d)")
                if d is not None:
                    mag = np.linalg.norm(d, axis=1)
                    arrow_factor = 0.08 * diam / (mag.max() + 1e-12)   # per frame: the largest arrow is 8 % of the cell diameter
                    if self.show_mesh and self.op_mesh > 0:
                        p.add_mesh(_poly(Vk, Fk), color="lightgray", opacity=self.op_mesh, smooth_shading=True)
                    if self.show_target and self.op_target > 0:
                        p.add_mesh(_poly(Vn, Fn), style="wireframe", color="steelblue", opacity=self.op_target, line_width=0.4)
                    sel = np.arange(0, Vk.shape[0], max(1, stride))
                    p.add_mesh(_arrows(Vk[sel], d[sel], scale=arrow_factor), scalars="mag", cmap="plasma", scalar_bar_args=self.scalar("|displacement|"))
                    txt = (f"{modes[0]}   mean |d| = {mag.mean():.3g}, max {mag.max():.3g}   (arrows magnified ×{arrow_factor:.3g})\n{state}")
                else:
                    if self.show_mesh and self.op_mesh > 0:
                        p.add_mesh(_poly(Vk, Fk), color="lightgray", opacity=max(self.op_mesh, 0.5), smooth_shading=True)
                    txt = modes[0] + " (last frame: no next mesh)\n" + state
                self.label(txt)
            elif mode in (1, 2):
                if mode == 1 or dense is None:
                    P = X[:k + 1, idx]
                else:
                    kk = int(np.searchsorted(tq, t_obs[k] + 1e-9)); P = dense[:max(kk, 2), idx]
                if self.show_mesh and self.op_traj_mesh > 0:
                    p.add_mesh(_poly(X[k], F0), color="lightgray", opacity=self.op_traj_mesh, smooth_shading=True)
                sel = self.sel
                if P.shape[0] >= 2 and sel.any():                 # trajectories of the selected dots only
                    p.add_mesh(polylines(P[:, sel]), scalars="time step", cmap="viridis", line_width=1.5 * self.line_scale,
                               scalar_bar_args=self.scalar("time step"))
                if self.show_dots:
                    if sel.any():
                        p.add_points(X[k, idx[sel]], color="red", point_size=3, render_points_as_spheres=True)
                    if not sel.all():
                        p.add_points(X[k, idx[~sel]], color="darkgray", point_size=3, render_points_as_spheres=True)
                self.label(modes[mode] + ("" if dense is not None or mode == 1 else "  (no dense interpolation on disk)")
                           + f"\nmesh {onoff(self.show_mesh)} (opacity {self.op_traj_mesh:.2f}, w/d) | dots {onoff(self.show_dots)} (t) | "
                             f"line width ×{self.line_scale:.2f} (p/l)\n{self._selection_text()}")
            else:
                step = max(1, k // 6)
                for j in range(0, k + 1, step):
                    op = (0.06 + 0.45 * (j / max(k, 1)) ** 2) if self.show_mesh else 0.02
                    op = float(np.clip(op * self.ghost_gain, 0.0, 1.0))
                    p.add_mesh(_poly(X[j], F0), color="steelblue" if j < k else "wheat", opacity=op, smooth_shading=True)
                if k >= 1:
                    p.add_mesh(polylines(X[max(k - 1, 0):k + 1, idx]), color="black", line_width=1.0 * self.line_scale)
                self.label(modes[3] + f"\nghost opacity ×{self.ghost_gain:.2f} (w/d) | line width ×{self.line_scale:.2f} (p/l)")

    _launch(Viewer(T, len(modes), names=names, screenshot_dir=res / "plots", screenshot_name="trajectories"), modes, t_obs)


def visualize_interpolation(results_path, scheme="arap_polar", compare_with=None, mesh_path=None):
    """
    Animates the dense interpolated sequence interpolation/<scheme>.npy; with `compare_with` a second
    scheme is shown side by side.  The frame counter runs over the query times (observed + interpolated).
    """
    res = Path(results_path) / "Trajectories"
    F = np.load(res / "reference_faces.npy"); tq = np.load(res / "interpolation" / "query_times.npy"); t = np.load(res / "time.npy")
    A = np.load(res / "interpolation" / f"{scheme}.npy")
    B = np.load(res / "interpolation" / f"{compare_with}.npy") if compare_with else None
    observed = np.isin(np.round(tq, 8), np.round(t, 8))
    obs_names = _frame_names(results_path, mesh_path, len(t))
    names = []
    for q, o in zip(tq, observed):
        j = int(np.argmin(np.abs(t - q))); nm = obs_names[j] if j < len(obs_names) else str(j)
        names.append(f"{nm} (observed)" if o else f"t = {q:.3f} (interpolated, near {nm})")

    class Viewer(_FrameViewer):
        title = "Interpolation: dense surface trajectory"
        help_text = (f"Left : {scheme}" + (f"\nRight : {compare_with}" if B is not None else "") +
                     "\nThe counter runs over the query times (observed frames + interpolated sub-steps).\n← → : steps | Enter : play")
        keys_line = "← → steps | Enter play | space GIF | s screenshot | i help | q quit"

        def draw(self):
            self.clear(); p = self.plotter
            p.subplot(0, 0); p.add_mesh(_poly(A[self.k], F), color="wheat", smooth_shading=True)
            p.add_text(scheme, name="t0", position="upper_edge", font_size=8)
            if B is not None:
                p.subplot(0, 1); p.add_mesh(_poly(B[self.k], F), color="lightblue", smooth_shading=True)
                p.add_text(compare_with, name="t1", position="upper_edge", font_size=8)
            self.label()

    _launch(Viewer(A.shape[0], 1, shape=(1, 2 if B is not None else 1), names=names, screenshot_dir=res / "plots",
                   screenshot_name=f"interp_{scheme}"), None, tq)


# --------------------------------------------------------------------------- #
#  (d) Hodge decomposition, FTLE, shape modes
# --------------------------------------------------------------------------- #

def visualize_hodge(results_path, mesh_path=None, stride=2, arrow_scale=None):
    """
    Helmholtz–Hodge decomposition of the surface velocity.  'm' cycles the component (total tangential →
    curl-free ∇α → divergence-free J∇β → harmonic → normal speed); the surface is coloured by the matching
    potential (α: sources/sinks, β: stream function).  Arrows are thin glyphs on every `stride`-th face.
    'v' switches to the VECTOR-FIELD view (only the vectors, no mesh); there 'x', 'y' or 'z' project the field
    on the plane perpendicular to that axis (parallel projection with a grid, one picture at a time), the same
    key again returns to 3-D and 'v' returns to the surface view.  'h' hides the mesh in the surface view.
    """
    res = Path(results_path); trj = res / "Trajectories"; hd = res / "DynamicAnalysis" / "Hodge"
    X = np.load(trj / "trajectories.npy"); F = np.load(trj / "reference_faces.npy")
    vel = np.load(trj / "kinematics" / "velocity.npy")
    T = X.shape[0]; names = _frame_names(results_path, mesh_path, T)
    # plain-text operators: VTK hands labels with symbols missing from its font to matplotlib's mathtext, whose default
    # font has no nabla ('Font default does not have a glyph for \u2207' printed for every rendered frame)
    comps = ["total tangential velocity", "curl-free  grad(α)  (sources / sinks)", "divergence-free  J grad(β)  (vortices)", "harmonic", "normal speed"]
    diam = np.linalg.norm(X[0].max(0) - X[0].min(0))
    vmax = max(np.linalg.norm(np.load(hd / f"components_T{k:04d}.npy")[0], axis=1).max() for k in range(0, T, max(1, T // 10))) + 1e-12
    scale = arrow_scale or 0.12 * diam / vmax                 # longest arrow ≈ 12 % of the cell diameter

    class Viewer(_FrameViewer):
        title = "Helmholtz–Hodge decomposition"
        help_text = ("m : component (total tangential → curl-free grad(α) → div-free J grad(β) → harmonic → normal speed)\n"
                     "Surface layout: mesh coloured by the potential (α, β) or the divergence, thin arrows on the faces\n"
                     "v : vector-field view (only the vectors, no surface); then x / y / z : projection on the plane perpendicular\n    to that axis (same key again = back to 3-D)\n"
                     "h : hide / show the mesh (surface layout)\n← → : frames")
        keys_line = "← → frames | Enter play | space GIF | m component | v vectors only | x/y/z projection | h mesh | s screenshot | i help | q quit"

        def _field(self):
            V = X[self.k]; comp = np.load(hd / f"components_T{self.k:04d}.npy"); c = self.mode % len(comps)
            centers = V[F].mean(axis=1)
            if c == 4:
                N = vertex_normals(V, F); vn = np.einsum("ij,ij->i", vel[self.k], N)
                field = (vn[F].mean(axis=1))[:, None] * (N[F].mean(axis=1))
            else:
                field = comp[c]
            return V, centers, field, c

        def vector_field(self):
            _, centers, field, c = self._field(); sel = slice(None, None, max(1, stride))
            frame_scale = 0.12 * diam / (np.linalg.norm(field, axis=1).max() + 1e-12)     # per frame: longest arrow 12 % of the cell
            return centers[sel], field[sel], f"component: {comps[c]}  (arrows ×{frame_scale:.3g})", frame_scale

        def draw(self):
            self.clear(); p = self.plotter
            V, centers, field, c = self._field()
            sel = slice(None, None, max(1, stride))
            if self.show_mesh:
                m = _poly(V, F); pot = np.load(hd / f"potentials_T{self.k:04d}.npy")
                if c == 4:
                    m["normal speed"] = np.einsum("ij,ij->i", vel[self.k], vertex_normals(V, F)); key = "normal speed"
                else:
                    key = "alpha (sources/sinks)" if c in (0, 1) else ("beta (stream function)" if c == 2 else "divergence")
                    m[key] = pot[:, 0] if c in (0, 1) else (pot[:, 1] if c == 2 else pot[:, 2])
                p.add_mesh(m, scalars=key, cmap="RdBu_r", smooth_shading=True, opacity=0.9, scalar_bar_args=self.scalar(key))
            p.add_mesh(_arrows(centers[sel], field[sel], scale), color="black")
            self.label(f"component: {comps[c]}    (rms |v| = {np.sqrt((field ** 2).sum(1).mean()):.3g})")

    _launch(Viewer(T, len(comps), names=names, screenshot_dir=res / "DynamicAnalysis" / "plots", screenshot_name="hodge"),
            comps, _observed_times(results_path, T))


def visualize_ftle(results_path, mesh_path=None):
    """Reference mesh coloured by the finite-time Lyapunov exponent of the flow map Φ(0 → k+1)."""
    res = Path(results_path); X = np.load(res / "Trajectories" / "trajectories.npy"); F = np.load(res / "Trajectories" / "reference_faces.npy")
    files = _sorted(res / "DynamicAnalysis" / "FTLE", "ftle_T*.npy")
    fields = [np.load(f) for f in files]
    clim = [np.percentile(np.concatenate(fields), 2), np.percentile(np.concatenate(fields), 98)]
    names = _frame_names(results_path, mesh_path, X.shape[0])

    class Viewer(_FrameViewer):
        title = "FTLE (d): Lagrangian coherent structures"
        help_text = ("Colour = finite-time Lyapunov exponent of the flow map from frame 1 to the current frame.\n"
                     "Ridges (bright) separate regions of the surface with different fate.\n← → : frames | h : deformed / reference geometry")
        keys_line = "← → frames | Enter play | space GIF | h geometry | s screenshot | i help | q quit"

        def draw(self):
            self.clear(); p = self.plotter
            V = X[self.k + 1] if (self.show_mesh and self.k + 1 < X.shape[0]) else X[0]
            m = _poly(V, F); m["FTLE"] = fields[self.k]
            p.add_mesh(m, scalars="FTLE", cmap="inferno", clim=clim, smooth_shading=True, scalar_bar_args=self.scalar("FTLE"))
            self.label(f"FTLE of Φ(frame 1 → frame {self.k + 2})  on the {'deformed' if self.show_mesh else 'reference'} geometry")

    t_all = _observed_times(results_path, X.shape[0])
    _launch(Viewer(len(fields), 1, names=names[1:] if len(names) > 1 else names, screenshot_dir=res / "DynamicAnalysis" / "plots",
                   screenshot_name="ftle"), None, t_all[1:len(fields) + 1] if t_all is not None and len(t_all) > len(fields) else None)


def visualize_shape_modes(results_path, n_modes=None):
    """Mean shape (centre) and −2σ / +2σ of each principal deformation mode (Right/Left cycles the modes)."""
    d = Path(results_path) / "DynamicAnalysis" / "ShapeSpace" / "mode_meshes"
    modes = sorted({int(re.search(r"mode(\d+)_", f.stem).group(1)) for f in d.glob("mode*_plus2sd.obj")})
    if n_modes:
        modes = modes[:n_modes]
    Vm, Fm = read_obj(d / "mean_shape.obj")

    class Viewer(_FrameViewer):
        title = "Shape space: principal deformation modes"
        help_text = "Left : mean − 2σ · mode | Centre : mean shape | Right : mean + 2σ · mode\n← → : cycle the modes (the 'frame' counter is the mode index)"
        keys_line = "← → modes | s screenshot | i help | q quit"

        def draw(self):
            self.clear(); p = self.plotter; j = modes[self.k]
            for col, (tag, color, ttl) in enumerate((("minus2sd", "lightblue", "−2σ"), (None, "wheat", "mean shape"), ("plus2sd", "salmon", "+2σ"))):
                p.subplot(0, col)
                V, F = (Vm, Fm) if tag is None else read_obj(d / f"mode{j}_{tag}.obj")
                p.add_mesh(_poly(V, F), color=color, smooth_shading=True)
                p.add_text(f"mode {j}: {ttl}", name=f"t{col}", position="upper_edge", font_size=8)
            self.label(f"shape mode {j}")

    _launch(Viewer(len(modes), 1, shape=(1, 3), names=[f"mode {j}" for j in modes], screenshot_dir=d.parent.parent / "plots",
                   screenshot_name="shape_modes"), None, None, animate=False)


#  (c) dynamic trajectories: the motion itself, not the meshes
# --------------------------------------------------------------------------- #

def visualize_dynamic_trajectories(results_path, mesh_path=None, scheme=None, stride=1, trail=12):
    """
    Animated view of the vertex trajectories as a MOVING PARTICLE FIELD (no surface unless asked):
    every vertex is a particle that travels along its (dense, interpolated) trajectory; press space to run.
    Modalities ('m'):
      1. particles + comet trails      : heads coloured by speed, trail of the last `trail` steps
      2. trajectory history            : the curves travelled so far, coloured by time, heads on top
      3. particles over the surface    : heads + trails over the translucent current surface
      4. streamers                     : all complete trajectories faint in the background + moving heads
    Up / Down lengthen / shorten the trail, 'h' toggles the surface in mode 3, 'p' / 'l' its opacity, 't' the
    particles (the trails stay), 'k' / 'j' the line width, ← → step the (dense) time steps; 'a' selects all / none of
    the particles and a left click on a particle selects / unselects it (only the trajectories of the selected
    particles are drawn, unselected particles are grey).
    The dense trajectories come from interpolation/<scheme>.npy (default: the best validated scheme on disk),
    so the motion between two meshes is shown continuously; the label tells which meshes bracket the time.
    """
    res = Path(results_path) / "Trajectories"
    X = np.load(res / "trajectories.npy"); F0 = np.load(res / "reference_faces.npy")
    T, n0, _ = X.shape
    t_obs = np.load(res / "time.npy") if (res / "time.npy").exists() else np.arange(T, dtype=float)
    obs_names = _frame_names(results_path, mesh_path, T)
    available = [f.stem for f in (res / "interpolation").glob("*.npy") if f.stem != "query_times"]
    if scheme is None and available:
        try:
            import pandas as pd
            v = pd.read_csv(res / "interpolation" / "validation.csv"); col = [c for c in v.columns if "error" in c.lower()][0]
            ranked = [str(x) for x in v.sort_values(col)[v.columns[0]]]
            scheme = next((r for r in ranked if r in available), available[0])
        except Exception:  # noqa: BLE001
            scheme = available[0]
    if scheme and (res / "interpolation" / f"{scheme}.npy").exists():
        P = np.load(res / "interpolation" / f"{scheme}.npy"); tq = np.load(res / "interpolation" / "query_times.npy")
    else:
        P, tq, scheme = X, t_obs, "observed frames only"
    S = P.shape[0]
    idx = np.arange(0, n0, max(1, stride)); P = P[:, idx]
    speed = np.zeros((S, len(idx)))
    if S > 1:
        speed[1:] = np.linalg.norm(np.diff(P, axis=0), axis=2) / np.maximum(np.diff(tq)[:, None], 1e-12); speed[0] = speed[1]
    smax = np.percentile(speed, 98) + 1e-12
    diam = np.linalg.norm(P[0].max(0) - P[0].min(0))
    names = []
    for q in tq:
        j = int(np.argmin(np.abs(t_obs - q)))
        if abs(t_obs[j] - q) < 1e-9:
            names.append(f"t = {q:.4g}  (mesh {obs_names[j] if j < len(obs_names) else j})")
        else:
            a = int(np.searchsorted(t_obs, q) - 1); b = min(a + 1, T - 1)
            names.append(f"t = {q:.4g}  (between {obs_names[a] if a < len(obs_names) else a} and {obs_names[b] if b < len(obs_names) else b})")
    modes = ["particles + comet trails", "trajectory history", "particles over the surface", "streamers"]
    trail0 = max(1, int(trail))

    def polylines(Q, scalars=None, name="v"):
        pv = _pv(); s_, m, _ = Q.shape
        pts = Q.transpose(1, 0, 2).reshape(-1, 3)
        lines = np.hstack([np.concatenate([[s_], np.arange(v * s_, (v + 1) * s_)]) for v in range(m)])
        pd_ = pv.PolyData(pts, lines=lines)
        if scalars is not None:
            pd_[name] = scalars.T.reshape(-1)
        return pd_

    class Viewer(_DotSelection, _FrameViewer):
        title = f"Dynamic trajectories: particle field of the surface motion  [{scheme}]"
        help_text = ("Every vertex is a particle moving along its interpolated trajectory (functional-map assignment).\n"
                     "space : run / pause the motion | ← → : single time steps\nm : modality\n"
                     "  1 particles coloured by speed + comet trails\n  2 trajectory history (curves travelled so far, coloured by time)\n"
                     "  3 particles + trails over the translucent surface (h hides it)\n"
                     "  4 streamers: complete trajectories faint in the background + moving heads\n"
                     "v : velocity-field view (only the particle velocity vectors); x / y / z : projection on a plane (again = 3-D)\n"
                     "Up / Down : longer / shorter trail\n"
                     "p / l : surface more opaque / more transparent (mode 3; steps of 0.05)\n"
                     "t : hide / show the particles (the trails / trajectory lines stay)\n"
                     "k / j : thicker / thinner trajectory lines (all modes, x1.25 per press)\n"
                     "a : select all <-> none of the particles | left click on a particle : select / unselect it\n"
                     "    (selected: original colour + trajectory; unselected: grey, no trajectory)\n"
                     "(values are kept between steps and modes and shown in the label)")
        keys_line = ("← → steps | Enter run | space GIF | m modality | v velocity vectors | x/y/z projection | Up/Down trail | h surface | "
                     "p/l surface opacity | t particles | k/j line width | a/click select | s screenshot | i help | q quit")
        trail = trail0

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.op_surface = 0.25         # p / l (mode 3)
            self.show_dots = True          # 't'
            self.line_scale = 1.0          # k / j
            self._init_selection(len(idx)) # 'a' / click

        def _selection_points(self):
            """Particle positions at the current step (every mode shows particles); None when they are hidden."""
            return P[self.k] if self.show_dots else None

        def vector_field(self):
            k = self.k
            if S < 2:
                return None
            a, b = (k, k + 1) if k + 1 < S else (k - 1, k)
            vec = (P[b] - P[a]) / max(tq[b] - tq[a], 1e-12)
            return P[k], vec, f"particle velocity field at {names[k]}", 0.08 * diam / (np.linalg.norm(vec, axis=1).max() + 1e-12)

        def _make_plotter(self):
            p = super()._make_plotter()
            self._install_selection(p)
            p.add_key_event("Up", lambda: self._trail(+2)); p.add_key_event("Down", lambda: self._trail(-2))
            p.add_key_event("p", lambda: self._opacity(+0.05)); p.add_key_event("l", lambda: self._opacity(-0.05))
            p.add_key_event("t", self._toggle_dots)
            p.add_key_event("k", lambda: self._width(1.25)); p.add_key_event("j", lambda: self._width(0.8))
            return p

        def _opacity(self, d):
            self.op_surface = float(np.clip(round(self.op_surface + d, 3), 0.0, 1.0)); self.redraw()

        def _toggle_dots(self):
            self.show_dots = not self.show_dots; self.redraw()

        def _width(self, f):
            self.line_scale = float(np.clip(self.line_scale * f, 0.2, 12.0)); self.redraw()

        def _trail(self, d):
            self.trail = int(np.clip(self.trail + d, 1, S)); self.redraw()

        def _heads(self, k, color=None):
            """Particles: the selected ones coloured by speed (or `color`), the unselected ones grey."""
            if not self.show_dots:
                return
            pv = _pv(); sel = self.sel
            if sel.any():
                pd_ = pv.PolyData(P[k][sel])
                if color is None:
                    pd_["speed"] = speed[k][sel]
                    self.plotter.add_mesh(pd_, scalars="speed", cmap="plasma", clim=[0, smax], point_size=6,
                                          render_points_as_spheres=True, scalar_bar_args=self.scalar("speed"))
                else:
                    self.plotter.add_mesh(pd_, color=color, point_size=5, render_points_as_spheres=True)
            if not sel.all():
                self.plotter.add_mesh(pv.PolyData(P[k][~sel]), color="darkgray", point_size=5 if color else 6,
                                      render_points_as_spheres=True)

        def _trails(self, k, color_by_speed=True):
            a = max(0, k - self.trail); sel = self.sel
            if k - a >= 1 and sel.any():                 # trails of the selected particles only
                Q = P[a:k + 1][:, sel]
                if color_by_speed:                       # the heads carry the speed bar; without heads the trails do
                    bar = {} if self.show_dots else {"scalar_bar_args": self.scalar("speed")}
                    self.plotter.add_mesh(polylines(Q, speed[a:k + 1][:, sel], "speed"), scalars="speed", cmap="plasma", clim=[0, smax],
                                          line_width=1.2 * self.line_scale, opacity=0.7, show_scalar_bar=not self.show_dots, **bar)
                else:
                    self.plotter.add_mesh(polylines(Q), color="black", line_width=1.0 * self.line_scale, opacity=0.5)

        def draw(self):
            self.clear(); p = self.plotter; k = self.k; mode = self.mode % len(modes)
            if mode == 0:
                self._trails(k); self._heads(k)
            elif mode == 1:
                if k >= 1 and self.sel.any():
                    Q = P[:k + 1][:, self.sel]; tt = np.tile(tq[:k + 1][:, None], (1, Q.shape[1]))
                    p.add_mesh(polylines(Q, tt, "time"), scalars="time", cmap="viridis", line_width=1.2 * self.line_scale,
                               scalar_bar_args=self.scalar("time"))
                self._heads(k, color="red")
            elif mode == 2:
                if self.show_mesh and self.op_surface > 0:
                    if stride == 1:                                   # dense surface at this very time step
                        p.add_mesh(_poly(P[k], F0), color="lightgray", opacity=self.op_surface, smooth_shading=True)
                    else:                                             # subsampled particles: nearest observed mesh
                        j = int(np.argmin(np.abs(t_obs - tq[k])))
                        p.add_mesh(_poly(X[j], F0), color="lightgray", opacity=self.op_surface, smooth_shading=True)
                self._trails(k); self._heads(k)
            else:
                if self.sel.any():
                    p.add_mesh(polylines(P[:, self.sel]), color="gray", line_width=0.6 * self.line_scale, opacity=0.12)
                self._trails(k, color_by_speed=False); self._heads(k)
            onoff = lambda b: "on" if b else "off"
            surf = (f"surface {onoff(self.show_mesh)} (opacity {self.op_surface:.2f}, p/l) | " if mode == 2
                    else f"surface opacity {self.op_surface:.2f} (p/l, shown in mode 3) | ")
            self.label(f"{modes[mode]}   trail = {self.trail} steps   ({len(idx)} particles)\n"
                       f"{surf}particles {onoff(self.show_dots)} (t) | line width ×{self.line_scale:.2f} (k/j)\n{self._selection_text()}")

    _launch(Viewer(S, len(modes), names=names, screenshot_dir=res / "plots", screenshot_name="dynamic_trajectories"), modes, tq)

# --------------------------------------------------------------------------- #
#  graph animation (graph_animation.py)
# --------------------------------------------------------------------------- #

GRAPH_TYPE_COLORS = {"maximum": (220, 30, 30), "minimum": (40, 90, 220), "saddle": (40, 180, 60), "center": (230, 180, 20)}
PANEL_SURFACE_COLORS = ("lightgray", "#cfe3ea", "#efe0c4", "#dcd3ec")


def visualize_graph_animation(results_path, mesh_path=None, graph=None, frame=None, model=None, scheme=None, trail=10):
    """
    A Reeb / Morse–Smale graph of one frame moved by the trajectory model (Results/<scene>/GraphAnimation, written by
    graph_animation.compute_graph_animation).

    scheme (or model) : None    -> every animation on disk, 'g' cycles them;
                        'name'  -> only the animations made with that trajectory model ('best' = the best validated);
                        ('a', 'b', ...) -> COMPARATIVE MODE: one panel per scheme side by side (like visualize_interpolation),
                                   the same graph and source frame, synchronised in time, linked cameras; every panel
                                   tells how far its nodes are from those of the first panel ('g' cycles graph / frame).
    graph / frame select the first graph kind ('reeb' | 'mscomplex') and source frame shown.
    Modalities ('m'):
      1. skeleton on the moving surface   : nodes + edges over the translucent surface of the trajectory model
      2. skeleton + node trails           : the path of every node over the last steps (Up / Down: trail length)
      3. edge strain                      : edges coloured by L / L_source − 1 (red stretched, blue shortened), nodes by speed
      4. animated vs actual               : the animated graph (blue) and the graph computed on the nearest observed
                                            frame (orange): where they differ, the structure changed, not only moved
    Keys: h surface | p / l surface more opaque / transparent | t nodes on/off (edges stay) | k / j thicker / thinner
    edges | u / n bigger / smaller nodes | g next animation | v node velocity field (x / y / z projection) | s screenshot.
    """
    from PynamicMesh.core.graph_animation import trajectory_model, graph_files
    import pickle as _pickle
    root = Path(results_path); ga_dir = root / "GraphAnimation"
    folders = sorted(p for p in ga_dir.glob("*") if (p / "animation.npz").exists()) if ga_dir.is_dir() else []
    if not folders:
        print(f"Error: no graph animation in {ga_dir} (run graph_animation.compute_graph_animation / compute_graph_animation=True).")
        return
    anims = []
    for f in folders:
        z = np.load(f / "animation.npz", allow_pickle=True)
        anims.append({"folder": f, **{k: z[k] for k in z.files}})

    # ---- requested schemes ('best' resolved to the scheme it designates) ------------------------------------------
    req = scheme if scheme is not None else model
    req = [] if req is None else ([req] if isinstance(req, str) else list(req))

    def _resolve(m):
        if m == "best":
            try:
                return trajectory_model(root, "best")[4]
            except Exception:  # noqa: BLE001
                return m
        return m
    req = list(dict.fromkeys(_resolve(str(m)) for m in req))
    on_disk = sorted({str(a["model"]) for a in anims})
    missing = [m for m in req if m not in on_disk]
    req = [m for m in req if m in on_disk]
    if missing and not req:
        print(f"Error: no graph animation with {missing} on disk (available models: {on_disk}); add them to "
              f"graph_animation_params['model'] and run the pipeline.")
        return
    compare = len(req) >= 2
    if len(req) > 4:
        req = req[:4]                                       # at most 4 panels side by side

    # ---- groups: (graph kind, source frame) -> {model: animation} --------------------------------------------------
    groups = {}
    for a in anims:
        groups.setdefault((str(a["graph"]), int(a["source_frame"])), {})[str(a["model"])] = a
    if compare:
        keys = [k for k, g in groups.items() if sum(m in g for m in req) >= 2]
        if not keys:
            print(f"Error: no graph animated with at least two of {req} (on disk: {on_disk}).")
            return
        views = [[groups[k][m] for m in req if m in groups[k]] for k in keys]
    else:
        views = [[a] for a in anims if not req or str(a["model"]) in req]
    def _match(v):
        a = v[0]
        return (graph is None or str(a["graph"]) == graph) and (frame is None or int(a["source_frame"]) == int(frame))
    start = next((i for i, v in enumerate(views) if _match(v)), 0)
    n_panels = max(len(v) for v in views)

    models = {}

    def surface_of(a):
        key = str(a["model"])
        if key not in models:
            P, times, F0, obs_step, name, X = trajectory_model(root, key)
            models[key] = (P, F0)
        return models[key]

    actual_cache = {}

    def actual_graph(a, j):
        key = (str(a["graph"]), j)
        if key not in actual_cache:
            files = graph_files(root, str(a["graph"]))
            if j in files:
                with open(files[j], "rb") as fh:
                    G = _pickle.load(fh)
                keys_ = list(G.nodes); idx = {k: i for i, k in enumerate(keys_)}
                P_ = np.array([np.asarray(G.nodes[k]["pos"], float) for k in keys_])
                E_ = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int64).reshape(-1, 2)
                actual_cache[key] = (P_, E_)
            else:
                actual_cache[key] = None
        return actual_cache[key]

    strain_cache = {}

    def strain_of(a):
        f = str(a["folder"])
        if f not in strain_cache:
            st = np.load(Path(f) / "edge_strain.npy") if len(a["edges"]) else np.zeros((len(a["times"]), 0))
            strain_cache[f] = (st, float(np.nanpercentile(np.abs(st), 99)) if st.size else 1.0)
        return strain_cache[f]

    obs_names = _frame_names(results_path, mesh_path, None)
    modes = ["skeleton on the moving surface", "skeleton + node trails", "edge strain", "animated vs actual graph"]

    def lines_poly(P, E, scal=None, name="s"):
        pv = _pv()
        if not len(E):
            return pv.PolyData(np.asarray(P, float))
        pd_ = pv.PolyData(np.asarray(P, float), lines=np.hstack([np.full((len(E), 1), 2), E]).ravel())   # line cells only
        if scal is not None:
            pd_.cell_data[name] = np.asarray(scal, float)
        return pd_

    class Viewer(_FrameViewer):
        title = ("Graph animation: comparison of trajectory models" if compare else
                 "Graph animation: a graph of one frame moved by the trajectory model")
        help_text = ("A Reeb / Morse-Smale graph of one frame is anchored to the moving reference mesh and follows the motion.\n"
                     + (f"COMPARATIVE MODE: one panel per trajectory model {req}, synchronised in time (linked cameras);\n"
                        "  each panel shows the mean / max distance of its nodes to those of the first panel.\n" if compare else "")
                     + "m : modality\n  1 skeleton on the moving surface\n  2 skeleton + node trails (Up / Down: trail length)\n"
                     "  3 edge strain: edges coloured by L / L_source - 1 (red stretched, blue shortened), nodes by speed\n"
                     "  4 animated graph (blue) vs the graph computed on the nearest observed frame (orange)\n"
                     "Enter : play / pause | ← → : steps\n"
                     + ("g : next graph kind / source frame (same models)\n" if compare else
                        "g : next animation (graph kind / source frame / model on disk)\n")
                     + "h : surface on / off | p / l : surface more opaque / more transparent\n"
                     "t : nodes on / off (the edges stay) | k / j : thicker / thinner edges | u / n : bigger / smaller nodes\n"
                     "v : node velocity field of the first panel (only the vectors); x / y / z : projection on a plane (again = 3-D)\n"
                     "(values are kept between steps, modes and animations and shown in the label)")
        keys_line = ("← → steps | Enter play | space GIF | m modality | g animation | h surface | p/l opacity | t nodes | k/j edge width | "
                     "u/n node size | Up/Down trail | v velocities | s screenshot | i help | q quit")

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.vi = start; self.op_surface = 0.18; self.show_nodes = True; self.line_scale = 1.0; self.node_scale = 1.0
            self.trail = int(trail); self.screenshot_name = self._shot_name()

        @property
        def V(self):
            return views[self.vi]

        @property
        def A(self):                                       # first panel (time base, velocity field, labels)
            return self.V[0]

        def _shot_name(self):
            a = self.A
            mods = "_vs_".join(str(x["model"]) for x in self.V)
            return f"{a['graph']}_T{int(a['source_frame']):04d}_{mods}"

        def _make_plotter(self):
            p = super()._make_plotter()
            for k_ in ("Up", "Down"):                          # pyvista's default zoom on these keys: the trail only
                try:
                    p.clear_events_for_key(k_)
                except Exception:  # noqa: BLE001
                    pass
            p.add_key_event("g", self._next_anim)
            p.add_key_event("p", lambda: self._op(+0.05)); p.add_key_event("l", lambda: self._op(-0.05))
            p.add_key_event("t", self._toggle_nodes)
            p.add_key_event("k", lambda: self._lw(1.25)); p.add_key_event("j", lambda: self._lw(0.8))
            p.add_key_event("u", lambda: self._ns(1.25)); p.add_key_event("n", lambda: self._ns(0.8))
            p.add_key_event("Up", lambda: self._trail(+2)); p.add_key_event("Down", lambda: self._trail(-2))
            return p

        def _next_anim(self):
            self.vi = (self.vi + 1) % len(views)
            self.n = int(self.A["positions"].shape[0]); self.k = min(self.k, self.n - 1)
            self.names = self._names(); self.screenshot_name = self._shot_name(); self.redraw()

        def _op(self, d):
            self.op_surface = float(np.clip(round(self.op_surface + d, 3), 0.0, 1.0)); self.redraw()

        def _toggle_nodes(self):
            self.show_nodes = not self.show_nodes; self.redraw()

        def _lw(self, f):
            self.line_scale = float(np.clip(self.line_scale * f, 0.2, 12.0)); self.redraw()

        def _ns(self, f):
            self.node_scale = float(np.clip(self.node_scale * f, 0.1, 10.0)); self.redraw()

        def _trail(self, d):
            self.trail = int(np.clip(self.trail + d, 1, self.n)); self.redraw()

        def _names(self):
            a = self.A; times = a["times"]; obs = a["observed_step"]
            out = []
            for s_ in range(len(times)):
                j = int(np.argmin(np.abs(obs - s_)))
                nm = obs_names[j] if j < len(obs_names) else f"T{j:04d}"
                out.append(f"t = {times[s_]:.4g}  ({'mesh ' + nm if obs[j] == s_ else 'interpolated, near ' + nm})")
            return out

        def _step_of(self, a):
            """Step of animation `a` at the time of the current step of the first panel (models can have other grids)."""
            if a is self.A:
                return self.k
            t = float(self.A["times"][self.k])
            return int(np.argmin(np.abs(a["times"] - t)))

        def vector_field(self):
            a = self.A; pos = a["positions"]; times = a["times"]; s_ = self.k
            if pos.shape[0] < 2:
                return None
            s0, s1 = (s_, s_ + 1) if s_ + 1 < pos.shape[0] else (s_ - 1, s_)
            vel = (pos[s1] - pos[s0]) / max(times[s1] - times[s0], 1e-12)
            mag = np.linalg.norm(vel, axis=1).max() + 1e-12
            return pos[s_], vel, f"node velocities of the {a['graph']} graph ({a['model']})", 0.08 * float(a["diag"]) / mag

        def _nodes(self, P, diag, colors=None, scalars=None, cmap=None, clim=None, bar=None, radius_mul=1.0, color=None):
            if not self.show_nodes or not len(P):
                return
            pv = _pv(); pts = pv.PolyData(np.asarray(P, float))
            r = 0.009 * float(diag) * self.node_scale * radius_mul
            if colors is not None:
                pts.point_data["rgb"] = np.asarray(colors, dtype=np.uint8)
            if scalars is not None:
                pts.point_data["val"] = np.asarray(scalars, float)
            g = pts.glyph(geom=pv.Sphere(radius=r, theta_resolution=12, phi_resolution=12), scale=False, orient=False)
            if colors is not None:
                self.plotter.add_mesh(g, scalars="rgb", rgb=True, smooth_shading=True)
            elif scalars is not None:
                self.plotter.add_mesh(g, scalars="val", cmap=cmap, clim=clim, smooth_shading=True,
                                      scalar_bar_args=self.scalar(bar) if bar else None, show_scalar_bar=bar is not None)
            else:
                self.plotter.add_mesh(g, color=color or "black", smooth_shading=True)

        def _node_colors(self, a):
            types = [str(t) for t in a["node_type"]]
            if str(a["graph"]) == "mscomplex":
                return np.array([GRAPH_TYPE_COLORS.get(t, (90, 90, 90)) for t in types]), None
            return None, a["node_f"]

        def draw_panel(self, c, a, mode):
            """One panel: animation `a` at the synchronised step, in modality `mode`; returns its label text."""
            p = self.plotter; p.subplot(0, c)
            s_ = self._step_of(a); pos = a["positions"]; E = a["edges"]; diag = float(a["diag"])
            first = c == 0
            if self.show_mesh and self.op_surface > 0 and mode != 3:
                P, F0 = surface_of(a)
                if s_ < P.shape[0]:
                    p.add_mesh(_poly(P[s_], F0), color=PANEL_SURFACE_COLORS[c % len(PANEL_SURFACE_COLORS)],
                               opacity=self.op_surface, smooth_shading=True)
            lw = 2.5 * self.line_scale
            extra = ""
            if mode in (0, 1):
                if len(E):
                    p.add_mesh(lines_poly(pos[s_], E), color="black", line_width=lw)
                cols, f = self._node_colors(a)
                if cols is not None:
                    self._nodes(pos[s_], diag, colors=cols)
                else:                                       # same colour range in every panel (same graph)
                    self._nodes(pos[s_], diag, scalars=f, cmap="viridis", clim=[float(np.min(f)), float(np.max(f)) + 1e-12],
                                bar="node f value" if first else None)
                if mode == 1 and s_ >= 1:
                    a0 = max(0, s_ - self.trail); Q = pos[a0:s_ + 1]
                    m_ = Q.shape[1]; S_ = Q.shape[0]
                    pts = Q.transpose(1, 0, 2).reshape(-1, 3)
                    lines = np.hstack([np.concatenate([[S_], np.arange(v * S_, (v + 1) * S_)]) for v in range(m_)])
                    tr = _pv().PolyData(pts, lines=lines); tr["step"] = np.tile(np.arange(a0, s_ + 1, dtype=float), m_)
                    p.add_mesh(tr, scalars="step", cmap="plasma", line_width=1.2 * self.line_scale, opacity=0.8, show_scalar_bar=False)
                    extra = f"trail {self.trail} steps (Up/Down)"
            elif mode == 2:
                st_all, v = strain_of(a)
                strain = st_all[s_] if len(E) else np.zeros(0)
                if compare:                                 # one strain range for all the panels
                    v = max(strain_of(b)[1] for b in self.V)
                if len(E):
                    p.add_mesh(lines_poly(pos[s_], E, strain, "edge strain"), scalars="edge strain", preference="cell",
                               cmap="RdBu_r", clim=[-max(v, 1e-6), max(v, 1e-6)], line_width=lw * 1.4,
                               scalar_bar_args=self.scalar("edge strain L/L0 - 1"), show_scalar_bar=first)
                times = a["times"]
                s0, s1 = (s_, s_ + 1) if s_ + 1 < pos.shape[0] else (max(s_ - 1, 0), s_)
                spd = np.linalg.norm(pos[s1] - pos[s0], axis=1) / max(times[s1] - times[s0], 1e-12)
                self._nodes(pos[s_], diag, scalars=spd, cmap="plasma", clim=[0, float(spd.max()) + 1e-12], bar=None)
                extra = f"mean |strain| {np.abs(strain).mean() if len(strain) else 0:.3f}, max {np.abs(strain).max() if len(strain) else 0:.3f}"
            else:
                obs = a["observed_step"]; j = int(np.argmin(np.abs(obs - s_)))
                if len(E):
                    p.add_mesh(lines_poly(pos[s_], E), color="royalblue", line_width=lw)
                self._nodes(pos[s_], diag, color="royalblue")
                act = actual_graph(a, j)
                if act is not None:
                    Pj, Ej = act
                    if len(Ej):
                        p.add_mesh(lines_poly(Pj, Ej), color="darkorange", line_width=lw)
                    self._nodes(Pj, diag, color="darkorange", radius_mul=0.8)
                extra = f"blue = animated, orange = graph of observed frame {j}" + ("" if obs[j] == s_ else " (nearest)")
                comp = Path(a["folder"]) / "comparison.csv"
                if comp.exists() and not compare:
                    try:
                        import pandas as _pd
                        C = _pd.read_csv(comp); row = C[C.frame == j]
                        if len(row):
                            extra += (f"\nChamfer {float(row.chamfer_rel.iloc[0]):.4f}, Hausdorff {float(row.hausdorff_rel.iloc[0]):.4f} of the size;"
                                      f" nodes animated {int(row.n_nodes_animated.iloc[0])} / actual {int(row.n_nodes_actual.iloc[0])}")
                    except Exception:  # noqa: BLE001
                        pass
            if compare:                                     # panel title: the model and its distance to the first panel
                if first:
                    dev = "(reference)"
                else:
                    d = np.linalg.norm(pos[s_] - self.A["positions"][self.k], axis=1) / diag
                    dev = f"nodes vs {self.A['model']}:\nmean {d.mean():.4f}  max {d.max():.4f} (/ size)"
                short = ""
                if mode == 2 and not first and len(E):
                    short = f"\n|strain| mean {np.abs(st_all[s_]).mean():.3f}  max {np.abs(st_all[s_]).max():.3f}"
                p.add_text(f"{a['model']}  {dev}{short}", name=f"panel{c}", position="upper_right", font_size=7)
            return extra

        def draw(self):
            self.clear(); mode = self.mode % len(modes)
            extras = [self.draw_panel(c, a, mode) for c, a in enumerate(self.V)]
            a = self.A; onoff = lambda b: "on" if b else "off"
            what = (f"{a['graph']} graph of frame {int(a['source_frame'])}: " +
                    (f"{len(self.V)} models side by side" if compare else f"moved by '{a['model']}'"))
            self.label(f"{what}  [{self.vi + 1}/{len(views)}]   {modes[mode]}" + (f"   {extras[0]}" if extras[0] else "") + "\n"
                       f"surface {onoff(self.show_mesh)} (opacity {self.op_surface:.2f}, p/l) | nodes {onoff(self.show_nodes)} (t) | "
                       f"edge width x{self.line_scale:.2f} (k/j) | node size x{self.node_scale:.2f} (u/n)")

    a0 = views[start][0]
    v = Viewer(int(a0["positions"].shape[0]), len(modes), shape=(1, n_panels),
               screenshot_dir=root / "GraphAnimation" / "screenshots", screenshot_name="graph_animation")
    v.names = v._names()

    def _variants():
        out_ = []
        for vi_ in range(len(views)):
            def setup(vi_=vi_):
                v.vi = vi_; v.n = int(v.A["positions"].shape[0]); v.k = 0
                v.names = v._names(); v.screenshot_name = v._shot_name(); v.gif_times = np.asarray(v.A["times"], float)
            out_.append((v_label(vi_), setup))
        return out_

    def v_label(vi_):
        a_ = views[vi_][0]
        return f"{a_['graph']}_T{int(a_['source_frame']):04d}_" + "_vs_".join(str(x["model"]) for x in views[vi_])
    v.gif_variants = _variants
    _launch(v, modes, np.asarray(v.A["times"], float))

def visualize_motion_analysis(results_path, mesh_path=None, stride=2):
    """
    Motion analysis on the tracked surface (Results/<scene>/MotionAnalysis, Physical_fields/from_trajectories).
    'm' cycles: shape-only motion (rigid motion removed, coloured by the deformation displacement, arrows = deformation),
    cumulative normal growth (growth atlas), growth anisotropy (arrows = principal stretch direction), strain and speed
    of the trajectory-based physical fields, sliding-window FTLE. 'v' shows only the vectors (modes with arrows).
    """
    res = Path(results_path); trj = res / "Trajectories"; ma = res / "MotionAnalysis"
    X = np.load(trj / "trajectories.npy"); F = np.load(trj / "reference_faces.npy"); T = X.shape[0]
    Xn = np.load(ma / "RigidMotion" / "nonrigid_trajectories.npy") if (ma / "RigidMotion" / "nonrigid_trajectories.npy").exists() else X
    G = np.load(ma / "GrowthAtlas" / "cumulative_normal_growth.npy") if (ma / "GrowthAtlas" / "cumulative_normal_growth.npy").exists() else None
    pf = res / "Physical_fields" / "from_trajectories"
    names = _frame_names(results_path, mesh_path, T)
    diam = np.linalg.norm(X[0].max(0) - X[0].min(0))
    modes = ["shape-only motion (rigid motion removed)", "cumulative normal growth", "growth anisotropy log(l1/l2)",
             "strain (trajectories)", "speed (trajectories)", "sliding-window FTLE"]

    def field(mode, k):
        """(vertices, per-vertex scalar or None, face centres, face vectors or None, scalar name)."""
        if mode == 0:
            d = Xn[k] - Xn[0]; return Xn[k], np.linalg.norm(d, axis=1), Xn[k], d, "deformation displacement"
        if mode == 1:
            return X[k], (G[k] if G is not None else None), None, None, "cumulative normal growth"
        if mode == 2:
            f = ma / "Anisotropy" / f"anisotropy_T{k:04d}.npz"
            if not f.exists():
                return Xn[k], np.zeros(len(X[0])), None, None, "anisotropy"
            z = np.load(f); c = Xn[k][F].mean(axis=1)
            return Xn[k], z["anisotropy_vertex"], c, z["direction"] * z["anisotropy"][:, None], "anisotropy"
        if mode in (3, 4):
            f = pf / f"frame_{k:04d}.npz"
            if not f.exists():
                return X[k], None, None, None, ""
            z = np.load(f)
            if mode == 3:
                return X[k], z["strain_norm"], None, None, "strain norm"
            return X[k], z["velocity"], X[k], z["velocity_vectors"], "speed"
        f = ma / "Anisotropy" / f"ftle_window_T{k:04d}.npy"
        return X[k], (np.load(f) if f.exists() else np.zeros(len(X[0]))), None, None, "FTLE (window)"

    clims = {}
    for m in range(len(modes)):
        vals = [field(m, k)[1] for k in range(0, T, max(1, T // 10))]
        vals = [v for v in vals if v is not None and len(v)]
        if vals:
            a = np.concatenate(vals); lo, hi = np.percentile(a, 2), np.percentile(a, 98)
            if m == 1:
                hi = max(abs(lo), abs(hi)); lo = -hi
            clims[m] = [lo, hi if hi > lo else lo + 1e-12]

    class Viewer(_FrameViewer):
        title = "Motion analysis"
        help_text = ("m : view (shape-only motion → cumulative growth → growth anisotropy → strain → speed → sliding-window FTLE)\n"
                     "Shape-only motion: the rigid motion (translation + rotation) of every frame is removed, so only the\n"
                     "deformation remains; arrows = deformation displacement.  Growth anisotropy: arrows = direction of the\n"
                     "largest stretch.  Strain / speed: physical fields computed from the trajectories.\n"
                     "v : vector-field view (views with arrows); then x / y / z : projection\n"
                     "h : hide / show the mesh\n← → : frames")
        keys_line = "← → frames | Enter play | space GIF | m view | v vectors only | h mesh | s screenshot | i help | q quit"

        def vector_field(self):
            V, s_, c, vec, name = field(self.mode % len(modes), self.k)
            if vec is None:
                return None
            sel = slice(None, None, max(1, stride))
            sc = 0.12 * diam / (np.linalg.norm(vec, axis=1).max() + 1e-12)
            return c[sel], vec[sel], f"{modes[self.mode % len(modes)]}  (arrows ×{sc:.3g})", sc

        def draw(self):
            self.clear(); p = self.plotter; m = self.mode % len(modes)
            V, s_, c, vec, name = field(m, self.k)
            if self.show_mesh:
                mesh = _poly(V, F)
                if s_ is not None:
                    mesh[name] = s_
                    p.add_mesh(mesh, scalars=name, cmap="RdBu_r" if m == 1 else "viridis", clim=clims.get(m),
                               smooth_shading=True, scalar_bar_args=self.scalar(name))
                else:
                    p.add_mesh(mesh, color="lightgray", smooth_shading=True)
            if vec is not None:
                sel = slice(None, None, max(1, stride))
                p.add_mesh(_arrows(c[sel], vec[sel], 0.12 * diam / (np.linalg.norm(vec, axis=1).max() + 1e-12)), color="black")
            self.label(f"view: {modes[m]}")

    _launch(Viewer(T, len(modes), names=names, screenshot_dir=ma / "plots", screenshot_name="motion_analysis"),
            modes, _observed_times(results_path, T))



def visualize_reeb_dynamics(results_path, mesh_path=None, stride=1):
    """
    Time-varying Reeb graph (Results/<scene>/ReebDynamics): the tracked mesh coloured by the Reeb field, the extremum of
    every branch (protrusion: maximum–saddle pair, dent: minimum–saddle pair) as a sphere coloured by its track; 'm'
    adds the critical-point paths (discrete Jacobi curve) as trails.
    """
    res = Path(results_path); trj = res / "Trajectories"; rd = res / "ReebDynamics"
    X = np.load(trj / "trajectories.npy"); F = np.load(trj / "reference_faces.npy"); T = X.shape[0]
    Fk = np.load(rd / "field_on_tracked_mesh.npy")
    import pandas as pd
    paths = pd.read_csv(rd / "paths.csv")
    t_obs = _observed_times(results_path, T)
    t_obs = np.arange(T, dtype=float) if t_obs is None else t_obs
    names = _frame_names(results_path, mesh_path, T)
    diam = np.linalg.norm(X[0].max(0) - X[0].min(0))
    import matplotlib
    # a branch is identified by (kind, track): track numbers restart for every kind (protrusion 'max', dent 'min'),
    # so 'max' track 0 and 'min' track 0 are two different branches
    paths["branch"] = paths["kind"].astype(str) + "_" + paths["track"].astype(str)
    branches = sorted(paths.branch.unique()); cmap = matplotlib.colormaps["tab20"]
    color = {k: cmap(i % 20)[:3] for i, k in enumerate(branches)}
    clim = [float(np.percentile(Fk, 1)), float(np.percentile(Fk, 99))]
    modes = ["branches on the moving surface", "branches + critical-point paths"]

    class Viewer(_FrameViewer):
        title = "Time-varying Reeb graph"
        help_text = ("m : view (branches / branches + paths)\n"
                     "Spheres: extremum of every branch of the Reeb graph (protrusion = maximum + saddle, dent = minimum +\n"
                     "saddle), coloured by branch identity; big spheres = the trunk (global extremum).\n"
                     "Paths: the trajectory of every extremum over time (discrete Jacobi curve).\n"
                     "h : hide / show the mesh\n← → : frames")
        keys_line = "← → frames | Enter play | space GIF | m view | h mesh | s screenshot | i help | q quit"

        def draw(self):
            self.clear(); p = self.plotter; k = self.k
            if self.show_mesh:
                m = _poly(X[k], F); m["Reeb field"] = Fk[k]
                p.add_mesh(m, scalars="Reeb field", cmap="coolwarm", clim=clim, opacity=0.85, smooth_shading=True,
                           scalar_bar_args=self.scalar("Reeb field"))
            near = paths[np.isclose(paths.time, t_obs[k])]
            for r in near.itertuples():
                pt = X[k][int(r.extremum)]
                rad = (0.035 if r.essential else 0.022) * diam
                p.add_mesh(self.pv.Sphere(radius=rad, center=pt), color=color[r.branch])
            if self.mode % 2 == 1:
                for bid, g in paths[paths.time <= t_obs[k] + 1e-12].groupby("branch"):
                    g = g.sort_values("sample")
                    if len(g) > 1:              # polyline through the sampled positions (a spline would overshoot)
                        p.add_mesh(self.pv.lines_from_points(g[["x", "y", "z"]].to_numpy()), color=color[bid], line_width=3)
            nm = int((near.kind == "max").sum()); nd = int((near.kind == "min").sum())
            self.label(f"view: {modes[self.mode % 2]}  |  protrusion branches {nm}, dent branches {nd}")

    _launch(Viewer(T, len(modes), names=names, screenshot_dir=rd / "plots", screenshot_name="reeb_dynamics"), modes, t_obs)


def visualize_field_comparison(results_path, mesh_path=None):
    """
    Physical fields: frame-to-frame maps (Physical_fields/, from the functional-map correspondences) vs the tracked mesh
    (Physical_fields/from_trajectories/comparison, the same definitions with the tracking correspondences).
    Left: the frame-to-frame field on its frame mesh; centre: the trajectory field on the tracked mesh (same colour
    scale); right: the per-vertex difference (trajectories − frame-to-frame). 'm' changes the field; views are linked.
    """
    import pandas as pd
    from PynamicMesh.core.motion_analysis import FIELD_PAIRS
    res = Path(results_path); trj = res / "Trajectories"; pf = res / "Physical_fields"
    cmpd = pf / "from_trajectories" / "comparison"
    X = np.load(trj / "trajectories.npy"); F0 = np.load(trj / "reference_faces.npy"); T = X.shape[0]
    ks = [k for k in range(1, T) if (cmpd / f"frame_{k:04d}.npz").exists() and (pf / f"frame_{k:04d}.npz").exists()]
    if not ks:
        print("No field comparison found: run the motion analysis with frame-to-frame physical fields "
              "(compute_physic_fields=True) and trajectories.")
        return
    keys = [key for key in FIELD_PAIRS if f"{key}_trajectories" in np.load(cmpd / f"frame_{ks[0]:04d}.npz").files]
    signed = {}
    for key in keys:                                  # signed fields (normal flow, strains) get a symmetric scale
        z0 = np.load(cmpd / f"frame_{ks[len(ks) // 2]:04d}.npz")
        signed[key] = bool(min(np.percentile(z0[f"{key}_frame_to_frame"], 5), np.percentile(z0[f"{key}_trajectories"], 5)) < 0)

    def frame_ranges(z, key):
        """Common colour range of the two sources for THIS frame (2-98th percentiles of both: an outlier transition
        or a few outlier vertices do not flatten the scale), and a symmetric range for the difference."""
        a, b = z[f"{key}_frame_to_frame"], z[f"{key}_trajectories"]
        v = np.concatenate([a, b])
        if signed[key]:
            m = float(np.percentile(np.abs(v), 98)) or 1.0; cl = (-m, m)
        else:
            lo, hi = float(np.percentile(v, 2)), float(np.percentile(v, 98)); cl = (lo, hi if hi > lo else lo + 1e-12)
        dm = float(np.percentile(np.abs(b - a), 98)) or 1.0
        return cl, (-dm, dm)
    sf = pf / "from_trajectories" / "comparison_fields.csv"
    stats = pd.read_csv(sf) if sf.exists() else None
    names_all = _frame_names(results_path, mesh_path, T)
    names = [names_all[k] for k in ks]
    t_all = _observed_times(results_path, T)
    t_obs = np.asarray(t_all)[ks] if t_all is not None else None
    modes = [FIELD_PAIRS[key] for key in keys]

    class Viewer(_FrameViewer):
        title = "Physical fields: frame-to-frame maps vs tracked mesh"
        help_text = ("Left : the field from the frame-to-frame functional maps, on the frame mesh\n"
                     "Centre : the same field from the tracked mesh (same definition, same colour scale as the left\n"
                     "         panel: 2-98th percentiles of both sources in this frame)\n"
                     "Right : difference trajectories − frame-to-frame at every tracked vertex\n\n"
                     "m : field (speed, acceleration, normal flow, strain, area strain)\n← → : transitions | views are linked")
        keys_line = "← → frames | Enter play | space GIF | m field | s screenshot | i help | q quit"

        def draw(self):
            self.clear(); p = self.plotter; k = ks[self.k]; key = keys[self.mode % len(keys)]; name = FIELD_PAIRS[key]
            z = np.load(cmpd / f"frame_{k:04d}.npz"); zf = np.load(pf / f"frame_{k:04d}.npz")
            cl, dl = frame_ranges(z, key)
            cmap = "coolwarm" if signed[key] else "viridis"
            p.subplot(0, 0)
            m1 = _poly(np.asarray(zf["vertices"], float), np.asarray(zf["faces"], int)); m1[name] = np.asarray(zf[key], float)
            p.add_mesh(m1, scalars=name, cmap=cmap, clim=cl, smooth_shading=True, scalar_bar_args=self.scalar(name))
            p.add_text("frame-to-frame maps", name="t0", position="upper_right", font_size=8)
            p.subplot(0, 1)
            m2 = _poly(X[k], F0); m2[name] = z[f"{key}_trajectories"]
            p.add_mesh(m2, scalars=name, cmap=cmap, clim=cl, smooth_shading=True, show_scalar_bar=False)
            p.add_text("tracked mesh (trajectories)", name="t1", position="upper_edge", font_size=8)
            p.subplot(0, 2)
            m3 = _poly(X[k], F0); m3["difference"] = z[f"{key}_trajectories"] - z[f"{key}_frame_to_frame"]
            p.add_mesh(m3, scalars="difference", cmap="PuOr", clim=dl, smooth_shading=True,
                       scalar_bar_args=self.scalar("difference"))
            p.add_text("difference (trajectories − frame-to-frame)", name="t2", position="upper_edge", font_size=8)
            txt = f"field: {name}"
            if stats is not None:
                r = stats[(stats.frame == k) & (stats.field == key)]
                if len(r):
                    r = r.iloc[0]
                    txt += f" | correlation r = {r.correlation:.2f} | mean ratio (trajectories / frame-to-frame) = {r.ratio:.2f}"
            self.label(txt)

    _launch(Viewer(len(ks), len(keys), shape=(1, 3), names=names, screenshot_dir=res / "MotionAnalysis" / "plots",
                   screenshot_name="field_comparison"), modes, t_obs)
