#!/usr/bin/env python3
# =============================================================================
#  hermes_gui.py  —  HermesLite 2  Spectrum Analyser GUI
# =============================================================================
#
#  Window layout
#  -------------
#    Native menu bar  :  Menu 1 / Menu 2 / Menu 3  (real OS pull-down menus,
#                        each with Option 1 / Option 2 / Option 3 — stubs)
#    Button toolbar   :  Capture / Clear / Export  (push buttons)
#    Centre canvas    :  matplotlib spectrum display  (dark theme)
#    Status bar       :  one-line text at bottom
#
#  Public API  (called from hermes_main.py)
#  ----------------------------------------
#    gui = HermesGUI(root, on_close_cb=fn)
#    gui.plot_spectrum(freqs_mhz, power_db, label)
#    gui.set_status(msg)
#
#  Dependencies:  matplotlib  numpy  (pip install matplotlib numpy)
# =============================================================================

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                NavigationToolbar2Tk)
import numpy as np


class HermesGUI:

    # ------------------------------------------------------------------
    # Spectrum display extents
    # ------------------------------------------------------------------
    FMIN =  6.75    # MHz  left  edge  (matches ±192 kHz @ 384 ksps / 7 MHz)
    FMAX =  7.25    # MHz  right edge
    YMIN = -120     # dBm  bottom of power axis  (well below -90 dBm noise floor)
    YMAX =   20     # dBm  top    of power axis  (headroom above carrier)

    # Dark colour palette
    BG_WIN    = "#1a1a2a"
    BG_PLOT   = "#0a0a18"
    COL_TRACE = "#00ff88"   # spectrum trace — green
    COL_PEAK  = "#ff4444"   # peak marker    — red
    COL_GRID  = "#2a2a3a"
    COL_TXT   = "white"

    # ------------------------------------------------------------------
    def __init__(self, root: tk.Tk, on_close_cb=None):
        """
        Parameters
        ----------
        root        : tkinter root window
        on_close_cb : optional callable, invoked before window is destroyed
                      (use to disconnect the radio cleanly)
        """
        self.root = root
        self.root.title("HermesLite 2  —  Spectrum Analyser")
        self.root.geometry("1024x700")
        self.root.configure(bg=self.BG_WIN)

        if on_close_cb:
            self.root.protocol(
                "WM_DELETE_WINDOW",
                lambda: (on_close_cb(), self.root.destroy()))

        self._build_menubar()       # native OS pull-down menus  (top of frame)
        self._build_button_toolbar()# push-button row below menu bar
        self._build_plot()          # matplotlib canvas  (fills remaining space)
        self._build_statusbar()     # one-line status at bottom

    # ==================================================================
    #  Layout builders
    # ==================================================================

    def _build_menubar(self):
        """
        Attach a native OS menu bar to the window.
        Three top-level pull-down menus, each with Option 1 / 2 / 3.
        All menu commands are stubs — wire them up in future work.
        """
        menubar = tk.Menu(self.root)

        # ---- Menu 1 --------------------------------------------------
        m1 = tk.Menu(menubar, tearoff=False)
        m1.add_command(label="Option 1",
                       command=lambda: self._on_menu1("1"))
        m1.add_command(label="Option 2",
                       command=lambda: self._on_menu1("2"))
        m1.add_command(label="Option 3",
                       command=lambda: self._on_menu1("3"))
        menubar.add_cascade(label="Menu 1", menu=m1)

        # ---- Menu 2 --------------------------------------------------
        m2 = tk.Menu(menubar, tearoff=False)
        m2.add_command(label="Option 1",
                       command=lambda: self._on_menu2("1"))
        m2.add_command(label="Option 2",
                       command=lambda: self._on_menu2("2"))
        m2.add_command(label="Option 3",
                       command=lambda: self._on_menu2("3"))
        menubar.add_cascade(label="Menu 2", menu=m2)

        # ---- Menu 3 --------------------------------------------------
        m3 = tk.Menu(menubar, tearoff=False)
        m3.add_command(label="Option 1",
                       command=lambda: self._on_menu3("1"))
        m3.add_command(label="Option 2",
                       command=lambda: self._on_menu3("2"))
        m3.add_command(label="Option 3",
                       command=lambda: self._on_menu3("3"))
        menubar.add_cascade(label="Menu 3", menu=m3)

        # Attach to window  (appears in native OS menu position)
        self.root.config(menu=menubar)

    # ------------------------------------------------------------------
    def _build_button_toolbar(self):
        """Toolbar row with three push buttons, directly under the menu bar."""
        bar = ttk.Frame(self.root, padding=(6, 3))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Separator(bar, orient=tk.HORIZONTAL)   # subtle top border

        for label, cmd in [
            ("Capture", self._on_capture),
            ("Clear",   self._on_clear),
            ("Export",  self._on_export),
        ]:
            ttk.Button(bar, text=label, command=cmd, width=10
                       ).pack(side=tk.LEFT, padx=4, pady=2)

        # Thin separator line below the buttons
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(
            side=tk.TOP, fill=tk.X)

    # ------------------------------------------------------------------
    def _build_plot(self):
        """Embed a matplotlib figure in a tkinter frame."""
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(facecolor=self.BG_WIN)
        self.ax  = self.fig.add_subplot(111)
        self._reset_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Standard matplotlib navigation toolbar (zoom / pan / save)
        tb = NavigationToolbar2Tk(self.canvas, frame)
        tb.update()

    # ------------------------------------------------------------------
    def _reset_axes(self):
        """Apply dark-theme formatting and restore default axis limits."""
        ax = self.ax
        ax.set_facecolor(self.BG_PLOT)
        ax.set_xlim(self.FMIN, self.FMAX)
        ax.set_ylim(self.YMIN, self.YMAX)
        ax.set_xlabel("Frequency  (MHz)", color=self.COL_TXT, fontsize=10)
        ax.set_ylabel("Power  (dBm)", color=self.COL_TXT, fontsize=10)
        ax.set_title("HL2 Spectrum Analyser  —  awaiting capture",
                     color=self.COL_TXT, fontsize=11)
        ax.tick_params(colors=self.COL_TXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        ax.grid(True, color=self.COL_GRID, linestyle="--", linewidth=0.5)
        self.fig.tight_layout(pad=1.5)

    # ------------------------------------------------------------------
    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root,
                  textvariable=self.status_var,
                  relief=tk.SUNKEN,
                  anchor=tk.W,
                  padding=(6, 2)
                  ).pack(side=tk.BOTTOM, fill=tk.X)

    # ==================================================================
    #  Public API
    # ==================================================================

    def plot_spectrum(self,
                      freqs_mhz: np.ndarray,
                      power_db:  np.ndarray,
                      label:     str = "RX spectrum"):
        """
        Redraw the spectrum display.

        Parameters
        ----------
        freqs_mhz : 1-D array, frequency axis in MHz
        power_db  : 1-D array, power in dB  (same length)
        label     : trace legend text  (should include RBW info)
        """
        self.ax.cla()
        self._reset_axes()

        # Fit x-axis exactly to data — no blank margins at edges
        f0 = float(freqs_mhz[0])
        f1 = float(freqs_mhz[-1])
        self.ax.set_xlim(f0, f1)

        peak_i = int(power_db.argmax())
        peak_f = float(freqs_mhz[peak_i])
        peak_p = float(power_db[peak_i])

        self.ax.plot(freqs_mhz, power_db,
                     color=self.COL_TRACE, linewidth=0.9, label=label)

        self.ax.axvline(peak_f,
                        color=self.COL_PEAK, linewidth=0.9, linestyle="--",
                        label=f"Peak  {peak_f:.4f} MHz  ({peak_p:.1f} dB)")

        self.ax.set_title(
            f"HL2 Spectrum  —  {len(freqs_mhz)} bins  |  "
            f"peak  {peak_p:.1f} dB  @  {peak_f:.4f} MHz",
            color=self.COL_TXT, fontsize=11)

        self.ax.legend(facecolor=self.BG_WIN, labelcolor=self.COL_TXT,
                       fontsize=8, loc="upper right")

        # RBW annotation — bottom-left corner of the plot
        self.ax.annotate(
            "RBW  2.2 kHz",
            xy=(0.01, 0.04),
            xycoords="axes fraction",
            color=self.COL_TXT,
            fontsize=8,
            alpha=0.75,
        )

        self.canvas.draw()
        self.set_status(
            f"Spectrum plotted  |  {len(freqs_mhz)} bins  |  "
            f"{f0:.4f} – {f1:.4f} MHz  |  "
            f"peak  {peak_p:.1f} dB  @  {peak_f:.4f} MHz")

    # ------------------------------------------------------------------
    def plot_time_domain(self,
                         iq_samples,
                         sample_rate: int = 384_000,
                         label: str = "RF3 loopback"):
        """
        Plot I channel, Q channel and envelope magnitude vs time.

        Parameters
        ----------
        iq_samples  : list / array of (I_float, Q_float), each normalised ±1.0
        sample_rate : ADC sample rate in Hz  (default 384 ksps)
        label       : legend / title suffix  (hardware config description)
        """
        iq_arr = np.array(iq_samples, dtype=np.float64)
        I_ch   = iq_arr[:, 0]
        Q_ch   = iq_arr[:, 1]
        mag    = np.sqrt(I_ch ** 2 + Q_ch ** 2)

        n    = len(I_ch)
        t_ms = np.arange(n) / sample_rate * 1000.0   # time axis in ms

        rms_val  = float(np.sqrt(np.mean(mag ** 2)))
        peak_val = float(mag.max())
        rms_dbfs  = 20.0 * np.log10(rms_val  + 1e-30)
        peak_dbfs = 20.0 * np.log10(peak_val + 1e-30)

        self.ax.cla()
        self.ax.set_facecolor(self.BG_PLOT)

        # Trace colours:  I = green,  Q = amber,  |IQ| = light grey
        self.ax.plot(t_ms, I_ch, color=self.COL_TRACE, linewidth=0.8,
                     label="I",    alpha=0.90)
        self.ax.plot(t_ms, Q_ch, color="#ffaa00",       linewidth=0.8,
                     label="Q",    alpha=0.90)
        self.ax.plot(t_ms, mag,  color="#aaaaaa",        linewidth=0.9,
                     label="|IQ|", alpha=0.70)

        # Zero-amplitude reference line
        self.ax.axhline(0, color="#555555", linewidth=0.5, linestyle="--")

        # Y-axis range — show signal clearly; at least ±0.05 so noise is visible
        y_lim = max(0.05, peak_val * 1.3)
        self.ax.set_ylim(-y_lim, y_lim)
        self.ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))

        self.ax.set_xlabel("Time  (ms)", color=self.COL_TXT, fontsize=10)
        self.ax.set_ylabel("Amplitude  (normalised)", color=self.COL_TXT, fontsize=10)
        self.ax.set_title(
            f"HL2 Time Domain  —  {n} samples  |  "
            f"RMS {rms_dbfs:.1f} dBFS  |  Peak {peak_dbfs:.1f} dBFS",
            color=self.COL_TXT, fontsize=11)

        self.ax.tick_params(colors=self.COL_TXT, labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#555555")
        self.ax.grid(True, color=self.COL_GRID, linestyle="--", linewidth=0.5)

        self.ax.legend(facecolor=self.BG_WIN, labelcolor=self.COL_TXT,
                       fontsize=8, loc="upper right")

        # Hardware config annotation — bottom-left corner
        self.ax.annotate(
            label,
            xy=(0.01, 0.04),
            xycoords="axes fraction",
            color=self.COL_TXT,
            fontsize=8,
            alpha=0.75,
        )

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()
        self.set_status(
            f"Time domain  |  {n} samples  |  "
            f"{float(t_ms[-1]):.2f} ms  |  "
            f"RMS {rms_dbfs:.1f} dBFS  |  Peak {peak_dbfs:.1f} dBFS  |  {label}")

    # ------------------------------------------------------------------
    def set_status(self, msg: str):
        """Update the status bar."""
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ==================================================================
    #  Push-button stubs
    # ==================================================================

    def _on_capture(self):
        # TODO: trigger a new IQ capture and re-plot
        print("[BTN]  Capture  (stub)")
        self.set_status("Capture pressed  —  stub, not yet wired to radio")

    def _on_clear(self):
        print("[BTN]  Clear")
        self.ax.cla()
        self._reset_axes()
        self.canvas.draw()
        self.set_status("Display cleared.")

    def _on_export(self):
        # TODO: save spectrum data to CSV / PNG
        print("[BTN]  Export  (stub)")
        self.set_status("Export pressed  —  stub, not yet implemented")

    # ==================================================================
    #  Pull-down menu stubs
    # ==================================================================

    def _on_menu1(self, option: str):
        # TODO: e.g. select sample rate
        print(f"[MENU1]  option {option}  (stub)")
        self.set_status(f"Menu 1  →  option {option}  (stub)")

    def _on_menu2(self, option: str):
        # TODO: e.g. select band / NCO frequency
        print(f"[MENU2]  option {option}  (stub)")
        self.set_status(f"Menu 2  →  option {option}  (stub)")

    def _on_menu3(self, option: str):
        # TODO: e.g. select display mode / averaging
        print(f"[MENU3]  option {option}  (stub)")
        self.set_status(f"Menu 3  →  option {option}  (stub)")
