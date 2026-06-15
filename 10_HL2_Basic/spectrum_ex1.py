#!/usr/bin/env python3
"""
spectrum_ex1.py — Welch spectrum computation + live matplotlib display.

Public API
----------
    compute_spectrum(iq, nco_hz, fs, seg, pad, dbfs_ref)
        -> (freqs_mhz, power_dbm, n_segments)

    LiveSpectrum(nco_hz, fs, dbfs_ref, seg, pad, ymin, ymax, lna_gain_db,
                 ptt, pa_enable, duplex, on_cc_change)
        .push(iq_samples)   — thread-safe, called from capture thread
        .run()              — blocks in plt.show() until window closed
"""

import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button


# ── Spectrum computation ───────────────────────────────────────────────────────

def compute_spectrum(iq_samples, nco_hz, fs, seg=256, pad=1024, dbfs_ref=0.0):
    """
    Welch averaged power spectrum.

    Parameters
    ----------
    iq_samples : list of (I_float, Q_float), normalised ±1.0
    nco_hz     : receiver LO frequency in Hz
    fs         : sample rate in Hz
    seg        : Hanning window / segment length  (sets RBW)
    pad        : zero-pad length  (sets bin spacing, not RBW)
    dbfs_ref   : empirical dBm calibration offset

    Returns
    -------
    freqs_mhz  : frequency axis in MHz  (length = pad)
    power_dbm  : power in dBm           (length = pad)
    n_segs     : number of segments averaged
    """
    z    = np.array([s[0] + 1j*s[1] for s in iq_samples], dtype=np.complex128)
    step = seg // 2          # 50 % overlap
    win  = np.hanning(seg)
    wpow = np.sum(win ** 2)
    pwr  = np.zeros(pad, dtype=np.float64)
    n    = 0

    for start in range(0, len(z) - seg + 1, step):
        buf       = np.zeros(pad, dtype=np.complex128)
        buf[:seg] = z[start:start + seg] * win
        pwr      += np.abs(np.fft.fft(buf)) ** 2
        n        += 1

    # PSD → power in RBW → dBm
    psd      = np.fft.fftshift(pwr / max(n, 1)) / (fs * wpow)
    rbw_hz   = (fs / seg) * 1.44          # Hanning 3 dB bandwidth
    power_db = 10.0 * np.log10(psd * rbw_hz / 50.0 / 1e-3 + 1e-40) + dbfs_ref

    freqs_mhz = (nco_hz + np.fft.fftshift(np.fft.fftfreq(pad, d=1.0 / fs))) / 1e6
    return freqs_mhz, power_db, n


# ── Live display ───────────────────────────────────────────────────────────────

class LiveSpectrum:
    """
    Live-updating spectrum plot with PA / PTT / Duplex toggle buttons.

    on_cc_change(ptt, pa_enable, duplex) is called on the main thread
    whenever a button is clicked — wire it to update the radio CC registers.
    """

    COL_ON  = "#003399"   # button active  — dark blue
    COL_OFF = "#001155"   # button inactive — darker blue

    def __init__(self, nco_hz, fs, dbfs_ref=0.0,
                 seg=256, pad=1024, ymin=-120, ymax=50, lna_gain_db=20,
                 ptt=True, pa_enable=True,
                 on_cc_change=None):

        self._nco_hz      = nco_hz
        self._fs          = fs
        self._dbfs_ref    = dbfs_ref
        self._seg         = seg
        self._pad         = pad
        self._lna_gain_db = lna_gain_db
        self._lock        = threading.Lock()
        self._iq          = None
        self._dirty       = False
        self._sweep       = 0
        self._on_cc_change = on_cc_change

        # Control state
        self._ptt       = ptt
        self._pa_enable = pa_enable

        rbw_hz = (fs / seg) * 1.44
        f_lo   = (nco_hz - fs / 2) / 1e6
        f_hi   = (nco_hz + fs / 2) / 1e6

        plt.style.use("dark_background")

        # Reserve space at bottom for buttons
        self.fig, self.ax = plt.subplots(figsize=(11, 5.5))
        self.fig.subplots_adjust(bottom=0.18)

        try:
            self.fig.canvas.manager.set_window_title("HL2 Spectrum Analyser")
        except Exception:
            pass

        self.ax.set_xlim(f_lo, f_hi)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_xlabel("Frequency  (MHz)")
        self.ax.set_ylabel("Power  (dBm)")
        self.ax.set_title("Hermes Lite 2 — waiting for data ...")
        self.ax.grid(True, color="#333333", linewidth=0.5)

        self.line,     = self.ax.plot([], [], color="#00ff88", linewidth=0.9,
                                      label=f"RBW {rbw_hz/1000:.1f} kHz")
        self.peak_line = self.ax.axvline(nco_hz / 1e6, color="#ff4444",
                                         linewidth=0.8, linestyle="--",
                                         visible=False)
        self.ax.legend(loc="upper right", fontsize=8)

        # ── Toggle buttons — lower left, small, close together ────────
        # [left, bottom, width, height]  in figure fraction
        ax_ptt = self.fig.add_axes([0.08, 0.04, 0.07, 0.05])
        ax_pa  = self.fig.add_axes([0.16, 0.04, 0.07, 0.05])

        self._btn_ptt = Button(ax_ptt, "PTT",   color=self._btn_col(ptt),
                               hovercolor="#0055cc")
        self._btn_pa  = Button(ax_pa,  "En PA", color=self._btn_col(pa_enable),
                               hovercolor="#0055cc")

        self._btn_ptt.label.set_color("white")
        self._btn_pa.label.set_color("white")

        self._btn_ptt.on_clicked(self._toggle_ptt)
        self._btn_pa.on_clicked(self._toggle_pa)

        self._ani = animation.FuncAnimation(
            self.fig, self._animate, interval=200, blit=False)

    # ── Button helpers ─────────────────────────────────────────────────────────

    def _btn_col(self, state):
        return self.COL_ON if state else self.COL_OFF

    def _toggle_ptt(self, _event):
        self._ptt = not self._ptt
        self._btn_ptt.color = self._btn_col(self._ptt)
        self._btn_ptt.ax.set_facecolor(self._btn_col(self._ptt))
        self.fig.canvas.draw_idle()
        self._fire_cc()

    def _toggle_pa(self, _event):
        self._pa_enable = not self._pa_enable
        self._btn_pa.color = self._btn_col(self._pa_enable)
        self._btn_pa.ax.set_facecolor(self._btn_col(self._pa_enable))
        self.fig.canvas.draw_idle()
        self._fire_cc()

    def _fire_cc(self):
        if self._on_cc_change:
            self._on_cc_change(self._ptt, self._pa_enable)

    # ── IQ push (capture thread) ───────────────────────────────────────────────

    def push(self, iq_samples):
        """Store latest IQ block — safe to call from any thread."""
        with self._lock:
            self._iq    = list(iq_samples)
            self._dirty = True

    # ── Animation (main thread) ────────────────────────────────────────────────

    def _animate(self, _frame):
        """Called by FuncAnimation on the main thread every 200 ms."""
        with self._lock:
            if not self._dirty:
                return
            iq          = self._iq
            self._dirty = False

        self._sweep += 1
        freqs, power, n_segs = compute_spectrum(
            iq, self._nco_hz, self._fs, self._seg, self._pad, self._dbfs_ref)

        peak_i = int(power.argmax())

        self.line.set_data(freqs, power)
        self.peak_line.set_xdata([freqs[peak_i], freqs[peak_i]])
        self.peak_line.set_visible(True)
        self.ax.set_title(
            f"Hermes Lite 2  |  peak {power[peak_i]:.1f} dBm @ {freqs[peak_i]:.4f} MHz")

        if self._sweep == 1:
            rx_dbm = power[peak_i]
            tx_est = rx_dbm + 46.0 - self._lna_gain_db
            print(f"  Peak Power: Rx {rx_dbm:+.1f} dBm  "
                  f"( Tx Est. {tx_est:+.1f} dBm )  "
                  f"@ {freqs[peak_i]:.4f} MHz")

    def run(self):
        """Block until the plot window is closed."""
        plt.show()
