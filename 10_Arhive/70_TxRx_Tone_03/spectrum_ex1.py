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
from matplotlib.widgets import Button, TextBox, Slider, RadioButtons


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

    on_cc_change(ptt, pa_enable, lna_gain_db) is called on the main thread
    whenever a button is clicked or the LNA gain box is edited — wire it to
    update the radio CC registers.
    """

    COL_ON  = "#990000"   # button active   — dark red
    COL_OFF = "#2266cc"   # button inactive — blue
    FWD_ADC_FULLSCALE = 4095   # AD7991 forward-power ADC full scale (12-bit)

    def __init__(self, nco_hz, fs, dbfs_ref=0.0,
                 seg=256, pad=1024, ymin=-120, ymax=50, lna_gain_db=20,
                 ptt=True, pa_enable=True, tx_drive=15,
                 on_cc_change=None, on_capture=None, on_freq_change=None):

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
        self._on_cc_change   = on_cc_change
        self._on_capture     = on_capture
        self._on_freq_change = on_freq_change

        # Control state
        self._ptt       = ptt
        self._pa_enable = pa_enable
        self._tx_drive  = tx_drive   # 0-15 TX drive level (HL2 reg 0x09)
        self._overlay   = False   # peak line + peak-power readout — default off

        # Forward-power telemetry readout
        self._pep_watts     = 0.0
        self._fwd_peak_adc  = 0
        self._fwd_avg_adc   = 0
        self._rev_avg_adc   = 0
        self._pa_current_ma = 0.0

        rbw_hz = (fs / seg) * 1.44
        f_lo   = (nco_hz - fs / 2) / 1e6
        f_hi   = (nco_hz + fs / 2) / 1e6

        plt.style.use("dark_background")

        # Reserve space at bottom for buttons + TX-power slider row.
        # top≈0.975 leaves a small (~8 px) margin above the plot.
        self.fig, self.ax = plt.subplots(figsize=(11, 5.5))
        self.fig.subplots_adjust(top=0.975, bottom=0.22)

        try:
            self.fig.canvas.manager.set_window_title("HL2 Spectrum Analyser")
        except Exception:
            pass

        self.ax.set_xlim(f_lo, f_hi)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_xlabel("Frequency  (MHz)")
        self.ax.set_ylabel("Power  (dBm)")
        self.ax.grid(True, color="#333333", linewidth=0.5)

        self.line,     = self.ax.plot([], [], color="#00ff88", linewidth=0.9)
        self.peak_line = self.ax.axvline(nco_hz / 1e6, color="#ff4444",
                                         linewidth=0.8, linestyle="--",
                                         visible=False)
        # RBW as plain text (upper-right) — not a legend.
        self._rbw_text = self.ax.text(
            0.99, 0.98, self._rbw_str(seg),
            transform=self.ax.transAxes, color="#888888", fontsize=9,
            va="top", ha="right", family="monospace")

        # Forward-power (PEP) readout — top-left of the plot
        self._pwr_text = self.ax.text(
            0.01, 0.98, "(0.0A) pep 0.0w",
            transform=self.ax.transAxes, color="#ffcc00", fontsize=11,
            va="top", ha="left", family="monospace")

        # Status / peak readout — top-centre (replaces the old title).
        self._status_text = self.ax.text(
            0.5, 0.985, "waiting for data ...",
            transform=self.ax.transAxes, color="#aaaaaa", fontsize=9,
            va="top", ha="center", family="monospace")

        # ── Toggle buttons — lower left, small, close together ────────
        # [left, bottom, width, height]  in figure fraction
        ax_ptt     = self.fig.add_axes([0.08, 0.04, 0.07, 0.05])
        ax_pa      = self.fig.add_axes([0.16, 0.04, 0.07, 0.05])
        ax_overlay = self.fig.add_axes([0.24, 0.04, 0.09, 0.05])

        self._btn_ptt     = Button(ax_ptt,     "PTT",     color=self._btn_col(ptt),
                                   hovercolor="#0055cc")
        self._btn_overlay = Button(ax_overlay, "Overlay", color=self._btn_col(self._overlay),
                                   hovercolor="#0055cc")
        self._btn_pa      = Button(ax_pa,      "En PA",   color=self._btn_col(pa_enable),
                                   hovercolor="#0055cc")

        self._btn_ptt.label.set_color("white")
        self._btn_overlay.label.set_color("white")
        self._btn_pa.label.set_color("white")

        self._btn_ptt.on_clicked(self._toggle_ptt)
        self._btn_overlay.on_clicked(self._toggle_overlay)
        self._btn_pa.on_clicked(self._toggle_pa)

        # ── LNA gain entry — right of Overlay ─────────────────────────
        # Editable 0–48 dB; sets the AD9866 LNA gain on the radio.
        ax_lna = self.fig.add_axes([0.43, 0.04, 0.06, 0.05])
        self._lna_busy = False
        self._txt_lna  = TextBox(ax_lna, "LNA Gn ",
                                 initial=str(int(self._lna_gain_db)),
                                 color="#001155", hovercolor="#0a1a55",
                                 textalignment="center")
        self._txt_lna.label.set_color("white")
        self._txt_lna.text_disp.set_color("white")
        self._txt_lna.on_submit(self._on_lna_submit)

        # Explicit Edit button — pops a dialog so it's clear the live plot
        # pauses on purpose while you enter a new gain.
        ax_lna_edit = self.fig.add_axes([0.50, 0.04, 0.06, 0.05])
        self._btn_lna_edit = Button(ax_lna_edit, "Edit", color="#003399",
                                    hovercolor="#0055cc")
        self._btn_lna_edit.label.set_color("white")
        self._btn_lna_edit.on_clicked(self._on_lna_edit)

        # Capture button — grab 100 ms of raw IQ to capture.npz
        ax_cap = self.fig.add_axes([0.58, 0.04, 0.08, 0.05])
        self._btn_cap = Button(ax_cap, "Capture", color="#003399",
                               hovercolor="#0055cc")
        self._btn_cap.label.set_color("white")
        self._btn_cap.on_clicked(self._on_capture_click)

        # Carrier-frequency (NCO) entry + Edit — right of Capture
        ax_freq = self.fig.add_axes([0.76, 0.04, 0.06, 0.05])
        self._freq_busy = False
        self._txt_freq  = TextBox(ax_freq, "MHz ",
                                  initial=f"{nco_hz/1e6:.3f}",
                                  color="#001155", hovercolor="#0a1a55",
                                  textalignment="center")
        self._txt_freq.label.set_color("white")
        self._txt_freq.text_disp.set_color("white")
        self._txt_freq.on_submit(self._on_freq_submit)

        ax_freq_edit = self.fig.add_axes([0.83, 0.04, 0.06, 0.05])
        self._btn_freq_edit = Button(ax_freq_edit, "Edit", color="#003399",
                                     hovercolor="#0055cc")
        self._btn_freq_edit.label.set_color("white")
        self._btn_freq_edit.on_clicked(self._on_freq_edit)

        # ── RBW selector — just above the Freq box + Edit ─────────────
        # seg chosen so the -3 dB RBW = (fs/seg)*1.44 lands on a clean decade
        # value (matches a spectrum-analyser's RBW convention).  10 kHz lets
        # you compare directly with a bench analyser at the same RBW.
        #             label,     seg,    pad     (-3 dB RBW, # of averages)
        self._rbw_options = [("10 Hz",  55296, 65536),   # 1 segment (no avg)
                             ("100 Hz",  5530,  8192),   # ~10 averages
                             ("1 kHz",    553,  1024),   # ~230 averages
                             ("10 kHz",    55,   256)]   # ~2400 averages
        active = next((i for i, o in enumerate(self._rbw_options)
                       if o[1] == self._seg), 2)
        self._rbw_label = self.fig.text(0.762, 0.185, "RBW", color="white",
                                        fontsize=8, family="monospace")
        ax_rbw = self.fig.add_axes([0.80, 0.075, 0.10, 0.105])
        ax_rbw.set_facecolor("#001133")
        self._radio_rbw = RadioButtons(
            ax_rbw, [o[0] for o in self._rbw_options], active=active,
            activecolor="#00ff88")
        for lbl in self._radio_rbw.labels:
            lbl.set_color("white")
            lbl.set_fontsize(8)
        self._radio_rbw.on_clicked(self._on_rbw_select)

        # ── TX power slider — second row above the buttons ────────────
        ax_txp = self.fig.add_axes([0.12, 0.135, 0.30, 0.03])
        self._sld_txp = Slider(ax_txp, "TX Pwr", 0, 15, valinit=tx_drive,
                               valstep=1, color="#003399")
        self._sld_txp.label.set_color("white")
        self._sld_txp.valtext.set_color("white")
        self._sld_txp.on_changed(self._on_tx_drive)

        self._ani = animation.FuncAnimation(
            self.fig, self._animate, interval=200, blit=False)

        # Pause the live redraw while the LNA box has focus — FuncAnimation's
        # periodic full repaint otherwise fights the TextBox cursor/typing.
        self._anim_paused = False
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ── Status text (replaces the old title) ────────────────────────────────────

    def _set_status(self, msg):
        """Update the top-centre status line (peak readout, paused/grab msgs)."""
        self._status_text.set_text(msg)

    def _rbw_str(self, seg):
        """Actual -3 dB RBW for a segment length, formatted Hz / kHz."""
        hz = (self._fs / seg) * 1.44
        return f"RBW {hz/1000:.1f} kHz" if hz >= 999.5 else f"RBW {hz:.0f} Hz"

    def _on_rbw_select(self, label):
        """RBW selector — switch Welch segment/pad and refresh the readout."""
        for lab, seg, pad in self._rbw_options:
            if lab == label:
                self._seg = seg
                self._pad = pad
                self._rbw_text.set_text(self._rbw_str(seg))
                with self._lock:
                    if self._iq is not None:
                        self._dirty = True      # force a recompute next frame
                self.fig.canvas.draw_idle()
                break

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

    def _toggle_overlay(self, _event):
        self._overlay = not self._overlay
        self._btn_overlay.color = self._btn_col(self._overlay)
        self._btn_overlay.ax.set_facecolor(self._btn_col(self._overlay))
        self.peak_line.set_visible(self._overlay)
        if not self._overlay:
            self._set_status("")
        self.fig.canvas.draw_idle()

    def _anim_pause(self):
        if not self._anim_paused:
            self._ani.event_source.stop()
            self._anim_paused = True

    def _anim_resume(self):
        if self._anim_paused:
            self._ani.event_source.start()
            self._anim_paused = False

    def _on_click(self, event):
        """Click in an entry box → pause redraw to edit; click anywhere else → resume."""
        if event.inaxes is self._txt_lna.ax:
            self._anim_pause()
            self._set_status("Paused — editing LNA gain (Enter to apply) ...")
            self.fig.canvas.draw_idle()
        elif event.inaxes is self._txt_freq.ax:
            self._anim_pause()
            self._set_status("Paused — editing carrier freq (Enter to apply) ...")
            self.fig.canvas.draw_idle()
        else:
            self._anim_resume()

    def _on_lna_edit(self, _event):
        """Edit button → pop a dialog for the LNA gain; pause makes the freeze clear."""
        from tkinter import simpledialog

        self._anim_pause()
        self._set_status("Paused — enter LNA gain (0–48 dB) ...")
        self.fig.canvas.draw_idle()
        try:
            parent = getattr(self.fig.canvas.manager, "window", None)
            val = simpledialog.askinteger(
                "LNA Gain", "Enter LNA gain (0–48 dB):",
                parent=parent, initialvalue=int(self._lna_gain_db),
                minvalue=0, maxvalue=48)
        finally:
            self._anim_resume()

        if val is not None:
            g = max(0, min(48, int(val)))
            self._lna_gain_db = g
            self._lna_busy = True
            self._txt_lna.set_val(str(g))   # update the box display
            self._lna_busy = False
            self._fire_cc()

    def _on_capture_click(self, _event):
        """Capture button — request a 100 ms file grab (done by the capture thread)."""
        if self._on_capture:
            self._set_status("Capturing 100 ms -> capture.npz ...")
            self.fig.canvas.draw_idle()
            self._on_capture()

    def _apply_freq(self, mhz):
        """Validate MHz, clamp 0.1–35, retune the radio and the spectrum x-axis."""
        try:
            f = float(mhz)
        except (ValueError, TypeError):
            f = self._nco_hz / 1e6
        f = max(0.1, min(35.0, f))
        self._nco_hz = f * 1e6

        norm = f"{f:.3f}"
        if self._txt_freq.text != norm:
            self._freq_busy = True
            self._txt_freq.set_val(norm)
            self._freq_busy = False

        f_lo = (self._nco_hz - self._fs / 2) / 1e6
        f_hi = (self._nco_hz + self._fs / 2) / 1e6
        self.ax.set_xlim(f_lo, f_hi)
        self.fig.canvas.draw_idle()

        if self._on_freq_change:
            self._on_freq_change(self._nco_hz)

    def _on_freq_submit(self, text):
        """Freq box edited directly — apply on Enter."""
        if self._freq_busy:
            return
        self._apply_freq(text)
        self._anim_resume()

    def _on_freq_edit(self, _event):
        """Edit button → pop a dialog for the carrier (NCO) frequency in MHz."""
        from tkinter import simpledialog

        self._anim_pause()
        self._set_status("Paused — enter carrier freq (MHz) ...")
        self.fig.canvas.draw_idle()
        try:
            parent = getattr(self.fig.canvas.manager, "window", None)
            val = simpledialog.askfloat(
                "Carrier Frequency", "Enter carrier / NCO frequency (MHz):",
                parent=parent, initialvalue=self._nco_hz / 1e6,
                minvalue=0.1, maxvalue=35.0)
        finally:
            self._anim_resume()

        if val is not None:
            self._apply_freq(val)

    def _on_lna_submit(self, text):
        """LNA gain box edited — validate 0–48 dB, clamp, push to radio."""
        if self._lna_busy:
            return
        try:
            g = int(round(float(text)))
        except (ValueError, TypeError):
            g = int(self._lna_gain_db)
        g = max(0, min(48, g))
        self._lna_gain_db = g

        # Reflect the clamped/normalised value back into the box
        if self._txt_lna.text != str(g):
            self._lna_busy = True
            self._txt_lna.set_val(str(g))
            self._lna_busy = False

        self._fire_cc()
        self._anim_resume()   # Enter submits — resume live redraw

    def _on_tx_drive(self, val):
        """TX power slider — set drive level 0-15 and push to radio."""
        self._tx_drive = int(round(val))
        self._fire_cc()

    def set_power(self, fwd_peak, fwd_avg, rev_avg, pep, pa_current_ma=0.0):
        """Called from the capture thread with HL2 forward-power telemetry."""
        self._fwd_peak_adc  = int(fwd_peak)
        self._fwd_avg_adc   = int(fwd_avg)
        self._rev_avg_adc   = int(rev_avg)
        self._pep_watts     = pep
        self._pa_current_ma = pa_current_ma

    def _fire_cc(self):
        if self._on_cc_change:
            self._on_cc_change(self._ptt, self._pa_enable, self._lna_gain_db,
                               self._tx_drive)

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
        if self._overlay:
            self.peak_line.set_xdata([freqs[peak_i], freqs[peak_i]])
            self.peak_line.set_visible(True)
            self._set_status(
                f"peak {power[peak_i]:.1f} dBm @ {freqs[peak_i]:.4f} MHz")
        else:
            self.peak_line.set_visible(False)
            self._set_status("")

        # Forward-power readout — PA current in amps (parenthesised) then PEP watts
        self._pwr_text.set_text(
            f"({self._pa_current_ma/1000.0:.1f}A) pep {self._pep_watts:.1f}w")

        if self._sweep == 1:
            rx_dbm = power[peak_i]
            tx_est = rx_dbm + 46.0 - self._lna_gain_db
            print(f"  Peak Power: Rx {rx_dbm:+.1f} dBm  "
                  f"( Tx Est. {tx_est:+.1f} dBm )  "
                  f"@ {freqs[peak_i]:.4f} MHz")

    def run(self):
        """Block until the plot window is closed."""
        plt.show()
