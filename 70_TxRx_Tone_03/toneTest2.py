#!/usr/bin/env python3
"""
toneTest2.py — HermesLite 2 tone-quality test bench.

Built on the example code (main_ex1.py + spectrum_ex1.py).  Adds a
**Grab 10s** button to the live spectrum window.  Pressing it:

    1. Grabs 10 seconds of the raw IQ signal shown in the plot.
    2. Saves that capture to memory (held in RAM) and to a file
       (tone_grab.npz) so it can be re-analysed later.
    3. Analyses the capture, treating it as a single (assumed) sine wave,
       and reports the quality metrics to the console:

           center frequency        (Hz / MHz, baseband offset + RF)
           magnitude               (RMS / peak, normalised + dBFS)
           frequency drift / sec    (max and average, Hz/s)
           phase jitter            (RMS, degrees)

Hardware / protocol behaviour is identical to main_ex1.py — RX-only at
startup; toggle PTT / En PA to transmit the test tone.

Usage
-----
    python toneTest2.py
"""

import os
import socket
import struct
import threading
import time

import numpy as np
from matplotlib.widgets import Button, TextBox

from spectrum_ex1 import LiveSpectrum
from main_ex1 import (
    freq_to_word, build_ep2, parse_ep6, make_cc, connect, init_radio,
    parse_ep6_power, adc_to_watts, adc_to_current_ma,
)

# =============================================================================
#  Configuration  (mirrors main_ex1.py)
# =============================================================================
HPSDR_PORT      = 1024
HL2_IP          = "169.254.19.221"
NCO_FREQ        = 6_880_000          # Hz  — 6.88 MHz default
TONE_OFFSET     = 870                # Hz  — TX tone = NCO + 870 Hz
TX_AMPLITUDE    = 0.800              # normalised digital level
RX_LNA_GAIN_DB  = -12                # dB  — AD9866 FAST_LNA (hardware floor; see note)
                                     #   Target was -16 dB, but the 6-bit gain field
                                     #   only reaches -12 dB; below that it WRAPS to
                                     #   +48 dB.  For a true -16 dB add a ~4 dB
                                     #   external pad on top of this -12 dB setting.
SAMPLE_RATE     = 384_000            # sps
CAPTURE_SIZE    = 65_536             # IQ samples per live sweep
PACKET_SAMPLES  = 126                # fixed by Protocol 1
PACKET_INTERVAL = PACKET_SAMPLES / SAMPLE_RATE

SEGMENT_SIZE    = 553             # -3 dB RBW = (fs/seg)*1.44 ≈ 1 kHz default
FFT_PAD_SIZE    = 1_024           # (matches the "1 kHz" RBW selector option)
DBFS_REF_DBM    = 14.0            # +6 dB cal (was 8.0) — trim to spec-an

GRAB_SECONDS    = 10.0               # length of the tone-quality grab
GRAB_FILE       = "tone_grab.npz"    # saved capture (also kept in RAM)

RF3_LOOPBACK    = True
DEBUG_TELEM     = False

# In-RAM store of the most recent grab: {"iq", "fs", "nco"}
LAST_GRAB = {}

# =============================================================================
#  Tone-quality analysis
# =============================================================================

def _moving_average(x, w):
    """Centred moving average of length len(x); edges replicated. w in samples."""
    if w <= 1:
        return x.astype(np.float64)
    c  = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    ma = (c[w:] - c[:-w]) / w               # length n-w+1
    pad_l = (w - 1) // 2
    pad_r = (w - 1) - pad_l
    return np.concatenate([np.full(pad_l, ma[0]), ma, np.full(pad_r, ma[-1])])


def _parabolic_peak(mag, k):
    """Sub-bin peak offset (in bins) from a 3-point log-parabola around index k."""
    n = len(mag)
    a = np.log(mag[(k - 1) % n] + 1e-20)
    b = np.log(mag[k]           + 1e-20)
    c = np.log(mag[(k + 1) % n] + 1e-20)
    denom = (a - 2.0 * b + c)
    if denom == 0.0:
        return 0.0
    return 0.5 * (a - c) / denom


def analyze_tone(iq, fs, nco_hz, dc_guard_hz=50.0, bp_halfwidth_hz=1000.0):
    """
    Analyse an (expected) tone within +/- bp_halfwidth_hz of the DEFINED freq
    (i.e. baseband DC = the MHz box / NCO), for characterising a tone received
    from another radio.

    Restricts the analysis to that +/- 1 kHz window (DC-guarded to reject LO
    leakage) rather than auto-detecting the strongest carrier anywhere in the
    band.  Returns a dict of metrics; printed by report_tone().
    """
    iq = np.asarray(iq, dtype=np.complex128)
    n  = len(iq)
    dt = 1.0 / fs
    t  = np.arange(n) * dt
    dur = n * dt

    # ── 1. Window = +/- bp_halfwidth around the DEFINED freq (baseband DC),
    #       DC-guarded to reject LO leakage.  The expected tone lives here.
    Z     = np.fft.fft(iq)
    freqs = np.fft.fftfreq(n, dt)
    mag   = np.abs(Z)
    win   = (np.abs(freqs) <= bp_halfwidth_hz) & (np.abs(freqs) >= dc_guard_hz)
    magw  = mag * win
    k        = int(np.argmax(magw))            # tone peak within the window
    bin_hz   = fs / n
    f0_fft   = float(freqs[k]) + _parabolic_peak(mag, k) * bin_hz
    tone_present = bool(np.any(win))

    # ── 2. Band-pass = that +/- window (isolates the tone for phase analysis) ─
    iq_bp = np.fft.ifft(Z * win)

    # ── 3. Unwrapped phase → precise center frequency (global linear fit) ────
    phase = np.unwrap(np.angle(iq_bp))
    slope, intercept = np.polyfit(t, phase, 1)
    f_center = slope / (2.0 * np.pi)            # precise baseband offset (Hz)
    rf_center = nco_hz + f_center

    # ── 4. Phase jitter = short-term RMS phase deviation ─────────────────────
    # Remove the ideal linear phase, then high-pass out slow drift/wander with a
    # moving-average trend so jitter reflects fast deviation, not frequency drift.
    resid    = phase - (slope * t + intercept)
    jit_win  = max(1, int(round(fs * 0.05)))          # 50 ms detrend window
    if n > 4 * jit_win:
        trend    = _moving_average(resid, jit_win)
        jitter   = (resid - trend)[jit_win:-jit_win]   # drop padded edges
    else:
        jitter   = resid
    phase_jitter_rad = float(np.std(jitter))
    phase_jitter_deg = float(np.degrees(phase_jitter_rad))

    # ── 5. Frequency drift per second (1-second block frequencies) ───────────
    blk   = int(round(fs))                     # 1-second blocks
    nblk  = n // blk
    blk_freqs = []
    for i in range(nblk):
        sl = slice(i * blk, (i + 1) * blk)
        s  = np.polyfit(t[sl], phase[sl], 1)[0]
        blk_freqs.append(s / (2.0 * np.pi))
    blk_freqs = np.asarray(blk_freqs)

    if len(blk_freqs) >= 2:
        drift_steps  = np.diff(blk_freqs)              # Hz change over each 1-s step
        drift_avg    = float(np.mean(np.abs(drift_steps)))
        drift_max    = float(np.max(np.abs(drift_steps)))
        secs         = np.arange(len(blk_freqs), dtype=float)
        overall_drift = float(np.polyfit(secs, blk_freqs, 1)[0])  # Hz/s linear trend
    else:
        drift_avg = drift_max = overall_drift = 0.0

    # ── 6. Magnitude of the windowed tone (level in the +/- 1 kHz window) ────
    mean_sq  = float(np.mean(np.abs(iq_bp) ** 2))
    rms      = float(np.sqrt(mean_sq))
    peak     = float(np.max(np.abs(iq_bp)))
    rms_dbfs = 20.0 * np.log10(rms + 1e-40)

    return {
        "n": n, "fs": fs, "dur": dur, "n_sec_blocks": int(nblk),
        "f_center_bb": f_center, "f_center_rf": rf_center, "f0_fft": f0_fft,
        "rms": rms, "peak": peak, "rms_dbfs": rms_dbfs,
        "drift_avg": drift_avg, "drift_max": drift_max,
        "overall_drift": overall_drift,
        "phase_jitter_deg": phase_jitter_deg,
        "phase_jitter_rad": phase_jitter_rad,
        "bp_halfwidth": bp_halfwidth_hz,
        "blk_freqs": blk_freqs,
    }


def report_tone(m):
    """Pretty-print the analyze_tone() result dict to the console."""
    print("\n" + "=" * 64)
    print("  RX TONE ANALYSIS  (+/- 1 kHz around the defined freq)")
    print("=" * 64)
    print(f"  Samples         : {m['n']:,}  ({m['dur']:.2f} s @ {m['fs']/1e3:.0f} ksps)")
    print(f"  Seconds measured: {m['n_sec_blocks']}")
    print(f"  Center freq     : {m['f_center_rf']/1e6:.6f} MHz "
          f"(offset {m['f_center_bb']:+.3f} Hz from defined freq)")
    print(f"  Level           : {m['rms']:.5f} rms  ({m['rms_dbfs']:+.1f} dBFS in +/-1 kHz)")
    print("-" * 64)
    print(f"  FREQ DRIFT/sec  : avg {m['drift_avg']:.4f} Hz/s   "
          f"max {m['drift_max']:.4f} Hz/s")
    print(f"  Overall drift   : {m['overall_drift']:+.4f} Hz/s (linear trend over {m['dur']:.0f} s)")
    print(f"  PHASE JITTER    : {m['phase_jitter_deg']:.4f} deg rms "
          f"({m['phase_jitter_rad']*1e3:.3f} mrad)  [in +/-1 kHz]")
    if len(m["blk_freqs"]) >= 2:
        fr = m["blk_freqs"]
        print("-" * 64)
        print("  Per-second freq (Hz offset from defined freq):")
        for i, f in enumerate(fr):
            step = (fr[i] - fr[i-1]) if i > 0 else 0.0
            print(f"    s{i:2d}: {f:+10.4f} Hz   (step {step:+.4f} Hz)")
    print("=" * 64 + "\n")


def analyze_noise(iq, fs, dbfs_ref=0.0):
    """
    Time-domain RMS / total-power of a capture — independent of the FFT display.

    Measures total power in the full IQ bandwidth (= fs), then derives the
    noise density and the equivalent level in common RBWs so you can cross-
    check the spectrum's noise floor.  Capture with the TONE OFF (and no strong
    in-band carrier) for this to read the noise.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    iq = iq - np.mean(iq)                       # strip DC / LO leakage
    n  = len(iq)

    mean_sq    = float(np.mean(np.abs(iq) ** 2))
    rms        = float(np.sqrt(mean_sq))
    peak       = float(np.max(np.abs(iq)))
    crest_db   = 20.0 * np.log10(peak / (rms + 1e-40))
    total_dbfs = 10.0 * np.log10(mean_sq + 1e-40)
    total_dbm  = 10.0 * np.log10(mean_sq / 50.0 / 1e-3 + 1e-40) + dbfs_ref
    density    = total_dbm - 10.0 * np.log10(fs)     # dBm/Hz

    return {
        "n": n, "fs": fs, "rms": rms, "peak": peak, "crest_db": crest_db,
        "total_dbfs": total_dbfs, "total_dbm": total_dbm, "density": density,
    }


def report_noise(m):
    """Pretty-print the analyze_noise() result dict."""
    c = m["crest_db"]
    if c < 6.0:
        kind = "CW/carrier-dominated (RMS is not noise)"
    elif c < 15.0:
        kind = "noise-like (Gaussian ~12 dB)"
    else:
        kind = "IMPULSIVE - man-made noise / carrier present (RMS not clean noise)"
    print("\n" + "-" * 64)
    print("  RMS / TOTAL-POWER  (full IQ bandwidth - grab with TONE OFF)")
    print("-" * 64)
    print(f"  Samples     : {m['n']:,}")
    print(f"  RMS         : {m['rms']:.6f}  ({m['total_dbfs']:+.2f} dBFS)")
    print(f"  Peak        : {m['peak']:.6f}   crest {m['crest_db']:.1f} dB  "
          f"-> {kind}")
    print(f"  Total power : {m['total_dbm']:+.2f} dBm  over {m['fs']/1e3:.0f} kHz")
    print(f"  Density     : {m['density']:+.2f} dBm/Hz")
    for rbw in (100, 1000, 10000):
        tag = "   <- compare to spec-an" if rbw == 10000 else ""
        print(f"   in {rbw:5d} Hz : {m['density'] + 10.0*np.log10(rbw):+.2f} dBm{tag}")
    print("-" * 64 + "\n")


# =============================================================================
#  GUI subclass — adds the "Grab 10s" button
# =============================================================================

class ToneSpectrum(LiveSpectrum):
    """LiveSpectrum + a Grab button that requests a 10-second tone grab.

    Also lowers the LNA-gain floor from 0 dB to the AD9866 hardware minimum of
    -12 dB so negative ("pad") settings are usable.
    """

    LNA_MIN = -12   # AD9866 gain floor (below this the 6-bit field wraps to +48 dB)
    LNA_MAX = 48

    def __init__(self, *args, on_grab=None, **kwargs):
        self._on_grab = on_grab
        super().__init__(*args, **kwargs)

        try:
            self.fig.canvas.manager.set_window_title("HL2 Tone Tool")
        except Exception:
            pass

        self._duplex = False   # C4[2]: lets RX see the -46 dB internal TX tap

        # Tear down example widgets we don't use: the bottom Capture button,
        # the LNA/Freq "Edit" buttons, and the TX-power Slider (replaced below
        # by a click-to-edit TxPwr box, consistent with LNA Gn / MHz).
        for attr in ("_btn_cap", "_btn_lna_edit", "_btn_freq_edit", "_sld_txp"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.disconnect_events()
                    w.ax.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        # TxPwr — click-to-edit text box (0–15 coarse drive), replaces the slider.
        ax_txp = self.fig.add_axes([0.085, 0.045, 0.05, 0.05])
        self._txp_busy = False
        self._txt_txp  = TextBox(ax_txp, "TxPwr ", initial=str(int(self._tx_drive)),
                                 color="#001155", hovercolor="#0a1a55",
                                 textalignment="center")
        self._txt_txp.label.set_color("white")
        self._txt_txp.text_disp.set_color("white")
        self._txt_txp.on_submit(self._on_txp_submit)

        # TxAmp — click-to-edit software TX amplitude (0.00–1.00 of full scale).
        ax_txa = self.fig.add_axes([0.16, 0.045, 0.05, 0.05])
        self._txamp_busy = False
        self._txt_txamp  = TextBox(ax_txa, "TxAmp ", initial=f"{_tx_amp[0]:.2f}",
                                   color="#001155", hovercolor="#0a1a55",
                                   textalignment="center")
        self._txt_txamp.label.set_color("white")
        self._txt_txamp.text_disp.set_color("white")
        self._txt_txamp.on_submit(self._on_txamp_submit)

        # Green Grab button.
        ax_grab = self.fig.add_axes([0.485, 0.045, 0.08, 0.05])
        self._btn_grab = Button(ax_grab, f"Grab {int(GRAB_SECONDS)}s",
                                color="#0a5a2a", hovercolor="#0e8a3f")
        self._btn_grab.label.set_color("white")
        self._btn_grab.on_clicked(self._on_grab_click)

        # Overlay -> "Pk" (peak marker/readout); narrower button.
        self._btn_overlay.label.set_text("Pk")

        # Duplex toggle — RX sees the -46 dB internal TX tap during TX.
        ax_dup = self.fig.add_axes([0.0, 0.0, 0.01, 0.01])   # positioned below
        self._btn_duplex = Button(ax_dup, "Duplex", color=self._btn_col(self._duplex),
                                  hovercolor="#0055cc")
        self._btn_duplex.label.set_color("white")
        self._btn_duplex.on_clicked(self._toggle_duplex)

        # Tone toggle — start/stop the test tone in the TX stream.
        ax_tone = self.fig.add_axes([0.0, 0.0, 0.01, 0.01])  # positioned below
        self._btn_tone = Button(ax_tone, "Tone", color=self._btn_col(_tone_on.is_set()),
                                hovercolor="#0055cc")
        self._btn_tone.label.set_color("white")
        self._btn_tone.on_clicked(self._toggle_tone)

        # All value boxes are display-only; a click opens their dialog.
        for tb in (self._txt_txp, self._txt_txamp, self._txt_lna, self._txt_freq):
            try:
                tb.disconnect_events()
            except Exception:
                pass

        # ── Lay out the bottom row:
        #   left-justified  : TxPwr, LNA Gn
        #   centred buttons : Tone, PTT, En PA, Duplex, Pk, Grab 10s
        #   right-justified : MHz, RBW
        # Widen the plot into the side margins (less empty space L/R).
        self.fig.subplots_adjust(left=0.085, right=0.985)

        Y, H = 0.045, 0.05
        # Left — TX level controls (TxPwr coarse drive, TxAmp fine digital) + LNA
        self._txt_txp.ax.set_position(    [0.040,  Y, 0.05,  H])
        self._txt_txamp.ax.set_position(  [0.110,  Y, 0.05,  H])
        self._txt_lna.ax.set_position(    [0.180,  Y, 0.05,  H])
        # Centre — signal-flow order: Tone -> PTT -> En PA -> Duplex, then Pk, Grab
        self._btn_tone.ax.set_position(   [0.3125, Y, 0.05,  H])
        self._btn_ptt.ax.set_position(    [0.3705, Y, 0.05,  H])
        self._btn_pa.ax.set_position(     [0.4285, Y, 0.055, H])
        self._btn_duplex.ax.set_position( [0.4915, Y, 0.065, H])
        self._btn_overlay.ax.set_position([0.5645, Y, 0.035, H])
        self._btn_grab.ax.set_position(   [0.6075, Y, 0.08,  H])
        # Right
        self._txt_freq.ax.set_position(   [0.790,  Y, 0.05,  H])
        self._radio_rbw.ax.set_position(  [0.850,  0.025, 0.10, 0.115])

        # Text-box labels above the box (centred) instead of to the left.
        for tb, name in ((self._txt_txp, "TxPwr"), (self._txt_txamp, "TxAmp"),
                         (self._txt_lna, "LNA Gn"), (self._txt_freq, "MHz")):
            tb.label.set_text(name)
            tb.label.set_position((0.5, 1.45))
            tb.label.set_horizontalalignment("center")
            tb.label.set_verticalalignment("bottom")

        # RBW label centred above its (now right-most) selector.
        self._rbw_label.set_position((0.90, 0.155))
        self._rbw_label.set_horizontalalignment("center")

    def _on_click(self, event):
        """Click a value box to open its edit dialog (boxes are display-only)."""
        if event.inaxes is self._txt_txp.ax:
            self._on_txp_edit(event)
        elif event.inaxes is self._txt_txamp.ax:
            self._on_txamp_edit(event)
        elif event.inaxes is self._txt_lna.ax:
            self._on_lna_edit(event)
        elif event.inaxes is self._txt_freq.ax:
            self._on_freq_edit(event)
        else:
            self._anim_resume()

    # ── Duplex + Tone toggles ───────────────────────────────────────────────

    def _fire_cc(self):
        """Push CC registers, including the Duplex bit (extends the base call)."""
        if self._on_cc_change:
            self._on_cc_change(self._ptt, self._pa_enable, self._lna_gain_db,
                               self._tx_drive, self._duplex)

    def _toggle_duplex(self, _event):
        self._duplex = not self._duplex
        self._btn_duplex.color = self._btn_col(self._duplex)
        self._btn_duplex.ax.set_facecolor(self._btn_col(self._duplex))
        self.fig.canvas.draw_idle()
        self._fire_cc()

    def _toggle_tone(self, _event):
        if _tone_on.is_set():
            _tone_on.clear()
        else:
            _tone_on.set()
        on = _tone_on.is_set()
        self._btn_tone.color = self._btn_col(on)
        self._btn_tone.ax.set_facecolor(self._btn_col(on))
        self.fig.canvas.draw_idle()

    # ── TX power: click-to-edit text box (0–15), replaces the slider ─────────

    def _on_txp_submit(self, text):
        if self._txp_busy:
            return
        try:
            v = int(round(float(text)))
        except (ValueError, TypeError):
            v = int(self._tx_drive)
        v = max(0, min(15, v))
        self._tx_drive = v
        if self._txt_txp.text != str(v):
            self._txp_busy = True
            self._txt_txp.set_val(str(v))
            self._txp_busy = False
        self._fire_cc()
        self._anim_resume()

    def _on_txp_edit(self, _event):
        from tkinter import simpledialog

        self._anim_pause()
        self._set_status("Paused — enter TX power (0–15) ...")
        self.fig.canvas.draw_idle()
        try:
            parent = getattr(self.fig.canvas.manager, "window", None)
            val = simpledialog.askinteger(
                "TX Power", "Enter TX drive level (0–15):",
                parent=parent, initialvalue=int(self._tx_drive),
                minvalue=0, maxvalue=15)
        finally:
            self._anim_resume()

        if val is not None:
            v = max(0, min(15, int(val)))
            self._tx_drive = v
            self._txp_busy = True
            self._txt_txp.set_val(str(v))
            self._txp_busy = False
            self._fire_cc()

    # ── TX amplitude: software sample level 0.00–1.00 (no register; live) ────

    def _on_txamp_submit(self, text):
        if self._txamp_busy:
            return
        try:
            v = float(text)
        except (ValueError, TypeError):
            v = _tx_amp[0]
        v = max(0.0, min(1.0, v))
        _tx_amp[0] = v
        norm = f"{v:.2f}"
        if self._txt_txamp.text != norm:
            self._txamp_busy = True
            self._txt_txamp.set_val(norm)
            self._txamp_busy = False
        self._anim_resume()

    def _on_txamp_edit(self, _event):
        from tkinter import simpledialog

        self._anim_pause()
        self._set_status("Paused — enter TX amplitude (0.00–1.00) ...")
        self.fig.canvas.draw_idle()
        try:
            parent = getattr(self.fig.canvas.manager, "window", None)
            val = simpledialog.askfloat(
                "TX Amplitude", "Enter TX amplitude (0.00 - 1.00 of full scale):",
                parent=parent, initialvalue=_tx_amp[0], minvalue=0.0, maxvalue=1.0)
        finally:
            self._anim_resume()

        if val is not None:
            v = max(0.0, min(1.0, float(val)))
            _tx_amp[0] = v
            self._txamp_busy = True
            self._txt_txamp.set_val(f"{v:.2f}")
            self._txamp_busy = False

    # ── LNA gain: allow -12..+48 dB (override the example's 0..48 floor) ─────

    def _on_lna_submit(self, text):
        if self._lna_busy:
            return
        try:
            g = int(round(float(text)))
        except (ValueError, TypeError):
            g = int(self._lna_gain_db)
        g = max(self.LNA_MIN, min(self.LNA_MAX, g))
        self._lna_gain_db = g
        if self._txt_lna.text != str(g):
            self._lna_busy = True
            self._txt_lna.set_val(str(g))
            self._lna_busy = False
        self._fire_cc()
        self._anim_resume()

    def _on_lna_edit(self, _event):
        from tkinter import simpledialog

        self._anim_pause()
        self._set_status(
            f"Paused — enter LNA gain ({self.LNA_MIN}..{self.LNA_MAX} dB) ...")
        self.fig.canvas.draw_idle()
        try:
            parent = getattr(self.fig.canvas.manager, "window", None)
            val = simpledialog.askinteger(
                "LNA Gain",
                f"Enter LNA gain ({self.LNA_MIN} to {self.LNA_MAX} dB):",
                parent=parent, initialvalue=int(self._lna_gain_db),
                minvalue=self.LNA_MIN, maxvalue=self.LNA_MAX)
        finally:
            self._anim_resume()

        if val is not None:
            g = max(self.LNA_MIN, min(self.LNA_MAX, int(val)))
            self._lna_gain_db = g
            self._lna_busy = True
            self._txt_lna.set_val(str(g))
            self._lna_busy = False
            self._fire_cc()

    def _on_grab_click(self, _event):
        if self._on_grab:
            self._set_status(
                f"Grabbing {int(GRAB_SECONDS)} s for tone analysis "
                f"(see console) ...")
            self.fig.canvas.draw_idle()
            self._on_grab()


# =============================================================================
#  TX keep-alive thread  (verbatim behaviour from main_ex1)
# =============================================================================

_tx_running    = threading.Event()
_grab_request  = threading.Event()
_tone_on       = threading.Event()   # set by the Tone button — gates tone output
_tx_amp        = [TX_AMPLITUDE]      # live software TX amplitude (0.0-1.0), TxAmp box


def _tx_thread_fn(sock, ip, cc_holder, start_seq):
    phase_step = 2.0 * np.pi * TONE_OFFSET / SAMPLE_RATE
    local_t    = np.arange(PACKET_SAMPLES)
    zeros      = np.zeros(PACKET_SAMPLES, dtype=np.int16)
    phase      = 0.0
    seq        = start_seq
    next_send  = time.monotonic()

    while _tx_running.is_set():
        # Tone button gates the test tone; otherwise stream silence (keep-alive).
        if _tone_on.is_set():
            amp  = _tx_amp[0]
            I_tx = (amp * np.cos(phase + phase_step * local_t) * 32767.0).astype(np.int16)
            Q_tx = (amp * np.sin(phase + phase_step * local_t) * 32767.0).astype(np.int16)
        else:
            I_tx = zeros
            Q_tx = zeros
        phase = (phase + phase_step * PACKET_SAMPLES) % (2.0 * np.pi)

        cc_regs = cc_holder[0]
        n_regs  = len(cc_regs)
        cc0 = cc_regs[ seq      % n_regs]
        cc1 = cc_regs[(seq + 1) % n_regs]
        pkt = build_ep2(seq, cc0, cc1, list(zip(I_tx.tolist(), Q_tx.tolist())))
        try:
            sock.sendto(pkt, (ip, HPSDR_PORT))
        except OSError:
            break
        seq += 1

        next_send += PACKET_INTERVAL
        wait = next_send - time.monotonic()
        if wait > 0:
            time.sleep(wait)


# =============================================================================
#  Grab + live-capture thread
# =============================================================================

def _do_grab(sock, spectrum, seconds=GRAB_SECONDS, outfile=GRAB_FILE):
    """Grab `seconds` of the displayed signal, save it, and analyse the tone."""
    n = int(round(SAMPLE_RATE * seconds))
    print(f"\nGrab: capturing {seconds:.0f} s ({n:,} samples) of the plotted signal ...")

    # Flush stale packets first.
    t_end = time.monotonic() + 0.3
    while time.monotonic() < t_end:
        try:    sock.recvfrom(2048)
        except socket.timeout: pass

    samples  = []
    deadline = time.monotonic() + seconds * 3.0 + 5.0
    while len(samples) < n and time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) == 1032 and data[0:2] == b"\xEF\xFE":
                samples.extend(parse_ep6(data))
        except socket.timeout:
            pass

    samples = samples[:n]
    if len(samples) < n:
        print(f"Grab: WARNING only {len(samples):,}/{n:,} samples received "
              f"(check link); analysing what we have.")
    if not samples:
        print("Grab: no samples received — nothing saved.")
        return

    arr = np.asarray(samples, dtype=np.float64)
    iq  = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    nco = getattr(spectrum, "_nco_hz", NCO_FREQ)

    # Save into memory (RAM) ...
    LAST_GRAB.clear()
    LAST_GRAB.update(iq=iq, fs=SAMPLE_RATE, nco=nco)
    # ... and to a file.
    np.savez(outfile, iq=iq, fs=SAMPLE_RATE, nco=nco,
             lna_db=getattr(spectrum, "_lna_gain_db", 0), seconds=seconds)
    print(f"Grab: saved {len(iq):,} samples to {outfile} "
          f"(and kept in memory as LAST_GRAB).")

    pk = float(np.max(np.abs(iq)))
    if pk > 0.98:
        print("  *** near/at full scale — CLIPPING; metrics may be degraded ***")

    # Analyse and report — RMS/noise first (valid tone-on or -off), then tone.
    report_noise(analyze_noise(iq, SAMPLE_RATE,
                               getattr(spectrum, "_dbfs_ref", 0.0)))
    metrics = analyze_tone(iq, SAMPLE_RATE, nco)
    report_tone(metrics)

    try:
        spectrum._set_status(
            f"RX tone: {metrics['f_center_rf']/1e6:.5f} MHz   "
            f"drift avg {metrics['drift_avg']:.3f} / max {metrics['drift_max']:.3f} Hz/s   "
            f"jitter {metrics['phase_jitter_deg']:.3f} deg  (see console)")
        spectrum.fig.canvas.draw_idle()
    except Exception:
        pass


def _capture_thread_fn(sock, spectrum):
    while _tx_running.is_set():

        # Grab request takes priority over the live sweep.
        if _grab_request.is_set():
            _do_grab(sock, spectrum)
            _grab_request.clear()
            continue

        # Flush stale packets.
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            try:    sock.recvfrom(2048)
            except socket.timeout: pass

        # Live IQ + forward-power telemetry for the spectrum display.
        samples  = []
        fwd_vals = []
        rev_vals = []
        cur_vals = []
        deadline = time.monotonic() + 2.0
        while len(samples) < CAPTURE_SIZE and time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(2048)
                if len(data) == 1032 and data[0:2] == b"\xEF\xFE":
                    samples.extend(parse_ep6(data))
                    f, r, c = parse_ep6_power(data)
                    if f is not None: fwd_vals.append(f)
                    if r is not None: rev_vals.append(r)
                    if c is not None: cur_vals.append(c)
            except socket.timeout:
                pass

        if fwd_vals:
            fwd_peak = max(fwd_vals)
            fwd_avg  = sum(fwd_vals) / len(fwd_vals)
            rev_avg  = (sum(rev_vals) / len(rev_vals)) if rev_vals else 0.0
            cur_avg  = (sum(cur_vals) / len(cur_vals)) if cur_vals else 0.0
            pep      = max(0.0, adc_to_watts(fwd_peak) - adc_to_watts(rev_avg))
            spectrum.set_power(fwd_peak, fwd_avg, rev_avg, pep,
                               adc_to_current_ma(cur_avg))

        if len(samples) >= CAPTURE_SIZE:
            spectrum.push(samples[:CAPTURE_SIZE])
        else:
            print(f"  WARNING: only {len(samples)}/{CAPTURE_SIZE} samples received")

        time.sleep(1.0)


# =============================================================================
#  Main
# =============================================================================

def main():
    import sys

    freq_word = freq_to_word(NCO_FREQ)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", HPSDR_PORT))
    except OSError as e:
        print(f"\nERROR: UDP port {HPSDR_PORT} is already in use.")
        print("Another copy of toneTest2 / main_ex1 is probably still running.")
        print("Close that window (or process) and run again.")
        print(f"  find it:  Get-NetUDPEndpoint -LocalPort {HPSDR_PORT} "
              f"| Select OwningProcess")
        print(f"  details:  {e}")
        sock.close()
        try:
            input("\nPress Enter to close ...")
        except Exception:
            pass
        return
    seq = connect(sock, HL2_IP)
    sock.settimeout(0.002)

    # RX-only at startup (no TX, no tone, duplex off) — connect a 50-ohm load
    # before enabling PA/PTT.
    seq      = init_radio(sock, HL2_IP, freq_word, seq, ptt=False, pa_enable=False,
                          duplex=False)
    cc_regs  = make_cc(freq_word, ptt=False, pa_enable=False, duplex=False)
    cc_holder = [cc_regs]

    print("toneTest2 — HL2 tone-quality bench")
    print(f"Mode  : PA_EN=0 PTT=0  (RX only; toggle buttons to enable TX)")
    print(f"NCO   : {NCO_FREQ/1e6:.3f} MHz  |  tone +{TONE_OFFSET} Hz")
    print(f"LNA   : +{RX_LNA_GAIN_DB} dB  |  {SAMPLE_RATE//1000} ksps  |  TX amp {TX_AMPLITUDE:.3f}")
    print(f"Grab  : press 'Grab {int(GRAB_SECONDS)}s' to capture + analyse the tone")

    freq_holder = [freq_word]

    def on_cc_change(ptt, pa_enable, lna_gain_db, tx_drive, duplex=False):
        fw = freq_holder[0]
        cc_holder[0] = make_cc(fw, ptt=ptt, pa_enable=pa_enable,
                               lna_gain_db=lna_gain_db, tx_drive=tx_drive,
                               duplex=duplex)
        init_radio(sock, HL2_IP, fw, 0, ptt=ptt, pa_enable=pa_enable,
                   lna_gain_db=lna_gain_db, tx_drive=tx_drive, duplex=duplex)

    def on_grab():
        _grab_request.set()

    def on_freq_change(freq_hz):
        fw = freq_to_word(freq_hz)
        freq_holder[0] = fw
        cc_holder[0] = make_cc(fw, ptt=spectrum._ptt, pa_enable=spectrum._pa_enable,
                               lna_gain_db=spectrum._lna_gain_db,
                               tx_drive=spectrum._tx_drive, duplex=spectrum._duplex)
        init_radio(sock, HL2_IP, fw, 0, ptt=spectrum._ptt,
                   pa_enable=spectrum._pa_enable, lna_gain_db=spectrum._lna_gain_db,
                   tx_drive=spectrum._tx_drive, duplex=spectrum._duplex)

    # Suppress stderr noise during matplotlib/tkinter init.
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    spectrum = ToneSpectrum(
        nco_hz       = NCO_FREQ,
        fs           = SAMPLE_RATE,
        dbfs_ref     = DBFS_REF_DBM,
        seg          = SEGMENT_SIZE,
        pad          = FFT_PAD_SIZE,
        ymin         = -120,
        ymax         = 20,
        lna_gain_db  = RX_LNA_GAIN_DB,
        ptt          = False,
        pa_enable    = False,
        tx_drive     = 15,
        on_cc_change   = on_cc_change,
        on_capture     = None,
        on_freq_change = on_freq_change,
        on_grab        = on_grab,
    )
    sys.stderr = _stderr

    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_sock.bind(("", 0))

    _tx_running.set()
    threading.Thread(target=_tx_thread_fn,
                     args=(tx_sock, HL2_IP, cc_holder, seq),
                     daemon=True, name="HL2-TX").start()

    print("Stabilising (1 s) ...")
    time.sleep(1.0)

    threading.Thread(target=_capture_thread_fn,
                     args=(sock, spectrum),
                     daemon=True, name="HL2-Capture").start()

    spectrum.run()   # blocks until the window is closed

    print("\nShutting down ...")
    _tx_running.clear()
    time.sleep(0.2)
    stop_cmd = b"\xEF\xFE\x04\x00" + bytes(60)
    sock.sendto(stop_cmd, (HL2_IP, HPSDR_PORT))
    sock.sendto(stop_cmd, (HL2_IP, HPSDR_PORT))
    sock.close()
    tx_sock.close()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # If the program "just opens and closes", the real cause is here.
        # Print the traceback, also save it to a log, and pause so a
        # double-click launch doesn't vanish before it can be read.
        import traceback
        tb = traceback.format_exc()
        print("\n" + "=" * 64)
        print("toneTest2 CRASHED — traceback follows:")
        print("=" * 64)
        print(tb)
        try:
            with open("toneTest2_error.log", "w") as _fh:
                _fh.write(tb)
            print("(also written to toneTest2_error.log)")
        except Exception:
            pass
        try:
            input("\nPress Enter to close ...")
        except Exception:
            pass
