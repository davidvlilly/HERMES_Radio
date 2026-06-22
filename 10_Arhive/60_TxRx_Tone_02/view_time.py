#!/usr/bin/env python3
"""
view_time.py — time-domain viewer for a capture.npz from capture_ex1.py
(or the Capture button in main_ex1.py).

Computes the average power of the capture, shows the data in VOLTS, and draws
the average (RMS) voltage as a reference line placed at 30% of the lower plot's
height.

    avg power   = mean(|IQ|^2)          (per-sample mean-square)
    avg volts   = sqrt(avg power) * FULL_SCALE_V   (RMS of the envelope)

Samples come out of the radio normalized to full scale (+/-1.0).  To read TRUE
volts, set FULL_SCALE_V to the ADC full-scale peak voltage; left at 1.0 the
axis is "fraction of full scale".

Zoom:
    * toolbar magnifier — drag a box
    * drag a region on the lower plot to zoom the shared time axis
    * Home button / 'h' — reset

Usage
-----
    python view_time.py                 # loads capture.npz
    python view_time.py other.npz
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

INFILE       = sys.argv[1] if len(sys.argv) > 1 else "capture.npz"
FULL_SCALE_V = 1.0      # volts at |IQ| = 1.0  (set to ADC full-scale; 1.0 = normalised)
AVG_FRAC     = 0.30     # place the average-volts line at this fraction of plot height

# Band-pass around the frequency of interest
BP_ENABLE        = True      # apply a band-pass before plotting
BP_HALFWIDTH_HZ  = 2000.0    # +/- this around the center (so 4 kHz wide total)
CENTER_OFFSET_HZ = None      # None = auto-detect the peak; or set a baseband offset (Hz)
DC_GUARD_HZ      = 50.0      # when auto-detecting, ignore +/-this around DC (LO leakage)


def bandpass(iq, fs, halfwidth, center=None, dc_guard=50.0):
    """Brick-wall band-pass on complex IQ: keep [f0-halfwidth, f0+halfwidth].
    f0 = `center` Hz, or the auto-detected spectral peak (ignoring +/-dc_guard
    around DC so LO leakage isn't mistaken for the signal). Returns (iq_bp, f0)."""
    n     = len(iq)
    Z     = np.fft.fft(iq)
    freqs = np.fft.fftfreq(n, 1.0 / fs)
    if center is None:
        mag = np.abs(Z).copy()
        mag[np.abs(freqs) < dc_guard] = 0.0
        f0 = float(freqs[int(np.argmax(mag))])
    else:
        f0 = float(center)
    mask = (freqs >= f0 - halfwidth) & (freqs <= f0 + halfwidth)
    iq_bp = np.fft.ifft(Z * mask).astype(np.complex64)
    return iq_bp, f0


def main():
    if not os.path.exists(INFILE):
        print(f"ERROR: '{INFILE}' not found. Run  python capture_ex1.py  first.")
        return
    d  = np.load(INFILE, allow_pickle=True)
    iq = d["iq"]
    if len(iq) == 0:
        print("ERROR: capture is empty — re-run the capture.")
        return
    fs  = float(d["fs"])
    nco = float(d["nco"])
    lna = float(d["lna_db"])

    # Band-pass around the frequency of interest (before any measurement)
    f0 = 0.0
    if BP_ENABLE:
        iq, f0 = bandpass(iq, fs, BP_HALFWIDTH_HZ, CENTER_OFFSET_HZ, DC_GUARD_HZ)
        print(f"Bandpass +/-{BP_HALFWIDTH_HZ/1000:.1f} kHz around {f0:+.0f} Hz "
              f"baseband  (RF {(nco + f0)/1e6:.5f} MHz)")

    n    = len(iq)
    t_ms = np.arange(n) / fs * 1e3

    # ── Average-power measurement ──────────────────────────────────────────────
    mean_sq      = float(np.mean(np.abs(iq) ** 2))          # normalised mean power
    avg_pwr_dbfs = 10.0 * np.log10(mean_sq + 1e-40)         # 0 dBFS = full scale
    avg_volts    = np.sqrt(mean_sq) * FULL_SCALE_V          # RMS of the envelope (V)

    # Data in volts
    v_i   = iq.real * FULL_SCALE_V
    v_q   = iq.imag * FULL_SCALE_V
    v_mag = np.abs(iq) * FULL_SCALE_V
    v_fs  = FULL_SCALE_V

    unit = "V" if FULL_SCALE_V != 1.0 else "V (norm.)"
    print(f"Loaded {n} samples ({t_ms[-1]:.1f} ms).")
    print(f"Average power = {avg_pwr_dbfs:+.2f} dBFS   |   "
          f"average volts = {avg_volts:.4g} {unit} (RMS)")

    plt.style.use("dark_background")
    fig, (ax_iq, ax_v) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    try:
        fig.canvas.manager.set_window_title("HL2 Time-Domain Viewer")
    except Exception:
        pass

    bp_txt = (f"BP +/-{BP_HALFWIDTH_HZ/1000:.0f}kHz @ {(nco+f0)/1e6:.4f}MHz"
              if BP_ENABLE else "no BP")
    fig.suptitle(f"HL2 capture   {n} samples @ {fs/1e3:.0f} ksps   "
                 f"NCO {nco/1e6:.3f} MHz   LNA {lna:.0f} dB   {bp_txt}   "
                 f"avg {avg_pwr_dbfs:+.1f} dBFS   avg {avg_volts:.4g} {unit} rms")

    # ── Top: I / Q in volts ────────────────────────────────────────────────────
    ax_iq.plot(t_ms, v_i, color="#00ff88", lw=0.7, label="I")
    ax_iq.plot(t_ms, v_q, color="#ff8800", lw=0.7, label="Q")
    ax_iq.axhline( v_fs, color="#ff4444", lw=0.6, ls="--")
    ax_iq.axhline(-v_fs, color="#ff4444", lw=0.6, ls="--", label="full scale")
    ax_iq.set_ylabel(f"I / Q  ({unit})")
    # Autoscale to the (filtered) I/Q peak so the tone fills the plot
    pk_iq = max(float(np.max(np.abs(v_i))), float(np.max(np.abs(v_q))), 1e-12)
    ax_iq.set_ylim(-1.2 * pk_iq, 1.2 * pk_iq)
    ax_iq.grid(True, color="#333", lw=0.5)
    ax_iq.legend(loc="upper right", fontsize=8)

    # ── Bottom: |IQ| envelope in volts, avg line at AVG_FRAC of the height ──────
    ax_v.plot(t_ms, v_mag, color="#66ccff", lw=0.7, label="|IQ|")
    ax_v.axhline(avg_volts, color="#ffdd00", lw=1.0, ls="--")
    # Scale so the average line sits at AVG_FRAC (30%) of the plot height
    ax_v.set_ylim(0, avg_volts / AVG_FRAC if avg_volts > 0 else 1.0)
    # Label the average, placed 30% across the width, sitting on the line
    ax_v.text(0.30, avg_volts,
              f" avg = {avg_volts:.4g} {unit} rms   ({avg_pwr_dbfs:+.1f} dBFS) ",
              transform=ax_v.get_yaxis_transform(),
              color="#ffdd00", fontsize=9, va="bottom", ha="left")
    ax_v.set_xlabel("time (ms)")
    ax_v.set_ylabel(f"|IQ|  ({unit})")
    ax_v.grid(True, color="#333", lw=0.5)
    ax_v.legend(loc="upper right", fontsize=8)

    # Drag a region on the lower axis to zoom the shared x-axis
    def on_select(xmin, xmax):
        if xmax - xmin > 1e-6:
            ax_iq.set_xlim(xmin, xmax)
            fig.canvas.draw_idle()

    fig._span = SpanSelector(ax_v, on_select, "horizontal", useblit=True,
                             props=dict(alpha=0.3, facecolor="#4444ff"))

    print("Zoom: toolbar magnifier, or drag a box on the lower plot. Home resets.")
    plt.show()


if __name__ == "__main__":
    main()
