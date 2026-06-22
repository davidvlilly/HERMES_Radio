#!/usr/bin/env python3
"""
sounder.py - HF channel sounder GUI for the Hermes-Lite 2.

Same tool both ends of the link.  Pick a SIGNAL type from the menu; on the TX
end key it up (TX), on the RX end press Grab+Analyze.  Each grab writes a
per-mode analysis .txt and a data .npz so the set can be fed to an LLM for the
modulation decision.  Self-test: Loopback feeds the internally generated signal
back into the LNA (duplex) so every mode can be validated before going on air.

    python sounder.py
"""

import os
import sys
import time
import threading

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox

import waveforms as wf
import channel as ch
from hl2 import HL2, SAMPLE_RATE

# ── Configuration ────────────────────────────────────────────────────────────
HL2_IP      = "169.254.19.221"
NCO_FREQ    = 6_880_000          # Hz
LNA_GAIN_DB = 0
TX_AMP      = 0.8                # DAC/FPGA digital level 0.0-1.0 (0.8 = headroom)
DBM_OFFSET  = 0.0                # dBm = dBFS + this; trim vs a known input for true dBm
GRAB_SEC    = 5.0                # sounding capture length (s)
SAVE_DIR    = os.path.dirname(os.path.abspath(__file__))
END_ID      = "A"               # which end this radio is (set per station)
CW_FREQ     = 1000.0            # CW diagnostic tone offset from NCO (Hz)

SIGNAL_MODES = ["CW", "LFM chirp", "Comb", "OFDM", "DSSS/PN"]


def make_waveform(mode, fs, amp=TX_AMP):
    d = wf.hf_defaults(fs)
    if mode == "CW":
        iq, spec = wf.cw_tone(fs, freq=CW_FREQ, amp=amp)
        spec["dbm_offset"] = DBM_OFFSET           # for dBm reporting (dBm = dBFS + offset)
        return iq, spec
    if mode == "LFM chirp":
        return wf.lfm_chirp(fs, amp=amp, **d["chirp"])
    if mode == "Comb":
        return wf.multitone_comb(fs, amp=amp, **d["comb"])
    if mode == "Two-tone":
        return wf.two_tone(fs, amp=amp, **d["two_tone"])
    if mode == "OFDM":
        return wf.ofdm_sounding(fs, amp=amp, **d["ofdm"])
    if mode == "DSSS/PN":
        return wf.pn_dsss(fs, amp=amp, **d["dsss"])
    raise ValueError(mode)


def analyse(mode, rx, spec):
    if mode == "CW":        return ch.analyse_cw(rx, spec)
    if mode == "LFM chirp": return ch.analyse_chirp(rx, spec)
    if mode == "Comb":      return ch.analyse_comb(rx, spec)
    if mode == "Two-tone":  return ch.analyse_two_tone(rx, spec)
    if mode == "OFDM":      return ch.analyse_ofdm(rx, spec)
    if mode == "DSSS/PN":   return ch.analyse_dsss(rx, spec)
    raise ValueError(mode)


def safe_name(mode):
    s = mode.lower()
    for ch in " -/\\":                      # strip path-unsafe / spacing chars
        s = s.replace(ch, "_")
    return s


class Sounder:
    def __init__(self, radio):
        self.radio = radio
        self.mode  = "CW"
        self.tx_amp = TX_AMP
        self.spec  = None
        self._iq   = None            # last live spectrum block
        self._lock = threading.Lock()
        self._dirty = False
        self._busy  = threading.Event()   # set while a sounding grab runs
        self._tone = False
        self._ptt = False
        self._pa = False
        self._duplex = False
        self._all_on = False
        self._wide_scale = True           # Scale toggles 0..-140 (default) / 0..-100 dBm
        self._last_S = None
        self.rbw_hz = 100.0               # live-spectrum resolution bandwidth
        self._live_rx = None              # latest live block (filled by bg thread)
        self._live_lock = threading.Lock()
        self._run_live = threading.Event(); self._run_live.set()

        plt.style.use("dark_background")
        self.fig, (self.ax_s, self.ax_d) = plt.subplots(
            2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]})
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.22,
                                 hspace=0.35)
        try:
            self.fig.canvas.manager.set_window_title("HL2 HF Sounder")
        except Exception:
            pass

        f_lo = (NCO_FREQ - SAMPLE_RATE / 2) / 1e6
        f_hi = (NCO_FREQ + SAMPLE_RATE / 2) / 1e6
        self.ax_s.set_xlim(f_lo, f_hi)
        self.ax_s.set_ylim(DBM_OFFSET - 140, DBM_OFFSET)             # 0..-140 dBm default (Scale toggles -100)
        self.ax_s.set_xlabel("Frequency (MHz)"); self.ax_s.set_ylabel("dBm")
        self.ax_s.grid(True, color="#333", lw=0.5)
        self.line, = self.ax_s.plot([], [], color="#00ff88", lw=0.8)
        self._rbw_text = self.ax_s.text(0.99, 0.94, "RBW 100 Hz",
                                        transform=self.ax_s.transAxes,
                                        ha="right", va="top", color="#888888",
                                        fontsize=9, family="monospace")

        self.ax_d.set_xlabel("delay (ms)"); self.ax_d.set_ylabel("power (dB)")
        self.ax_d.set_title("Detection Power")
        self.ax_d.set_ylim(-20, 0)
        self.ax_d.grid(True, color="#333", lw=0.5)
        self.pdp_line, = self.ax_d.plot([], [], color="#ffcc00", lw=1.0)
        self.pdp_line2, = self.ax_d.plot([], [], color="#ff66cc", lw=1.0,
                                         visible=False)   # 2nd tone (Two-tone mode)
        # Comb mode: coherence-curve reference lines (0.5 level + BW marker).
        self._coh_half = self.ax_d.axhline(0.5, color="#888888", lw=0.8,
                                           ls="--", visible=False)
        self._coh_bw_marker = self.ax_d.axvline(0.0, color="#ff5555", lw=1.0,
                                                ls=":", visible=False)

        # OFDM data-fidelity sub-panels (constellation + sent/received sequence),
        # overlaying the lower-panel region; shown only in OFDM mode.
        posd = self.ax_d.get_position()
        cw = posd.height * 7.0 / 11.0                    # square box (fig is 11x7)
        self.ax_const = self.fig.add_axes([posd.x0, posd.y0, cw, posd.height])
        self.ax_seq   = self.fig.add_axes([posd.x0 + cw + 0.07, posd.y0,
                                           posd.x1 - (posd.x0 + cw + 0.07), posd.height])
        self.ax_const.set_visible(False); self.ax_seq.set_visible(False)

        self._status = self.ax_s.text(0.5, 1.04, "", transform=self.ax_s.transAxes,
                                      ha="center", va="bottom", color="#aaa",
                                      fontsize=9, family="monospace")

        self._build_controls()
        self._build_menus()
        self._select_mode(self.mode)

        self._ani = animation.FuncAnimation(self.fig, self._animate,
                                            interval=200, blit=False,
                                            cache_frame_data=False)
        # Background live-capture thread keeps the GUI responsive (grabs block).
        threading.Thread(target=self._live_loop, daemon=True, name="live-rx").start()

    def _async(self, fn, **kw):
        """Run a (possibly slow) radio call off the GUI thread."""
        threading.Thread(target=lambda: fn(**kw), daemon=True).start()

    def _live_loop(self):
        while self._run_live.is_set():
            if self._busy.is_set():               # yield the socket to a sounding grab
                time.sleep(0.1); continue
            seg = max(64, int(round(1.44 * SAMPLE_RATE / self.rbw_hz)))
            grab_len = min(0.5, max(0.05, 4.0 * seg / SAMPLE_RATE))
            try:
                rx = self.radio.grab(grab_len)
            except Exception:
                time.sleep(0.2); continue
            with self._live_lock:
                self._live_rx = rx

    # ── bottom controls (matplotlib widgets) ─────────────────────────────────
    def _build_controls(self):
        Y, H = 0.050, 0.05
        # Value boxes (display-only; click opens a dialog), labels ABOVE the box.
        ax_freq = self.fig.add_axes([0.045, Y, 0.055, H])
        self.tb_freq = TextBox(ax_freq, "MHz", initial=f"{NCO_FREQ/1e6:.3f}",
                               color="#001155", hovercolor="#0a1a55",
                               textalignment="center")
        ax_lna = self.fig.add_axes([0.125, Y, 0.05, H])
        self.tb_lna = TextBox(ax_lna, "LNA Gn", initial=str(LNA_GAIN_DB),
                              color="#001155", hovercolor="#0a1a55",
                              textalignment="center")
        ax_amp = self.fig.add_axes([0.205, Y, 0.05, H])
        self.tb_amp = TextBox(ax_amp, "TxAmp", initial=f"{TX_AMP:.2f}",
                              color="#001155", hovercolor="#0a1a55",
                              textalignment="center")
        ax_pwr = self.fig.add_axes([0.285, Y, 0.05, H])
        self.tb_pwr = TextBox(ax_pwr, "TxPwr", initial="15",
                              color="#001155", hovercolor="#0a1a55",
                              textalignment="center")
        for tb, name in ((self.tb_freq, "MHz"), (self.tb_lna, "LNA Gn"),
                         (self.tb_amp, "TxAmp"), (self.tb_pwr, "TxPwr")):
            tb.label.set_text(name)
            tb.label.set_position((0.5, 1.4))          # label on top of the box
            tb.label.set_horizontalalignment("center")
            tb.label.set_verticalalignment("bottom")
            tb.label.set_color("white")
            tb.text_disp.set_color("white")
            try:
                tb.disconnect_events()
            except Exception:
                pass

        # Toggle buttons (Tone / PTT / En PA / Duplex / Auto) + Grab action.
        Hb = 0.0375                     # 25% shorter than the boxes
        Yb = Y + H - Hb                 # all buttons aligned along the top
        ax_tone = self.fig.add_axes([0.345, Yb, 0.05,  Hb])
        ax_ptt  = self.fig.add_axes([0.405, Yb, 0.045, Hb])
        ax_pa   = self.fig.add_axes([0.460, Yb, 0.055, Hb])
        ax_dup  = self.fig.add_axes([0.525, Yb, 0.065, Hb])
        ax_auto = self.fig.add_axes([0.600, Yb, 0.05,  Hb])
        ax_grab = self.fig.add_axes([0.660, Yb, 0.18,  Hb])
        self.b_tone = Button(ax_tone, "Tone",   color="#2266cc")
        self.b_ptt  = Button(ax_ptt,  "PTT",    color="#2266cc")
        self.b_pa   = Button(ax_pa,   "En PA",  color="#2266cc")
        self.b_dup  = Button(ax_dup,  "Duplex", color="#2266cc")
        self.b_auto = Button(ax_auto, "Scale",  color="#2266cc")
        self.b_grab = Button(ax_grab, "Grab + Analyze", color="#0a5a2a",
                             hovercolor="#0e8a3f")
        for b in (self.b_tone, self.b_ptt, self.b_pa, self.b_dup,
                  self.b_auto, self.b_grab):
            b.label.set_color("white")
        self.b_tone.on_clicked(self._toggle_tone)
        self.b_ptt.on_clicked(self._toggle_ptt)
        self.b_pa.on_clicked(self._toggle_pa)
        self.b_dup.on_clicked(self._toggle_duplex)
        self.b_auto.on_clicked(self._toggle_auto)
        self.b_grab.on_clicked(self._on_grab)

        # Master TX bar (no label) under Tone..Duplex: one click keys all four.
        # Half the button height, slid up snug under the button row.
        x0 = ax_tone.get_position().x0
        x1 = ax_dup.get_position().x1
        ax_all = self.fig.add_axes([x0, Yb - 0.0225 - 0.004, x1 - x0, 0.0225])
        self.b_all = Button(ax_all, "", color="#444444", hovercolor="#aa0000")
        self.b_all.on_clicked(self._toggle_all)

        # "Grab in progress" indicator: red dot just right of the Grab button.
        gx = ax_grab.get_position().x1
        self._grab_dot = self.fig.text(gx + 0.012, Yb + Hb / 2.0, "●",
                                       color="red", fontsize=18, ha="center",
                                       va="center", visible=False)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ── Tk File / Signal / Help menus ────────────────────────────────────────
    def _build_menus(self):
        try:
            import tkinter as tk
            root = self.fig.canvas.manager.window
            bar  = tk.Menu(root)

            filem = tk.Menu(bar, tearoff=0)
            filem.add_command(label=f"Save folder: {SAVE_DIR}", state="disabled")
            filem.add_command(label="Quit", command=root.quit)
            bar.add_cascade(label="File", menu=filem)

            self._mode_var = tk.StringVar(value=self.mode)
            sigm = tk.Menu(bar, tearoff=0)
            for m in SIGNAL_MODES:
                sigm.add_radiobutton(label=m, variable=self._mode_var,
                                     command=lambda mm=m: self._select_mode(mm))
            bar.add_cascade(label="Signal", menu=sigm)

            self._rbw_var = tk.StringVar(value="100 Hz")
            viewm = tk.Menu(bar, tearoff=0)
            for label, hz in (("10 Hz", 10), ("100 Hz", 100),
                              ("1 kHz", 1000), ("10 kHz", 10000)):
                viewm.add_radiobutton(label=f"RBW {label}", variable=self._rbw_var,
                                      value=label,
                                      command=lambda h=hz, l=label: self._set_rbw(h, l))
            bar.add_cascade(label="View", menu=viewm)

            helpm = tk.Menu(bar, tearoff=0)
            helpm.add_command(label="About", command=self._about)
            bar.add_cascade(label="Help", menu=helpm)
            root.config(menu=bar)
        except Exception as e:
            print("menu setup skipped:", e)

    def _about(self):
        try:
            from tkinter import messagebox
            messagebox.showinfo("HL2 HF Sounder",
                "Pick a Signal type; on the TX end key PTT + En PA, on the RX end "
                "press Grab+Analyze.\nEach grab saves <type>_analysis.txt and "
                "<type>_data.npz.\nDuplex routes the internal TX tap to the LNA for "
                "self-test.\nView menu sets the live-spectrum RBW.")
        except Exception:
            pass

    # ── control callbacks ────────────────────────────────────────────────────
    def _select_mode(self, mode):
        if self._busy.is_set():           # don't switch mid-grab (corrupts analysis)
            self._set_status("grab in progress - signal change ignored")
            mv = getattr(self, "_mode_var", None)
            if mv is not None:
                mv.set(self.mode)         # keep the menu in sync with reality
            self.fig.canvas.draw_idle()
            return
        self.mode = mode
        iq, self.spec = make_waveform(mode, SAMPLE_RATE, self.tx_amp)
        self.radio.set_waveform(iq)
        self.pdp_line.set_data([], [])                       # drop stale profile
        self._setup_lower_axes(mode)                         # mode-appropriate panel
        self._set_status(f"mode = {mode}  TxAmp={self.tx_amp:.2f}")
        self.fig.canvas.draw_idle()

    def _setup_lower_axes(self, mode):
        """Lower panel: coherence curve (Comb), fade envelopes (Two-tone), or PDP."""
        self.pdp_line2.set_visible(False)
        # OFDM uses its own constellation+sequence sub-axes instead of self.ax_d
        ofdm = (mode == "OFDM")
        self.ax_d.set_visible(not ofdm)
        self.ax_const.set_visible(ofdm); self.ax_seq.set_visible(ofdm)
        if ofdm:
            for ax in (self.ax_const, self.ax_seq):
                ax.clear(); ax.grid(True, color="#333", lw=0.4)
            self.ax_const.set_title("Constellation"); self.ax_const.set_aspect("equal")
            self.ax_const.set_xlim(-1.8, 1.8); self.ax_const.set_ylim(-1.8, 1.8)
            self.ax_const.set_xlabel("I"); self.ax_const.set_ylabel("Q")
            self.ax_seq.set_title("sent vs received data")
            self.ax_seq.set_ylim(-0.5, 3.5); self.ax_seq.set_yticks([0, 1, 2, 3])
            self.ax_seq.set_xlabel("subcarrier"); self.ax_seq.set_ylabel("QPSK sym")
            return
        if mode == "CW":
            self.ax_d.set_title("CW Spectrum (100 Hz RBW)")
            self.ax_d.set_xlabel("baseband frequency (kHz)")
            self.ax_d.set_ylabel("dBFS")
            self.ax_d.set_xlim(-SAMPLE_RATE / 2 / 1e3, SAMPLE_RATE / 2 / 1e3)
            self.ax_d.set_ylim(-140, 0)
            self.pdp_line.set_color("#00ff88"); self.pdp_line.set_marker("")
            self._coh_half.set_visible(False)
        elif mode == "Comb":
            self.ax_d.set_title("Coherence Curve - Comb")
            self.ax_d.set_xlabel("frequency offset (Hz)")
            self.ax_d.set_ylabel("|correlation|")
            self.ax_d.set_xlim(0, 2000); self.ax_d.set_ylim(0, 1.05)
            self.pdp_line.set_color("#33ccff")
            self.pdp_line.set_marker("o"); self.pdp_line.set_markersize(3)
            self._coh_half.set_visible(True)
        elif mode == "LFM chirp":
            self.ax_d.set_title("LFM Pulse Compression")
            self.ax_d.set_xlabel("delay (ms)")
            self.ax_d.set_ylabel("matched filter (dB)")
            self.ax_d.set_xlim(-1.5, 1.5); self.ax_d.set_ylim(-50, 3)
            self.pdp_line.set_color("#00ff88"); self.pdp_line.set_marker("")
            self._coh_half.set_visible(False)
        elif mode == "OFDM":
            self.ax_d.set_title("OFDM Channel Response")
            self.ax_d.set_xlabel("subcarrier frequency (kHz)")
            self.ax_d.set_ylabel("|H| (dB)")
            self.ax_d.set_xlim(-6, 6); self.ax_d.set_ylim(-40, 3)
            self.pdp_line.set_color("#ffcc00"); self.pdp_line.set_marker("")
            self.pdp_line2.set_color("#ff5555"); self.pdp_line2.set_visible(True)
            self._coh_half.set_visible(False)
        else:
            self.ax_d.set_title(f"Detection Power - {mode}")
            self.ax_d.set_xlabel("delay (ms)")
            self.ax_d.set_ylabel("power (dB)")
            self.ax_d.set_ylim(-20, 0)
            self.pdp_line.set_color("#ffcc00"); self.pdp_line.set_marker("")
            self._coh_half.set_visible(False)
        self._coh_bw_marker.set_visible(False)

    def _on_click(self, event):
        """Click a value box -> edit dialog (boxes are display-only)."""
        from tkinter import simpledialog
        parent = getattr(self.fig.canvas.manager, "window", None)
        if event.inaxes is self.tb_freq.ax:
            v = simpledialog.askfloat("Frequency", "Carrier (MHz):", parent=parent,
                                      initialvalue=self.radio.nco_hz / 1e6,
                                      minvalue=0.1, maxvalue=35.0)
            if v is not None:
                self.radio.set_freq(v * 1e6)
                self.tb_freq.set_val(f"{v:.3f}")
                f = self.radio.nco_hz / 1e6
                self.ax_s.set_xlim(f - SAMPLE_RATE / 2e6, f + SAMPLE_RATE / 2e6)
        elif event.inaxes is self.tb_lna.ax:
            v = simpledialog.askinteger("LNA Gain", "dB (-12 to 48):", parent=parent,
                                        initialvalue=self.radio.lna_gain_db,
                                        minvalue=-12, maxvalue=48)
            if v is not None:
                self.radio.set_lna(v)                 # pushed to radio -> shifts display
                self.tb_lna.set_val(str(v))
                self._set_status(f"LNA = {v} dB (applied)")
        elif event.inaxes is self.tb_amp.ax:
            v = simpledialog.askfloat("TX Amplitude", "DAC level 0.0-1.0:",
                                      parent=parent, initialvalue=self.tx_amp,
                                      minvalue=0.0, maxvalue=1.0)
            if v is not None:
                self.tx_amp = v
                self.tb_amp.set_val(f"{v:.2f}")
                self._select_mode(self.mode)        # regenerate waveform at new amp
        elif event.inaxes is self.tb_pwr.ax:
            v = simpledialog.askinteger("TX Drive", "drive 0-15 (HL2 reg 0x09):",
                                        parent=parent, initialvalue=self.radio.tx_drive,
                                        minvalue=0, maxvalue=15)
            if v is not None:
                self.radio.set_tx_drive(v)
                self.tb_pwr.set_val(str(v))

    def _toggle_all(self, _e):
        """Master bar: key Tone + PTT + En PA + Duplex together (or drop them)."""
        self._all_on = not self._all_on
        on = self._all_on
        self._tone = self._ptt = self._pa = self._duplex = on
        col = "#990000" if on else "#2266cc"
        for b in (self.b_tone, self.b_ptt, self.b_pa, self.b_dup):
            b.color = col
        self.b_all.color = "#aa0000" if on else "#444444"
        self._set_status(f"ALL TX {'ON' if on else 'off'} (Tone+PTT+En PA+Duplex)")
        self.fig.canvas.draw_idle()                       # instant visual feedback
        self.radio.set_tone(on)
        self._async(self.radio.set_tx, ptt=on, pa=on, duplex=on)

    def _toggle_tone(self, _e):
        self._tone = not self._tone
        self.b_tone.color = "#990000" if self._tone else "#2266cc"
        self._set_status(f"Tone {'ON' if self._tone else 'off'}  ({self.mode})")
        self.fig.canvas.draw_idle()
        self.radio.set_tone(self._tone)                   # instant (just a flag)

    def _toggle_ptt(self, _e):
        self._ptt = not self._ptt
        self.b_ptt.color = "#990000" if self._ptt else "#2266cc"
        self._set_status(f"PTT {'ON' if self._ptt else 'off'}")
        self.fig.canvas.draw_idle()
        self._async(self.radio.set_tx, ptt=self._ptt)

    def _toggle_pa(self, _e):
        self._pa = not self._pa
        self.b_pa.color = "#990000" if self._pa else "#2266cc"
        self._set_status(f"En PA {'ON' if self._pa else 'off'}")
        self.fig.canvas.draw_idle()
        self._async(self.radio.set_tx, pa=self._pa)

    def _toggle_duplex(self, _e):
        self._duplex = not self._duplex
        self.b_dup.color = "#990000" if self._duplex else "#2266cc"
        self._set_status(f"Duplex {'ON (RX hears TX tap)' if self._duplex else 'off'}")
        self.fig.canvas.draw_idle()
        self._async(self.radio.set_tx, duplex=self._duplex)

    def _set_rbw(self, hz, label):
        self.rbw_hz = float(hz)
        self._rbw_text.set_text(f"RBW {label}")
        self.fig.canvas.draw_idle()

    def _toggle_auto(self, _e):
        """Toggle the spectrum y-scale between 0..-140 dBm and 0..-100 dBm."""
        self._wide_scale = not self._wide_scale
        floor = 140 if self._wide_scale else 100
        self.ax_s.set_ylim(DBM_OFFSET - floor, DBM_OFFSET)
        self._set_status(f"Spectrum scale 0..-{floor} dBm")
        self.fig.canvas.draw_idle()

    def _set_status(self, msg):
        self._status.set_text(msg)

    # ── live spectrum ────────────────────────────────────────────────────────
    def _animate(self, _f):
        # While a sounding grab runs, skip the live recompute/redraw so it can't
        # starve the TX streaming thread (GIL) and cause a TX FIFO underrun.
        if self._busy.is_set():
            return
        # Plot the latest block captured by the background thread (no blocking).
        with self._live_lock:
            rx = self._live_rx
        if rx is None or len(rx) < 64:
            return
        seg = max(64, int(round(1.44 * SAMPLE_RATE / self.rbw_hz)))
        if seg > len(rx):
            seg = 1 << int(np.floor(np.log2(len(rx))))
        w = np.hanning(seg)
        wp2 = np.sum(w) ** 2
        step = max(1, seg // 2)
        acc = np.zeros(seg)
        cnt = 0
        for s0 in range(0, len(rx) - seg + 1, step):
            acc += np.abs(np.fft.fft(rx[s0:s0 + seg] * w)) ** 2
            cnt += 1
        if cnt == 0:
            return
        P = np.fft.fftshift(acc / cnt)
        S = 10.0 * np.log10(P / wp2 + 1e-24) + DBM_OFFSET   # dBFS+offset -> dBm
        f = (self.radio.nco_hz +
             np.fft.fftshift(np.fft.fftfreq(seg, 1.0 / SAMPLE_RATE))) / 1e6
        self.line.set_data(f, S)
        self._last_S = S          # y-scale is fixed; Auto toggles -100/-145 floor

    # ── grab + analyze + save ────────────────────────────────────────────────
    def _on_grab(self, _e):
        if self._busy.is_set():
            return
        threading.Thread(target=self._grab_worker, daemon=True).start()

    def _grab_worker(self):
        self._busy.set()
        self._grab_dot.set_visible(True)      # red "in progress" dot
        time.sleep(0.6)   # let the live-capture thread release the RX socket
        self._set_status(f"Grabbing {GRAB_SEC:.0f}s of {self.mode} ...")
        self.fig.canvas.draw_idle()
        try:
            rx = self.radio.grab(GRAB_SEC)
            if len(rx) < SAMPLE_RATE:
                self._set_status("Grab failed - too few samples")
                return
            m = analyse(self.mode, rx, self.spec)
            self._save(rx, m)
            self._plot_pdp(m)
            self._print_and_status(m)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_status(f"analyze error: {e}")
        finally:
            self._busy.clear()
            self._grab_dot.set_visible(False)
            self.fig.canvas.draw_idle()

    def _plot_ofdm_fidelity(self, m):
        """OFDM lower panel: received constellation + sent-vs-received sequence."""
        ac, aq = self.ax_const, self.ax_seq
        ac.clear(); aq.clear()
        ac.grid(True, color="#333", lw=0.4); aq.grid(True, color="#333", lw=0.4)
        g = 1.0 / np.sqrt(2.0)
        ac.scatter([g, -g, -g, g], [g, g, -g, -g], s=130, marker="+",
                   c="#00aa55", linewidths=1.2, zorder=1)         # ideal (behind)
        ac.scatter(m["ofdm_const_i"], m["ofdm_const_q"], s=6, c="#66bbff",
                   alpha=0.4, zorder=2)                           # received cloud
        ac.scatter(m["ofdm_cmean_i"], m["ofdm_cmean_q"], s=16, c="#ff6633",
                   edgecolors="white", linewidths=0.3, zorder=3)  # averaged (on top)
        ac.set_aspect("equal"); ac.set_xlim(-1.8, 1.8); ac.set_ylim(-1.8, 1.8)
        ac.set_xlabel("I"); ac.set_ylabel("Q")
        ac.set_title(f"Constellation  EVM {m['ofdm_evm1']*100:.0f}%"
                     f"->{m['ofdm_evm_avg']*100:.1f}% (avg {m['ofdm_nsym']})",
                     fontsize=9)
        sent = np.asarray(m["ofdm_seq_sent"]); recv = np.asarray(m["ofdm_seq_recv"])
        x = np.arange(len(sent))
        aq.step(x, sent, where="mid", color="#00ff88", lw=1.4)
        aq.plot(x, recv, "o", color="#ff6633", ms=3)
        bad = np.where(sent != recv)[0]
        if len(bad):
            aq.scatter(bad, recv[bad], s=80, facecolors="none",
                       edgecolors="red", linewidths=1.3)
        aq.set_ylim(-0.5, 3.5); aq.set_yticks([0, 1, 2, 3])
        aq.set_xlabel("subcarrier"); aq.set_ylabel("QPSK sym")
        match = "PERFECT match" if not len(bad) else f"{len(bad)} mismatch"
        aq.set_title(f"sent(green) vs received(orange)   SNR {m['ofdm_snr1']:.0f}->"
                     f"{m['ofdm_snr_avg']:.0f}dB,  SER {m['ofdm_ser']*100:.1f}%,  "
                     f"{match}", fontsize=9)

    def _plot_pdp(self, m):
        if self.mode == "CW" and m.get("cw_spec_db") is not None:
            f = np.asarray(m["cw_spec_f"]) / 1e3                  # kHz
            s = np.asarray(m["cw_spec_db"])
            self.pdp_line.set_data(f, s)
            self.ax_d.set_xlim(float(f[0]), float(f[-1]))
            pk = float(np.max(s))
            self.ax_d.set_ylim(pk - 110, pk + 5)
            jumps = m.get("cw_phase_jumps", 0); drops = m.get("cw_dropouts", 0)
            flag = "CLEAN" if (jumps == 0 and drops == 0) else \
                   f"ERRORS: {jumps} phase jumps, {drops} drops"
            self.ax_d.set_title(
                f"CW  f={m['cw_freq']:.1f} Hz (off {m['cw_freq_offset']:+.1f})  "
                f"SNR {m['snr_db']:.0f} dB  Pwr {m['cw_power_dbfs']:.0f} dBFS  [{flag}]")
        elif self.mode == "Comb" and m.get("coh_curve") is not None:
            f = np.asarray(m["coh_curve_freq"]); c = np.asarray(m["coh_curve"])
            self.pdp_line.set_data(f, c)
            cb = m.get("coh_bw_direct")
            if cb is not None and np.isfinite(cb):
                self._coh_bw_marker.set_xdata([cb, cb])
                self._coh_bw_marker.set_visible(True)
                self.ax_d.set_xlim(0, min(float(f[-1]), max(5.0 * cb, 500.0)))
                self.ax_d.set_title(
                    f"Coherence Curve - Comb  (coherence BW = {cb:.0f} Hz @ 0.5)")
            else:
                self._coh_bw_marker.set_visible(False)
                self.ax_d.set_xlim(0, float(f[-1]))
                self.ax_d.set_title("Coherence Curve - Comb  (BW > sounding span)")
            self.ax_d.set_ylim(0, 1.05)
            self._coh_half.set_visible(True)
        elif self.mode == "LFM chirp" and m.get("mf_db") is not None:
            d = np.asarray(m["mf_delay"]) * 1e3                   # ms
            self.pdp_line.set_data(d, np.asarray(m["mf_db"]))
            self.ax_d.set_xlim(float(d[0]), float(d[-1]))
            self.ax_d.set_ylim(-50, 3)
            psnr = m.get("mf_peaksnr_db", 0.0)
            if psnr < 12.0:        # no compression peak -> receiving noise only
                self.ax_d.set_title(
                    f"LFM - NO CHIRP DETECTED (noise, peak SNR {psnr:.0f} dB) - "
                    f"key Tone+PTT+En PA+Duplex, or check the far end is sending")
            else:
                self.ax_d.set_title(
                    f"LFM Pulse Compression  (PG {m['mf_pg_db']:.0f} dB, "
                    f"res {m['mf_res_us']:.0f} us, peak SNR {psnr:.0f} dB, "
                    f"sidelobe {m['mf_pslr_db']:.0f} dB)")
        elif self.mode == "OFDM" and m.get("ofdm_const_i") is not None:
            self._plot_ofdm_fidelity(m)
        elif "delay_axis" in m and m.get("pdp_db") is not None:
            self.pdp_line.set_data(np.asarray(m["delay_axis"]) * 1e3, m["pdp_db"])
            self.ax_d.set_xlim(m["delay_axis"][0] * 1e3, m["delay_axis"][-1] * 1e3)
            self.ax_d.set_ylim(-20, 0)
            self.ax_d.set_title(f"Detection Power - {self.mode}")
        self.fig.canvas.draw_idle()

    def _header(self):
        loss = self.radio.last_loss
        quality = "OK" if loss["pct"] < 1.0 else "DEGRADED - high packet loss"
        return (f"HF SOUNDER  end={END_ID}  "
                f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"RF={self.radio.nco_hz/1e6:.4f} MHz  fs={SAMPLE_RATE} sps  "
                f"LNA={self.radio.lna_gain_db} dB  "
                f"grab={GRAB_SEC:.0f}s  duplex={'yes' if self._duplex else 'no'}\n"
                f"packet loss={loss['lost']}/{loss['recv']+loss['lost']} "
                f"({loss['pct']:.2f}%)  reorder={loss.get('reorder',0)} "
                f"dup={loss.get('dup',0)}  spliced={loss.get('spliced',0)}"
                f"  data quality={quality}")

    def _save(self, rx, m):
        base = os.path.join(SAVE_DIR, safe_name(self.mode))
        with open(base + "_analysis.txt", "w") as fh:
            fh.write(ch.format_report(m, self._header()) + "\n")
        np.savez(base + "_data.npz", iq=rx.astype(np.complex64),
                 fs=SAMPLE_RATE, nco=self.radio.nco_hz, mode=self.mode)
        print("saved:", base + "_analysis.txt", "and", base + "_data.npz")

    def _print_and_status(self, m):
        print("\n" + ch.format_report(m, self._header()) + "\n")
        if m.get("kind") == "CW":
            err = m['cw_phase_jumps'] + m['cw_dropouts']
            self._set_status(
                f"CW: f={m['cw_freq']:.1f}Hz off {m['cw_freq_offset']:+.1f}  "
                f"SNR {m['snr_db']:.0f}dB  "
                f"{'CLEAN' if err==0 else str(err)+' DATA ERRORS'}  (saved)")
        elif m.get("ofdm_evm1") is not None:
            self._set_status(
                f"OFDM: EVM {m['ofdm_evm1']*100:.0f}%->{m['ofdm_evm_avg']*100:.1f}%  "
                f"eff.SNR {m['ofdm_snr_avg']:.0f}dB  SER {m['ofdm_ser']*100:.1f}%  (saved)")
        elif "rms_delay" in m:
            self._set_status(
                f"{self.mode}: delay {m['rms_delay']*1e3:.2f} ms  "
                f"cohBW {m['coh_bw']:.0f} Hz  Doppler {m.get('dop_spread',0):.2f} Hz  "
                f"(saved)")
        else:
            self._set_status(f"{self.mode}: rho {m.get('fade_corr',0):.2f}  (saved)")
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


def main():
    radio = HL2(HL2_IP, NCO_FREQ, LNA_GAIN_DB)
    print(f"Connecting to HL2 at {HL2_IP} ...")
    radio.open()
    print("Connected. Pick a Signal from the menu, then TX (one end) / "
          "Grab+Analyze (other end).")
    gui = Sounder(radio)
    try:
        gui.run()
    finally:
        print("Shutting down ...")
        radio.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        print("\n" + "=" * 64 + "\nSOUNDER CRASHED:\n" + "=" * 64)
        print(tb)
        try:
            with open(os.path.join(SAVE_DIR, "sounder_error.log"), "w") as fh:
                fh.write(tb)
        except Exception:
            pass
        try:
            input("\nPress Enter to close ...")
        except Exception:
            pass
