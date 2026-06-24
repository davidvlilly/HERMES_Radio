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
from matplotlib.patches import Rectangle

import waveforms as wf
import channel as ch
from hl2 import HL2, SAMPLE_RATE

# ── Configuration ────────────────────────────────────────────────────────────
HL2_IP      = "169.254.19.221"
NCO_FREQ    = 7_130_000          # Hz (40 m; OTA test frequency)
LNA_GAIN_DB = 0
TX_AMP      = 0.7                # DAC/FPGA digital level; capped at 0.7 for OFDM PAPR headroom
DBM_OFFSET  = 0.0                # dBm = dBFS + this; trim vs a known input for true dBm
GRAB_SEC    = 5.0                # sounding capture length (s)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = os.path.join(SCRIPT_DIR, "capture")     # captured IQ + analysis land here
os.makedirs(SAVE_DIR, exist_ok=True)
END_ID      = "A"               # which end this radio is (set per station)
CW_FREQ     = 1000.0            # CW diagnostic tone offset from NCO (Hz)

# TX modes (Tx Modes menu).  Broadcast = full level for over-the-air; the two
# loopback modes self-test through the internal TX->RX tap (Duplex) at a low
# level, optionally with noise added on the RX grab to emulate a weak link.
TX_AMP_BROADCAST = TX_AMP               # full digital level (never higher than 0.8)
LOOPBACK_POWER   = 0.03                 # loopback level = 3% of broadcast
LOOPBACK_SNR_DB  = 0.0                  # noise added in "Loopback w/ noise" (~0 dB SNR)
TX_MODES = ["Broadcast", "Loopback w/o noise", "Loopback w/ noise"]

SIGNAL_MODES = ["CW", "LFM chirp", "Comb", "OFDM", "DSSS/PN"]


def make_waveform(mode, fs, amp=TX_AMP):
    d = wf.hf_defaults(fs)
    if   mode == "CW":        iq, spec = wf.cw_tone(fs, freq=CW_FREQ, amp=amp)
    elif mode == "LFM chirp": iq, spec = wf.lfm_chirp(fs, amp=amp, **d["chirp"])
    elif mode == "Comb":      iq, spec = wf.multitone_comb(fs, amp=amp, **d["comb"])
    elif mode == "Two-tone":  iq, spec = wf.two_tone(fs, amp=amp, **d["two_tone"])
    elif mode == "OFDM":      iq, spec = wf.ofdm_sounding(fs, amp=amp, **d["ofdm"])
    elif mode == "DSSS/PN":   iq, spec = wf.pn_dsss(fs, amp=amp, **d["dsss"])
    else: raise ValueError(mode)
    spec["dbm_offset"] = DBM_OFFSET               # dBm = dBFS + offset (all modes)
    return iq, spec


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
        self.tx_mode = "Loopback w/ noise"  # Tx Modes default (Broadcast / Loopback w/o|w/ noise)
        self._lb_noise = True             # add ~0 dB-SNR noise to RX grab (loopback w/ noise)
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

        # ON-AIR warning: dark-red bar across the top of the spectrum (~16 px),
        # shown ONLY when actually broadcasting -- Broadcast Tx mode AND Tone+PTT+
        # En PA all keyed -- so an operator can't miss that they are on the air.
        self._tx_bar = Rectangle((0.0, 0.95), 1.0, 0.05, transform=self.ax_s.transAxes,
                                 facecolor="#7a0000", edgecolor="none", zorder=20,
                                 visible=False)
        self.ax_s.add_patch(self._tx_bar)
        self._tx_bar_text = self.ax_s.text(0.5, 0.975, "ON AIR — BROADCASTING",
                                           transform=self.ax_s.transAxes, ha="center",
                                           va="center", color="white", fontsize=10,
                                           fontweight="bold", zorder=21, visible=False)

        # ── Lower row: the SAME two panels for every mode ──────────────────────
        #   left  = Constellation (populated only for OFDM (QPSK) and DSSS (BPSK))
        #   right = Detection power-delay profile, fixed -25..0 dB over -1..20 ms
        self.ax_d.set_visible(False)                     # replaced by the two panels below
        posd = self.ax_d.get_position()
        sq = posd.height * 7.0 / 11.0                    # square box (fig is 11x7)
        self.ax_const = self.fig.add_axes([posd.x0, posd.y0, sq, posd.height])
        self.ax_det   = self.fig.add_axes([posd.x0 + sq + 0.10, posd.y0 - 0.0414,
                                           posd.x1 - (posd.x0 + sq + 0.10),
                                           posd.height + 0.0714])  # taller; bottom lowered ~15 px
        self.ax_det.set_xlabel("delay (ms)"); self.ax_det.set_ylabel("power (dB)")
        self.ax_det.set_xlim(-1.0, 20.0); self.ax_det.set_ylim(-35.0, 0.0)
        self.ax_det.grid(True, color="#333", lw=0.5)
        # "Detection" label inside the plot, upper-right (no title above the axes).
        self.ax_det.text(0.98, 0.94, "Detection", transform=self.ax_det.transAxes,
                         ha="right", va="top", color="#cccccc", fontsize=10,
                         fontweight="bold")
        self.pdp_line, = self.ax_det.plot([], [], color="#ffcc00", lw=1.0)
        self._draw_constellation(None)                   # blank constellation panel

        # Persistent Signal-Mode label: the ONLY thing ever shown above the top display.
        self._mode_text = self.ax_s.text(0.5, 1.04, "", transform=self.ax_s.transAxes,
                                         ha="center", va="bottom", color="#00ddff",
                                         fontsize=14, fontweight="bold",
                                         family="monospace")

        self._build_controls()
        self._build_menus()
        self._select_mode(self.mode)
        self._set_tx_mode(self.tx_mode)   # apply the default Tx mode (loopback level + Duplex)

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
            if self._lb_noise:                    # loopback w/ noise: show it live too
                rx = self._add_noise(rx, LOOPBACK_SNR_DB)
            with self._live_lock:
                self._live_rx = rx

    # ── bottom controls (matplotlib widgets) ─────────────────────────────────
    def _build_controls(self):
        Y, H = 0.021, 0.05          # row dropped to ~5 px off the bottom
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
            tb.label.set_position((0.5, 1.15))         # label just above the box (snug)
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

            self._txmode_var = tk.StringVar(value=self.tx_mode)
            txm = tk.Menu(bar, tearoff=0)
            for label in TX_MODES:
                txm.add_radiobutton(label=label, variable=self._txmode_var,
                                    command=lambda l=label: self._set_tx_mode(l))
            bar.add_cascade(label="Tx Modes", menu=txm)

            self._rbw_var = tk.StringVar(value="100 Hz")
            viewm = tk.Menu(bar, tearoff=0)
            for label, hz in (("10 Hz", 10), ("100 Hz", 100), ("1 kHz", 1000)):
                viewm.add_radiobutton(label=f"RBW {label}", variable=self._rbw_var,
                                      value=label,
                                      command=lambda h=hz, l=label: self._set_rbw(h, l))
            bar.add_cascade(label="RBW", menu=viewm)

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
                "Sounder Test Tool\n\n"
                "Sends & Receives Modulated Soundings pulses;\n"
                "CW, LFM, Comb, OFDM, DSSS(PN)\n\n"
                f"Interface to the Hermes Lite 2  (IP Address: {HL2_IP})\n\n"
                "Chip Watson & David Lilly    Version 1.0")
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
        self._setup_lower_axes(mode)                         # clears panel + sets the mode title
        self._mode_text.set_text(f"MODE: {mode}")            # persistent mode label
        self._set_status("")
        self.fig.canvas.draw_idle()

    def _set_tx_mode(self, mode):
        """Tx Modes menu: Broadcast (full level, over-the-air) or one of the two
        Loopback self-tests (internal TX->RX tap via Duplex, at 3% level; the
        'w/ noise' variant adds ~0 dB-SNR noise to the RX grab to emulate a weak
        link so the matched-filter processing gain can be exercised on the bench)."""
        if self._busy.is_set():               # don't disturb a running grab
            self._set_status("grab in progress - TX mode change ignored")
            self._txmode_var.set(self.tx_mode)
            return
        self.tx_mode  = mode
        loopback      = mode.startswith("Loopback")
        self._lb_noise = (mode == "Loopback w/ noise")
        # power: broadcast = full digital level; loopback = 3% of it
        self.tx_amp = TX_AMP_BROADCAST * (LOOPBACK_POWER if loopback else 1.0)
        iq, self.spec = make_waveform(self.mode, SAMPLE_RATE, self.tx_amp)
        self.radio.set_waveform(iq)
        self.tb_amp.set_val(f"{self.tx_amp:.3f}")
        # Loopback REQUIRES the internal TX->RX tap, so force Duplex on.  Broadcast
        # leaves Duplex under your control (keep it on to watch your own TX signal).
        if loopback:
            self._duplex = True
            self.b_dup.color = "#990000"
            self._async(self.radio.set_tx, duplex=True)
        self._update_tx_bar()           # ON-AIR bar depends on the Tx mode
        self.fig.canvas.draw_idle()

    def _add_noise(self, rx, snr_db):
        """Add complex AWGN to a capture to hit a target SNR (loopback w/ noise)."""
        rx = np.asarray(rx, dtype=np.complex64)
        sp = float(np.mean(np.abs(rx) ** 2))
        if sp <= 0.0 or len(rx) == 0:
            return rx
        npow = sp / (10.0 ** (snr_db / 10.0))
        n = (np.random.standard_normal(len(rx)) +
             1j * np.random.standard_normal(len(rx))) * np.sqrt(npow / 2.0)
        return (rx + n).astype(np.complex64)

    def _draw_constellation(self, m):
        """Left panel. Empty for CW/LFM/Comb; QPSK for OFDM, BPSK (vertical) for DSSS."""
        ac = self.ax_const
        ac.clear()
        ac.grid(True, color="#333", lw=0.4)
        ac.set_aspect("equal"); ac.set_xlim(-1.8, 1.8); ac.set_ylim(-1.8, 1.8)
        ac.set_xlabel("I"); ac.set_ylabel("Q")
        if m is not None and self.mode == "OFDM" and m.get("ofdm_cmean_i") is not None:
            g = 1.0 / np.sqrt(2.0)
            ac.scatter([g, -g, -g, g], [g, g, -g, -g], s=130, marker="+",
                       c="#00aa55", linewidths=1.2, zorder=1)            # ideal QPSK
            ac.scatter(m["ofdm_cmean_i"], m["ofdm_cmean_q"], s=14,
                       c="#66bbff", alpha=0.85, zorder=2)                # symbol-averaged
            ac.set_title(f"Constellation - OFDM (avg {m.get('ofdm_nsym','')})")
        elif m is not None and self.mode == "DSSS/PN" and m.get("dsss_const_i") is not None:
            ac.scatter([0, 0], [1, -1], s=130, marker="+",
                       c="#00aa55", linewidths=1.2, zorder=1)            # ideal BPSK +/-90
            ac.scatter(m["dsss_const_i"], m["dsss_const_q"], s=8,
                       c="#66bbff", alpha=0.5, zorder=2)                 # despread peaks
            ac.set_title("Constellation - DSSS (BPSK)")
        else:
            ac.set_title("Constellation")

    def _setup_lower_axes(self, mode):
        """Same two panels for every mode: blank them and reset the fixed detection axes."""
        self.pdp_line.set_data([], [])
        self.ax_det.set_xlim(-1.0, 20.0); self.ax_det.set_ylim(-35.0, 0.0)
        self._draw_constellation(None)

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
            v = simpledialog.askfloat("TX Amplitude", f"DAC level 0.0-{TX_AMP:.1f}:",
                                      parent=parent, initialvalue=self.tx_amp,
                                      minvalue=0.0, maxvalue=TX_AMP)
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
        self._update_tx_bar()
        self.fig.canvas.draw_idle()                       # instant visual feedback
        self.radio.set_tone(on)
        self._async(self.radio.set_tx, ptt=on, pa=on, duplex=on)

    def _update_tx_bar(self):
        """Show the dark-red ON-AIR bar only when truly broadcasting over the air:
        Broadcast Tx mode AND Tone + PTT + En PA all keyed."""
        on = (self.tx_mode == "Broadcast" and self._tone and self._ptt and self._pa)
        self._tx_bar.set_visible(on)
        self._tx_bar_text.set_visible(on)
        self.fig.canvas.draw_idle()

    def _toggle_tone(self, _e):
        self._tone = not self._tone
        self.b_tone.color = "#990000" if self._tone else "#2266cc"
        self._update_tx_bar()
        self.fig.canvas.draw_idle()
        self.radio.set_tone(self._tone)                   # instant (just a flag)

    def _toggle_ptt(self, _e):
        self._ptt = not self._ptt
        self.b_ptt.color = "#990000" if self._ptt else "#2266cc"
        self._update_tx_bar()
        self.fig.canvas.draw_idle()
        self._async(self.radio.set_tx, ptt=self._ptt)

    def _toggle_pa(self, _e):
        self._pa = not self._pa
        self.b_pa.color = "#990000" if self._pa else "#2266cc"
        self._update_tx_bar()
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
        # On-screen status line removed (it collided with the detection panel);
        # status/progress still goes to the terminal via _print_and_status.
        pass

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
            if self._lb_noise:                         # loopback w/ noise: emulate a weak link
                rx = self._add_noise(rx, LOOPBACK_SNR_DB)
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

    def _plot_pdp(self, m):
        """Uniform lower display: detection power-delay (right) + constellation (left)."""
        # Detection (right): normalised power-delay profile on the fixed -25..0 dB,
        # -1..20 ms axes.  Present for every mode that yields an impulse response
        # (LFM / DSSS / Comb / OFDM); CW has none, so it stays blank.
        if m.get("pdp_db") is not None and m.get("delay_axis") is not None:
            self.pdp_line.set_data(np.asarray(m["delay_axis"]) * 1e3,
                                   np.asarray(m["pdp_db"]))
        else:
            self.pdp_line.set_data([], [])
        self.ax_det.set_xlim(-1.0, 20.0); self.ax_det.set_ylim(-35.0, 0.0)
        # Constellation (left): OFDM (QPSK) / DSSS (BPSK), blank otherwise.
        self._draw_constellation(m)
        self.fig.canvas.draw_idle()

    def _header(self):
        loss = self.radio.last_loss
        q = "OK" if loss["pct"] < 1.0 else "DEGRADED"
        return (f"HF SOUNDER  end={END_ID}  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"RF={self.radio.nco_hz/1e6:.4f} MHz  fs={SAMPLE_RATE//1000}k  "
                f"LNA={self.radio.lna_gain_db} dB  grab={GRAB_SEC:.0f}s  "
                f"loss={loss['pct']:.2f}%  [{q}]")

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
            self._set_status(f"CW: SNR {m['snr_db']:.0f} dB  (saved)")
        elif m.get("ofdm_evm1") is not None:
            self._set_status(f"OFDM: EVM {m['ofdm_evm_avg']*100:.1f}% (avg of {m['ofdm_nsym']})  (saved)")
        elif "rms_delay" in m:
            snr = m.get("snr_db")
            head = f"SNR {snr:.0f} dB  " if snr is not None else ""
            self._set_status(f"{self.mode}: {head}(saved)")
        else:
            self._set_status(f"{self.mode}: (saved)")
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
            with open(os.path.join(SCRIPT_DIR, "sounder_error.log"), "w") as fh:
                fh.write(tb)
        except Exception:
            pass
        try:
            input("\nPress Enter to close ...")
        except Exception:
            pass
