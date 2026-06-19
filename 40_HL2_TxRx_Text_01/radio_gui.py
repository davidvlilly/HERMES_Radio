#!/usr/bin/env python3
"""
radio_gui.py — Tkinter GUI for the HL2 text radio.

Layout (DESIGN.md §6):
    menubar: File | Config            (stubs for now)
    top:     spectrum (embedded matplotlib, ±48 kHz)
    mid:     [Call] [Detect]  MODE(red)   [TX](red box)   LNA  TX-Pwr  PEP  offset
    bottom:  scrolling message log  +  compose entry  +  [Send]

Display markers in the log:
    '.'  incoming beacon      '/'  outgoing beacon      '#'  missed burst
    text messages get their own line ( "<" incoming,  ">" outgoing ).

The GUI drains `engine.events` on a timer and never blocks; all radio work
happens in the engine threads.
"""

import queue
import time
import tkinter as tk
from collections import deque
from tkinter import scrolledtext

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import radio_engine as eng

CORR_SPAN_S = 60.0       # visible time span of the Sync-Corr strip chart (s)
CORR_MAXPTS = 2000       # max retained strip-chart points
TITLE_Y     = 0.92       # title position inside the plot (1.0 = top, 0 = bottom)


class RadioGUI:
    RED = "#cc1111"
    GREY = "#444444"
    SPEC_COLOR     = "#00ff88"   # spectrum trace: green
    CORR_MAX_COLOR = "#4aa3ff"   # corr max: blue
    CORR_AVG_COLOR = "#cc6600"   # corr avg: darkish orange
    FROZEN_COLOR   = "#cccccc"   # plot held during TX: light gray

    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        root.title("HL2 Text Radio")
        root.geometry("1000x760")

        self._disp_mode = "corr"
        self._corr_t_hist   = deque(maxlen=CORR_MAXPTS)
        self._corr_max_hist = deque(maxlen=CORR_MAXPTS)
        self._corr_avg_hist = deque(maxlen=CORR_MAXPTS)
        self._build_menu()
        self._build_spectrum()
        self._build_controls()
        self._build_log()
        self._set_display(self._disp_var.get())   # apply default plot mode

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll_events)

    def _set_display(self, mode):
        """Switch the upper plot between the Sync-Corr (processing-gain) view
        and the raw Spectrum view."""
        self._disp_mode = mode
        self.engine.set_display_mode(mode)
        spectrum = (mode == "spectrum")
        self.spec_line.set_visible(spectrum)
        self.corr_max_line.set_visible(not spectrum)
        self.corr_avg_line.set_visible(not spectrum)
        self.corr_thresh_line.set_visible(not spectrum)
        if spectrum:
            self.spec_line.set_data([], [])
            f_lo = (self.engine._nco_hz - eng.SAMPLE_RATE / 2) / 1e6
            f_hi = (self.engine._nco_hz + eng.SAMPLE_RATE / 2) / 1e6
            self.ax.set_xlim(f_lo, f_hi)
            self.ax.set_ylim(-130, 10)
            self.ax.set_xlabel("Frequency (MHz)", color="#cccccc")
            self.ax.set_ylabel("Power (dBm)", color="#cccccc")
            self.ax.set_title(f"Spectrum   (RBW {eng.SPEC_RBW_HZ/1000:.1f} kHz)",
                              color="#cccccc", y=TITLE_Y)
        else:
            self._corr_t_hist.clear()
            self._corr_max_hist.clear()
            self._corr_avg_hist.clear()
            self.corr_max_line.set_data([], [])
            self.corr_avg_line.set_data([], [])
            self.ax.set_xlim(0, CORR_SPAN_S)
            self.ax.set_ylim(-5, 30)
            self.ax.set_xlabel("receive time (s)", color="#cccccc")
            self.ax.set_ylabel("corr (dB over noise)", color="#cccccc")
            self.ax.set_title(f"Sync Corr   Thresh {eng.ACQ_THRESH_DB:.1f} dB",
                              color="#cccccc", y=TITLE_Y)
        self.canvas.draw_idle()

    # ── menus ────────────────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)

        dispmenu = tk.Menu(menubar, tearoff=0)
        self._disp_var = tk.StringVar(value="corr")
        dispmenu.add_radiobutton(label="Sync Corr", variable=self._disp_var,
                                 value="corr",
                                 command=lambda: self._set_display("corr"))
        dispmenu.add_radiobutton(label="Spectrum", variable=self._disp_var,
                                 value="spectrum",
                                 command=lambda: self._set_display("spectrum"))
        menubar.add_cascade(label="Display", menu=dispmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Usage Modes", command=self._help_usage)
        helpmenu.add_command(label="Symbols", command=self._help_symbols)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.root.config(menu=menubar)

    # ── help ───────────────────────────────────────────────────────────────────

    def _left_text(self, block):
        """Dump a block of text into the left pane (used by Help)."""
        w = self.log_left
        w.configure(state="normal")
        if w.index("end-1c") != "1.0":
            w.insert(tk.END, "\n")
        w.insert(tk.END, block.rstrip() + "\n")
        w.see(tk.END)
        w.configure(state="disabled")

    def _help_usage(self):
        self._left_text(
            "Command-line options:\n"
            "  python main.py            connect to the HL2 and open the GUI\n"
            "  --ip <addr>               HL2 IP address (default 169.254.19.221)\n"
            "  --freq <MHz>              carrier frequency (default 7.000)\n"
            "  --selftest                offline DSP self-test (no radio)\n"
            "  --sim                     two-station simulation (two windows)\n"
            "  --simtest                 headless two-station integration test\n"
            "  --sim-snr <dB>            simulated channel SNR (with --sim/--simtest)")

    def _help_symbols(self):
        self._left_text(
            "Symbols  (right pane = per-burst activity):\n"
            "  ^  (white)        Tx empty   - beacon sent (no text this cycle)\n"
            "  ^  (red)          Tx data    - message fragment sent\n"
            "  *  (white)        Rx empty   - beacon received\n"
            "  [chars] (cyan)    Rx data    - received message fragment (letters)\n"
            "  #                 missed expected burst\n"
            "Left pane = full messages:\n"
            "  > text (blue)     full message you sent\n"
            "  < text (white)    full message received\n"
            "  <end connection>  link dropped")

    # ── spectrum pane ──────────────────────────────────────────────────────────

    def _build_spectrum(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 3.6), facecolor="#101010")
        self.ax = self.fig.add_subplot(111, facecolor="#101010")
        f_lo = (self.engine._nco_hz - eng.SAMPLE_RATE / 2) / 1e6
        f_hi = (self.engine._nco_hz + eng.SAMPLE_RATE / 2) / 1e6
        self.ax.set_xlim(f_lo, f_hi)
        self.ax.set_ylim(-130, 10)
        self.ax.set_xlabel("Frequency (MHz)", color="#cccccc")
        self.ax.set_ylabel("Power (dBm)", color="#cccccc")
        self.ax.tick_params(colors="#cccccc")
        self.ax.grid(True, color="#333333", linewidth=0.5)
        (self.spec_line,) = self.ax.plot([], [], color=self.SPEC_COLOR, linewidth=0.8)
        # Sync-Corr strip-chart traces + threshold line (shown in corr mode)
        (self.corr_max_line,) = self.ax.plot([], [], color=self.CORR_MAX_COLOR,
                                             linestyle="none", marker=".",
                                             markersize=3, label="max")
        (self.corr_avg_line,) = self.ax.plot([], [], color=self.CORR_AVG_COLOR,
                                             linestyle="none", marker=".",
                                             markersize=2, label="avg")
        self.corr_thresh_line = self.ax.axhline(eng.ACQ_THRESH_DB, color="#dddd00",
                                                linestyle=":", linewidth=0.9)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── control row ──────────────────────────────────────────────────────────

    def _build_controls(self):
        bar = tk.Frame(self.root, pady=4)
        bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_call = tk.Button(bar, text="Call", width=7,
                                  command=self._on_call)
        self.btn_call.pack(side=tk.LEFT, padx=3)
        self.btn_detect = tk.Button(bar, text="Detect", width=7,
                                    command=self._on_detect)
        self.btn_detect.pack(side=tk.LEFT, padx=3)
        self.btn_end = tk.Button(bar, text="End", width=5, command=self._on_end)
        self.btn_end.pack(side=tk.LEFT, padx=3)

        self.mode_lbl = tk.Label(bar, text="", fg=self.RED, width=8,
                                 font=("TkDefaultFont", 11, "bold"))
        self.mode_lbl.pack(side=tk.LEFT, padx=6)

        self.tx_lbl = tk.Label(bar, text=" TX ", bg=self.GREY, fg="white",
                               font=("TkDefaultFont", 10, "bold"))
        self.tx_lbl.pack(side=tk.LEFT, padx=6)

        tk.Label(bar, text="LNA dB:").pack(side=tk.LEFT, padx=(12, 0))
        self.lna_var = tk.StringVar(value=str(self.engine._lna_db))
        lna_entry = tk.Entry(bar, textvariable=self.lna_var, width=4)
        lna_entry.pack(side=tk.LEFT)
        lna_entry.bind("<Return>", self._on_lna)

        tk.Label(bar, text="TX Pwr:").pack(side=tk.LEFT, padx=(12, 0))
        self.txp = tk.Scale(bar, from_=0, to=15, orient=tk.HORIZONTAL, length=120,
                            command=self._on_txp)
        self.txp.set(self.engine._tx_drive)
        self.txp.pack(side=tk.LEFT)

        tk.Label(bar, text="Freq MHz:").pack(side=tk.LEFT, padx=(12, 0))
        self.freq_var = tk.StringVar(value=f"{self.engine._nco_hz/1e6:.4f}")
        freq_entry = tk.Entry(bar, textvariable=self.freq_var, width=8)
        freq_entry.pack(side=tk.LEFT)
        freq_entry.bind("<Return>", self._on_freq)

        self.pep_lbl = tk.Label(bar, text="PEP -.- w", fg="black",
                                font=("TkFixedFont", 13))
        self.pep_lbl.pack(side=tk.RIGHT, padx=8)
        self._pep_last = 0.0          # rate-limit the PEP readout to ~10 Hz
        # (carrier offset is shown in the lower-left status line, not here)

    # ── message log + compose ─────────────────────────────────────────────────

    def _build_log(self):
        # Two panes: LEFT (65%) = full messages (sent + received);
        #            RIGHT (35%) = beacons / misses / outgoing fragments.
        logframe = tk.Frame(self.root)
        logframe.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=4, pady=2)
        logframe.columnconfigure(0, weight=65)
        logframe.columnconfigure(1, weight=35)
        logframe.rowconfigure(0, weight=1)

        def _mklog(col, padx):
            w = scrolledtext.ScrolledText(logframe, width=1, height=12,
                                          wrap=tk.WORD,   # roll over when full
                                          bg="#0a0a0a", fg="#dddddd",
                                          insertbackground="#dddddd",
                                          font=("TkFixedFont", 11))
            w.grid(row=0, column=col, sticky="nsew", padx=padx)
            return w

        self.log_left  = _mklog(0, (0, 2))
        self.log_left.tag_configure("out", foreground="#4aa3ff")   # sent text: blue
        self.log_left.tag_configure("in",  foreground="#ffffff")   # recv text: white
        self.log_left.configure(state="disabled")

        self.log_right = _mklog(1, (2, 0))
        self.log_right.tag_configure("white",  foreground="#ffffff")  # empty in/out
        self.log_right.tag_configure("outmsg", foreground="#ff4444")  # msg out: red ^
        self.log_right.tag_configure("inmsg",  foreground="#00d5d5")  # msg in: cyan [..]
        self.log_right.configure(state="disabled")

        compose = tk.Frame(self.root)
        compose.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)
        self.entry = tk.Entry(compose)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self._on_send)
        tk.Button(compose, text="Send", command=self._on_send).pack(
            side=tk.LEFT, padx=4)
        tk.Button(compose, text="Clear", command=self._on_clear).pack(
            side=tk.LEFT, padx=4)

        self.status = tk.Label(self.root, text="", anchor="w", fg="#999999")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ── log helpers ────────────────────────────────────────────────────────────

    def _left_line(self, text, tag=None):
        """Full message line (sent/received) in the left pane."""
        w = self.log_left
        w.configure(state="normal")
        if w.index("end-1c") != "1.0":
            w.insert(tk.END, "\n")
        w.insert(tk.END, text, (tag,) if tag else ())
        w.see(tk.END)
        w.configure(state="disabled")

    def _right_append(self, text, tag=None):
        """Append one activity token (+ a space) to the right pane. The pane
        word-wraps, so it only breaks lines on width rollover or an explicit
        newline (inserted when an incoming message finishes)."""
        w = self.log_right
        w.configure(state="normal")
        w.insert(tk.END, text + " ", (tag,) if tag else ())
        w.see(tk.END)
        w.configure(state="disabled")

    def _right_newline(self):
        """CRLF in the right pane — only when an incoming message completes."""
        w = self.log_right
        w.configure(state="normal")
        if w.index("end-1c") != "1.0":
            w.insert(tk.END, "\n")
        w.see(tk.END)
        w.configure(state="disabled")

    # ── button / control callbacks ──────────────────────────────────────────

    def _on_call(self):
        self.engine.start_call()

    def _on_detect(self):
        self.engine.start_detect()

    def _on_end(self):
        self.engine.end_mode()

    def _on_send(self, _event=None):
        text = self.entry.get().strip()
        if text:
            self.engine.send_text(text)
            self.entry.delete(0, tk.END)

    def _on_clear(self):
        for w in (self.log_left, self.log_right):
            w.configure(state="normal")
            w.delete("1.0", tk.END)
            w.configure(state="disabled")
        # Restart the plot histories too (they refill from the live RX).
        self._corr_t_hist.clear()
        self._corr_max_hist.clear()
        self._corr_avg_hist.clear()
        self.corr_max_line.set_data([], [])
        self.corr_avg_line.set_data([], [])
        self.spec_line.set_data([], [])
        self.canvas.draw_idle()

    def _on_lna(self, _event=None):
        try:
            self.engine.set_lna(int(float(self.lna_var.get())))
        except ValueError:
            pass
        self.lna_var.set(str(self.engine._lna_db))

    def _on_txp(self, val):
        self.engine.set_tx_power(int(float(val)))

    def _on_freq(self, _event=None):
        try:
            hz = float(self.freq_var.get()) * 1e6
            self.engine.set_freq(hz)
            if self._disp_mode == "spectrum":
                f_lo = (self.engine._nco_hz - eng.SAMPLE_RATE / 2) / 1e6
                f_hi = (self.engine._nco_hz + eng.SAMPLE_RATE / 2) / 1e6
                self.ax.set_xlim(f_lo, f_hi)
                self.canvas.draw_idle()
        except ValueError:
            pass
        self.freq_var.set(f"{self.engine._nco_hz/1e6:.4f}")

    # ── event pump ───────────────────────────────────────────────────────────

    def _poll_events(self):
        try:
            while True:
                ev = self.engine.events.get_nowait()
                self._handle_event(ev)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_events)

    def _handle_event(self, ev):
        kind = ev[0]
        if kind == "spectrum":
            if self._disp_mode != "spectrum":
                return
            _, freqs, power = ev
            self.spec_line.set_data(freqs, power)
            self.canvas.draw_idle()
        elif kind == "corr":
            if self._disp_mode != "corr":
                return
            _, t, max_db, avg_db = ev
            self._corr_t_hist.append(t)
            self._corr_max_hist.append(max_db)
            self._corr_avg_hist.append(avg_db)
            ts = list(self._corr_t_hist)
            self.corr_max_line.set_data(ts, list(self._corr_max_hist))
            self.corr_avg_line.set_data(ts, list(self._corr_avg_hist))
            t_latest = max(ts)
            self.ax.set_xlim(max(0.0, t_latest - CORR_SPAN_S),
                             max(CORR_SPAN_S, t_latest))
            self.canvas.draw_idle()
        elif kind == "pep":
            _, watts, ma, adc = ev
            now = time.monotonic()
            if now - self._pep_last >= 0.1:        # at most ~10 updates/sec
                self.pep_lbl.config(text=f"PEP {watts:.1f} w")
                self._pep_last = now
        elif kind == "offset":
            self.status.config(text=f"Linked — offset {ev[1]:+.0f} Hz")
        elif kind == "rx_beacon":
            self._right_append("*", tag="white")        # empty in: white *
        elif kind == "tx_beacon":
            self._right_append("^", tag="white")        # empty out: white ^
        elif kind == "rx_frag":
            self._right_append(f"[{ev[1]}]", tag="inmsg")  # msg in: cyan [letters]
        elif kind == "tx_frag":
            self._right_append("^", tag="outmsg")       # msg out: red ^
        elif kind == "miss":
            self._right_append("#")
        elif kind == "rx_text":
            self._left_line(f"< {ev[1]}", tag="in")     # full received message
            self._right_newline()                       # close this msg's fragments
        elif kind == "tx_text":
            self._left_line(f"> {ev[1]}", tag="out")    # full sent message
        elif kind == "end_connection":
            self._left_line("<end connection>")
        elif kind == "mode":
            name = ev[1]
            self.mode_lbl.config(text=name or "")
        elif kind == "tx_active":
            self.tx_lbl.config(bg=self.RED if ev[1] else self.GREY)
            # Spectrum freezes + grays during TX; the corr strip keeps its
            # colors and just shows a blank gap (it keeps scrolling in time).
            self.spec_line.set_color(self.FROZEN_COLOR if ev[1] else self.SPEC_COLOR)
            self.canvas.draw_idle()
        elif kind == "freq":
            self.freq_var.set(f"{ev[1]/1e6:.4f}")
        elif kind == "status":
            self.status.config(text=ev[1])

    def _on_close(self):
        self.engine.shutdown()
        self.root.after(200, self.root.destroy)
