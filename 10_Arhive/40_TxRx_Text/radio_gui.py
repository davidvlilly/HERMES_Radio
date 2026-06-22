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
import tkinter as tk
from tkinter import scrolledtext

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import radio_engine as eng


class RadioGUI:
    RED = "#cc1111"
    GREY = "#444444"

    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        root.title("HL2 Text Radio")
        root.geometry("1000x760")

        self._build_menu()
        self._build_spectrum()
        self._build_controls()
        self._build_log()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll_events)

    # ── menus ────────────────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=filemenu)

        cfgmenu = tk.Menu(menubar, tearoff=0)
        cfgmenu.add_command(label="(no options yet)", state="disabled")
        menubar.add_cascade(label="Config", menu=cfgmenu)
        self.root.config(menu=menubar)

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
        (self.spec_line,) = self.ax.plot([], [], color="#00ff88", linewidth=0.8)
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

        self.pep_lbl = tk.Label(bar, text="-- w", fg="black",
                                font=("TkFixedFont", 10))
        self.pep_lbl.pack(side=tk.RIGHT, padx=8)
        # (carrier offset is shown in the lower-left status line, not here)

    # ── message log + compose ─────────────────────────────────────────────────

    def _build_log(self):
        self.log = scrolledtext.ScrolledText(self.root, height=12,
                                             bg="#0a0a0a", fg="#dddddd",
                                             insertbackground="#dddddd",
                                             font=("TkFixedFont", 11))
        self.log.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=4, pady=2)
        self.log.tag_configure("out",  foreground="#4aa3ff")  # outgoing text: blue
        self.log.tag_configure("in",   foreground="#ffffff")  # incoming text: white
        self.log.tag_configure("frag", foreground="#00d5d5")  # sent fragment: cyan
        self.log.tag_configure("alert", foreground="#ff4444") # incoming fragment: red *
        self._last_was_text = False
        self.log.configure(state="disabled")

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

    def _log_inline(self, ch, tag=None):
        self.log.configure(state="normal")
        # Separate a run of notifications from the text that preceded it.
        if self._last_was_text:
            self.log.insert(tk.END, " " * 10)
            self._last_was_text = False
        self.log.insert(tk.END, ch, (tag,) if tag else ())
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _log_line(self, text, tag=None):
        self.log.configure(state="normal")
        if self.log.index("end-1c") != "1.0":
            self.log.insert(tk.END, "\n")
        self.log.insert(tk.END, text, (tag,) if tag else ())
        self.log.see(tk.END)
        self.log.configure(state="disabled")
        self._last_was_text = True

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
        self.log.configure(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.configure(state="disabled")
        self._last_was_text = False

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
            _, freqs, power = ev
            self.spec_line.set_data(freqs, power)
            self.canvas.draw_idle()
        elif kind == "pep":
            _, watts, ma, adc = ev
            self.pep_lbl.config(text=f"{watts:.2g}w")
        elif kind == "offset":
            self.status.config(text=f"Linked — offset {ev[1]:+.0f} Hz")
        elif kind == "rx_beacon":
            self._log_inline(".")
        elif kind == "tx_beacon":
            self._log_inline("/")
        elif kind == "rx_frag":
            self._log_inline("*", tag="alert")     # incoming message fragment
        elif kind == "miss":
            self._log_inline("#")
        elif kind == "rx_text":
            self._log_line(f"< {ev[1]}", tag="in")
        elif kind == "tx_text":
            self._log_line(f"> {ev[1]}", tag="out")
        elif kind == "tx_frag":
            self._log_line(f"    > {ev[1]}", tag="frag")
        elif kind == "end_connection":
            self._log_line("<end connection>")
        elif kind == "mode":
            name = ev[1]
            self.mode_lbl.config(text=name or "")
        elif kind == "tx_active":
            self.tx_lbl.config(bg=self.RED if ev[1] else self.GREY)
        elif kind == "freq":
            self.freq_var.set(f"{ev[1]/1e6:.4f}")
        elif kind == "status":
            self.status.config(text=ev[1])

    def _on_close(self):
        self.engine.shutdown()
        self.root.after(200, self.root.destroy)
