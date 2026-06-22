#!/usr/bin/env python3
"""
radio_engine.py — orchestration for the HL2 text radio.

Owns the radio session and ties everything together:
    * a TX thread that paces EP2 packets (modem samples when transmitting,
      silence otherwise),
    * an engine thread that drains EP6, runs the 5-second scheduler and the
      call/detect state machine, demodulates received bursts, and computes the
      spectrum,
    * a thread-safe event queue the GUI drains to update the display.

See DESIGN.md §4–6 for the framing, fragmentation, cadence and connection rules.

The engine never touches Tk — it only emits events. The GUI never touches the
socket — it only calls the public methods (set_mode_*, send_text, set_lna, ...).
"""

import queue
import socket
import threading
import time
from collections import deque

import numpy as np

import hl2_transport as hl2
from modem import (DbpskModem, END_CODES, MSG_TERMINATOR,
                   CODE_CALL_CONTINUE, CODE_CALL_END,
                   CODE_DETECT_CONTINUE, CODE_DETECT_END)

# ── Configuration ────────────────────────────────────────────────────────────
HL2_IP        = "169.254.19.221"
NCO_FREQ      = 7_000_000          # Hz
SAMPLE_RATE   = 96_000             # HL2 speed bits = 01  (±48 kHz spectrum view)
BAUD          = 40
PEDESTAL_HZ   = 2000.0
TX_AMPLITUDE  = 0.80
RX_LNA_GAIN_DB = 30
DEFAULT_TX_DRIVE = 15              # 0–15

# Cadence (DESIGN.md §5)
CYCLE_S       = 5.0
HALF_S        = CYCLE_S / 2.0     # each station owns one 2.5 s half
TX_SLOT_S     = 2.0               # max burst length (sets max payload); the
                                  # ~0.5 s slack in each half lets the burst
                                  # finish + T/R settle before the handoff.
BURST_ARM_S   = 0.3               # only start a burst within this much of the
                                  # half's start (avoids a truncated burst if we
                                  # lock mid-half).

# The cadence runs off the HL2 sample clock (EP6 sequence numbers), not the PC
# wall-clock, so schedule + RX timing + drift all share the radio's one crystal
# and survive ethernet jitter / packet loss (gaps are zero-filled).
CYCLE_SAMPLES    = int(CYCLE_S * SAMPLE_RATE)
TRACK_DEADBAND_S = 0.050          # responder ignores |error| <= 50 ms
TRACK_STEP_S     = 0.050          # ... and nudges its clock by 50 ms beyond that

# Detection / connection
ACQ_THRESH_DB = 14.5              # just above the noise-only correlation ceiling
MAX_MISS      = 4
NARROW_SEARCH_HZ = 15.0           # once locked, re-search only +/-15 Hz around the
                                  # known offset (vs full +/-255 Hz) — much cheaper
                                  # per cycle, frees CPU for packet capture.
VALID_CODES   = (CODE_CALL_CONTINUE, CODE_CALL_END,
                 CODE_DETECT_CONTINUE, CODE_DETECT_END)

PACKET_INTERVAL = hl2.PACKET_SAMPLES / SAMPLE_RATE

# Wall-clock time scaling. 1.0 = real time (hardware). The simulation sets this
# > 1.0 to run the radios slower than real time, giving the CPU more processing
# headroom per radio-second. The cadence is driven by the received sample count,
# so slowing transmit production slows the whole system consistently; nothing in
# the radio logic depends on wall-clock, so this only affects pacing.
TIME_SCALE = 1.0

# Spectrum (Welch)
SPEC_RBW_HZ = 2500.0              # spectrum resolution bandwidth (SSB channel width)
SPEC_SEG   = max(16, round(SAMPLE_RATE * 1.44 / SPEC_RBW_HZ))  # Welch segment for 2.5 kHz RBW
SPEC_PAD   = 2048
DBFS_REF   = 8.0
SPEC_PERIOD_S = 0.2
DISP_BUF_SAMPLES = int(1.3 * SAMPLE_RATE)   # rolling RX held for the displays
                                            # (long enough for the Sync-Corr view)


def compute_spectrum(iq, nco_hz, fs, seg=SPEC_SEG, pad=SPEC_PAD, dbfs_ref=DBFS_REF):
    """Welch averaged power spectrum -> (freqs_mhz, power_dbm)."""
    arr = np.asarray(iq, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:          # list of (I, Q) pairs
        z = arr[:, 0] + 1j * arr[:, 1]
    else:
        z = np.asarray(iq, dtype=np.complex128)
    if len(z) < seg:
        return None, None
    step = seg // 2
    win  = np.hanning(seg)
    wpow = np.sum(win ** 2)
    pwr  = np.zeros(pad)
    n = 0
    for s in range(0, len(z) - seg + 1, step):
        buf = np.zeros(pad, dtype=np.complex128)
        buf[:seg] = z[s:s + seg] * win
        pwr += np.abs(np.fft.fft(buf)) ** 2
        n += 1
    psd = np.fft.fftshift(pwr / max(n, 1)) / (fs * wpow)
    rbw = (fs / seg) * 1.44
    power_db = 10.0 * np.log10(psd * rbw / 50.0 / 1e-3 + 1e-40) + dbfs_ref
    freqs = (nco_hz + np.fft.fftshift(np.fft.fftfreq(pad, d=1.0 / fs))) / 1e6
    return freqs, power_db


class RadioEngine:
    def __init__(self, ip=HL2_IP, nco_freq=NCO_FREQ, log_path=None):
        self.ip       = ip
        self.events   = queue.Queue()
        # Optional timing-diagnostic log (expected vs actual burst start, etc.)
        self._logf    = open(log_path, "w", buffering=1) if log_path else None
        self._log_t0  = time.monotonic()
        self.modem    = DbpskModem(sample_rate=SAMPLE_RATE, baud=BAUD,
                                   pedestal_hz=PEDESTAL_HZ)
        self.max_payload = self.modem.max_payload_for_window(TX_SLOT_S)

        # Radio control state
        self._nco_hz     = float(nco_freq)
        self._freq_word  = hl2.freq_to_word(nco_freq)
        self._lna_db     = RX_LNA_GAIN_DB
        self._tx_drive   = DEFAULT_TX_DRIVE
        self._ptt        = False
        self._pa_enable  = True

        # Mode / connection state machine
        self._mode      = None        # None | 'call' | 'detect'
        self._linked    = False
        self._miss      = 0
        self._cycle_anchor = None     # sample index of phase 0 (caller burst start)
        self._pending_end = 0         # >0: send this many more end bursts

        # Outgoing text buffer + incoming reassembly
        self._tx_buf    = bytearray()
        self._rx_msg    = bytearray()
        self._lock      = threading.Lock()

        # TX sample source shared with the TX thread
        self._tx_state  = {"active": False, "iq": np.zeros(0, np.complex64), "pos": 0}
        self._tx_lock   = threading.Lock()

        # Sample-clock state (from EP6 sequence numbers)
        self._seq_start = None         # seq of first EP6 frame (anchor)
        self._prev_seq  = None
        self._rx_count  = 0            # absolute samples elapsed (incl. zero-fills)

        # Spectrum rolling buffer + listen accumulator. deque(maxlen) gives O(1)
        # append with automatic discard — NEVER slice-copy this in the hot path.
        self._spec_buf  = deque(maxlen=DISP_BUF_SAMPLES)   # recent (I,Q) for displays
        self._rx_accum  = []          # samples during the current listen window
        self._listening = False
        self._accum_sample0 = None    # abs sample index of _rx_accum[0]

        # Demod runs on its own thread so the ~0.4 s modem.receive() never
        # stalls the 2 ms scheduler loop (which would drift the cadence).
        self._demod_q   = queue.Queue()
        self._result_q  = queue.Queue()
        self._demod_busy   = False
        self._search_pending = False

        self._cc_regs   = None
        self._display_mode = "corr"    # "corr" (processing gain) | "spectrum"
        self._last_df   = 0.0          # last measured carrier offset (for corr view)
        # Raw-RX capture-to-file (diagnostic)
        self._capturing = False
        self._cap_buf   = None
        self._cap_need  = 0
        self._cap_path  = "capture.npz"
        self._tone      = False        # continuous-carrier (CW) test transmit
        self._inv_rx    = False        # InvFrq: conjugate RX I/Q (fix sideband mirror)
        self._running   = threading.Event()
        self._sock      = None
        self._tx_sock   = None
        self._seq       = 0

    # ── public API (called from the GUI thread) ─────────────────────────────

    def emit(self, *event):
        self.events.put(event)

    def start_call(self):
        with self._lock:
            self._mode = "call"
            self._linked = False
            self._miss = 0
            self._pending_end = 0
            self._cycle_anchor = self._rx_count    # we define the cadence (now)
        self._log(f"--- CALL start; anchor={self._cycle_anchor} cycle={CYCLE_SAMPLES} "
                  f"half={HALF_S}s (TX first half, RX second) ---")
        self.emit("mode", "CALL")
        self.emit("status", "Calling — transmitting every 5 s")

    def start_detect(self):
        with self._lock:
            self._mode = "detect"
            self._linked = False
            self._miss = 0
            self._pending_end = 0
            self._cycle_anchor = None               # learn it from the caller
        self._log(f"--- DETECT start; searching (cycle={CYCLE_SAMPLES} "
                  f"half={HALF_S}s; RX first half, TX second) ---")
        self.emit("mode", "DETECT")
        self.emit("status", "Detecting — searching for a caller …")

    def end_mode(self):
        """Operator-requested end: immediately drop any active mode. (The engine
        thread cleans up TX / PTT on its next tick.)"""
        self._drop_mode("Connection ended by operator")

    def _drop_mode(self, reason=""):
        with self._lock:
            self._mode = None
            self._linked = False
            self._miss = 0
            self._pending_end = 0
        self.emit("mode", None)
        self.emit("tx_active", False)
        if reason:
            self.emit("status", reason)

    def send_text(self, text):
        if not text:
            return
        data = text.encode("ascii", errors="replace")
        with self._lock:
            self._tx_buf.extend(data)
            self._tx_buf.append(MSG_TERMINATOR)
        self.emit("tx_text", text)

    def set_display_mode(self, mode):
        self._display_mode = mode

    def request_capture(self, seconds=6.0, path=None):
        """Capture `seconds` of *received* IQ to a mode-named .npz for offline
        analysis (capture_call.npz / capture_detect.npz / capture_idle.npz).

        Only RX samples are recorded — our own TX halves are skipped — so the
        file is purely the partner's incoming band. Because the TX halves don't
        count, the capture spans more than `seconds` of wall-clock in Call mode."""
        if path is None:
            path = f"capture_{self._mode or 'idle'}.npz"
        self._cap_buf  = []
        self._cap_need = int(seconds * SAMPLE_RATE)
        self._cap_path = path
        self._capturing = True
        self.emit("capturing", True)
        self.emit("note", f"--- CAPTURE START (RX only): {seconds:.0f}s -> {path} ---")
        self.emit("status", f"Recording {seconds:.0f}s of RX -> {path} ...")

    def _save_capture(self):
        try:
            arr = np.asarray(self._cap_buf[:self._cap_need], dtype=np.float64)
            iq  = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
            np.savez(self._cap_path, iq=iq, fs=SAMPLE_RATE, nco=self._nco_hz,
                     lna=self._lna_db, pedestal=PEDESTAL_HZ, baud=BAUD)
            self.emit("note", f"--- CAPTURE DONE: {len(iq)} samples "
                              f"({len(iq)/SAMPLE_RATE:.1f}s) -> {self._cap_path} ---")
            self.emit("status", f"Saved -> {self._cap_path}")
        except Exception as e:                          # noqa: BLE001
            self.emit("note", f"--- CAPTURE FAILED: {e} ---")
        self._capturing = False
        self._cap_buf = None
        self.emit("capturing", False)

    def set_tone(self, on):
        """Key a continuous carrier (CW) at the pedestal so a partner can find you
        on their spectrum. Diagnostic only — use when idle (not in Call/Detect)."""
        if on and self._mode:
            self.emit("note", "--- TONE: End the Call/Detect mode first ---")
            self.emit("tone", False)
            return
        if on:
            # 48384 = 1008*48 samples: an exact number of 2 kHz cycles AND of
            # 126-sample packets, so the looped carrier is phase-continuous.
            n    = np.arange(48384)
            tone = (TX_AMPLITUDE * np.exp(1j * 2.0 * np.pi * PEDESTAL_HZ * n
                                          / SAMPLE_RATE)).astype(np.complex64)
            with self._tx_lock:
                self._tx_state = {"active": True, "iq": tone, "pos": 0, "loop": True}
            self._tone = True
            self._push_cc(ptt=True)
            self.emit("tx_active", True)
            self.emit("tone", True)
            self.emit("note", "--- TONE ON: continuous CW carrier ---")
            self.emit("status", "TONE ON — transmitting CW carrier")
        else:
            self._tone = False
            self._stop_tx_burst()
            self._push_cc(ptt=False)
            self.emit("tx_active", False)
            self.emit("tone", False)
            self.emit("note", "--- TONE OFF ---")
            self.emit("status", "Tone off")

    def set_invert(self, on):
        """Conjugate the received I/Q so the spectrum reads the right way round.
        Use if the partner's signal moves the wrong way / the link won't lock due
        to a TX/RX sideband mismatch."""
        self._inv_rx = bool(on)
        self.emit("invert", self._inv_rx)
        self.emit("note", f"--- RX I/Q INVERT {'ON' if self._inv_rx else 'OFF'} ---")
        self.emit("status", f"RX freq invert {'ON' if self._inv_rx else 'off'}")

    def set_lna(self, db):
        self._lna_db = max(0, min(48, int(db)))
        self._push_cc()

    def set_tx_power(self, drive):
        self._tx_drive = max(0, min(15, int(drive)))
        self._push_cc()

    def set_freq(self, hz):
        self._nco_hz = max(0.1e6, min(35e6, float(hz)))
        self._freq_word = hl2.freq_to_word(self._nco_hz)
        self._push_cc()
        self.emit("freq", self._nco_hz)

    def shutdown(self):
        self._running.clear()

    # ── control-register push ────────────────────────────────────────────────

    def _push_cc(self, ptt=None):
        if ptt is not None:
            self._ptt = ptt
        regs = hl2.make_cc(self._freq_word, ptt=self._ptt,
                           pa_enable=self._pa_enable, lna_gain_db=self._lna_db,
                           tx_drive=self._tx_drive, sample_rate=SAMPLE_RATE)
        self._cc_regs = regs
        # Push promptly so PTT/gain changes take effect without waiting on the
        # TX thread's register cycling.
        if self._sock is not None:
            try:
                hl2.init_radio(self._sock, self.ip, self._freq_word, 0,
                               ptt=self._ptt, pa_enable=self._pa_enable,
                               lna_gain_db=self._lna_db, tx_drive=self._tx_drive,
                               sample_rate=SAMPLE_RATE)
            except OSError:
                pass

    # ── connect + threads ────────────────────────────────────────────────────

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind(("", hl2.HPSDR_PORT))
        self._seq = hl2.connect(self._sock, self.ip, self._freq_word,
                                sample_rate=SAMPLE_RATE)
        self._sock.settimeout(0.002)
        self._push_cc(ptt=False)

        self._tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tx_sock.bind(("", 0))
        self.emit("status", f"Connected to HL2 at {self.ip}")

    def run(self):
        """Start threads. Returns immediately; engine runs in the background."""
        self._running.set()
        threading.Thread(target=self._tx_thread, daemon=True,
                         name="HL2-TX").start()
        threading.Thread(target=self._engine_thread, daemon=True,
                         name="HL2-Engine").start()
        threading.Thread(target=self._demod_worker, daemon=True,
                         name="HL2-Demod").start()
        threading.Thread(target=self._display_worker, daemon=True,
                         name="HL2-Display").start()

    # ── TX thread: pace EP2 packets ──────────────────────────────────────────

    def _tx_thread(self):
        seq = self._seq
        next_send = time.monotonic()
        while self._running.is_set():
            with self._tx_lock:
                st = self._tx_state
                if st["active"] and len(st["iq"]):
                    if st["pos"] >= len(st["iq"]) and st.get("loop"):
                        st["pos"] = 0           # loop the buffer (CW tone)
                    if st["pos"] < len(st["iq"]):
                        chunk = st["iq"][st["pos"]:st["pos"] + hl2.PACKET_SAMPLES]
                        st["pos"] += hl2.PACKET_SAMPLES
                    else:
                        chunk = None
                else:
                    chunk = None

            if chunk is not None:
                I = np.clip(np.real(chunk) * 32767.0, -32767, 32767).astype(np.int16)
                Q = np.clip(np.imag(chunk) * 32767.0, -32767, 32767).astype(np.int16)
                tx_iq = list(zip(I.tolist(), Q.tolist()))
            else:
                tx_iq = [(0, 0)] * hl2.PACKET_SAMPLES

            regs = self._cc_regs
            if regs is not None:
                n = len(regs)
                cc0 = regs[seq % n]
                cc1 = regs[(seq + 1) % n]
                try:
                    self._tx_sock.sendto(
                        hl2.build_ep2(seq, cc0, cc1, tx_iq),
                        (self.ip, hl2.HPSDR_PORT))
                except OSError:
                    break
                seq += 1

            next_send += PACKET_INTERVAL * TIME_SCALE
            wait = next_send - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                next_send = time.monotonic()

    # ── engine thread: recv + schedule + state machine ───────────────────────

    def _load_tx_burst(self, burst_code, payload):
        iq = self.modem.modulate(burst_code, payload, amplitude=TX_AMPLITUDE)
        with self._tx_lock:
            self._tx_state = {"active": True, "iq": iq, "pos": 0}

    def _stop_tx_burst(self):
        with self._tx_lock:
            self._tx_state = {"active": False, "iq": np.zeros(0, np.complex64),
                              "pos": 0}

    # ── display worker (off the time-critical engine loop) ───────────────────

    def _display_worker(self):
        """Compute the upper-plot data (spectrum or matched-filter corr) on its
        own thread — its FFT/FIR cost must not stall recv/scheduling. Frozen
        while transmitting; holds the last received view."""
        last = time.monotonic()
        while self._running.is_set():
            time.sleep(0.05)
            now = time.monotonic()
            if now - last < SPEC_PERIOD_S:
                continue
            last = now
            # During TX: spectrum just freezes; the corr strip keeps its time
            # axis moving with a blank (NaN) gap so the timeline stays honest.
            if self._ptt:
                if self._display_mode == "corr":
                    self.emit("corr", self._rx_count / SAMPLE_RATE,
                              float("nan"), float("nan"))
                continue
            if len(self._spec_buf) < SPEC_PAD:
                continue
            buf = list(self._spec_buf)                  # snapshot (display thread)
            try:
                if self._display_mode == "spectrum":
                    # Antenna-referred: subtract the LNA gain so the displayed
                    # dBm is at the antenna, not after the LNA (fixes the ~LNA-dB
                    # hot noise floor, and tracks the gain if it's changed).
                    freqs, power = compute_spectrum(
                        buf[-SPEC_PAD * 2:], self._nco_hz, SAMPLE_RATE,
                        dbfs_ref=DBFS_REF - self._lna_db)
                    if freqs is not None:
                        self.emit("spectrum", freqs, power)
                else:  # "corr" — matched-filter detection strip chart
                    arr = np.asarray(buf, dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        iqc = arr[:, 0] + 1j * arr[:, 1]
                        _, mag_db, _ = self.modem.corr_db(
                            self.modem.to_baseband_fast(iqc), self._last_df)
                        if mag_db is not None and len(mag_db):
                            # Full-window peak (always finds the burst), reported
                            # at its TRUE receive time (window-start + lag). Every
                            # overlapping window that sees a burst reports the same
                            # time, so the points stack → one sharp mark, no drops.
                            L = int(np.argmax(mag_db))
                            max_db = float(mag_db[L])
                            avg_db = float(np.mean(mag_db))
                            peak_samp = (self._rx_count - len(iqc)
                                         + L * self.modem.decim)
                            peak_t = peak_samp / SAMPLE_RATE
                            self.emit("corr", peak_t, max_db, avg_db)
            except Exception:
                pass

    # ── demod worker + result dispatch ───────────────────────────────────────

    def _demod_worker(self):
        while self._running.is_set():
            try:
                kind, partner, iq, accum_start, center, search = \
                    self._demod_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                acq, dec = self.modem.receive(iq, center=center, search_hz=search)
            except Exception:                       # never let the worker die
                acq, dec = {"found": False, "snr_db": -99.0, "df_hz": 0.0,
                            "start": 0}, None
            self._result_q.put((kind, partner, acq, dec, accum_start))
            self._demod_busy = False

    def _drain_results(self):
        while True:
            try:
                kind, partner, acq, dec, sample0 = self._result_q.get_nowait()
            except queue.Empty:
                return
            good = (acq["found"] and acq["snr_db"] >= ACQ_THRESH_DB and
                    dec is not None and dec["ok"] and
                    dec["burst_code"] in VALID_CODES)
            # A genuine end burst is a beacon (no payload). Reject any "end
            # code" that arrives with a payload — it's a corrupt/noise decode,
            # and a single false end-code would wrongly drop the link.
            if good and dec["burst_code"] in END_CODES and dec["n_payload"] != 0:
                good = False
            # Absolute sample index of the burst start (baseband -> full rate).
            burst_abs = sample0 + int(acq["start"] * self.modem.decim)
            if kind == "search":
                self._search_pending = False
                if good and self._mode == "detect" and self._cycle_anchor is None:
                    self._cycle_anchor = burst_abs          # phase 0 = caller burst
                    self._listening = False                 # cadence branch owns it
                    self._rx_accum = []
                    self._accum_sample0 = None
                    self._log(f"LOCK  offset={acq['df_hz']:+.0f}Hz snr={acq['snr_db']:.1f} "
                              f"code={dec['burst_code']} anchor={burst_abs}")
                    self.emit("status", f"Caller found — offset "
                                        f"{acq['df_hz']:+.0f} Hz; replying at +2.5 s")
                    self._handle_burst(dec, acq, "caller")
                else:
                    self._log(f"SEARCH no-lock (snr={acq['snr_db']:.1f} "
                              f"ok={dec['ok'] if dec else None})")
            else:  # listen
                expected = 0.0 if self._mode == "detect" else HALF_S
                if good:
                    ph, err = self._burst_phase_err(burst_abs, expected)
                    errtxt = f"err={err*1000:+.0f}ms" if err is not None else "err=n/a"
                    self._log(f"DET   from={partner:9s} exp={expected:.2f}s "
                              f"act={ph if ph is not None else -1:.3f}s {errtxt} "
                              f"snr={acq['snr_db']:.1f} code={dec['burst_code']} "
                              f"npay={dec['n_payload']}")
                    if self._mode == "detect" and self._cycle_anchor is not None:
                        self._track_cadence(burst_abs)      # responder trims to caller
                    self._handle_burst(dec, acq, partner)
                else:
                    if dec is None or not acq["found"]:
                        reason = "noframe"
                    elif acq["snr_db"] < ACQ_THRESH_DB:
                        reason = "lowSNR"
                    elif not dec["ok"]:
                        reason = "badCRC"
                    else:
                        reason = "rejected"
                    self._log(f"MISS  from={partner:9s} snr={acq['snr_db']:.1f} "
                              f"({reason}) linked={self._linked}")
                    self._register_miss(partner)

    def _engine_thread(self):
        prev_slot = None
        # Per-role: which slot is "mine" to transmit and which is the partner's.
        while self._running.is_set():
            # 1) Drain a batch of EP6 packets (sample clock tracked in _ingest).
            fwd = rev = cur = None
            for _ in range(96):
                try:
                    data, _addr = self._sock.recvfrom(2048)
                except socket.timeout:
                    break
                if len(data) != 1032 or data[0:2] != b"\xEF\xFE":
                    continue
                f, r, c = self._ingest(data, accumulate=True)
                if f is not None: fwd = f
                if r is not None: rev = r
                if c is not None: cur = c

            # 2) Telemetry -> PEP.
            if fwd is not None:
                pep = max(0.0, hl2.adc_to_watts(fwd) - hl2.adc_to_watts(rev or 0))
                ma  = hl2.adc_to_current_ma(cur or 0)
                self.emit("pep", pep, ma, int(fwd))

            # 3) Dispatch any completed demods, then run the scheduler.
            #    (The upper-display compute runs on its own thread so its FFT/FIR
            #    cost never stalls this time-critical recv + scheduler loop.)
            self._drain_results()
            mode = self._mode
            if mode == "call":
                self._tick_caller()
            elif mode == "detect":
                prev_slot = self._tick_responder(prev_slot)
            else:
                # Idle: make sure we're not stuck keyed (so PTT drops, the
                # display un-freezes, and RX/plots keep progressing).
                if getattr(self, "_tx_armed", False):
                    self._end_my_burst()
                    self._tx_armed = False
                self._listening = False

            time.sleep(0.002)

    # ── sample-clock ingest / flush / phase ──────────────────────────────────

    def _ingest(self, data, accumulate):
        """Account one EP6 frame against the sample clock and, while listening,
        append it to the demod accumulator (zero-filling any dropped frames so
        the burst stays a contiguous waveform). Returns (fwd, rev, cur) ADC."""
        seq6 = hl2.ep6_seq(data)
        if self._seq_start is None:
            self._seq_start = seq6
            self._prev_seq  = (seq6 - 1) & 0xFFFF_FFFF
        gap = (seq6 - self._prev_seq - 1) & 0xFFFF_FFFF
        if gap > 100_000:                       # wrap / garbage guard
            gap = 0
        self._prev_seq = seq6
        abs_idx = ((seq6 - self._seq_start) & 0xFFFF_FFFF) * hl2.PACKET_SAMPLES
        self._rx_count = abs_idx + hl2.PACKET_SAMPLES

        samp = hl2.parse_ep6(data)
        if self._inv_rx:                    # InvFrq: conjugate I/Q -> un-mirror the
            samp = [(i, -q) for (i, q) in samp]   # received spectrum (whole RX chain)
        # Capture and spectrum use RECEIVED samples only — never our own TX. So a
        # capture is purely the partner's incoming signal; the TX halves (where
        # the RX stream is just our own transmit looping back) are skipped.
        if not self._ptt:
            if self._capturing:
                self._cap_buf.extend(samp)
                if len(self._cap_buf) >= self._cap_need:
                    self._save_capture()
            self._spec_buf.extend(samp)     # deque(maxlen) auto-discards old

        if accumulate and self._listening:
            if self._accum_sample0 is None:
                self._accum_sample0 = abs_idx       # anchor the window
            elif gap > 0:
                self._rx_accum.extend([(0.0, 0.0)] * (gap * hl2.PACKET_SAMPLES))
            self._rx_accum.extend(samp)
        return hl2.parse_ep6_power(data)

    def _flush_rx(self):
        """Discard currently-buffered (stale) packets so the next accumulated
        sample is current. The sample clock keeps advancing from the sequence
        numbers, so only the backlog is dropped — timing stays continuous."""
        try:
            self._sock.settimeout(0.0)
        except OSError:
            pass
        drained = 0
        while drained < 5000:
            try:
                data, _addr = self._sock.recvfrom(2048)
            except (socket.timeout, BlockingIOError, OSError):
                break
            if len(data) == 1032 and data[0:2] == b"\xEF\xFE":
                self._ingest(data, accumulate=False)
            drained += 1
        try:
            self._sock.settimeout(0.002)
        except OSError:
            pass
        self._rx_accum = []
        self._accum_sample0 = None

    def _phase_s(self):
        """Current cadence phase in seconds, from the sample clock."""
        if self._cycle_anchor is None:
            return 0.0
        ps = (self._rx_count - self._cycle_anchor) % CYCLE_SAMPLES
        return ps / SAMPLE_RATE

    def _log(self, msg):
        f = self._logf
        if f is None:
            return
        try:
            f.write(f"{time.monotonic() - self._log_t0:8.2f}s  {msg}\n")
        except Exception:
            pass

    def _burst_phase_err(self, burst_abs, expected_phase_s):
        """(actual_phase_s, error_s) of a detected burst vs where it was
        expected, error wrapped to +/- CYCLE_S/2. None if not yet locked."""
        if self._cycle_anchor is None:
            return None, None
        ph  = ((burst_abs - self._cycle_anchor) % CYCLE_SAMPLES) / SAMPLE_RATE
        err = ph - expected_phase_s
        if err >  CYCLE_S / 2: err -= CYCLE_S
        if err < -CYCLE_S / 2: err += CYCLE_S
        return ph, err

    def _track_cadence(self, burst_abs):
        """Responder-only deadband tracker: nudge the cadence clock toward the
        caller's measured burst time (DESIGN.md §5). Master (caller) never moves."""
        err = (burst_abs - self._cycle_anchor) % CYCLE_SAMPLES
        if err > CYCLE_SAMPLES // 2:
            err -= CYCLE_SAMPLES
        dead = int(TRACK_DEADBAND_S * SAMPLE_RATE)
        step = int(TRACK_STEP_S * SAMPLE_RATE)
        if err > dead:
            self._cycle_anchor += step
            self._log(f"TRACK err={err/SAMPLE_RATE*1000:+.0f}ms -> nudge "
                      f"+{TRACK_STEP_S*1000:.0f}ms")
        elif err < -dead:
            self._cycle_anchor -= step
            self._log(f"TRACK err={err/SAMPLE_RATE*1000:+.0f}ms -> nudge "
                      f"-{TRACK_STEP_S*1000:.0f}ms")

    # ── caller cadence ───────────────────────────────────────────────────────

    def _tick_caller(self):
        # Half-cycle model: caller transmits the whole first half, listens the
        # whole second half. _manage_listen runs first so the just-finished RX
        # half is processed before the next TX burst begins.
        phase = self._phase_s()
        in_tx = phase < HALF_S
        self._manage_listen(not in_tx, partner="responder")
        if in_tx and phase < BURST_ARM_S and not getattr(self, "_tx_armed", False):
            self._begin_my_burst(role="call")
            self._tx_armed = True
        elif not in_tx and getattr(self, "_tx_armed", False):
            self._end_my_burst()
            self._tx_armed = False

    def _tick_responder(self, prev_slot):
        # Searching: no cadence yet — listen continuously and acquire on a
        # window longer than (cycle + longest burst), guaranteeing it contains a
        # complete burst regardless of the caller's phase.
        if self._cycle_anchor is None:
            if not self._listening:
                self._rx_accum = []
                self._accum_sample0 = None
                self._listening = True
            if (not self._demod_busy and not self._search_pending and
                    self._accum_sample0 is not None and
                    len(self._rx_accum) >= int((CYCLE_S + TX_SLOT_S + 0.5)
                                               * SAMPLE_RATE)):
                iq = np.array([s[0] + 1j * s[1] for s in self._rx_accum],
                              dtype=np.complex64)
                sample0 = self._accum_sample0
                self._rx_accum = []
                self._accum_sample0 = None          # re-anchor on next packet
                self._search_pending = True
                self._demod_busy = True
                self._demod_q.put(("search", "caller", iq, sample0, 0.0, None))
            return prev_slot

        # Linked: responder listens the whole first half, transmits the second.
        phase = self._phase_s()
        in_tx = phase >= HALF_S
        self._manage_listen(not in_tx, partner="caller")
        if (in_tx and phase < HALF_S + BURST_ARM_S
                and not getattr(self, "_tx_armed", False)):
            self._begin_my_burst(role="detect")
            self._tx_armed = True
        elif not in_tx and getattr(self, "_tx_armed", False):
            self._end_my_burst()
            self._tx_armed = False
        return prev_slot

    # ── listen window management ──────────────────────────────────────────────

    def _manage_listen(self, in_listen, partner):
        if in_listen and not self._listening:
            self._flush_rx()             # drop stale backlog -> anchor "now"
            self._listening = True
        elif not in_listen and self._listening:
            self._listening = False
            self._process_listen(partner)

    def _process_listen(self, partner):
        if len(self._rx_accum) < int(0.5 * SAMPLE_RATE) or self._accum_sample0 is None:
            self._register_miss(partner)
            return
        if self._demod_busy:            # previous demod still running — skip
            self._rx_accum = []
            return
        iq = np.array([s[0] + 1j * s[1] for s in self._rx_accum],
                      dtype=np.complex64)
        sample0 = self._accum_sample0
        self._rx_accum = []
        self._demod_busy = True
        # Once linked, the offset is known — narrow search (cheap). Otherwise
        # full sweep.
        if self._linked:
            center, search = self._last_df, NARROW_SEARCH_HZ
        else:
            center, search = 0.0, None
        self._demod_q.put(("listen", partner, iq, sample0, center, search))

    def _register_miss(self, partner):
        if not self._linked:
            return                       # not connected yet — silence is normal
        self._miss += 1
        self.emit("miss")
        if self._miss >= MAX_MISS:
            self.emit("end_connection")
            self._drop_mode(f"Connection lost ({MAX_MISS} missed bursts)")

    def _handle_burst(self, dec, acq, partner):
        self._miss = 0
        if not self._linked:
            self._linked = True
            self.emit("status", f"Linked — offset {acq['df_hz']:+.0f} Hz")
        self._last_df = acq["df_hz"]
        self.emit("offset", acq["df_hz"])

        code = dec["burst_code"]
        if code in END_CODES:
            self.emit("rx_text", "<end connection>")
            self._drop_mode("Partner ended the connection")
            return

        if dec["n_payload"] == 0:
            self.emit("rx_beacon")       # empty in
        else:
            # show the received fragment's letters (terminator stripped)
            self.emit("rx_frag", dec["payload"].decode("ascii", errors="replace")
                                                .replace("\n", ""))
            self._reassemble(dec["payload"])

    def _reassemble(self, payload):
        with self._lock:
            self._rx_msg.extend(payload)
            while MSG_TERMINATOR in self._rx_msg:
                idx = self._rx_msg.index(MSG_TERMINATOR)
                line = self._rx_msg[:idx].decode("ascii", errors="replace")
                del self._rx_msg[:idx + 1]
                self.emit("rx_text", line)

    # ── building my outgoing burst ────────────────────────────────────────────

    def _begin_my_burst(self, role):
        self._push_cc(ptt=True)
        self.emit("tx_active", True)

        # End-of-connection takes priority (sent twice).
        if self._pending_end > 0:
            code = CODE_CALL_END if role == "call" else CODE_DETECT_END
            self._load_tx_burst(code, b"")
            self._pending_end -= 1
            if self._pending_end == 0:
                # schedule drop after this burst finishes (handled on slot exit)
                self._end_after_burst = True
            return

        cont = CODE_CALL_CONTINUE if role == "call" else CODE_DETECT_CONTINUE
        with self._lock:
            if self._tx_buf:
                payload = bytes(self._tx_buf[:self.max_payload])
                del self._tx_buf[:self.max_payload]
            else:
                payload = b""
        if payload:
            self._load_tx_burst(cont, payload)
            # Show the operator each fragment as it actually goes out (the
            # terminating newline shows as a trailing '.').
            disp = payload.decode("ascii", errors="replace").replace("\n", ".")
            self.emit("tx_frag", disp)
        else:
            self._load_tx_burst(cont, b"")
            self.emit("tx_beacon")       # '/'

    def _end_my_burst(self):
        self._stop_tx_burst()
        self._push_cc(ptt=False)
        self.emit("tx_active", False)
        if getattr(self, "_end_after_burst", False):
            self._end_after_burst = False
            self._drop_mode("Connection ended")
