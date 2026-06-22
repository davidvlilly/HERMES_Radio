#!/usr/bin/env python3
"""
modem.py — DBPSK text modem for the HL2 text radio.  Pure NumPy.

Waveform (see DESIGN.md §2–3):
    * 40 baud differential BPSK on a 2 kHz pedestal.
    * Complex (SSB): tx[n] = d[n] * exp(j 2 pi f_ped n / fs), d[n] = +/-1.
    * Frame: 20-sym preamble | 4-sym burst code | 8-sym length | payload*8.
    * Differential encoding is continuous across the whole burst; the last
      preamble symbol is the phase reference for the first data symbol.
    * bit 1 = 180 deg phase flip, bit 0 = no change.

RX (see DESIGN.md §1, §9):
    * Mix the pedestal to baseband, low-pass, decimate 96 k -> 2 kHz.
    * Acquire with a noncoherent sub-block matched filter on the preamble,
      swept over a carrier-offset search (handles HL2 oscillator offset).
    * Extract per-symbol complex values (rectangular matched filter) and
      differentially decode.

The class `DbpskModem` bundles the parameters and the encode/decode methods.
"""

import numpy as np

# ── Burst codes (DESIGN.md §3) ───────────────────────────────────────────────
CODE_CALL_CONTINUE = 0
CODE_CALL_END      = 1
CODE_DETECT_CONTINUE = 2
CODE_DETECT_END    = 4

END_CODES = (CODE_CALL_END, CODE_DETECT_END)

# Newline sentinel marks the end of a logical (possibly multi-burst) message.
MSG_TERMINATOR = 0x0A


# ── small bit helpers ────────────────────────────────────────────────────────

def _int_to_bits(value, nbits):
    """MSB-first list of bits."""
    return [(value >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def _bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    return v


def gen_preamble(n=20, seed=0xACE1):
    """Deterministic +/-1 preamble of length n (same in every copy of the SW).

    A 16-bit Fibonacci LFSR gives a balanced, good-autocorrelation sequence.
    """
    reg = seed & 0xFFFF
    syms = []
    for _ in range(n):
        bit = reg & 1
        syms.append(1.0 if bit else -1.0)
        # x^16 + x^14 + x^13 + x^11 + 1 (a standard maximal-length tap set)
        nb = ((reg >> 0) ^ (reg >> 2) ^ (reg >> 3) ^ (reg >> 5)) & 1
        reg = (reg >> 1) | (nb << 15)
    return np.array(syms, dtype=np.float64)


class DbpskModem:
    def __init__(self, sample_rate=96_000, baud=40, pedestal_hz=2000.0,
                 preamble_syms=20, decim=48,
                 freq_search_hz=255.0, freq_step_hz=0.5, lpf_cutoff_hz=400.0):
        self.fs        = float(sample_rate)
        self.baud      = float(baud)
        self.f_ped     = float(pedestal_hz)
        self.decim     = int(decim)
        self.fs_bb     = self.fs / self.decim          # baseband sample rate
        self.sps_tx    = int(round(self.fs / self.baud))     # samples/sym at fs
        self.sps_bb    = int(round(self.fs_bb / self.baud))  # samples/sym at baseband
        self.preamble  = gen_preamble(preamble_syms)
        self.npre      = preamble_syms
        self.search_hz = float(freq_search_hz)
        self.step_hz   = float(freq_step_hz)

        # Overhead symbols = preamble + 4 burst-code + 8 length + 8 checksum.
        self.overhead_syms = self.npre + 4 + 8 + 8

        # Baseband low-pass FIR for mix+decimate (windowed sinc).
        self._lpf = self._design_lpf(self.fs, lpf_cutoff_hz, ntaps=193)

        # Baseband preamble template (rectangular, +/-1 per symbol).
        self._pre_template = np.repeat(self.preamble, self.sps_bb).astype(np.complex128)

    # ── geometry helpers ────────────────────────────────────────────────────

    def max_payload_for_window(self, window_s):
        """Max payload chars that fit in a transmit window of `window_s` seconds."""
        total_syms = int(window_s * self.baud)
        return max(0, (total_syms - self.overhead_syms) // 8)

    def burst_duration_s(self, payload_len):
        return (self.overhead_syms + 8 * payload_len) / self.baud

    # ── TX: build a burst of complex baseband-at-fs samples ───────────────────

    @staticmethod
    def _checksum(burst_code, payload_bytes):
        return (burst_code + len(payload_bytes) + sum(payload_bytes)) & 0xFF

    def encode_symbols(self, burst_code, payload_bytes):
        """Return the +/-1 absolute-phase symbol sequence for a burst.

        Frame data bits: code(4) | length(8) | payload(8*len) | checksum(8).
        The checksum lets the RX reject corrupt/partial decodes (DESIGN.md §3).
        """
        bits  = _int_to_bits(burst_code & 0x0F, 4)
        bits += _int_to_bits(len(payload_bytes) & 0xFF, 8)
        for ch in payload_bytes:
            bits += _int_to_bits(ch & 0xFF, 8)
        bits += _int_to_bits(self._checksum(burst_code, payload_bytes), 8)

        # Differential encode the data bits, starting from the last preamble sym.
        syms = list(self.preamble)
        phase = self.preamble[-1]
        for b in bits:
            if b:
                phase = -phase          # bit 1 -> 180 deg flip
            syms.append(phase)
        return np.array(syms, dtype=np.float64)

    def modulate(self, burst_code, payload_bytes, amplitude=0.8):
        """Build a burst as complex IQ at the HL2 sample rate (Re=I, Im=Q)."""
        syms = self.encode_symbols(burst_code, payload_bytes)
        d    = np.repeat(syms, self.sps_tx)                  # rectangular NRZ
        n    = np.arange(len(d))
        carrier = np.exp(1j * 2.0 * np.pi * self.f_ped * n / self.fs)
        return (amplitude * d * carrier).astype(np.complex64)

    # ── RX: filtering, acquisition, demod ────────────────────────────────────

    @staticmethod
    def _design_lpf(fs, fc, ntaps=193):
        n = np.arange(ntaps)
        m = (ntaps - 1) / 2.0
        h = np.sinc(2.0 * fc / fs * (n - m)) * np.hamming(ntaps)
        h /= np.sum(h)
        return h.astype(np.float64)

    def to_baseband(self, iq):
        """Mix the pedestal to DC, low-pass, decimate to fs_bb."""
        iq = np.asarray(iq, dtype=np.complex128)
        n  = np.arange(len(iq))
        x  = iq * np.exp(-1j * 2.0 * np.pi * self.f_ped * n / self.fs)
        x  = np.convolve(x, self._lpf, mode="same")
        return x[::self.decim]

    def acquire(self, x_bb, center=0.0, search_hz=None, step_hz=None):
        """Coherent preamble matched filter over a carrier-offset search.

        The full 20-symbol preamble is correlated coherently (sharp peak, the
        whole ~13 dB of preamble gain) via FFT, so the frequency step has to be
        fine (~0.5 Hz) — FFT correlation keeps that affordable. `center` and
        `search_hz` allow a narrow re-search once a link is locked.

        Returns {found, start, df_hz, peak, noise, snr_db}; `start` is the
        baseband index of the first preamble symbol.
        """
        tmpl = self._pre_template
        P = len(tmpl)
        if len(x_bb) < P + self.sps_bb:
            return {"found": False, "start": 0, "df_hz": 0.0,
                    "peak": 0.0, "noise": 1.0, "snr_db": -99.0}

        search = self.search_hz if search_hz is None else float(search_hz)
        step   = self.step_hz   if step_hz   is None else float(step_hz)

        L = len(x_bb)
        nfft = 1
        while nfft < L + P:                          # avoid circular wraparound
            nfft <<= 1
        metric_len = L - P + 1
        t = np.arange(L)
        Tconj = np.conj(np.fft.fft(tmpl, nfft))      # matched-filter kernel
        # Per-symbol energy normaliser so peak ~ amplitude (not P-scaled).
        norm = 1.0 / P

        best = None
        df = center - search
        while df <= center + search + 0.5 * step:
            y    = x_bb * np.exp(-1j * 2.0 * np.pi * df * t / self.fs_bb)
            corr = np.fft.ifft(np.fft.fft(y, nfft) * Tconj)[:metric_len]
            mag  = np.abs(corr) * norm
            pk_i = int(np.argmax(mag))
            pk   = float(mag[pk_i])
            if best is None or pk > best[0]:
                best = (pk, df, pk_i, mag)
            df += step

        peak, df_hz, start, mag = best
        noise = float(np.median(mag)) + 1e-12
        snr_db = 20.0 * np.log10(peak / noise + 1e-12)
        return {"found": True, "start": start, "df_hz": df_hz,
                "peak": peak, "noise": noise, "snr_db": snr_db}

    def to_baseband_fast(self, iq):
        """Cheap baseband for the *display* only: mix the pedestal to DC and
        boxcar-decimate (mean of `decim` samples) — ~100x lighter than the FIR
        to_baseband. Good enough for the correlation display; the demod path
        still uses the proper FIR to_baseband."""
        iq = np.asarray(iq, dtype=np.complex64)
        n = np.arange(len(iq))
        x = iq * np.exp(-1j * 2.0 * np.pi * self.f_ped * n / self.fs).astype(np.complex64)
        L = (len(x) // self.decim) * self.decim
        if L == 0:
            return np.zeros(0, dtype=np.complex64)
        return x[:L].reshape(-1, self.decim).mean(axis=1)

    def corr_db(self, x_bb, df=0.0):
        """Single-frequency preamble matched-filter profile for the display.

        Returns (lag_ms, mag_db, peak_db): the correlation magnitude in dB above
        the noise floor across the window. With no signal it's a flat noisy
        baseline; once a synced preamble is present a peak rises whose height IS
        the realized processing gain (detection SNR)."""
        tmpl = self._pre_template
        P = len(tmpl)
        L = len(x_bb)
        if L < P + self.sps_bb:
            return None, None, -99.0
        nfft = 1
        while nfft < L + P:
            nfft <<= 1
        t = np.arange(L)
        y = x_bb * np.exp(-1j * 2.0 * np.pi * df * t / self.fs_bb)
        corr = np.fft.ifft(np.fft.fft(y, nfft) *
                           np.conj(np.fft.fft(tmpl, nfft)))[:L - P + 1]
        mag = np.abs(corr) / P
        noise = float(np.median(mag)) + 1e-12
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12) / noise)
        lag_ms = np.arange(len(mag)) / self.fs_bb * 1e3
        return lag_ms, mag_db, float(mag_db.max())

    def demod_symbols(self, x_bb, start, df_hz, n_syms):
        """Rectangular matched filter -> n_syms complex per-symbol values."""
        t = np.arange(len(x_bb))
        y = x_bb * np.exp(-1j * 2.0 * np.pi * df_hz * t / self.fs_bb)
        sps = self.sps_bb
        out = np.zeros(n_syms, dtype=np.complex128)
        for k in range(n_syms):
            a = start + k * sps
            b = a + sps
            if b > len(y):
                out = out[:k]
                break
            out[k] = np.sum(y[a:b])
        return out

    def decode(self, x_bb, acq):
        """Full decode after acquisition. Returns dict or None on failure.

        dict: {burst_code, payload (bytes), n_payload, ok}
        """
        if not acq["found"]:
            return None
        # Read enough symbols to cover preamble + header, then the payload.
        avail = (len(x_bb) - acq["start"]) // self.sps_bb
        if avail < self.npre + 4 + 8 + 1:
            return None

        syms = self.demod_symbols(x_bb, acq["start"], acq["df_hz"], int(avail))
        if len(syms) < self.npre + 12:
            return None

        # Differential decode every symbol after the preamble.
        # data bit k uses syms[npre-1 + k] vs its predecessor.
        bits = []
        for k in range(self.npre, len(syms)):
            d = syms[k] * np.conj(syms[k - 1])
            bits.append(1 if d.real < 0 else 0)

        if len(bits) < 12:
            return None
        burst_code = _bits_to_int(bits[0:4])
        n_payload  = _bits_to_int(bits[4:12])
        need_bits  = 12 + 8 * n_payload + 8       # + checksum
        if len(bits) < need_bits:
            # Not all payload/checksum symbols were captured.
            return {"burst_code": burst_code, "payload": b"",
                    "n_payload": n_payload, "ok": False}

        payload = bytearray()
        for i in range(n_payload):
            base = 12 + 8 * i
            payload.append(_bits_to_int(bits[base:base + 8]) & 0xFF)
        rx_sum = _bits_to_int(bits[12 + 8 * n_payload:20 + 8 * n_payload])
        ok = (rx_sum == self._checksum(burst_code, payload))
        return {"burst_code": burst_code, "payload": bytes(payload),
                "n_payload": n_payload, "ok": ok}

    def receive(self, iq, center=0.0, search_hz=None, step_hz=None):
        """Convenience: baseband -> acquire -> decode. Returns (acq, decoded).
        Pass a narrow center/search_hz once the offset is known (locked) to skip
        the full carrier sweep — far cheaper per cycle."""
        x_bb = self.to_baseband(iq)
        acq  = self.acquire(x_bb, center=center, search_hz=search_hz, step_hz=step_hz)
        dec  = self.decode(x_bb, acq) if acq["found"] else None
        return acq, dec
