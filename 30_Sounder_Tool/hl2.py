#!/usr/bin/env python3
"""
hl2.py - minimal Hermes-Lite 2 Protocol-1 interface for the sounder.

Self-contained (protocol verbatim from the example code).  Adds an HL2 class
that streams an ARBITRARY complex IQ waveform (the sounding signal, looped) and
captures N seconds of RX IQ.
"""

import gc
import socket
import struct
import threading
import time

import numpy as np

HPSDR_PORT      = 1024
# Protocol-1 TX DUC is FIXED at 48 ksps; only RX speed varies.  Run RX at 48k too
# so transmitted waveforms and the matched-filter references are at the same rate
# (any higher RX rate plays our TX waveforms at half speed -> nothing correlates).
SAMPLE_RATE     = 48_000
PACKET_SAMPLES  = 126
PACKET_INTERVAL = PACKET_SAMPLES / SAMPLE_RATE

# Protocol speed bits (C1[1:0]) kept consistent with SAMPLE_RATE.
_SPEED_BITS = {48_000: 0b00, 96_000: 0b01, 192_000: 0b10, 384_000: 0b11}


def freq_to_word(freq_hz):
    return int(freq_hz) & 0xFFFF_FFFF


def lpf_c2(freq_hz):
    if   freq_hz <  2_500_000: code = 0b000_0001
    elif freq_hz <  5_500_000: code = 0b100_0010
    elif freq_hz <  9_500_000: code = 0b100_0100
    elif freq_hz < 16_000_000: code = 0b100_1000
    elif freq_hz < 22_000_000: code = 0b101_0000
    else:                      code = 0b110_0000
    return (code & 0x7F) << 1


def build_ep2(seq, cc0, cc1, tx_iq):
    pkt = bytearray(1032)
    pkt[0:4] = b"\xEF\xFE\x01\x02"
    struct.pack_into(">I", pkt, 4, seq & 0xFFFF_FFFF)
    for blk, cc in enumerate((cc0, cc1)):
        base = 8 + blk * 512
        pkt[base:base + 3]     = b"\x7F\x7F\x7F"
        pkt[base + 3:base + 8] = cc
        for i in range(63):
            idx = blk * 63 + i
            Iv  = tx_iq[idx][0] if idx < len(tx_iq) else 0
            Qv  = tx_iq[idx][1] if idx < len(tx_iq) else 0
            struct.pack_into(">hhhh", pkt, base + 8 + i * 8, 0, 0, Iv, Qv)
    return bytes(pkt)


def pack_waveform(iq):
    """Pre-pack complex IQ into EP2 sample bytes (8 B/sample: 0,0,I,Q big-endian),
    with a 1-packet wrap pad so any 126-sample window is a contiguous slice.
    Done once per waveform so the TX hot loop never touches per-sample packing."""
    iq = np.asarray(iq)
    I = np.clip(np.real(iq) * 32767.0, -32767, 32767).astype(">i2")
    Q = np.clip(np.imag(iq) * 32767.0, -32767, 32767).astype(">i2")
    n = len(iq)
    arr = np.zeros((n, 4), dtype=">i2")
    arr[:, 2] = I; arr[:, 3] = Q
    packed = arr.tobytes()                       # n * 8 bytes
    return packed + packed[:PACKET_SAMPLES * 8], n


def build_ep2_fast(seq, cc0, cc1, iqb):
    """EP2 packet from a pre-packed 1008-byte IQ block (126 samples) - no per-sample work."""
    pkt = bytearray(1032)
    pkt[0:4] = b"\xEF\xFE\x01\x02"
    struct.pack_into(">I", pkt, 4, seq & 0xFFFF_FFFF)
    for blk, cc in enumerate((cc0, cc1)):
        base = 8 + blk * 512
        pkt[base:base + 3]     = b"\x7F\x7F\x7F"
        pkt[base + 3:base + 8] = cc
        pkt[base + 8:base + 8 + 504] = iqb[blk * 504:blk * 504 + 504]
    return bytes(pkt)


def splice_silence(iq, min_run=240):
    """Excise long near-zero runs (inserted TX-FIFO-underrun silence) and rejoin
    the waveform.  Only valid when no RX packets were lost (else a zero run is a
    real-loss zero-fill that must stay).  Returns (iq, n_spliced)."""
    if len(iq) < min_run * 2:
        return iq, 0
    amp = np.abs(iq)
    med = np.median(amp)
    if med <= 0:
        return iq, 0
    low = amp < 0.2 * med
    if not low.any():
        return iq, 0
    d = np.diff(low.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1
    if low[0]:  starts = np.insert(starts, 0, 0)
    if low[-1]: ends = np.append(ends, len(low))
    keep = np.ones(len(iq), dtype=bool)
    spliced = 0
    for s, e in zip(starts, ends):
        if e - s >= min_run:                       # long silence -> excise
            keep[s:e] = False
            spliced += e - s
    return (iq[keep], spliced) if spliced else (iq, 0)


def parse_ep6(data):
    if len(data) < 1032 or data[0:2] != b"\xEF\xFE":
        return []
    samples = []
    for blk in range(2):
        base = 8 + blk * 512 + 8
        for i in range(63):
            pos   = base + i * 8
            I_raw = int.from_bytes(data[pos    :pos + 3], "big", signed=True)
            Q_raw = int.from_bytes(data[pos + 3:pos + 6], "big", signed=True)
            samples.append((I_raw / 8_388_608.0, Q_raw / 8_388_608.0))
    return samples


def make_cc(freq_word, ptt=False, pa_enable=False, lna_gain_db=20,
            tx_drive=0x0F, duplex=False):
    mox = 0x01 if ptt else 0x00
    nco = struct.pack(">I", freq_word)
    cc_cfg = bytes([(0x00 << 1) | mox, 0x10 | _SPEED_BITS[SAMPLE_RATE],
                    lpf_c2(freq_word), 0x00, 0x04 if duplex else 0x00])
    cc_tx_nco = bytes([(0x01 << 1) | mox]) + nco
    cc_rx_nco = bytes([(0x02 << 1) | mox]) + nco
    cc_pa     = bytes([(0x09 << 1) | mox,
                       ((int(tx_drive) & 0x0F) << 4) if ptt else 0x00,
                       0x08 if (ptt and pa_enable) else 0x00, 0x00, 0x00])
    lna_code  = 0x40 | ((int(lna_gain_db) + 12) & 0x3F)
    cc_lna    = bytes([(0x0a << 1) | mox, 0x00, 0x00, 0x00, lna_code])
    return (cc_cfg, cc_tx_nco, cc_rx_nco, cc_pa, cc_lna)


def connect(sock, ip):
    probe     = b"\xEF\xFE\x02" + bytes(60)
    stop_cmd  = b"\xEF\xFE\x04\x00" + bytes(60)
    start_cmd = b"\xEF\xFE\x04\x01" + bytes(60)
    sock.settimeout(0.5)
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    time.sleep(0.3)
    while True:
        try:    sock.recvfrom(2048)
        except socket.timeout: break
    for _ in range(40):
        sock.sendto(probe, (ip, HPSDR_PORT))
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) >= 10 and data[0:2] == b"\xEF\xFE" and data[2] in (0x02, 0x03):
                break
        except socket.timeout:
            pass
    else:
        raise RuntimeError("HL2 not found - check cable and power")
    sock.sendto(stop_cmd,  (ip, HPSDR_PORT))
    time.sleep(0.2)
    sock.sendto(start_cmd, (ip, HPSDR_PORT))
    cc_cfg = make_cc(freq_to_word(7_000_000))[0]
    silent = build_ep2(0, cc_cfg, cc_cfg, [(0, 0)] * PACKET_SAMPLES)
    deadline  = time.monotonic() + 5.0
    ep6_count = 0
    seq       = 0
    while time.monotonic() < deadline:
        pkt = bytearray(silent)
        struct.pack_into(">I", pkt, 4, seq)
        sock.sendto(bytes(pkt), (ip, HPSDR_PORT))
        seq += 1
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) == 1032 and data[0:2] == b"\xEF\xFE":
                ep6_count += 1
                if ep6_count >= 3:
                    return seq
        except socket.timeout:
            pass
        time.sleep(0.010)
    raise RuntimeError("No EP6 after START - check firewall (UDP 1024)")


def init_radio(sock, ip, freq_word, start_seq, ptt=False, pa_enable=False,
               lna_gain_db=20, tx_drive=0x0F, duplex=False):
    cc_regs = make_cc(freq_word, ptt, pa_enable, lna_gain_db, tx_drive, duplex)
    for i, cc in enumerate(cc_regs):
        sock.sendto(build_ep2(start_seq + i, cc, cc, [(0, 0)] * PACKET_SAMPLES),
                    (ip, HPSDR_PORT))
        time.sleep(0.020)
    return start_seq + len(cc_regs)


def iq_to_int16(iq):
    """Complex float (|.|<=1) -> list of (I,Q) int16 for build_ep2."""
    I = np.clip(np.real(iq) * 32767.0, -32767, 32767).astype(np.int16)
    Q = np.clip(np.imag(iq) * 32767.0, -32767, 32767).astype(np.int16)
    return list(zip(I.tolist(), Q.tolist()))


class HL2:
    """Connect, stream a looped sounding waveform, and capture RX IQ."""

    def __init__(self, ip, nco_hz, lna_gain_db=20):
        self.ip = ip
        self.nco_hz = nco_hz
        self.lna_gain_db = lna_gain_db
        self.sock = None
        self.tx_sock = None
        self._seq = 0
        self._tx_run = threading.Event()
        self._tx_packed_ext = bytes((PACKET_SAMPLES + PACKET_SAMPLES) * 8)  # pre-packed IQ
        self._tx_n = PACKET_SAMPLES                   # current waveform length (samples)
        self._tx_lock = threading.Lock()
        self._ptt = self._pa = self._duplex = False
        self.tx_drive = 15                          # HL2 drive register 0-15
        self.tone_on = False                        # stream waveform vs silence
        self._cc_lock = threading.Lock()            # serialise concurrent applies
        self._cc = make_cc(freq_to_word(nco_hz), lna_gain_db=lna_gain_db)
        self.last_loss = {"recv": 0, "lost": 0, "pct": 0.0}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def open(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", HPSDR_PORT))
        self._seq = connect(self.sock, self.ip)
        self.sock.settimeout(0.002)
        self._apply_cc()
        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tx_sock.bind(("", 0))
        self._tx_run.set()
        threading.Thread(target=self._tx_loop, daemon=True, name="HL2-TX").start()

    def close(self):
        self._tx_run.clear()
        time.sleep(0.2)
        if self.sock:
            stop = b"\xEF\xFE\x04\x00" + bytes(60)
            self.sock.sendto(stop, (self.ip, HPSDR_PORT))
            self.sock.close()
        if self.tx_sock:
            self.tx_sock.close()

    # ── control ──────────────────────────────────────────────────────────────
    def _apply_cc(self):
        with self._cc_lock:
            self._cc = make_cc(freq_to_word(self.nco_hz), self._ptt, self._pa,
                               self.lna_gain_db, self.tx_drive, self._duplex)
            init_radio(self.sock, self.ip, freq_to_word(self.nco_hz), 0,
                       self._ptt, self._pa, self.lna_gain_db, self.tx_drive,
                       self._duplex)

    def set_freq(self, hz):
        self.nco_hz = hz;  self._apply_cc()

    def set_lna(self, db):
        self.lna_gain_db = db;  self._apply_cc()

    def set_tx_drive(self, v):
        self.tx_drive = max(0, min(15, int(v)));  self._apply_cc()

    def set_tone(self, on):
        """Gate the sounding waveform: True streams it, False streams silence."""
        self.tone_on = bool(on)

    def set_tx(self, ptt=None, pa=None, duplex=None):
        if ptt is not None:    self._ptt = ptt
        if pa is not None:     self._pa = pa
        if duplex is not None: self._duplex = duplex
        self._apply_cc()

    def set_waveform(self, iq):
        """Set the looped TX waveform (complex float, peak <= 1.0).  Pre-packs the
        whole waveform here (off the hot loop) so streaming can't stall/underrun."""
        ext, n = pack_waveform(np.asarray(iq))
        with self._tx_lock:
            self._tx_packed_ext = ext
            self._tx_n = n

    # ── TX streaming thread (loops the waveform) ─────────────────────────────
    def _tx_loop(self):
        seq = self._seq
        idx = 0
        next_send = time.monotonic()
        silence = bytes(1008)                        # pre-packed zeros (126 samples)
        while self._tx_run.is_set():
            with self._tx_lock:
                ext = self._tx_packed_ext            # pre-packed -> just slice bytes
                n = self._tx_n
                cc = self._cc
            if self.tone_on:                         # Tone button gates the waveform
                off = (idx % n) * 8
                iqb = ext[off:off + 1008]
                idx = (idx + PACKET_SAMPLES) % n
            else:
                iqb = silence                        # silence (keep-alive)
            cc0 = cc[seq % len(cc)]
            cc1 = cc[(seq + 1) % len(cc)]
            try:
                self.tx_sock.sendto(build_ep2_fast(seq, cc0, cc1, iqb),
                                    (self.ip, HPSDR_PORT))
            except OSError:
                break
            seq += 1
            next_send += PACKET_INTERVAL
            wait = next_send - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    # ── RX capture ───────────────────────────────────────────────────────────
    def grab(self, seconds):
        """
        Capture `seconds` of RX IQ -> complex64 array.

        Detects lost EP6 packets via the sequence number and ZERO-FILLS the gaps
        so the sample time grid stays correct (splicing gaps out would shift the
        periodic sounding structure and corrupt delay/Doppler).  Loss stats are
        stored in self.last_loss and a warning is printed.
        """
        n = int(round(SAMPLE_RATE * seconds))
        npkt = n // PACKET_SAMPLES + 2
        # Disable the cyclic GC during the capture: the grab allocates ~240k
        # tuples, and a GC scan-pause (it holds the GIL) would stall the TX
        # streaming thread -> HL2 TX FIFO underrun -> ~20 ms silence burst.
        # Safe: these tuples are reclaimed by reference counting, not the GC.
        gc.disable()
        try:
            t_end = time.monotonic() + 0.3
            while time.monotonic() < t_end:            # flush stale
                try:    self.sock.recvfrom(2048)
                except socket.timeout: pass

            # Collect packets keyed by EP6 sequence number so out-of-order and
            # duplicate datagrams can't shift the timeline; assemble in order.
            pkts = {}
            prev_seq = None
            reorder = dup = 0
            deadline = time.monotonic() + seconds * 3.0 + 5.0
            while len(pkts) < npkt and time.monotonic() < deadline:
                try:
                    data, _ = self.sock.recvfrom(2048)
                except socket.timeout:
                    continue
                if len(data) != 1032 or data[0:2] != b"\xEF\xFE":
                    continue
                seq = int.from_bytes(data[4:8], "big")
                if seq in pkts:
                    dup += 1
                    continue
                if prev_seq is not None and ((seq - prev_seq) & 0xFFFF_FFFF) > 0x8000_0000:
                    reorder += 1                       # arrived after a higher seq
                prev_seq = seq
                pkts[seq] = data

            # assemble in sequence order from the earliest packet, zero-fill gaps
            samples = []
            recv = lost = 0
            seq = min(pkts) if pkts else 0
            while len(samples) < n:
                d = pkts.get(seq)
                if d is not None:
                    samples.extend(parse_ep6(d)); recv += 1
                else:
                    samples.extend([(0.0, 0.0)] * PACKET_SAMPLES); lost += 1
                seq = (seq + 1) & 0xFFFF_FFFF

            total = recv + lost
            pct = 100.0 * lost / total if total else 0.0
            self.last_loss = {"recv": recv, "lost": lost, "pct": pct,
                              "reorder": reorder, "dup": dup}
            if lost or reorder or dup:
                print(f"WARNING: lost={lost} ({pct:.2f}%)  reordered={reorder}  "
                      f"duplicate={dup}  - timeline preserved (zero-filled)")

            arr = np.asarray(samples[:n], dtype=np.float64)
        finally:
            gc.enable()

        if len(arr) == 0:
            return np.zeros(0, dtype=np.complex64)
        iq = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
        # Gap-splice recovery: with no RX loss, a long near-zero run is inserted
        # TX-silence (FIFO underrun) -> excise it to rejoin the waveform.
        self.last_loss["spliced"] = 0
        if lost == 0:
            iq, spliced = splice_silence(iq)
            if spliced:
                self.last_loss["spliced"] = spliced
                print(f"INFO: spliced {spliced} samples ({spliced/SAMPLE_RATE*1e3:.1f} ms) "
                      f"of inserted TX silence - waveform rejoined")
        return iq
