#!/usr/bin/env python3
"""
sim.py — no-hardware two-station simulation.

Stands in for the HL2 + the over-the-air link so a full Call<->Detect exchange
can be watched in the GUI (or run headless as an integration test) without any
radio. A `SimMedium` is a virtual RF channel shared by two `RadioEngine`s; each
engine gets a `FakeSocket` instead of a real UDP socket.

    python main.py --sim                 # two GUI windows, auto Call + Detect
    python main.py --simtest             # ~16 s headless exchange, prints events

The medium models:
    * half-duplex coupling — each station receives the *other* station's
      transmitted I/Q (silence + noise when the partner isn't keying),
    * a fixed inter-radio carrier offset (default 37 Hz),
    * additive white Gaussian noise at a chosen channel SNR (2.5 kHz ref),
    * just enough forward-power telemetry to drive the PEP readout while keyed.

Timing is real-time: the fake RX socket hands out one EP6 packet per
PACKET_INTERVAL, so the engines' 5-second cadence runs at wall-clock speed.
"""

import socket
import struct
import threading
import time

import numpy as np

import hl2_transport as hl2
import radio_engine as eng


FS = eng.SAMPLE_RATE
PKT = hl2.PACKET_SAMPLES
PACKET_INTERVAL = PKT / FS

# Wall-clock time scale for the simulated radios. 1.0 = real time (best demo).
# Raise above 1.0 only if the host can't keep two full radios fed in one
# process (you'll see dropped cycles in the logs). Real hardware always uses 1.0.
SIM_TIME_SCALE = 2.0


def _s24(v):
    iv = int(round(float(v) * 8388607.0))
    iv = max(-8388608, min(8388607, iv))
    return (iv & 0xFFFFFF).to_bytes(3, "big")


def _build_fake_ep6(seq, iq, fwd_adc, rev_adc, cur_adc):
    """Build a 1032-byte EP6 (radio->host) frame, 24-bit I/Q + telemetry."""
    pkt = bytearray(1032)
    pkt[0:4] = b"\xEF\xFE\x01\x06"
    struct.pack_into(">I", pkt, 4, seq & 0xFFFF_FFFF)
    for blk in range(2):
        hbase = 8 + blk * 512
        pkt[hbase:hbase + 3] = b"\x7F\x7F\x7F"
        if blk == 0:                       # RADDR 1: C3:C4 = forward power
            pkt[hbase + 3] = 0x01 << 3
            pkt[hbase + 6] = (fwd_adc >> 8) & 0xFF
            pkt[hbase + 7] = fwd_adc & 0xFF
        else:                              # RADDR 2: C1:C2 rev, C3:C4 current
            pkt[hbase + 3] = 0x02 << 3
            pkt[hbase + 4] = (rev_adc >> 8) & 0xFF
            pkt[hbase + 5] = rev_adc & 0xFF
            pkt[hbase + 6] = (cur_adc >> 8) & 0xFF
            pkt[hbase + 7] = cur_adc & 0xFF
        sbase = hbase + 8
        for i in range(63):
            idx = blk * 63 + i
            s = iq[idx] if idx < len(iq) else 0.0
            pos = sbase + i * 8
            pkt[pos:pos + 3]     = _s24(s.real)
            pkt[pos + 3:pos + 6] = _s24(s.imag)
    return bytes(pkt)


def _parse_tx_iq(data):
    """Extract complex I/Q from a host->radio EP2 packet (16-bit format)."""
    out = np.zeros(PKT, dtype=np.complex64)
    for blk in range(2):
        sbase = 8 + blk * 512 + 8
        for i in range(63):
            pos = sbase + i * 8
            _, _, I, Q = struct.unpack(">hhhh", data[pos:pos + 8])
            out[blk * 63 + i] = (I / 32768.0) + 1j * (Q / 32768.0)
    return out


class SimMedium:
    def __init__(self, offset_hz=37.0, snr_db=-8.0, amplitude=eng.TX_AMPLITUDE):
        self.offset = {0: +offset_hz, 1: -offset_hz}
        self.air    = {0: [], 1: []}          # station s -> samples it transmitted
        self.tx_at  = {0: 0.0, 1: 0.0}        # last time station s keyed nonzero
        self.rx_n   = {0: 0, 1: 0}            # rx sample counter (offset rotation)
        self.seq    = {0: 0, 1: 0}            # delivered-block count = sample clock
        self.lock   = threading.Lock()
        # Model realistic RX levels: a real receiver sets gain so signal+noise
        # fit inside the ADC's +/-1.0 full scale. Pick a small noise floor with
        # headroom, then scale the (full-amplitude) transmitted signal down to
        # whatever gives the requested channel SNR (2.5 kHz ref).
        self.noise_std = 0.05                 # per-component RX noise (fits ADC)
        snr_lin = 10 ** (snr_db / 10.0)
        a_rx = np.sqrt(snr_lin * 2.0 * self.noise_std ** 2 * 2500.0 / FS)
        a_rx = min(a_rx, 0.6)                  # cap so the signal fits the ADC
        self.sig_scale = a_rx / float(amplitude)

    # ── called from a station's TX sends ──────────────────────────────────────
    def transmit(self, sid, data):
        if len(data) != 1032 or data[0:2] != b"\xEF\xFE":
            return
        iq = _parse_tx_iq(data)
        with self.lock:
            if np.any(np.abs(iq) > 1e-4):
                self.tx_at[sid] = time.monotonic()
            self.air[sid].extend(iq.tolist())
            if len(self.air[sid]) > FS * 4:
                self.air[sid] = self.air[sid][-FS * 2:]

    # ── called from a station's RX reads ──────────────────────────────────────
    def receive(self, sid):
        now = time.monotonic()
        with self.lock:
            partner = 1 - sid
            buf = self.air[partner]
            # Deliver the partner's transmitted stream in contiguous full blocks
            # as fast as the consumer asks, until it is caught up to production
            # (then there is nothing new -> timeout). The transmit threads are
            # real-time paced, so "production" tracks the wall clock; a consumer
            # that stalls simply drains the backlog and catches its sample clock
            # back up to real time. NO rate-limit here: rate-limiting delivery is
            # what let the cadence clock slip behind real time under load.
            if len(buf) < PKT:
                raise socket.timeout()
            chunk = np.array(buf[:PKT], dtype=np.complex64)
            del buf[:PKT]

            n0 = self.rx_n[sid]
            self.rx_n[sid] += PKT
            keyed = (now - self.tx_at[sid]) < 0.15
            seq = self.seq[sid]
            self.seq[sid] += 1
            off = self.offset[sid]
            std = self.noise_std

        idx = np.arange(n0, n0 + PKT)
        rx = chunk * self.sig_scale * np.exp(1j * 2 * np.pi * off * idx / FS)
        rx = rx + (np.random.randn(PKT) + 1j * np.random.randn(PKT)) * std
        fwd = 2000 if keyed else 0          # ~3 W on the HL2 power table
        rev = 60 if keyed else 0
        cur = 1500 if keyed else 0
        pkt = _build_fake_ep6(seq, rx, fwd, rev, cur)
        return pkt, ("sim", 0)


class FakeSocket:
    """Duck-typed UDP socket backed by a SimMedium (one per station)."""

    def __init__(self, medium, sid):
        self.medium = medium
        self.sid = sid

    def sendto(self, data, _addr):
        self.medium.transmit(self.sid, data)
        return len(data)

    def recvfrom(self, _bufsize):
        return self.medium.receive(self.sid)

    def settimeout(self, _t):        pass
    def setsockopt(self, *_a):       pass
    def bind(self, _a):              pass
    def close(self):                 pass


def build_sim_pair(offset_hz=37.0, snr_db=-8.0):
    """Create two engines wired through a shared medium. No real sockets."""
    eng.TIME_SCALE = SIM_TIME_SCALE          # run the radios slower than real time
    medium = SimMedium(offset_hz=offset_hz, snr_db=snr_db)
    engines = []
    for sid in range(2):
        e = eng.RadioEngine(log_path=f"log{'AB'[sid]}.txt")
        sock = FakeSocket(medium, sid)
        e._sock = sock
        e._tx_sock = sock
        e._seq = 0
        e._push_cc(ptt=False)        # populate cc registers (sends harmless idle)
        engines.append(e)
    return engines[0], engines[1], medium


# ── headless integration test ────────────────────────────────────────────────

def run_simtest(seconds=None, snr_db=-8.0, offset_hz=37.0):
    import queue as _q
    # Radio time runs SIM_TIME_SCALE x slower than wall-clock, so scale the
    # wall-clock test window and the text-injection time to match.
    seconds = (30.0 * SIM_TIME_SCALE) if seconds is None else seconds
    text_at = 16.0 * SIM_TIME_SCALE
    print(f"Sim test: 2 stations, offset {offset_hz:+.0f} Hz, "
          f"channel SNR {snr_db:+.0f} dB (2.5 kHz ref), "
          f"time scale {SIM_TIME_SCALE:.0f}x ({seconds:.0f}s wall-clock)\n")
    A, B, _medium = build_sim_pair(offset_hz, snr_db)
    A.run()
    B.run()

    def drain(tag, e):
        out = []
        try:
            while True:
                ev = e.events.get_nowait()
                if ev[0] in ("spectrum", "pep"):
                    continue
                out.append((tag, ev))
        except _q.Empty:
            pass
        return out

    log = []
    A.start_call()
    B.start_detect()
    t0 = time.monotonic()
    sent_text = False
    while time.monotonic() - t0 < seconds:
        time.sleep(0.1)
        log += drain("A", A)
        log += drain("B", B)
        # Send text only once the link is established (B locks after its ~7.5
        # radio-second search; allow a couple of reply cycles first).
        if not sent_text and time.monotonic() - t0 > text_at:
            A.send_text("HI")        # fits one burst at 40 baud
            sent_text = True
    A.shutdown(); B.shutdown()
    time.sleep(0.2)
    log += drain("A", A) + drain("B", B)

    for tag, ev in log:
        print(f"  [{tag}] {ev[0]:<14} {ev[1:] if len(ev) > 1 else ''}")
    # crude pass check
    b_linked = any(t == "B" and ev[0] == "status" and "offset" in str(ev[1])
                   for t, ev in log)
    a_got = any(t == "A" and ev[0] in ("rx_beacon", "offset") for t, ev in log)
    b_text = any(t == "B" and ev[0] == "rx_text" and "HI" in str(ev[1])
                 for t, ev in log)
    print(f"\n  B locked to caller : {b_linked}")
    print(f"  A heard responder  : {a_got}")
    print(f"  B received 'HI'    : {b_text}")
