#!/usr/bin/env python3
"""
hl2_transport.py — Hermes-Lite 2 Protocol-1 (HPSDR / Metis) UDP transport.

Adapted (largely verbatim) from the working reference `05_Reference/main_ex1.py`.
This module has **no modem logic** — it only knows how to:

    * find and start an HL2 (`connect`)
    * build EP2 (host -> radio) packets carrying I/Q + control (`build_ep2`)
    * parse EP6 (radio -> host) packets back into I/Q (`parse_ep6`)
    * assemble the five control-register frames (`make_cc`)
    * extract forward/reverse-power + PA-current telemetry (`parse_ep6_power`)

The only change from the reference is that the sample rate is configurable
(48/96/192/384 kHz via the C1 speed bits) so the app can run the HL2 at 96 kHz.

Protocol-1 fixed facts (do not change):
    * EP2 / EP6 frames are 1032 bytes, magic 0xEFFE.
    * Each frame carries 2 blocks of 63 I/Q samples => 126 samples/packet.
    * I/Q are 24-bit big-endian signed; control sits in the 5 "C&C" bytes.
"""

import socket
import struct
import time

# ── Fixed Protocol-1 constants ───────────────────────────────────────────────
HPSDR_PORT     = 1024
PACKET_SAMPLES = 126               # samples per EP2/EP6 frame (fixed by Protocol 1)

# Speed bits (C1[1:0]) -> sample rate
_SPEED_BITS = {48_000: 0b00, 96_000: 0b01, 192_000: 0b10, 384_000: 0b11}


def speed_bits_for(sample_rate):
    try:
        return _SPEED_BITS[int(sample_rate)]
    except KeyError:
        raise ValueError(f"Unsupported HL2 sample rate {sample_rate}; "
                         f"use one of {sorted(_SPEED_BITS)}")


# ── Control helpers ──────────────────────────────────────────────────────────

def freq_to_word(freq_hz):
    return int(freq_hz) & 0xFFFF_FFFF


def lpf_c2(freq_hz):
    """Low-pass-filter board selection by band (Quisk HL2FilterE3 mapping)."""
    if   freq_hz <  2_500_000: code = 0b000_0001
    elif freq_hz <  5_500_000: code = 0b100_0010
    elif freq_hz <  9_500_000: code = 0b100_0100
    elif freq_hz < 16_000_000: code = 0b100_1000
    elif freq_hz < 22_000_000: code = 0b101_0000
    else:                      code = 0b110_0000
    return (code & 0x7F) << 1


def make_cc(freq_word, ptt=True, pa_enable=True, lna_gain_db=20, tx_drive=0x0F,
            sample_rate=96_000):
    """Build the five control-register (C&C) frames the radio cycles through.

    Returns (cc_cfg, cc_tx_nco, cc_rx_nco, cc_pa, cc_lna) — each 5 bytes.
    """
    mox = 0x01 if ptt else 0x00
    nco = struct.pack(">I", freq_word)
    spd = speed_bits_for(sample_rate)

    cc_cfg = bytes([
        (0x00 << 1) | mox,
        0x10 | spd,                 # C1: bit4 misc + speed bits [1:0]
        lpf_c2(freq_word),
        0x00,
        0x04,
    ])
    cc_tx_nco = bytes([(0x01 << 1) | mox]) + nco
    cc_rx_nco = bytes([(0x02 << 1) | mox]) + nco
    cc_pa     = bytes([
        (0x09 << 1) | mox,
        ((int(tx_drive) & 0x0F) << 4) if ptt else 0x00,   # C1[31:28] TX drive 0-15
        0x08 if (ptt and pa_enable) else 0x00,
        0x00, 0x00,
    ])
    lna_code  = 0x40 | ((int(lna_gain_db) + 12) & 0x3F)
    cc_lna    = bytes([(0x0a << 1) | mox, 0x00, 0x00, 0x00, lna_code])

    return (cc_cfg, cc_tx_nco, cc_rx_nco, cc_pa, cc_lna)


# ── EP2 build / EP6 parse ────────────────────────────────────────────────────

def build_ep2(seq, cc0, cc1, tx_iq):
    """Build a 1032-byte EP2 frame: seq + two 512-byte blocks of control+I/Q.

    `tx_iq` is a sequence of (I, Q) int16 pairs, up to PACKET_SAMPLES long.
    """
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


def parse_ep6(data):
    """Parse a 1032-byte EP6 frame into a list of (I, Q) floats in ±1.0."""
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


# ── Forward-power telemetry (AD7991) ─────────────────────────────────────────
HL2_PWR_TABLE = [
    [0, 0.0], [25.87, 0.00255], [101.02, 0.01275], [265.29, 0.05060],
    [647.92, 0.21646], [1196.59, 0.66548], [1603.70, 1.15572], [2012.33, 1.81189],
    [2616.77, 3.00858], [3173.82, 4.39274], [3382.79, 4.97913], [3721.07, 6.02475],
    [4093.18, 7.28995], [4502.50, 8.82084], [4952.75, 10.67321],
]


def adc_to_watts(adc):
    tbl = HL2_PWR_TABLE
    if adc <= tbl[0][0]:
        return 0.0
    for i in range(1, len(tbl)):
        a0, w0 = tbl[i - 1]
        a1, w1 = tbl[i]
        if adc <= a1:
            return w0 + (w1 - w0) * (adc - a0) / (a1 - a0)
    return tbl[-1][1]


def parse_ep6_power(data):
    """Extract (fwd_adc, rev_adc, cur_adc) from EP6 telemetry; None where absent."""
    fwd = rev = cur = None
    for blk in range(2):
        base  = 8 + blk * 512
        raddr = data[base + 3] >> 3          # C0[7:3]
        if   raddr == 0x01:
            fwd = (data[base + 6] << 8) | data[base + 7]   # C3:C4 forward power
        elif raddr == 0x02:
            rev = (data[base + 4] << 8) | data[base + 5]   # C1:C2 reverse power
            cur = (data[base + 6] << 8) | data[base + 7]   # C3:C4 PA current
    return fwd, rev, cur


def adc_to_current_ma(adc):
    """HL2 PA-current ADC -> milliamps (Quisk Code2Current)."""
    a = ((3.26 * (adc / 4096.0)) / 50.0) / 0.04
    a = a / (1000.0 / 1270.0)
    return a * 1000.0


def ep6_seq(data):
    """EP6 frame sequence number (used to detect/zero-fill dropped packets)."""
    return int.from_bytes(data[4:8], "big")


# ── Session bring-up ─────────────────────────────────────────────────────────

def connect(sock, ip, freq_word, sample_rate=96_000):
    """Discover the HL2, stop any stale session, START streaming, and wait for
    EP6 to flow. Returns the next EP2 sequence number to use."""
    probe     = b"\xEF\xFE\x02" + bytes(60)
    stop_cmd  = b"\xEF\xFE\x04\x00" + bytes(60)
    start_cmd = b"\xEF\xFE\x04\x01" + bytes(60)
    sock.settimeout(0.5)

    # Stop any leftover session and drain the socket
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    time.sleep(0.3)
    while True:
        try:    sock.recvfrom(2048)
        except socket.timeout: break

    # Wait for the HL2 to answer a discovery probe
    for _ in range(40):
        sock.sendto(probe, (ip, HPSDR_PORT))
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) >= 10 and data[0:2] == b"\xEF\xFE" and data[2] in (0x02, 0x03):
                break
        except socket.timeout:
            pass
    else:
        raise RuntimeError("HL2 not found — check cable and power")

    # Start streaming
    sock.sendto(stop_cmd,  (ip, HPSDR_PORT))
    time.sleep(0.2)
    sock.sendto(start_cmd, (ip, HPSDR_PORT))

    # Send silent EP2 packets until EP6 flows back
    cc_cfg = make_cc(freq_word, ptt=False, pa_enable=False,
                     sample_rate=sample_rate)[0]
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

    raise RuntimeError("No EP6 response after START — check firewall (UDP 1024)")


def init_radio(sock, ip, freq_word, start_seq, ptt=True, pa_enable=True,
               lna_gain_db=20, tx_drive=0x0F, sample_rate=96_000):
    """Push the five control-register frames once (cycled, silent I/Q)."""
    cc_regs = make_cc(freq_word, ptt=ptt, pa_enable=pa_enable,
                      lna_gain_db=lna_gain_db, tx_drive=tx_drive,
                      sample_rate=sample_rate)
    for i, cc in enumerate(cc_regs):
        sock.sendto(build_ep2(start_seq + i, cc, cc, [(0, 0)] * PACKET_SAMPLES),
                    (ip, HPSDR_PORT))
        time.sleep(0.020)
    return start_seq + len(cc_regs)


def stop(sock, ip):
    """Tell the HL2 to stop streaming."""
    stop_cmd = b"\xEF\xFE\x04\x00" + bytes(60)
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
