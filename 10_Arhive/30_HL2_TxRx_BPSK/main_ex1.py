#!/usr/bin/env python3
"""
main_ex1.py — HermesLite 2  spectrum analyser

Hardware
--------
    RF3_LOOPBACK = True  (default)
        PA off.  RF1 SMA → 30 dB attenuator → RF3 SMA.
        DAC drives RF1; no ANT radiation.
        Clean controlled RX test — 70 dB dynamic range confirmed.

    RF3_LOOPBACK = False
        PA on.  ANT transmits (~0 dBm with TX_AMPLITUDE = 0.450).
        Pure-Signal tap feeds PA output back to RX internally.
        Use for transmitter testing only.

Calibration
-----------
    With RF3_LOOPBACK = True and cable connected, inject a known signal
    at RF3, read the displayed peak, then set:
        DBFS_REF_DBM += (known_dBm - displayed_dBm)

Usage
-----
    python main_ex1.py
"""

import socket
import struct
import time
import threading

import numpy as np

from spectrum_ex1 import LiveSpectrum

# =============================================================================
#  Configuration  (copied from v04 working values)
# =============================================================================
HPSDR_PORT     = 1024
HL2_IP         = "169.254.19.221"
NCO_FREQ       = 7_000_000         # Hz
TONE_OFFSET    = 870               # Hz  — (unused in PSK mode; kept for reference)
TX_AMPLITUDE   = 0.800             # normalised  — digital level; slider/drive sets power
RX_LNA_GAIN_DB = 30                # dB  — AD9866 FAST_LNA
SAMPLE_RATE    = 48_000            # sps  (PSK mode: 48 kHz, HL2 speed bits = 00)
PACKET_SAMPLES = 126               # fixed by Protocol 1
PACKET_INTERVAL = PACKET_SAMPLES / SAMPLE_RATE

SEGMENT_SIZE   = 512               # Welch segment for the spectrum (<= CAPTURE_SIZE)
FFT_PAD_SIZE   = 1024              # zero-pad length

# ── PSK / DSSS (example 2) ──────────────────────────────────────────────────
CODE_LEN       = 1023              # code repeat period in samples (1 sample/chip; 2^10-1 m-seq)
NUM_PERIODS    = 1                 # code periods summed per update (1 = fastest; >1 = +SNR)
CAPTURE_SIZE   = NUM_PERIODS * CODE_LEN   # samples per correlation / display update
FREQ_SEARCH_HZ = 500.0            # +/- inter-radio carrier-offset search (7 MHz, low-cost xtals)
FREQ_STEP_HZ   = 10.0            # search step (Hz)
TRACK_HZ       = 40.0             # narrow +/- search once locked (follows slow drift)
LOCK_DB        = 25.0             # peak/noise (dB) to acquire frequency lock
DROP_DB        = 15.0             # peak/noise (dB) to hold lock (hysteresis)
MAX_MISS       = 8                # frames below DROP_DB before re-acquiring (full search)

DBFS_REF_DBM   = 8.0               # calibration offset (RX mode, ADC-referred / after LNA)
                                   #   cal: -50 dBm in @ LNA 0 -> displayed -50 dBm
                                   #   re-trim with: DBFS_REF_DBM += (known_dBm - displayed_dBm)

# Loopback mode
# -------------
# True  : PA off, DAC active on RF1.  Connect RF1 → attenuator → RF3.
#         No ANT radiation.  Clean controlled RX test path.
# False : PA on, ANT transmits.  Pure-Signal tap provides internal loopback.
#         Use when you need ANT output (e.g. transmitter testing).
RF3_LOOPBACK   = True

DEBUG_TELEM    = False             # print all EP6 status fields each sweep (telemetry diagnosis)

# =============================================================================
#  Protocol-1 helpers  (verbatim from v04)
# =============================================================================

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


# ── Forward-power telemetry (AD7991) ─────────────────────────────────────────
# HL2 reports ADC telemetry in the EP6 status bytes (round-robin by RADDR = C0>>3):
#   RADDR 0x01: C1:C2 = temperature, C3:C4 = forward power
#   RADDR 0x02: C1:C2 = reverse power, C3:C4 = supply current
# Calibration table (ADC code -> watts) from Quisk's "HL2FilterE3" filter board.
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
    """Extract (fwd_adc, rev_adc, cur_adc) from EP6 telemetry; None where absent.
    Verified against live HL2: R1 C3:C4 = fwd power, R2 C1:C2 = rev power,
    R2 C3:C4 = PA current, R1 C1:C2 = temperature."""
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
    """HL2 PA current ADC -> milliamps (Quisk Code2Current)."""
    a = ((3.26 * (adc / 4096.0)) / 50.0) / 0.04
    a = a / (1000.0 / 1270.0)
    return a * 1000.0


# ── DSSS code + matched filter (example 2) ───────────────────────────────────

def gen_code(pad_len=CODE_LEN):
    """Maximal-length LFSR (deg-10, primitive x^10 + x^7 + 1) -> 1023-chip +/-1
    m-sequence. Ideal periodic autocorrelation (peak 1023, sidelobes -1, ~60 dB);
    near-balanced (sum +1, so DC/hum rejection is also ~60 dB). Deterministic:
    identical in every copy of the software, so all stations share one code.
    (pad_len > 1023 would pad with -1, but that degrades the autocorrelation.)"""
    TAPS  = (10, 7)
    reg   = [1] * 10
    chips = []
    for _ in range((1 << 10) - 1):          # 1023 chips
        chips.append(1.0 if reg[-1] else -1.0)
        fb = 0
        for t in TAPS:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]
    code = np.array(chips, dtype=np.float32)
    if pad_len > len(code):                  # pad and balance to zero mean
        code = np.concatenate(
            [code, np.full(pad_len - len(code), -1.0, dtype=np.float32)])
    return code[:pad_len]


TX_CODE = gen_code()


def matched_filter(iq, code, fs, search_hz, step_hz, center=0.0):
    """De-spread complex iq against the repeating code, searching carrier offset
    over center +/- search_hz. Per-period circular correlation, magnitudes summed
    noncoherently. Returns (corr_mag[code_len], best_df_hz, peak, noise_floor)."""
    clen = len(code)
    nper = len(iq) // clen
    if nper < 1:
        m = np.abs(np.asarray(iq[:clen], dtype=np.complex64))
        return m, 0.0, float(m.max() if m.size else 0.0), 1.0
    z     = np.asarray(iq[:nper * clen], dtype=np.complex64)
    t     = np.arange(len(z), dtype=np.float64)
    Rconj = np.conj(np.fft.fft(code.astype(np.complex64)))      # length clen
    best  = None
    df    = center - search_hz
    while df <= center + search_hz + 0.5 * step_hz:
        zr     = z * np.exp(-1j * 2.0 * np.pi * df * t / fs).astype(np.complex64)
        blocks = zr.reshape(nper, clen)
        C      = np.fft.ifft(np.fft.fft(blocks, axis=1) * Rconj[None, :], axis=1)
        mag    = np.abs(C).sum(axis=0)                          # noncoherent sum
        pk     = float(mag.max())
        if best is None or pk > best[0]:
            best = (pk, df, mag)
        df += step_hz
    pk, best_df, mag = best
    noise = float(np.median(mag)) + 1e-12
    return mag, best_df, pk, noise


def make_cc(freq_word, ptt=True, pa_enable=True, lna_gain_db=RX_LNA_GAIN_DB, tx_drive=0x0F):
    mox = 0x01 if ptt else 0x00
    nco = struct.pack(">I", freq_word)

    cc_cfg = bytes([
        (0x00 << 1) | mox,
        0x10,                       # C1: speed bits [1:0]=00 -> 48 kHz
        lpf_c2(freq_word),
        0x00,
        0x04,
    ])
    cc_tx_nco = bytes([(0x01 << 1) | mox]) + nco
    cc_rx_nco = bytes([(0x02 << 1) | mox]) + nco
    cc_pa     = bytes([
        (0x09 << 1) | mox,
        ((int(tx_drive) & 0x0F) << 4) if ptt else 0x00,   # C1[31:28] = TX drive level 0-15
        0x08 if (ptt and pa_enable) else 0x00,
        0x00, 0x00,
    ])
    lna_code  = 0x40 | ((int(lna_gain_db) + 12) & 0x3F)
    cc_lna    = bytes([(0x0a << 1) | mox, 0x00, 0x00, 0x00, lna_code])

    return (cc_cfg, cc_tx_nco, cc_rx_nco, cc_pa, cc_lna)

# =============================================================================
#  Connect  (verbatim from v04)
# =============================================================================

def connect(sock, ip):
    probe     = b"\xEF\xFE\x02" + bytes(60)
    stop_cmd  = b"\xEF\xFE\x04\x00" + bytes(60)
    start_cmd = b"\xEF\xFE\x04\x01" + bytes(60)
    sock.settimeout(0.5)

    # Stop any leftover session and drain buffer
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    time.sleep(0.3)
    while True:
        try:    sock.recvfrom(2048)
        except socket.timeout: break

    # Wait for HL2
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

    # Send silent EP2 packets and wait for EP6
    cc_cfg = make_cc(freq_to_word(NCO_FREQ))[0]
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
               lna_gain_db=RX_LNA_GAIN_DB, tx_drive=0x0F):
    cc_regs   = make_cc(freq_word, ptt=ptt, pa_enable=pa_enable,
                        lna_gain_db=lna_gain_db, tx_drive=tx_drive)
    reg_names = ["0x00 cfg", "0x01 TX NCO", "0x02 RX NCO", "0x09 PA", "0x0a LNA"]
    for i, (cc, name) in enumerate(zip(cc_regs, reg_names)):
        sock.sendto(build_ep2(start_seq + i, cc, cc, [(0, 0)] * PACKET_SAMPLES),
                    (ip, HPSDR_PORT))
        time.sleep(0.020)
    return start_seq + len(cc_regs)

# =============================================================================
#  TX keep-alive thread  (verbatim from v04)
# =============================================================================

_tx_running      = threading.Event()
_capture_request = threading.Event()   # set by the Capture button


def _tx_thread_fn(sock, ip, cc_holder, start_seq):
    code = TX_CODE
    clen = len(code)
    amp  = TX_AMPLITUDE * 32767.0
    seq  = start_seq
    idx  = 0                      # running chip index, wraps every CODE_LEN
    next_send = time.monotonic()

    while _tx_running.is_set():
        # BPSK at DC: I = +/-code, Q = 0; code repeats seamlessly across packets
        k    = (idx + np.arange(PACKET_SAMPLES)) % clen
        I_tx = (code[k] * amp).astype(np.int16)
        Q_tx = np.zeros(PACKET_SAMPLES, dtype=np.int16)
        idx  = (idx + PACKET_SAMPLES) % clen

        cc_regs = cc_holder[0]
        n_regs  = len(cc_regs)
        cc0 = cc_regs[ seq      % n_regs]
        cc1 = cc_regs[(seq + 1) % n_regs]
        pkt = build_ep2(seq, cc0, cc1, list(zip(I_tx.tolist(), Q_tx.tolist())))
        try:
            sock.sendto(pkt, (ip, HPSDR_PORT))
        except OSError:
            break
        seq += 1

        next_send += PACKET_INTERVAL
        wait = next_send - time.monotonic()
        if wait > 0:
            time.sleep(wait)

# =============================================================================
#  Capture loop thread
# =============================================================================

def _do_file_capture(sock, spectrum, capture_ms=100.0, outfile="capture.npz"):
    """Grab capture_ms of raw IQ and save to a .npz for view_time.py."""
    n = int(round(SAMPLE_RATE * capture_ms / 1000.0))
    print(f"Capture: grabbing {capture_ms:.0f} ms ({n} samples) ...")

    # Bounded flush of stale packets
    t_end = time.monotonic() + 0.3
    while time.monotonic() < t_end:
        try:    sock.recvfrom(2048)
        except socket.timeout: pass

    samples  = []
    deadline = time.monotonic() + max(5.0, capture_ms / 1000.0 * 20)
    while len(samples) < n and time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) == 1032 and data[0:2] == b"\xEF\xFE":
                samples.extend(parse_ep6(data))
        except socket.timeout:
            pass

    samples = samples[:n]
    if not samples:
        print("Capture: no samples received — nothing saved.")
        return

    arr = np.asarray(samples, dtype=np.float64)
    iq  = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    np.savez(outfile, iq=iq, fs=SAMPLE_RATE,
             nco=getattr(spectrum, "_nco_hz", NCO_FREQ),
             lna_db=getattr(spectrum, "_lna_gain_db", 0), ms=capture_ms)

    pk = float(np.max(np.abs(iq)))
    print(f"Capture: saved {len(iq)} samples to {outfile}  (peak |IQ| = {pk:.4f}).")
    if pk > 0.98:
        print("  *** near/at full scale — CLIPPING ***")
    print("  View with:  python view_time.py")


def _capture_thread_fn(sock, spectrum):
    # Contiguous capture: read back-to-back, code-aligned blocks (CAPTURE_SIZE is
    # an exact multiple of CODE_LEN) so the correlation peak stays at a stable lag
    # instead of jumping each sweep. EP6 sequence is tracked continuously and lost
    # packets are zero-filled to hold the time/phase grid.
    gap_cap  = CAPTURE_SIZE // PACKET_SAMPLES
    buf      = []          # carry-over samples -> next block continues seamlessly
    prev_seq = None

    # Frequency-lock tracking state
    locked   = False
    df_lock  = 0.0
    miss     = 0

    # One-time flush of any startup backlog, then capture continuously
    t_end = time.monotonic() + 0.3
    while time.monotonic() < t_end:
        try:    sock.recvfrom(2048)
        except socket.timeout: pass

    while _tx_running.is_set():

        # File-capture request takes priority; it consumes packets -> resync after
        if _capture_request.is_set():
            _do_file_capture(sock, spectrum)
            _capture_request.clear()
            buf, prev_seq = [], None
            continue

        fwd_vals = []; rev_vals = []; cur_vals = []; telem = {}
        dropped  = 0
        deadline = time.monotonic() + 3.0
        while len(buf) < CAPTURE_SIZE and time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            if len(data) != 1032 or data[0:2] != b"\xEF\xFE":
                continue
            seq6 = int.from_bytes(data[4:8], "big")
            if prev_seq is not None:
                gap = (seq6 - prev_seq - 1) & 0xFFFF_FFFF
                if gap > 0:
                    fill = min(gap, gap_cap) * PACKET_SAMPLES
                    buf.extend([(0.0, 0.0)] * fill)            # hold timing grid
                    dropped += gap
            prev_seq = seq6
            buf.extend(parse_ep6(data))
            f, r, c = parse_ep6_power(data)
            if f is not None: fwd_vals.append(f)
            if r is not None: rev_vals.append(r)
            if c is not None: cur_vals.append(c)
            if DEBUG_TELEM:
                for blk in range(2):
                    b = 8 + blk * 512
                    telem[data[b + 3] >> 3] = (data[b + 4], data[b + 5],
                                               data[b + 6], data[b + 7])

        if len(buf) < CAPTURE_SIZE:
            print("  WARNING: stream stalled — resyncing")
            buf, prev_seq = [], None
            continue

        if DEBUG_TELEM and telem:
            print("TELEM raddr->(C1,C2,C3,C4):",
                  {r: telem[r] for r in sorted(telem)})

        block = buf[:CAPTURE_SIZE]
        buf   = buf[CAPTURE_SIZE:]          # keep remainder -> contiguous next block

        if fwd_vals:
            fwd_peak = max(fwd_vals)
            fwd_avg  = sum(fwd_vals) / len(fwd_vals)
            rev_avg  = (sum(rev_vals) / len(rev_vals)) if rev_vals else 0.0
            cur_avg  = (sum(cur_vals) / len(cur_vals)) if cur_vals else 0.0
            pep      = max(0.0, adc_to_watts(fwd_peak) - adc_to_watts(rev_avg))
            spectrum.set_power(fwd_peak, fwd_avg, rev_avg, pep,
                               adc_to_current_ma(cur_avg))

        arr = np.asarray(block, dtype=np.float64)
        iqc = arr[:, 0] + 1j * arr[:, 1]

        # Acquire with a full search; once locked, track a narrow window (hysteresis)
        if locked:
            mag, df, pk, noise = matched_filter(iqc, TX_CODE, SAMPLE_RATE,
                                                TRACK_HZ, FREQ_STEP_HZ, center=df_lock)
        else:
            mag, df, pk, noise = matched_filter(iqc, TX_CODE, SAMPLE_RATE,
                                                FREQ_SEARCH_HZ, FREQ_STEP_HZ)
        snr = 20.0 * np.log10(pk / noise + 1e-12)
        if locked:
            if snr >= DROP_DB:
                df_lock += 0.5 * (df - df_lock)      # smooth-track slow drift
                miss = 0
            else:
                miss += 1
                if miss > MAX_MISS:
                    locked = False
        elif snr >= LOCK_DB:
            locked, df_lock, miss = True, df, 0

        report_df = df_lock if locked else df
        spectrum.set_corr(mag, report_df, pk, noise, dropped, locked)
        spectrum.push(block)

# =============================================================================
#  Main
# =============================================================================

def main():
    import sys, os

    freq_word = freq_to_word(NCO_FREQ)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1 MB, fewer drops
    sock.bind(("", HPSDR_PORT))
    seq = connect(sock, HL2_IP)
    sock.settimeout(0.002)

    # Initial mode: PA_EN=0, PTT=0  (no TX — RX only at startup)
    # *** Connect a 50-ohm dummy load to ANT before enabling PA/PTT. ***
    seq      = init_radio(sock, HL2_IP, freq_word, seq, ptt=False, pa_enable=False)
    cc_regs  = make_cc(freq_word, ptt=False, pa_enable=False)
    # Mutable holder so button callback can swap in new cc_regs safely
    cc_holder = [cc_regs]

    print(f"Mode  : PA_EN=0 PTT=0  (no TX — RX only; toggle buttons to enable)")
    print(f"NCO   : {NCO_FREQ/1e6:.3f} MHz  |  DSSS BPSK {CODE_LEN}-chip code @ {SAMPLE_RATE//1000} kchip/s")
    print(f"LNA   : +{RX_LNA_GAIN_DB} dB  |  {SAMPLE_RATE//1000} ksps  |  TX amp {TX_AMPLITUDE:.3f}")
    print(f"RBW   : {(SAMPLE_RATE/SEGMENT_SIZE)*1.44:.0f} Hz  |  FFT pad {FFT_PAD_SIZE}  |  cal {DBFS_REF_DBM:+.1f} dBm")

    # Mutable NCO holder so the Freq box can retune at runtime
    freq_holder = [freq_word]

    def on_cc_change(ptt, pa_enable, lna_gain_db, tx_drive):
        """Called from matplotlib main thread when a button, LNA box, or TX-power slider changes."""
        fw = freq_holder[0]
        new_regs = make_cc(fw, ptt=ptt, pa_enable=pa_enable,
                           lna_gain_db=lna_gain_db, tx_drive=tx_drive)
        cc_holder[0] = new_regs
        init_radio(sock, HL2_IP, fw, 0, ptt=ptt, pa_enable=pa_enable,
                   lna_gain_db=lna_gain_db, tx_drive=tx_drive)

    def on_capture():
        """Capture button — request a 100 ms file grab from the capture thread."""
        _capture_request.set()

    def on_freq_change(freq_hz):
        """Freq box — retune the NCO (TX + RX) on the radio."""
        fw = freq_to_word(freq_hz)
        freq_holder[0] = fw
        new_regs = make_cc(fw, ptt=spectrum._ptt, pa_enable=spectrum._pa_enable,
                           lna_gain_db=spectrum._lna_gain_db, tx_drive=spectrum._tx_drive)
        cc_holder[0] = new_regs
        init_radio(sock, HL2_IP, fw, 0, ptt=spectrum._ptt,
                   pa_enable=spectrum._pa_enable, lna_gain_db=spectrum._lna_gain_db,
                   tx_drive=spectrum._tx_drive)

    # Suppress stderr during matplotlib/tkinter init to hide folder path messages
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    spectrum = LiveSpectrum(
        nco_hz       = NCO_FREQ,
        fs           = SAMPLE_RATE,
        dbfs_ref     = DBFS_REF_DBM,
        seg          = SEGMENT_SIZE,
        pad          = FFT_PAD_SIZE,
        ymin         = -120,
        ymax         = 20,
        lna_gain_db  = RX_LNA_GAIN_DB,
        ptt          = False,
        pa_enable    = False,
        tx_drive     = 15,
        on_cc_change   = on_cc_change,
        on_capture     = on_capture,
        on_freq_change = on_freq_change,
    )
    sys.stderr = _stderr

    # Dedicated TX socket
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_sock.bind(("", 0))

    _tx_running.set()
    threading.Thread(target=_tx_thread_fn,
                     args=(tx_sock, HL2_IP, cc_holder, seq),
                     daemon=True, name="HL2-TX").start()

    print("Stabilising (1 s) ...")
    time.sleep(1.0)

    # Start capture thread
    threading.Thread(target=_capture_thread_fn,
                     args=(sock, spectrum),
                     daemon=True, name="HL2-Capture").start()

    spectrum.run()   # blocks until plot window closed

    # Shutdown
    print("\nShutting down ...")
    _tx_running.clear()
    time.sleep(0.2)
    stop_cmd = b"\xEF\xFE\x04\x00" + bytes(60)
    sock.sendto(stop_cmd, (HL2_IP, HPSDR_PORT))
    sock.sendto(stop_cmd, (HL2_IP, HPSDR_PORT))
    sock.close()
    tx_sock.close()
    print("Done.")


if __name__ == "__main__":
    main()
