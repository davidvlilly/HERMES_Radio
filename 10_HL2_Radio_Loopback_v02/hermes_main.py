#!/usr/bin/env python3
# =============================================================================
#  hermes_main.py  —  HermesLite 2  Main Application
# =============================================================================
#
#  What this file does (in order):
#  --------------------------------
#  1. Discover the HL2 radio on the local network (UDP broadcast)
#  2. Connect  — send 64-byte start command, confirm EP6 response
#  3. Initialise registers:
#       • NCO  7.000 MHz  (TX and RX)
#       • Sample rate  384 ksps  (maximum for Protocol 1)
#       • LPF relay    40 m filter  (J16 GPIO via reg 0x00 C2)
#       • TX drive     100 %  (C1 = 0xF0)
#       • PA enable    yes   (reg 0x09 bit 19)
#  4. Print all settings to the terminal
#  5. Start a background thread that continuously sends the 7.001 MHz
#     TX tone EP2 packets  (keeps the HL2 watchdog satisfied)
#  6. Capture 4096 IQ samples from EP6
#       (Pure-Signal loopback: TX leaks back through the AD9866 RX path
#        at the RF1 / pre-amp monitor tap — no external cable required)
#  7. Compute power spectrum  (Welch's averaged method):
#       4096 samples  →  31 × 256-pt Hanning segments (50 % overlap)
#       each zero-padded to 1024  →  averaged |FFT|²  →  dB normalised
#       RBW = (384000 / 256) × 1.44 ≈ 2.2 kHz
#  8. Open the GUI  (hermes_gui.HermesGUI) and plot the spectrum
#  9. Close cleanly when the GUI window is closed
#
#  Display range : 6.808 – 7.192 MHz  (exact data span, ±192 kHz at 384 ksps)
#
#  Dependencies:  numpy  matplotlib  (pip install numpy matplotlib)
#  Python 3.10+  required for X | Y type-union syntax
# =============================================================================

import socket
import struct
import time
import sys
import threading
import tkinter as tk
import numpy as np

from hermes_gui import HermesGUI


# =============================================================================
#  Configuration constants
# =============================================================================
HPSDR_PORT      = 1024
NCO_FREQ        = 7_000_000        # Hz  — carrier / LO frequency
TONE_OFFSET     = 1_000            # Hz  — TX tone above NCO → RF 7.001 MHz
SAMPLE_RATE     = 384_000          # Hz  — 384 ksps = Protocol 1 maximum
TX_AMPLITUDE    = 0.450            # ~1 W at ANT (confirmed 0 dBm on SA with 30 dB atten)
HL2_KNOWN_IP    = "169.254.19.221" # static link-local IP (from Quisk config)
CAPTURE_SIZE    = 4096             # total IQ samples captured per run

# Welch averaged spectrum parameters
SEGMENT_SIZE    = 256              # samples per Hanning-windowed segment
FFT_PAD_SIZE    = 1024             # zero-pad each segment to this FFT length
#  RBW  = (SAMPLE_RATE / SEGMENT_SIZE) × 1.44  (Hanning 3 dB BW)
#       = (384000 / 256) × 1.44  =  1500 × 1.44  ≈  2160 Hz  ≈  2.2 kHz
#  Overlap = 50%  →  step = SEGMENT_SIZE // 2 = 128 samples
#  Segments from 4096 samples = (4096 - 256) / 128 + 1 = 31 averages

PACKET_SAMPLES  = 126              # IQ samples per EP2/EP6 packet (fixed)
PACKET_INTERVAL = PACKET_SAMPLES / SAMPLE_RATE   # ~0.328 ms at 384 ksps

# RX LNA gain  (register 0x0a, HL2 FAST_LNA mode)
# -------------------------------------------------
# The AD9866 internal LNA MUST be set — without gain the HL2 is deaf.
# FAST_LNA mode (C4 bit 6 = 1): gain range −12 to +48 dB in 1 dB steps.
# C4 byte = 0x40 | (RX_LNA_GAIN_DB + 12)
# During TX the TR switch routes ANT to the PA; RX sees TX leakage
# through the switch isolation (~−35 dB).  At +20 dB LNA the leakage
# from a 400 mW TX should still be visible above the noise floor.
# If the ADC clips during TX (D4/D5 LEDs), reduce this value.
RX_LNA_GAIN_DB  = 20               # dB   LNA test step 2 of 2

# Cable loopback mode
# -------------------
# LOOPBACK_CABLE_MODE = True  (use this to verify TX signal in software)
#   Physical wiring:  RF1 SMA → CABLE_ATT_DB attenuator → ANT SMA
#   PTT is de-asserted (MOX=0): TR switch routes ANT → LNA → ADC
#   AD9866 DAC continues to output the TX tone on RF1 regardless of PTT.
#   Signal path: DAC → RF1 → cable → attenuator → ANT → TR sw → LNA → ADC → EP6
#
# LOOPBACK_CABLE_MODE = False  (original mode)
#   PTT asserted, PA on, TX tone radiated from ANT.
#   RX uses internal Pure Signal loopback tap (requires hardware resistor on board).
LOOPBACK_CABLE_MODE = False        # legacy cable loopback (PTT=0) — not functional
CABLE_ATT_DB        = 30           # attenuator in cable (dB) — for log only
                                   # RF1 (+15 dBm) → 30 dB atten → RF3 (−15 dBm)

# RF3 loopback mode
# -----------------
# Physical wiring:  RF1 SMA → 30 dB attenuator → RF3 SMA
# RF3 connects to pin 6 of the filter-to-mainboard connector, injecting
# directly into the RX ADC path — bypasses the TR switch entirely.
# PTT=1 so the DAC outputs on RF1.  PA is disabled (no radiation from ANT).
# Signal path:  DAC → RF1 → cable → 30 dB atten → RF3 → LNA → ADC → EP6
# Display:  time domain (I and Q vs time) rather than spectrum.
RF3_LOOPBACK_MODE   = True         # RF1 → 30 dB atten → RF3 → LNA → ADC
RX_TEST_MODE        = False        # PTT on, TX tone active on RF1
PA_ENABLE           = True         # PA on — connect dummy load or antenna to ANT

# Absolute power calibration
# ---------------------------
# IQ samples arrive normalised ±1.0 = ADC full scale.
# We assume full scale = 1 V peak into 50 Ω (real-sine convention):
#   P_fullscale = (1 / √2)² / 50 / 0.001  =  10 mW  =  +10 dBm
#
# DBFS_REF_BASE is the empirical trim at 0 dB LNA gain, measured by SA.
# The LNA gain is subtracted automatically so the display always reads
# true power at the RF3 (or ANT) input regardless of LNA gain setting.
#
#   DBFS_REF_DBM = DBFS_REF_BASE - RX_LNA_GAIN_DB
#
# Calibration (2026-05-31):
#   SA measured 0 dBm at RF3 input with RX_LNA_GAIN_DB = 20 dB
#   Python showed −31 dBm  →  offset needed = +31 dB total
#   DBFS_REF_BASE = 31 - 20 = 11 dB  (input-referred, LNA-independent)
DBFS_REF_DBM    = 31.0             # SA-calibrated constant for RF3 injection path
                                   # RF3 injects after the LNA/PGA gain stage — LNA gain
                                   # has no effect on RF3 signal amplitude.
                                   # Calibrated: SA = 0 dBm at RF3, Python = 0 dBm display.
                                   # NOTE: for ANT receive (RX_TEST_MODE), LNA gain DOES
                                   # affect the signal — re-calibrate if using ANT path.


# =============================================================================
#  Helper: frequency word
# =============================================================================
def freq_to_word(freq_hz: int) -> int:
    """Return frequency as raw 32-bit Hz integer for C1-C4 bytes."""
    return int(freq_hz) & 0xFFFF_FFFF


# =============================================================================
#  Helper: LPF relay code  →  register 0x00 C2 byte
# =============================================================================
def lpf_c2(freq_hz: int) -> int:
    """
    Return the C2 byte for register address 0x00 that selects the correct
    HL2 low-pass filter relay for the given transmit frequency.

    The HL2 has a switched LPF bank; relay coils are driven by 7 GPIO lines
    on the J16 connector, controlled by C2[7:1] of address 0x00.

    Band codes come from Quisk quisk_conf_defaults.py (HermesLite2 profile,
    Hermes_BandDict).  The 7-bit code is shifted left 1 into C2[7:1].

    Band     Freq range       code (7-bit)   C2 byte
    ----     ----------       ------------   -------
    160 m    < 2.5 MHz        0b000_0001     0x02
    80 m     2.5 – 5.5 MHz    0b100_0010     0x84
    60/40 m  5.5 – 9.5 MHz    0b100_0100     0x88   ← 7 MHz is here
    30/20 m  9.5 – 16 MHz     0b100_1000     0x90
    17/15 m  16 – 22 MHz      0b101_0000     0xA0
    12/10 m  22 – 35 MHz      0b110_0000     0xC0
    """
    if   freq_hz <  2_500_000: code = 0b000_0001
    elif freq_hz <  5_500_000: code = 0b100_0010
    elif freq_hz <  9_500_000: code = 0b100_0100   # 40 m  ← our band
    elif freq_hz < 16_000_000: code = 0b100_1000
    elif freq_hz < 22_000_000: code = 0b101_0000
    else:                      code = 0b110_0000
    return (code & 0x7F) << 1


# =============================================================================
#  EP2 packet builder  (PC → HL2, 1032 bytes)
# =============================================================================
def build_ep2(seq: int, cc0: bytes, cc1: bytes, tx_iq: list) -> bytes:
    """
    Build a 1032-byte HPSDR Protocol-1 EP2 packet.

    Layout
    ------
    Bytes   0– 3 : EF FE 01 02  (sync + endpoint)
    Bytes   4– 7 : sequence number  (big-endian uint32)
    Bytes   8–519: Block 0  (512 bytes)
                   [0:3]  7F 7F 7F  block sync
                   [3:8]  C0-C4     command-and-control bytes
                   [8:512] 63 samples × 8 bytes
    Bytes 520–1031: Block 1  (same structure)

    Each TX sample (8 bytes, big-endian signed 16-bit):
      left-audio(16) | right-audio(16) | I-tx(16) | Q-tx(16)
    """
    pkt = bytearray(1032)
    pkt[0:4] = b"\xEF\xFE\x01\x02"
    struct.pack_into(">I", pkt, 4, seq & 0xFFFF_FFFF)

    for blk, cc in enumerate((cc0, cc1)):
        base = 8 + blk * 512
        pkt[base:base + 3]   = b"\x7F\x7F\x7F"
        pkt[base + 3:base + 8] = cc
        for i in range(63):
            idx = blk * 63 + i
            Iv  = tx_iq[idx][0] if idx < len(tx_iq) else 0
            Qv  = tx_iq[idx][1] if idx < len(tx_iq) else 0
            struct.pack_into(">hhhh", pkt, base + 8 + i * 8, 0, 0, Iv, Qv)

    return bytes(pkt)


# =============================================================================
#  EP6 parser  (HL2 → PC, 1032 bytes)
# =============================================================================
def parse_ep6(data: bytes) -> list:
    """
    Extract IQ sample pairs from a 1032-byte EP6 packet (1 receiver).

    Each sample:  I(24-bit signed)  Q(24-bit signed)  audio(16-bit)  = 8 bytes
    Returns list of (I_float, Q_float) normalised ±1.0
    """
    if len(data) < 1032 or data[0:2] != b"\xEF\xFE":
        return []
    samples = []
    for blk in range(2):
        base = 8 + blk * 512 + 8      # skip 8-byte pkt hdr + 8-byte blk hdr
        for i in range(63):
            pos   = base + i * 8
            I_raw = int.from_bytes(data[pos    :pos + 3], "big", signed=True)
            Q_raw = int.from_bytes(data[pos + 3:pos + 6], "big", signed=True)
            samples.append((I_raw / 8_388_608.0,    # 2^23
                            Q_raw / 8_388_608.0))
    return samples


def read_ep6_status(data: bytes) -> dict:
    """
    Extract C&C status bytes from a 1032-byte EP6 packet.

    Returns a dict with status fields decoded from both blocks.
    Key field: tx_inhibit — True means the HL2 PA thermal protection
    has fired and is blocking transmission.

    EP6 block header (8 bytes):
      [0:3]  7F 7F 7F  sync
      [3]    C0  — bits[6:1]=address, bit[0]=hardware PTT
      [4]    C1
      [5]    C2
      [6]    C3
      [7]    C4

    At C0 address 0 (C0[6:1] == 0):
      C1 bit 0 = ADC overflow
      C1 bit 1 = TX inhibit  (0 = inhibited / PA protection active,
                               1 = OK to transmit)
    """
    status = {"tx_inhibit": None, "adc_overflow": None, "raw_c0": []}
    if len(data) < 1032 or data[0:2] != b"\xEF\xFE":
        return status

    for blk in range(2):
        hdr_base = 8 + blk * 512     # start of 8-byte block header
        c0 = data[hdr_base + 3]
        c1 = data[hdr_base + 4]
        status["raw_c0"].append(c0)
        dindex = (c0 >> 1) & 0x3F    # address field
        if dindex == 0:
            status["adc_overflow"] = bool(c1 & 0x01)
            status["tx_inhibit"]   = not bool(c1 & 0x02)  # bit1=0 → inhibited

    return status


# =============================================================================
#  C&C byte sets
# =============================================================================
def make_cc(freq_word: int, ptt: bool = True, pa_enable: bool = True) -> tuple:
    """
    Return (cc_cfg, cc_tx_nco, cc_rx_nco, cc_pa_on) — the four C&C byte
    tuples that configure the HL2 for transmit at freq_word Hz.

    Register 0x00  — general config
      C1 = 0x13  → bit[4]=Hermes board  bits[1:0]=11=384 ksps
      C2 = lpf_c2(freq_word)  → LPF relay select via J16 GPIO C2[7:1]
      C4 = 0x04  → bit[2]=duplex on  bits[5:3]=000=1 receiver
      NOTE: duplex is C4 bit 2, NOT a C1 bit (confirmed from Quisk source)

    Register 0x01  — TX NCO frequency   (raw Hz, big-endian 32-bit)
    Register 0x02  — RX NCO frequency   (raw Hz, big-endian 32-bit)

    Register 0x09  — TX drive + PA enable
      C1 = 0xF0  → bits[7:4] = 15  → drive = 15/15 = 100 %
      C2 = 0x08  → bit 3 = bit 19 of register  → PA enable

    Register 0x0a  — RX LNA gain  (HL2 FAST_LNA mode)
      C4 bit 6 = 1  → FAST_LNA mode (full −12 to +48 dB range)
      C4 bits[5:0]  → gain code = RX_LNA_GAIN_DB + 12
    """
    mox = 0x01 if ptt else 0x00
    nco = struct.pack(">I", freq_word)

    cc_cfg = bytes([
        (0x00 << 1) | mox,          # C0: addr 0x00, PTT bit
        0x13,                        # C1: Hermes board (bit 4) + 384 ksps (bits 1:0)
        lpf_c2(freq_word),           # C2: LPF relay select (J16 GPIO)
        0x00,                        # C3: unused
        0x04,                        # C4: duplex on (bit 2) + 1 receiver (bits 5:3 = 000)
    ])

    cc_tx_nco = bytes([(0x01 << 1) | mox]) + nco   # addr 0x01: TX NCO
    cc_rx_nco = bytes([(0x02 << 1) | mox]) + nco   # addr 0x02: RX NCO

    cc_pa_on = bytes([
        (0x09 << 1) | mox,                        # C0: addr 0x09, PTT bit
        0xF0 if ptt else 0x00,                    # C1: TX drive 100 % when PTT
        0x08 if (ptt and pa_enable) else 0x00,    # C2: PA enable — PTT AND pa_enable
        0x00, 0x00,
    ])

    lna_code = 0x40 | ((RX_LNA_GAIN_DB + 12) & 0x3F)   # FAST_LNA: bit6=1, gain code
    cc_lna = bytes([
        (0x0a << 1) | mox,          # C0: addr 0x0a, PTT bit
        0x00, 0x00, 0x00,           # C1-C3: unused
        lna_code,                   # C4: FAST_LNA mode + gain code
    ])

    return cc_cfg, cc_tx_nco, cc_rx_nco, cc_pa_on, cc_lna


# =============================================================================
#  Discovery
# =============================================================================
def discover() -> str | None:
    """
    Broadcast HPSDR discovery probe and return the HL2 IP address string,
    or None if no radio responds.  Falls back to the known static IP.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", HPSDR_PORT))
    sock.settimeout(3.0)

    probe = b"\xEF\xFE\x02" + bytes(60)
    sock.sendto(probe, ("255.255.255.255", HPSDR_PORT))
    print("  Broadcasting discovery probe ...")

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            my_ips = socket.gethostbyname_ex(socket.gethostname())[2]
            if addr[0] in my_ips:
                continue                          # skip own echo
            if data[0:2] == b"\xEF\xFE" and data[2] in (0x02, 0x03):
                mac = ":".join(f"{b:02X}" for b in data[3:9])
                fw  = data[9]
                print(f"  Found HL2:  IP={addr[0]}  MAC={mac}  FW={fw}")
                sock.close()
                return addr[0]
    except socket.timeout:
        print(f"  Broadcast timed out — trying static IP {HL2_KNOWN_IP}")

    sock.close()

    # Direct probe to known link-local IP
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2.settimeout(3.0)
    try:
        sock2.sendto(probe, (HL2_KNOWN_IP, HPSDR_PORT))
        data, addr = sock2.recvfrom(1024)
        if data[0:2] == b"\xEF\xFE" and data[2] in (0x02, 0x03):
            mac = ":".join(f"{b:02X}" for b in data[3:9])
            fw  = data[9]
            print(f"  Found HL2 at static IP  {addr[0]}  MAC={mac}  FW={fw}")
            return addr[0]
        print(f"  No valid response at {HL2_KNOWN_IP}")
    except socket.timeout:
        print(f"  No response at {HL2_KNOWN_IP}")
    finally:
        sock2.close()

    return None


# =============================================================================
#  Connect
# =============================================================================
def connect(sock: socket.socket, ip: str) -> tuple[bool, int]:
    """
    Four-phase startup sequence.  Returns (success, next_seq_number).

    Phase 1  Check    — confirm HL2 is reachable via discovery probe
    Phase 2  Reboot   — Protocol 2 packet (port 1025) sets hl2_reset_state=1.
                        Since run is already 0 (STOP state), hl2_reset fires
                        immediately → FPGA reloads from flash → AD9866 SPI init
                        re-runs (63 writes, identical to power cycle).
                        NOTE: Protocol 1 EP2 C&C does NOT handle addr 0x3a —
                        usopenhpsdr1.v only decodes 0x00,0x09,0x10,0x17,0x39.
    Phase 3  Wait     — poll discovery until HL2 comes back up (expect 3–5 s).
                        If response arrives in < 2 s the reboot did NOT happen.
    Phase 4  Start    — send START, confirm EP6 is flowing

    Doing the reboot HERE (at startup) rather than at shutdown means we can
    confirm the HL2 received the command and wait the correct amount of time
    before proceeding.  UDP gives us no confirmation at shutdown.
    """
    probe     = b"\xEF\xFE\x02" + bytes(60)
    stop_cmd  = b"\xEF\xFE\x04\x00" + bytes(60)
    start_cmd = b"\xEF\xFE\x04\x01" + bytes(60)
    sock.settimeout(0.5)

    # ------------------------------------------------------------------
    # Phase 1: Confirm HL2 is reachable
    # ------------------------------------------------------------------
    print("  [1/4] Checking HL2 is reachable ...", end="", flush=True)
    found = False
    for _ in range(10):   # up to 5 s
        sock.sendto(probe, (ip, HPSDR_PORT))
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) >= 10 and data[0:2] == b"\xEF\xFE" and data[2] in (0x02, 0x03):
                mac = ":".join(f"{b:02X}" for b in data[3:9])
                fw  = data[9]
                print(f" OK  (MAC={mac}  FW={fw})")
                found = True
                break
        except socket.timeout:
            print(".", end="", flush=True)
    if not found:
        print(" FAILED — HL2 not responding")
        return False, 0

    # ------------------------------------------------------------------
    # Phase 2: Trigger FPGA reboot via Protocol 2 command
    # ------------------------------------------------------------------
    # Protocol 1 EP2 C&C (usopenhpsdr1.v) only decodes addresses:
    #   0x00, 0x09, 0x10, 0x17, 0x39
    # Register 0x3a (hl2_reset_state) is a Protocol 2 / HL2-extended
    # command.  It is reached by sending a 60-byte packet to port 1025.
    #
    # The HL2 FPGA logic is combinatorial:
    #   assign hl2_reset = hl2_reset_state & ~run;
    # Because the HL2 is already in STOP state (run=0) when we get here,
    # writing hl2_reset_state=1 immediately fires the reset — no START/STOP
    # dance needed.  The FPGA reloads from flash and the AD9866 SPI init
    # sequence (63 writes: TX path enable, DC cal, IAMP, etc.) re-runs from
    # scratch, just as it does after a power cycle.
    #
    # Packet format (from hermeslite.py reference library):
    #   EF FE 05 7F  [reg_addr = 0x74 = 0x3a<<1]  [4-byte data = 0x00000001]
    #   + 51 zero-pad bytes  →  60 bytes total
    #   Sent to port 1025 (Protocol 2 port) from our port-1024 socket.
    print("  [2/4] Triggering FPGA reboot (Protocol 2 reg 0x3a → port 1025) ...",
          end="", flush=True)

    reboot_p2 = (bytes([0xEF, 0xFE, 0x05, 0x7F, 0x74, 0x00, 0x00, 0x00, 0x01])
                 + bytes(51))
    sock.sendto(reboot_p2, (ip, 1025))   # Protocol 2 port — sets hl2_reset_state=1
    time.sleep(0.050)

    # Belt-and-suspenders: STOP on Protocol 1 ensures run=0 in case HL2
    # was somehow left in run=1 state (harmless if already stopped).
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    print(" reboot triggered")
    sock.settimeout(0.5)

    # ------------------------------------------------------------------
    # Phase 3: Wait for FPGA to finish rebooting
    # ------------------------------------------------------------------
    # The Xilinx Spartan 6 FPGA + HL2 network stack takes ~10-15 s to reload
    # from SPI flash and re-announce itself on the network.
    # Budget: 6 s quiet sleep + 60 × 0.5 s probes = 36 s total maximum.
    #
    # Diagnostic:
    #   elapsed < 7 s  → responded before a real FPGA reload could finish
    #                    → reboot did NOT happen (Protocol 2 was ignored?)
    #   elapsed > 7 s  → genuine FPGA reload confirmed
    print("  [3/4] Waiting for HL2 to reboot (expect 8-15 s) ", end="", flush=True)
    time.sleep(6.0)    # FPGA silent during flash load — don't probe yet
    found   = False
    t_start = time.monotonic()
    for _ in range(60):   # up to 30 more s  (total budget 36 s)
        sock.sendto(probe, (ip, HPSDR_PORT))
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) >= 10 and data[0:2] == b"\xEF\xFE" and data[2] in (0x02, 0x03):
                mac     = ":".join(f"{b:02X}" for b in data[3:9])
                fw      = data[9]
                elapsed = time.monotonic() - t_start + 6.0
                if elapsed < 7.0:
                    print(f" OK  ({elapsed:.1f} s)  MAC={mac}  FW={fw}")
                    print("  *** WARNING: responded in < 7 s — FPGA reboot may NOT have triggered.")
                    print("  *** Protocol 2 packet may have been ignored (port 1025 blocked?)")
                    print("  *** Run 2 will likely fail.  Check Windows Firewall for port 1025.")
                else:
                    print(f" OK  ({elapsed:.1f} s)  MAC={mac}  FW={fw}  "
                          f"[FPGA reloaded — AD9866 SPI init re-ran]")
                found = True
                break
        except socket.timeout:
            print(".", end="", flush=True)
    if not found:
        print(" TIMEOUT — HL2 did not come back up after reboot")
        return False, 0

    # ------------------------------------------------------------------
    # Phase 4: Send START and confirm EP6 is flowing
    # ------------------------------------------------------------------
    print("  [4/4] Starting radio ...", end="", flush=True)
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))   # ensure clean idle before START
    time.sleep(0.2)
    sock.sendto(start_cmd, (ip, HPSDR_PORT))

    cc_silent  = bytes([0x00, 0x13, lpf_c2(NCO_FREQ), 0x00, 0x04])
    silent_pkt = build_ep2(0, cc_silent, cc_silent, [(0, 0)] * PACKET_SAMPLES)
    deadline   = time.monotonic() + 5.0
    ep6_count  = 0
    seq        = 0

    while time.monotonic() < deadline:
        pkt = bytearray(silent_pkt)
        struct.pack_into(">I", pkt, 4, seq)
        sock.sendto(bytes(pkt), (ip, HPSDR_PORT))
        seq += 1
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) == 1032 and data[0:2] == b"\xEF\xFE":
                ep6_count += 1
                if ep6_count >= 3:
                    print(f" OK  ({ep6_count} EP6 packets)  RUN LED should be solid.")
                    return True, seq
        except socket.timeout:
            pass
        time.sleep(0.010)

    print(" FAILED — no EP6 response after START")
    return False, 0


# =============================================================================
#  Initialise registers
# =============================================================================
def initialise_radio(sock: socket.socket, ip: str,
                     freq_word: int, start_seq: int = 0,
                     ptt: bool = True, pa_enable: bool = True) -> int:
    """
    Write all four C&C register sets with silent IQ (no RF during init).
    Returns the next sequence number to use.

    Registers written:
      0x00  general config  (sample rate, duplex, LPF relay)
      0x01  TX NCO frequency
      0x02  RX NCO frequency
      0x09  TX drive level + PA enable  (PA disabled when ptt=False)
      0x0a  RX LNA gain
    """
    # Safe-state write is handled by main() (after connect, before this call)
    # and by disconnect() (before STOP on clean exit).  No pre-write needed here.
    cc_cfg, cc_tx_nco, cc_rx_nco, cc_pa_on, cc_lna = make_cc(freq_word, ptt=ptt,
                                                             pa_enable=pa_enable)
    reg_names = ["0x00 config", "0x01 TX NCO", "0x02 RX NCO",
                 "0x09 PA/drive", "0x0a RX LNA"]

    for i, (cc, name) in enumerate(zip(
            [cc_cfg, cc_tx_nco, cc_rx_nco, cc_pa_on, cc_lna], reg_names)):
        pkt = build_ep2(start_seq + 1 + i, cc, cc, [(0, 0)] * PACKET_SAMPLES)
        sock.sendto(pkt, (ip, HPSDR_PORT))
        print(f"    reg {name}  written")
        time.sleep(0.020)   # longer gap — give FPGA time to process each write

    return start_seq + 6


# =============================================================================
#  Safe-state register write  (call before STOP and after START on recovery)
# =============================================================================
def write_safe_registers(sock: socket.socket, ip: str, start_seq: int = 0) -> int:
    """
    Write key FPGA registers to a known-safe/off state via EP2 packets.

    The STOP command only halts EP6 streaming — it does NOT reset register
    values.  Without this call, the FPGA retains PA-enable, PTT=1 and
    TX-drive settings from the previous run.  On the next run those dirty
    values can prevent clean re-initialisation.

    Call this:
      • at the END of every session (inside disconnect, before STOP)
      • at the START of every session (in main, after connect, before init)

    Safe state written
    ------------------
      reg 0x09 : PA off, TX drive = 0, PTT = 0
      reg 0x00 : general config, PTT = 0   (sample rate / duplex / LPF)
      reg 0x0a : LNA minimum gain (−12 dB), PTT = 0

    Returns next sequence number.
    """
    freq_word = freq_to_word(NCO_FREQ)

    safe_regs = [
        # Most important first: kill PA and TX drive
        ("reg 0x09  PA=off  drive=0  PTT=0",
         bytes([(0x09 << 1) | 0x00, 0x00, 0x00, 0x00, 0x00])),
        # General config with PTT=0
        ("reg 0x00  config  PTT=0",
         bytes([(0x00 << 1) | 0x00, 0x13, lpf_c2(freq_word), 0x00, 0x04])),
        # LNA minimum gain (code 0 = −12 dB), PTT=0
        ("reg 0x0a  LNA=-12dB  PTT=0",
         bytes([(0x0a << 1) | 0x00, 0x00, 0x00, 0x00, 0x40 | 0x00])),
    ]

    for i, (name, cc) in enumerate(safe_regs):
        pkt = build_ep2(start_seq + i, cc, cc, [(0, 0)] * PACKET_SAMPLES)
        sock.sendto(pkt, (ip, HPSDR_PORT))
        print(f"    safe write {i+1}/{len(safe_regs)}: {name}  "
              f"[{' '.join(f'{b:02X}' for b in cc)}]")
        time.sleep(0.020)

    return start_seq + len(safe_regs)


# =============================================================================
#  Disconnect
# =============================================================================
def disconnect(sock: socket.socket, ip: str):
    """
    Clean shutdown sequence: write safe register state then send STOP.

    We do NOT try to trigger a reboot here — UDP gives no delivery
    confirmation, so we can't tell if the packet reached the HL2 before
    the Python process exits.  Instead the FPGA reboot is triggered at the
    START of the next run (connect() Phase 2) via a Protocol 2 packet to
    port 1025, where we can confirm it by measuring the reboot time in
    Phase 3.

    Steps
    -----
    1. Write safe registers: PA off, TX drive=0, PTT=0, LNA minimum.
       The STOP command does not reset FPGA registers — without this the
       FPGA retains PA-on / PTT=1 / full drive settings across runs.
    2. Send STOP × 2 to halt EP6 streaming and put HL2 into idle state.
    """
    # Safe register writes — PA off, TX drive=0, PTT=0
    print("  [disconnect] Writing safe register state ...")
    write_safe_registers(sock, ip, start_seq=0)
    time.sleep(0.050)
    print("  [disconnect] Safe registers written — PA off, PTT=0, drive=0")

    # STOP — halts EP6 streaming
    stop_cmd = b"\xEF\xFE\x04\x00" + bytes(60)
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    sock.sendto(stop_cmd, (ip, HPSDR_PORT))
    print("  [disconnect] STOP sent.")


# =============================================================================
#  TX keep-alive thread
#  Sends phase-continuous 7.001 MHz tone EP2 packets at 384 ksps rate.
#  Keeps the HL2 watchdog satisfied and the PA transmitting.
# =============================================================================
_tx_running = threading.Event()


def _tx_thread_fn(sock: socket.socket, ip: str,
                  cc_regs: tuple, start_seq: int):
    """
    Background thread: continuously pump TX-tone EP2 packets.

    Cycles through ALL five C&C register sets in round-robin, two per
    packet (one per 512-byte block).  This matches Quisk / Protocol-1
    spec behaviour and ensures every register is continuously refreshed —
    critical for reliable operation after STOP / START cycles where the
    FPGA may not hold one-shot writes.

    Register rotation order:
      0x00  general config  (PTT, sample rate, LPF relay, duplex)
      0x01  TX NCO frequency
      0x02  RX NCO frequency
      0x09  TX drive + PA enable
      0x0a  RX LNA gain
    Each register is re-sent every 5 packets  (~1.6 ms at 384 ksps).
    """
    phase_step = 2.0 * np.pi * TONE_OFFSET / SAMPLE_RATE
    local_t    = np.arange(PACKET_SAMPLES)
    phase      = 0.0
    seq        = start_seq
    next_send  = time.monotonic()
    n_regs     = len(cc_regs)

    while _tx_running.is_set():
        # Generate one packet of phase-continuous IQ samples
        I_tx = (TX_AMPLITUDE
                * np.cos(phase + phase_step * local_t)
                * 32767.0).astype(np.int16)
        Q_tx = (TX_AMPLITUDE
                * np.sin(phase + phase_step * local_t)
                * 32767.0).astype(np.int16)
        phase = (phase + phase_step * PACKET_SAMPLES) % (2.0 * np.pi)

        # Rotate C&C: two consecutive registers per packet (one per block)
        cc0 = cc_regs[ seq         % n_regs]
        cc1 = cc_regs[(seq + 1)    % n_regs]

        pairs = list(zip(I_tx.tolist(), Q_tx.tolist()))
        pkt   = build_ep2(seq, cc0, cc1, pairs)
        try:
            sock.sendto(pkt, (ip, HPSDR_PORT))
        except OSError:
            break
        seq += 1

        # Pace to exact sample rate
        next_send += PACKET_INTERVAL
        wait = next_send - time.monotonic()
        if wait > 0:
            time.sleep(wait)


# =============================================================================
#  IQ capture
# =============================================================================
def capture_iq(sock: socket.socket,
               n_samples: int = CAPTURE_SIZE,
               timeout_s: float = 4.0) -> tuple:
    """
    Accumulate n_samples IQ pairs from EP6 packets.

    The TX thread must already be running so the HL2 is sending EP6.
    Returns (samples, last_raw_packet) where:
      samples          — list of (I_float, Q_float) tuples, length ≤ n_samples
      last_raw_packet  — last raw 1032-byte EP6 bytes received, or None

    The raw packet is used by read_ep6_status() to check TX inhibit / ADC
    overflow status bits from the HL2 C&C response stream.
    """
    samples  = []
    last_pkt = None
    deadline = time.monotonic() + timeout_s

    while len(samples) < n_samples and time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(2048)
            if len(data) == 1032 and data[0:2] == b"\xEF\xFE":
                samples.extend(parse_ep6(data))
                last_pkt = data          # keep most recent raw packet for status read
        except socket.timeout:
            pass

    return samples[:n_samples], last_pkt


# =============================================================================
#  Spectrum calculation  —  Welch's averaged method
# =============================================================================
def compute_spectrum(iq_samples: list,
                     nco_hz:  int = NCO_FREQ,
                     fs:      int = SAMPLE_RATE,
                     seg:     int = SEGMENT_SIZE,
                     pad:     int = FFT_PAD_SIZE
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Welch's averaged power spectrum.

    Processing chain
    ----------------
    1.  Form complex signal  z = I + jQ  from captured IQ list
    2.  Slice into overlapping 256-sample segments  (50 % overlap → 128-step)
    3.  Apply Hanning window to each segment
    4.  Zero-pad each windowed segment from 256 → 1024 samples
          (interpolates the FFT bins without changing RBW)
    5.  FFT each padded segment  (1024-point complex FFT)
    6.  Accumulate |FFT|² power, average across all segments  (31 averages
          from 4096 samples with 50 % overlap)
    7.  Shift DC to centre  →  absolute frequency axis
    8.  Convert averaged power to dB, normalise peak = 0 dB

    RBW
    ---
    RBW is set by the segment size, NOT the zero-pad size:
      RBW = (fs / seg) × 1.44  =  (384000 / 256) × 1.44  ≈  2.2 kHz

    Frequency axis
    --------------
    Spans exactly  NCO ± fs/2  with  pad  bins  (1024 bins at 375 Hz spacing).
    Absolute range at 384 ksps / 7 MHz:  6.808 – 7.192 MHz

    Returns
    -------
    freqs_mhz  : frequency axis in MHz  (length = pad)
    power_db   : averaged power in dB relative to peak  (length = pad)
    n_segments : number of segments averaged
    """
    z         = np.array([s[0] + 1j * s[1] for s in iq_samples], dtype=np.complex128)
    step      = seg // 2                    # 50 % overlap = 128 samples
    win       = np.hanning(seg)
    win_power = np.sum(win ** 2)            # window power — used for PSD normalisation
    pwr       = np.zeros(pad, dtype=np.float64)
    n         = 0

    for start in range(0, len(z) - seg + 1, step):
        buf        = np.zeros(pad, dtype=np.complex128)
        buf[:seg]  = z[start:start + seg] * win   # window then zero-pad
        pwr       += np.abs(np.fft.fft(buf)) ** 2
        n         += 1

    # --- Calibrate to absolute dBm ---
    # Step 1 — average and shift DC to centre
    pwr_avg = np.fft.fftshift(pwr / max(n, 1))

    # Step 2 — PSD normalisation → V²/Hz
    #   Dividing by (fs × win_power) converts raw |FFT|² to power spectral
    #   density in (normalised-ADC-units)²/Hz.  With our 1 V full-scale
    #   assumption the units become V²/Hz directly.
    psd = pwr_avg / (fs * win_power)

    # Step 3 — integrate over RBW → V² in the measurement bandwidth
    #   RBW = Hanning 3 dB BW = 1.44 × fs/seg  (≈ 2 160 Hz)
    rbw_hz = (fs / seg) * 1.44
    p_rbw  = psd * rbw_hz

    # Step 4 — convert V² → dBm  (50 Ω load assumed)
    #   P_dBm = 10·log10( V²_rms / 50 Ω / 0.001 W/mW )
    pwr_db = 10.0 * np.log10(p_rbw / 50.0 / 1e-3 + 1e-40)

    # Step 5 — empirical trim for actual ADC full-scale voltage
    #   Default 0.0 = 1 V full-scale assumed.  Increase if readings appear
    #   too low; decrease if too high.  (DBFS_REF_DBM is a module constant.)
    pwr_db += DBFS_REF_DBM

    # Absolute frequency axis (exact span, no gaps)
    f_bb      = np.fft.fftshift(np.fft.fftfreq(pad, d=1.0 / fs))
    freqs_mhz = (nco_hz + f_bb) / 1.0e6

    return freqs_mhz, pwr_db, n


# =============================================================================
#  Settings summary  (terminal printout)
# =============================================================================
def print_settings(ip: str):
    bw_hz  = SAMPLE_RATE // 2
    bw_khz = bw_hz / 1e3
    f_lo   = (NCO_FREQ - bw_hz) / 1e6
    f_hi   = (NCO_FREQ + bw_hz) / 1e6
    res_hz = SAMPLE_RATE / FFT_PAD_SIZE   # bin spacing (display resolution)

    print()
    print("=" * 60)
    print("  HermesLite 2  —  Spectrum Analyser  settings")
    print("=" * 60)
    print(f"  Radio IP          : {ip}")
    print(f"  HPSDR UDP port    : {HPSDR_PORT}")
    print()
    print(f"  NCO frequency     : {NCO_FREQ / 1e6:.3f} MHz")
    print(f"  TX tone offset    : +{TONE_OFFSET:,} Hz  →  RF = "
          f"{(NCO_FREQ + TONE_OFFSET) / 1e6:.4f} MHz")
    print(f"  TX amplitude      : {TX_AMPLITUDE:.3f}  (~{TX_AMPLITUDE**2*100:.0f} % of full power)")
    print()
    print(f"  Sample rate       : {SAMPLE_RATE // 1000} ksps  (Protocol-1 max)")
    print(f"  RX bandwidth      : ±{bw_khz:.0f} kHz  "
          f"({f_lo:.3f} – {f_hi:.3f} MHz)")
    print()
    if RF3_LOOPBACK_MODE:
        print(f"  TX drive          : 100 %  (DAC active on RF1; PA disabled)")
        print(f"  PA enable         : no     (PTT=1, DAC on; PA off — no ANT radiation)")
    elif LOOPBACK_CABLE_MODE:
        print(f"  TX drive          : 100 %  (DAC active on RF1; PA disabled)")
        print(f"  PA enable         : no     (PTT=0, TR switch routes ANT → LNA)")
    else:
        print(f"  TX drive          : 100 %  (reg 0x09 C1 = 0xF0)")
        print(f"  PA enable         : yes    (reg 0x09 bit 19 = C2 bit 3 = 0x08)")
    print(f"  LPF relay (40 m)  : C2 = 0x{lpf_c2(NCO_FREQ):02X}"
          f"  (code 0b{lpf_c2(NCO_FREQ) >> 1:07b}, via J16 GPIO)")
    print(f"  Duplex mode       : full   (reg 0x00 C4 bit 2 = 0x04)")
    lna_code = 0x40 | ((RX_LNA_GAIN_DB + 12) & 0x3F)
    print(f"  RX LNA gain       : +{RX_LNA_GAIN_DB} dB  "
          f"(reg 0x0a C4 = 0x{lna_code:02X}, FAST_LNA mode)")
    if RF3_LOOPBACK_MODE:
        print(f"  RX source         : RF3 direct injection  "
              f"(RF1 → {CABLE_ATT_DB} dB atten → RF3 SMA → pin 6, bypasses TR switch)")
    elif LOOPBACK_CABLE_MODE:
        print(f"  RX source         : Cable loopback  "
              f"(RF1 → {CABLE_ATT_DB} dB attenuator → ANT)")
    else:
        print(f"  RX source         : Pure-Signal loopback  "
              f"(RF1 / pre-amp tap, no cable needed)")
    print()
    rbw_hz  = (SAMPLE_RATE / SEGMENT_SIZE) * 1.44
    n_segs  = (CAPTURE_SIZE - SEGMENT_SIZE) // (SEGMENT_SIZE // 2) + 1
    print(f"  Capture size      : {CAPTURE_SIZE} IQ samples")
    print(f"  Segment size      : {SEGMENT_SIZE} samples  (Hanning windowed)")
    print(f"  Zero-pad to       : {FFT_PAD_SIZE} points  (interpolates bins)")
    print(f"  Overlap           : 50 %  →  step = {SEGMENT_SIZE // 2} samples")
    print(f"  Averages          : {n_segs} segments")
    print(f"  RBW               : {rbw_hz:.0f} Hz  ≈  {rbw_hz/1000:.1f} kHz")
    print(f"  Bin spacing       : {SAMPLE_RATE / FFT_PAD_SIZE:.2f} Hz  "
          f"(display resolution, not RBW)")
    print(f"  Display range     : {f_lo:.3f} – {f_hi:.3f} MHz  "
          f"(exact data span, no gaps)")
    print("=" * 60)
    print()


# =============================================================================
#  Main
# =============================================================================
def main():
    print()
    print("HermesLite 2  —  Spectrum Analyser")
    print("-" * 38)

    # ------------------------------------------------------------------
    # 1. Discover radio
    # ------------------------------------------------------------------
    ip = discover()
    if ip is None:
        print("\nERROR: HL2 not found.  Check Ethernet cable and power.")
        print("\nDiagnostics to try:")
        print("  ping 169.254.19.221")
        print("  ipconfig /all  (verify a 169.254.x.x adapter is up)")
        print("  Check Windows Firewall — UDP port 1024 must be allowed")
        input("\nPress Enter to exit ...")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Open socket and connect
    # ------------------------------------------------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", HPSDR_PORT))
    sock.settimeout(0.5)

    print("Connecting ...")
    ok, seq = connect(sock, ip)
    if not ok:
        print("ERROR: connection failed.  Check firewall (UDP port 1024).")
        print("\nDiagnostics to try:")
        print("  Confirm RUN LED on HL2 is flickering (not solid = not connected)")
        print("  Check Windows Firewall — UDP port 1024 inbound must be allowed")
        print("  Try:  ping 169.254.19.221")
        sock.close()
        input("\nPress Enter to exit ...")
        sys.exit(1)

    # Short timeout for capture phase (TX thread only calls sendto)
    sock.settimeout(0.002)

    # ------------------------------------------------------------------
    # 2b. Write safe register state (crash recovery)
    # ------------------------------------------------------------------
    # The STOP command does not reset FPGA registers.  If the previous run
    # was killed without a clean disconnect(), the HL2 still holds PA-enable,
    # PTT=1 and TX-drive from that session.  Writing safe defaults here
    # guarantees a clean slate before initialise_radio() programs the real
    # operating values.  (disconnect() does the same thing on clean exit,
    # so both paths leave the FPGA in a known state.)
    print("Writing safe register state ...")
    seq = write_safe_registers(sock, ip, start_seq=seq)
    time.sleep(0.050)   # allow FPGA to settle before operational init

    # ------------------------------------------------------------------
    # 3. Initialise registers
    # ------------------------------------------------------------------
    freq_word  = freq_to_word(NCO_FREQ)
    ptt        = not RX_TEST_MODE              # PTT off in RX test → ANT routed to LNA
    pa_enable  = ptt and PA_ENABLE             # PA controlled by PA_ENABLE constant
    print("Initialising registers ...")
    seq = initialise_radio(sock, ip, freq_word, start_seq=seq,
                           ptt=ptt, pa_enable=pa_enable)

    # ------------------------------------------------------------------
    # 4. Print all settings
    # ------------------------------------------------------------------
    print_settings(ip)

    # ------------------------------------------------------------------
    # 5. Start TX keep-alive thread
    # ------------------------------------------------------------------
    # Pass ALL five register sets so the thread can rotate through them.
    # This matches Protocol-1 / Quisk behaviour and keeps every register
    # continuously refreshed — not just reg 0x00.
    cc_regs = make_cc(freq_word, ptt=ptt, pa_enable=pa_enable)  # 5-tuple
    _tx_running.set()
    tx_thread = threading.Thread(
        target=_tx_thread_fn,
        args=(sock, ip, cc_regs, seq),
        daemon=True,
        name="HL2-TX")
    tx_thread.start()

    if RX_TEST_MODE:
        print(f"RX TEST MODE  —  PTT off, DAC muted, ANT → LNA → ADC")
        print(f"  Connect: NanoVNA PORT1 → 20 dB atten → ANT")
        print(f"  NanoVNA: set start=stop={NCO_FREQ / 1e6:.3f} MHz for CW output")
    elif RF3_LOOPBACK_MODE:
        print(f"TX tone on RF1  —  {(NCO_FREQ + TONE_OFFSET) / 1e6:.4f} MHz  "
              f"(PTT=1, PA off, RF3 loopback: RF1 → {CABLE_ATT_DB} dB atten → RF3)")
    elif LOOPBACK_CABLE_MODE:
        print(f"TX tone on RF1  —  {(NCO_FREQ + TONE_OFFSET) / 1e6:.4f} MHz  "
              f"(PTT=0, PA off, cable loopback: RF1 → {CABLE_ATT_DB} dB atten → ANT)")
    else:
        print(f"TX running  —  {(NCO_FREQ + TONE_OFFSET) / 1e6:.4f} MHz  "
              f"@ ~10 % power  (PA on, ~400 mW est.)")
    print("Allowing 1 s for HL2 to stabilise ...")
    time.sleep(1.0)

    # ------------------------------------------------------------------
    # 6. Capture IQ samples
    # ------------------------------------------------------------------
    cap_ms = CAPTURE_SIZE / SAMPLE_RATE * 1000.0
    print(f"Capturing {CAPTURE_SIZE} IQ samples "
          f"({SAMPLE_RATE // 1000} ksps, "
          f"~{cap_ms:.1f} ms of data) ...")

    iq, last_pkt = capture_iq(sock, n_samples=CAPTURE_SIZE)
    n  = len(iq)
    print(f"  Received {n} samples  "
          f"({n / SAMPLE_RATE * 1000:.2f} ms)")

    if n < CAPTURE_SIZE:
        print(f"  WARNING: expected {CAPTURE_SIZE}, got {n}  "
              f"— spectrum will be noisier")

    # --- HL2 status diagnostic -------------------------------------------
    if last_pkt:
        st = read_ep6_status(last_pkt)
        tx_inh = st.get("tx_inhibit")
        adc_ov = st.get("adc_overflow")
        if tx_inh is True:
            print(f"  *** TX INHIBIT ACTIVE — PA thermal protection has fired ***")
            print(f"  *** Transmitter is blocked. Power cycle HL2 to reset.   ***")
        elif tx_inh is False:
            print(f"  TX inhibit   : OK  (transmitter enabled)")
        if adc_ov:
            print(f"  ADC overflow : YES  (input too strong — reduce LNA gain)")
        elif adc_ov is False:
            print(f"  ADC overflow : no")
    # --- IQ amplitude diagnostic ------------------------------------------
    iq_c  = np.array([s[0] + 1j * s[1] for s in iq], dtype=np.complex128)
    rms   = float(np.sqrt(np.mean(np.abs(iq_c) ** 2)))
    peak  = float(np.abs(iq_c).max())
    print(f"  IQ RMS  : {rms:.5f}  ({20*np.log10(rms  + 1e-30):.1f} dBFS)")
    print(f"  IQ Peak : {peak:.5f}  ({20*np.log10(peak + 1e-30):.1f} dBFS)")
    print(f"  (Signal present if RMS > -40 dBFS; pure noise if < -50 dBFS)")
    # -----------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 7. Compute spectrum  (Welch's averaged method)
    # ------------------------------------------------------------------
    print("Computing power spectrum  (Welch's method) ...")
    freqs_mhz, power_db, n_segs = compute_spectrum(iq)

    peak_i = int(power_db.argmax())
    rbw_hz  = (SAMPLE_RATE / SEGMENT_SIZE) * 1.44
    print(f"  Segments averaged : {n_segs}")
    print(f"  RBW               : {rbw_hz:.0f} Hz  ≈  {rbw_hz/1000:.1f} kHz")
    expected_mhz = (NCO_FREQ + TONE_OFFSET) / 1e6
    print(f"  Peak: {power_db[peak_i]:.1f} dB  @  "
          f"{freqs_mhz[peak_i]:.4f} MHz  "
          f"(expected {expected_mhz:.4f} MHz)")

    # ------------------------------------------------------------------
    # 8. Open GUI and plot
    # ------------------------------------------------------------------
    def on_close():
        """
        Called when the GUI window is closed.
        Reads one last EP6 status packet for diagnostics before shutting down.
        """
        print()
        print("=" * 60)
        print("  GUI closed — running shutdown diagnostics")
        print("=" * 60)

        # --- Step 1: read EP6 status BEFORE stopping TX ---
        # TX thread is still running here so the HL2 is still sending EP6.
        print("  [shutdown] Step 1: reading EP6 status (TX still running) ...")
        sock.settimeout(0.5)
        ep6_diag_ok = False
        for attempt in range(5):
            try:
                data, _ = sock.recvfrom(2048)
                if len(data) == 1032 and data[0:2] == b"\xEF\xFE":
                    st       = read_ep6_status(data)
                    tx_inh   = st.get("tx_inhibit")
                    adc_ov   = st.get("adc_overflow")
                    raw_c0   = st.get("raw_c0", [])
                    print(f"  [EP6 status]  raw C0: "
                          f"{[hex(b) for b in raw_c0]}")
                    if tx_inh is True:
                        print("  [EP6 status]  TX inhibit : *** ACTIVE ***"
                              "  (PA thermal protection has fired)")
                    elif tx_inh is False:
                        print("  [EP6 status]  TX inhibit : OK"
                              "  (transmitter enabled)")
                    else:
                        print("  [EP6 status]  TX inhibit : unknown"
                              "  (dindex=0 not seen in this packet)")
                    if adc_ov is True:
                        print("  [EP6 status]  ADC overflow : YES")
                    elif adc_ov is False:
                        print("  [EP6 status]  ADC overflow : no")
                    ep6_diag_ok = True
                    break
            except socket.timeout:
                print(f"  [EP6 status]  attempt {attempt+1} timed out")
        if not ep6_diag_ok:
            print("  [EP6 status]  could not read EP6 status")

        # --- Step 2: stop TX thread ---
        print("  [shutdown] Step 2: stopping TX thread ...")
        _tx_running.clear()
        time.sleep(0.1)
        print("  [shutdown] TX thread stopped")

        # --- Step 3: disconnect (safe regs + STOP) ---
        print("  [shutdown] Step 3: disconnecting radio ...")
        disconnect(sock, ip)

        # --- Step 4: close socket ---
        print("  [shutdown] Step 4: closing socket ...")
        sock.close()
        print("  [shutdown] Done.  Socket closed.")
        print()

    root = tk.Tk()
    gui  = HermesGUI(root, on_close_cb=on_close)

    gui.set_status("Rendering spectrum ...")
    root.update()   # show window before heavy matplotlib draw
    gui.plot_spectrum(
        freqs_mhz, power_db,
        label=(f"RX  {SAMPLE_RATE // 1000} ksps  "
               f"| Welch  {SEGMENT_SIZE}-pt Hanning  "
               f"| RBW {rbw_hz/1000:.1f} kHz  "
               f"| {n_segs} avg"))

    print("GUI open.  Close the window to run shutdown diagnostics.")
    root.mainloop()

    # ------------------------------------------------------------------
    # 9. Post-GUI: keep terminal open for review
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Terminal staying open — review shutdown messages above.")
    print("  If run 2 fails, compare TX inhibit and safe-write output")
    print("  between run 1 and run 2.")
    print("=" * 60)
    input("\nPress Enter to exit ...")


# =============================================================================
if __name__ == "__main__":
    main()
