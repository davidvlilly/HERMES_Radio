#!/usr/bin/env python3
"""
analyze_capture.py — inspect a capture.npz saved by the GUI **Capture** button.

Tells you whether the partner's signal/preamble is actually present in the
received IQ, and at what carrier offset and SNR — so you can decide whether the
problem is on the TX side (nothing arriving) or the RX/decode side (signal is
there but not decoding).

    python analyze_capture.py capture_detect.npz     # captured in Detect mode
    python analyze_capture.py capture_call.npz        # captured in Call mode

The Capture button records RECEIVED IQ only (your own TX halves are skipped), so
the file is always the partner's incoming band — never your own transmit. The
filename tells you which role you were in when you recorded it.
"""

import sys
import numpy as np

import radio_engine as eng
from modem import DbpskModem, _bits_to_int


def decode_detail(m, x_bb, acq):
    """Print exactly how the frame decodes: header fields, payload, checksum, and
    a 'decision crispness' measure that tells garbled bits (timing/clock/
    distortion) apart from clean-but-wrong framing."""
    print("\n--- DECODE DETAIL ---")
    avail = (len(x_bb) - acq["start"]) // m.sps_bb
    syms  = m.demod_symbols(x_bb, acq["start"], acq["df_hz"], int(avail))
    if len(syms) < m.npre + 12:
        print("  too few symbols captured for a header"); return

    prod = np.real(syms[m.npre:] * np.conj(syms[m.npre - 1:-1]))   # differential
    bits = [1 if p < 0 else 0 for p in prod]

    # crispness: |decision| relative to its own median. Clean BPSK -> all ~1.0;
    # garbled (timing/clock/distortion) -> many near 0.
    mag = np.abs(prod[:60])
    med = np.median(mag) if len(mag) else 0.0
    weak = int(np.sum(mag < 0.25 * med)) if med > 0 else 0
    print(f"  symbols: {len(syms)}   data bits: {len(bits)}")
    print(f"  decision crispness: {weak}/{len(mag)} of the first bits are weak/ambiguous "
          f"(0 = crisp, many = garbled symbols)")

    code = _bits_to_int(bits[0:4])
    nlen = _bits_to_int(bits[4:12])
    print(f"  burst_code field : {code}")
    print(f"  length field     : {nlen} bytes")
    need = 12 + 8 * nlen + 8
    if len(bits) < need:
        print(f"  -> need {need} data bits for that length, only have {len(bits)}")
        print("  => the length field is garbage -> bit errors in the header.")
        return
    payload = bytes(_bits_to_int(bits[12 + 8 * i:20 + 8 * i]) & 0xFF for i in range(nlen))
    rx_sum  = _bits_to_int(bits[12 + 8 * nlen:20 + 8 * nlen])
    calc    = (code + nlen + sum(payload)) & 0xFF
    print(f"  payload bytes    : {payload!r}")
    print(f"  payload as text  : {payload.decode('latin1')!r}")
    print(f"  checksum  rx={rx_sum}  computed={calc}  "
          f"{'MATCH' if rx_sum == calc else 'MISMATCH'}")


def main(path="capture.npz"):
    d   = np.load(path)
    iq  = d["iq"]
    fs  = float(d["fs"])
    nco = float(d["nco"])
    lna = float(d["lna"])
    print(f"Loaded {len(iq)} samples = {len(iq)/fs:.1f} s   "
          f"fs={fs/1e3:.0f} kHz  NCO={nco/1e6:.4f} MHz  LNA={lna:.0f} dB\n")

    # 1) Where is the energy? (antenna-referred spectrum)
    seg = min(len(iq), eng.SPEC_PAD * 2)
    freqs, power = eng.compute_spectrum(
        [(x.real, x.imag) for x in iq[:seg]], nco, fs, dbfs_ref=eng.DBFS_REF - lna)
    if freqs is not None:
        i = int(np.argmax(power))
        off_hz = (freqs[i] - nco / 1e6) * 1e6
        print(f"Spectrum : peak {power[i]:+.0f} dBm at {off_hz:+.0f} Hz from NCO, "
              f"noise floor ~{np.median(power):+.0f} dBm")

    # 2) Is the preamble there? (wide matched-filter search)
    m    = DbpskModem(sample_rate=int(fs))
    x_bb = m.to_baseband(iq)
    acq  = m.acquire(x_bb)                       # wide carrier search (modem default)
    print(f"Matched filter : peak/noise {acq['snr_db']:.1f} dB "
          f"at offset {acq['df_hz']:+.0f} Hz  (threshold {eng.ACQ_THRESH_DB:.1f} dB)\n")

    # 3) Verdict
    if acq["snr_db"] >= eng.ACQ_THRESH_DB:
        dec = m.decode(x_bb, acq)
        ok  = dec and dec["ok"]
        print("VERDICT: PREAMBLE PRESENT — the partner's signal IS arriving.")
        print(f"         carrier offset = {acq['df_hz']:+.0f} Hz, "
              f"decode {'OK' if ok else 'failed CRC'}"
              + (f", code={dec['burst_code']} payload={dec['payload']!r}" if ok else ""))
        print("  => RX path is receiving. If the live radio still won't link, the")
        print("     issue is in timing/cadence or the offset is outside the live")
        print("     search range — set the search to cover this offset.")
        decode_detail(m, x_bb, acq)
    else:
        print("VERDICT: NO preamble found above threshold.")
        print("  => Look at the spectrum peak above:")
        print("     - energy bump near the pedestal+offset  -> signal is there but weak/garbled")
        print("       (levels, LNA, or a real but sub-threshold link).")
        print("     - just a flat noise floor               -> nothing arriving: partner TX not")
        print("       keying, wrong frequency/band, antenna, or propagation.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "capture.npz")
