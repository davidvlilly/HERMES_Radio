#!/usr/bin/env python3
"""
waveforms.py - sounding-waveform generators for the HF channel sounder.

All generators return complex baseband IQ (numpy complex64), peak-normalised to
`amp` (<= 1.0 for the DAC).  Each also returns a small `spec` dict that the
matching analyser in channel.py needs to recover the channel.

Methods
-------
    lfm_chirp        - linear-FM sweep; matched-filter -> impulse response
    multitone_comb   - equally-spaced tones; FFT -> frequency response
    two_tone         - two tones; fade-correlation vs spacing
    ofdm_sounding    - known QPSK-loaded OFDM symbols; per-carrier H estimate

HF skywave defaults: sounding BW ~24 kHz (delay resolution ~42 us), repetition
window > a few ms (covers multi-hop delay spread), observed over seconds for the
~0.1-2 Hz Doppler.
"""

import numpy as np


def cw_tone(fs, freq=1000.0, amp=0.8):
    """
    Single continuous-wave tone at baseband `freq` Hz - the simplest probe, used
    to find data-path errors (phase discontinuities, dropped/repeated samples,
    packet glitches) and to measure raw SNR / power / frequency offset.

    The period is an integer number of cycles, so the looped waveform is perfectly
    phase-continuous: ANY phase jump in the capture comes from the data path, not
    from the transmitted signal.
    """
    n   = int(round(fs * 0.040))                    # 40 ms period (looped)
    cyc = max(1, int(round(freq * n / fs)))         # integer cycles -> seamless loop
    f   = cyc * fs / n
    t   = np.arange(n) / fs
    iq  = amp * np.exp(1j * 2.0 * np.pi * f * t)
    spec = {"kind": "cw", "fs": fs, "freq": f, "n": n, "amp": amp}
    return iq.astype(np.complex64), spec


def lfm_chirp(fs, bw, dur, amp=0.8):
    """
    Linear-FM chirp sweeping -bw/2 .. +bw/2 over `dur` seconds.

    Returns (iq, spec).  Delay resolution after pulse compression ~ 1/bw;
    unambiguous delay window = dur (the repetition interval).
    """
    n  = int(round(fs * dur))
    t  = np.arange(n) / fs
    k  = bw / dur                                   # chirp rate, Hz/s
    ph = 2.0 * np.pi * (-0.5 * bw * t + 0.5 * k * t * t)
    iq = amp * np.exp(1j * ph)
    spec = {"kind": "chirp", "fs": fs, "bw": bw, "dur": dur, "n": n,
            "ref": iq.astype(np.complex64)}
    return iq.astype(np.complex64), spec


def multitone_comb(fs, bw, dur, n_tones, amp=0.8):
    """
    Comb of `n_tones` equally spaced across +/-bw/2, Schroeder-phased for low
    peak-to-average.  FFT of the RX at the comb bins gives the frequency
    response H(f); IFFT(H) -> impulse response.

    Tone spacing = bw/(n_tones-1) sets the unambiguous delay window = 1/spacing;
    total span (bw) sets the delay resolution = 1/bw.
    """
    n     = int(round(fs * dur))
    t     = np.arange(n) / fs
    freqs = np.linspace(-bw / 2.0, bw / 2.0, n_tones)
    # Schroeder phases minimise PAPR for a multitone
    phases = np.pi * np.arange(n_tones) ** 2 / n_tones
    sig = np.zeros(n, dtype=np.complex128)
    for f, p in zip(freqs, phases):
        sig += np.exp(1j * (2.0 * np.pi * f * t + p))
    sig = sig / np.max(np.abs(sig)) * amp
    spec = {"kind": "comb", "fs": fs, "bw": bw, "dur": dur, "n": n,
            "freqs": freqs, "phases": phases}
    return sig.astype(np.complex64), spec


def two_tone(fs, dur, spacing, amp=0.8, center=0.0):
    """
    Two equal tones at center +/- spacing/2.  Watching whether the two fade
    together or independently, vs `spacing`, maps out the coherence bandwidth.
    """
    n  = int(round(fs * dur))
    t  = np.arange(n) / fs
    f1 = center - spacing / 2.0
    f2 = center + spacing / 2.0
    sig = np.exp(1j * 2.0 * np.pi * f1 * t) + np.exp(1j * 2.0 * np.pi * f2 * t)
    sig = sig / np.max(np.abs(sig)) * amp
    spec = {"kind": "two_tone", "fs": fs, "dur": dur, "spacing": spacing,
            "f1": f1, "f2": f2}
    return sig.astype(np.complex64), spec


def ofdm_sounding(fs, n_fft, n_used, cp, n_sym, amp=0.8, seed=0):
    """
    OFDM sounder: every used subcarrier carries a KNOWN QPSK pilot, with a
    cyclic prefix longer than the expected delay spread.  Per symbol the RX
    estimates H[k] = RX[k]/pilot[k] across the band -> frequency response;
    IFFT -> impulse response.  Variation across symbols -> Doppler.

    n_fft   : FFT size  (subcarrier spacing = fs/n_fft)
    n_used  : number of active subcarriers (centred, DC included)
    cp      : cyclic-prefix length in samples  (must exceed delay spread)
    n_sym   : number of OFDM symbols to transmit
    """
    rng  = np.random.default_rng(seed)
    half = n_used // 2
    # baseband subcarrier indices, mapped to FFT bins (negative -> high bins)
    idx  = np.arange(-half, n_used - half)
    bins = idx % n_fft
    # ONE fixed known QPSK pilot symbol, repeated every symbol.  A repeated
    # known training symbol lets the RX estimate H from ANY captured symbol with
    # no sequence sync (robust to an arbitrary grab start on the looped TX).
    qpsk = np.exp(1j * (np.pi / 4 + (np.pi / 2) * rng.integers(0, 4, size=n_used)))

    sym_len = n_fft + cp
    X = np.zeros(n_fft, dtype=np.complex128)
    X[bins] = qpsk
    x = np.fft.ifft(X) * n_fft / np.sqrt(n_used)         # time-domain symbol
    x = np.concatenate([x[n_fft - cp:], x])              # prepend cyclic prefix
    iq = np.tile(x, n_sym)

    iq = iq / np.max(np.abs(iq)) * amp
    spec = {"kind": "ofdm", "fs": fs, "n_fft": n_fft, "n_used": n_used,
            "cp": cp, "n_sym": n_sym, "bins": bins, "idx": idx,
            "qpsk": qpsk, "sym_len": sym_len, "bw": n_used * fs / n_fft}
    return iq.astype(np.complex64), spec


def _mseq(m):
    """Maximal-length sequence of length 2^m-1 as +/-1 chips (primitive taps)."""
    taps = {5: [5, 3], 6: [6, 5], 7: [7, 6], 8: [8, 6, 5, 4],
            9: [9, 5], 10: [10, 7], 11: [11, 9]}[m]
    N   = (1 << m) - 1
    reg = [1] * m
    out = np.empty(N)
    for i in range(N):
        out[i] = reg[-1]
        fb = 0
        for t in taps:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]
    return 1.0 - 2.0 * out                       # 0/1 -> +1/-1


def pn_dsss(fs, chip_rate, m=8, amp=0.8):
    """
    Direct-sequence spread spectrum: a BPSK m-sequence, matched-filtered at the
    RX -> channel impulse response with processing gain = code length.  Directly
    exercises the spread-spectrum gain that closes a negative-SNR link.

    Delay resolution = 1/chip_rate; unambiguous delay window = code_len/chip_rate
    (the code period, which loops on TX); processing gain = 10log10(code_len) dB.
    """
    spc      = int(round(fs / chip_rate))        # samples per chip
    code     = _mseq(m)
    code_len = len(code)
    chips    = np.repeat(code, spc).astype(np.complex128)
    one      = chips / np.max(np.abs(chips)) * amp
    spec = {"kind": "dsss", "fs": fs, "chip_rate": chip_rate, "code_len": code_len,
            "spc": spc, "ref": one.astype(np.complex64), "period_n": len(one),
            "bw": chip_rate, "pg_db": 10.0 * np.log10(code_len)}
    return one.astype(np.complex64), spec


# Convenience: HF-skywave default parameter sets (tune as needed).
def hf_defaults(fs):
    """Sensible HF-skywave sounding parameters for sample rate `fs`."""
    return {
        "chirp": dict(bw=24_000.0, dur=0.020),            # 24 kHz, 20 ms PRI
        "comb":  dict(bw=6_000.0, dur=0.040, n_tones=241),  # 25 Hz spacing -> fine coherence-BW resolution
        "two_tone": dict(dur=0.5, spacing=500.0),
        "ofdm":  dict(n_fft=4096, n_used=512, cp=2048, n_sym=64),
        #            spacing fs/4096=93.75 Hz, CP 2048/fs=5.3 ms window
        "dsss":  dict(chip_rate=24_000.0, m=9),   # 511 chips, ~21.3 ms, 27 dB PG
    }
