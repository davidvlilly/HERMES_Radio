#!/usr/bin/env python3
"""
channel.py - sounding analysis engine.

Each analyse_*() takes the received IQ plus the waveform `spec` (from
waveforms.py) and returns a metrics dict via channel_metrics().  All methods
ultimately produce a set of channel impulse responses over time, from which we
derive:

    rms delay spread      -> coherence bandwidth   (frequency selectivity)
    Doppler spread        -> coherence time        (fading rate)
    in-band SNR

format_report() renders a metrics dict to the plain-text block that gets saved
per signal type and later fed to an LLM for the modulation decision.
"""

import numpy as np

C_DB_FLOOR = -60.0


def _matched_filter(rep, ref):
    """Circular matched filter (pulse compression): IFFT(FFT(rep)·conj(FFT(ref)))."""
    return np.fft.ifft(np.fft.fft(rep) * np.conj(np.fft.fft(ref)))


def _compressed_pulse(rep, ref, fs, bw):
    """Range-windowed pulse compression: Hann weighting across the chirp band so
    the sidelobes drop from the unwindowed ~-13 dB sinc to ~-32 dB (clean pulse).
    Costs ~1.5x main-lobe width; used only for the quality display, not the
    delay-spread metrics (those keep the full-resolution unwindowed MF)."""
    L  = len(ref)
    Rp = np.fft.fft(rep) * np.conj(np.fft.fft(ref))
    fb = np.fft.fftfreq(L, 1.0 / fs)
    idx = np.where(np.abs(fb) <= bw / 2.0)[0]      # bins inside the chirp band
    win = np.zeros(L)
    win[idx[np.argsort(fb[idx])]] = np.hanning(len(idx))
    return np.fft.ifft(Rp * win)


def _coarse_sync(rx, ref):
    """Sample offset of the first waveform period in rx (circular MF over 3 periods)."""
    L = len(ref)
    seg = rx[:min(len(rx), 3 * L)].astype(np.complex128)
    m = len(seg)
    c = np.abs(np.fft.ifft(np.fft.fft(seg) *
                           np.conj(np.fft.fft(ref, n=m))))
    return int(np.argmax(c)) % L


def _doppler_autocorr(tap, rep_interval):
    """
    Doppler spread + coherence time from the tap's complex autocorrelation.

    Coherence time T_c = lag where |R(tau)| falls to 0.5 (model-free, standard
    definition).  Additive noise is white, so it only inflates lag 0 -- we
    normalise by lag 1 and search from there, making this robust to noise and
    free of the window-broadening bias of the spectral-RMS method.

    For a Gaussian Doppler PSD of std sigma_f, R(tau)=exp(-2 pi^2 sigma_f^2 tau^2)
    so |R(T_c)|=0.5  =>  sigma_f = sqrt(ln2/2)/(pi*T_c) = 0.18739 / T_c.
    """
    n = len(tap)
    if n < 32:
        return 0.0, np.inf
    fmax = 10.0                          # plausible HF Doppler band (Hz)
    nseg = 4 if n >= 256 else 1          # Welch averaging cuts the variance
    seg  = n // nseg
    acc  = np.zeros(seg)
    w    = np.hanning(seg)
    for i in range(nseg):
        s = tap[i * seg:(i + 1) * seg] * w
        acc += np.abs(np.fft.fft(s)) ** 2
    S  = np.fft.fftshift(acc / nseg)
    fa = np.fft.fftshift(np.fft.fftfreq(seg, rep_interval))
    S  = np.clip(S - np.median(S), 0.0, None)        # subtract white-noise floor
    band = np.abs(fa) <= fmax                          # ignore far noise tails
    Sb, fb = S[band], fa[band]
    if Sb.sum() <= 0:
        return 0.0, np.inf
    sp = Sb / Sb.sum()
    dmean = float(np.sum(fb * sp))
    sigma = float(np.sqrt(np.sum((fb - dmean) ** 2 * sp)))
    if sigma <= 1e-6:
        return 0.0, np.inf
    return sigma, 0.18739 / sigma                      # coherence time (=0.5 autocorr)


def channel_metrics(H, delay_axis, rep_interval, snr_db=None, kind="",
                    floor_db=-20.0):
    """
    H            : complex impulse responses, shape [n_reps, n_delays]
    delay_axis   : delays (s) for the n_delays axis
    rep_interval : time (s) between rows of H (for the Doppler estimate)
    floor_db     : taps below (peak + floor_db) are excluded from the spread
    """
    H = np.atleast_2d(np.asarray(H))
    n_reps, n_del = H.shape

    # ── Power-delay profile (averaged over time) ─────────────────────────────
    pdp = np.mean(np.abs(H) ** 2, axis=0)
    pk  = pdp.max() + 1e-30
    pdp_db = 10.0 * np.log10(pdp / pk + 1e-30)

    # Significant taps: within floor_db of the peak AND at least 10 dB above the
    # bulk noise floor, so that when peak/noise is low the noise bins aren't
    # mistaken for multipath (that inflated the delay spread to ms with a sharp
    # peak).  Always keep the peak itself so the set is never empty.
    noise = float(np.median(pdp))
    sig = (pdp >= pk * 10.0 ** (floor_db / 10.0)) & (pdp >= noise * 10.0)   # noise + 10 dB
    sig[int(np.argmax(pdp))] = True
    p   = pdp[sig]
    tau = delay_axis[sig]
    mean_delay = float(np.sum(tau * p) / np.sum(p))
    rms_delay  = float(np.sqrt(np.sum((tau - mean_delay) ** 2 * p) / np.sum(p)))
    coh_bw     = (1.0 / (2.0 * np.pi * rms_delay)) if rms_delay > 1e-12 else np.inf
    # span between first and last significant tap (-X dB), a practical "max delay"
    idx = np.where(sig)[0]
    delay_span = float(delay_axis[idx[-1]] - delay_axis[idx[0]]) if len(idx) else 0.0

    # ── Doppler from the strongest tap's time series ─────────────────────────
    k0  = int(np.argmax(pdp))
    dop_spread = 0.0
    coh_time   = np.inf
    dop_axis = dop_psd = None
    if n_reps >= 8:
        tap = H[:, k0]
        dop_spread, coh_time = _doppler_autocorr(tap, rep_interval)
        # Doppler PSD kept for optional display
        w   = np.hanning(n_reps)
        S   = np.fft.fftshift(np.abs(np.fft.fft(tap * w)) ** 2)
        fa  = np.fft.fftshift(np.fft.fftfreq(n_reps, rep_interval))
        dop_axis, dop_psd = fa, 10.0 * np.log10(S / S.max() + 1e-30)

    return {
        "kind": kind, "n_reps": n_reps, "rep_interval": rep_interval,
        "snr_db": snr_db,
        "delay_axis": delay_axis, "pdp_db": pdp_db,
        "mean_delay": mean_delay, "rms_delay": rms_delay,
        "delay_span": delay_span, "coh_bw": coh_bw,
        "dop_spread": dop_spread, "coh_time": coh_time,
        "dop_axis": dop_axis, "dop_psd": dop_psd,
    }


def _band_snr(rep, ref):
    """Rough in-band SNR (dB) from the matched-filter peak vs off-peak floor."""
    mf = np.abs(_matched_filter(rep, ref)) ** 2
    pk = mf.max()
    noise = np.median(mf)                      # off-peak ~ noise+sidelobes
    return 10.0 * np.log10(pk / (noise + 1e-30) + 1e-30)


def _power_dbfs(rx):
    """Mean capture power in dBFS (0 dB = full scale)."""
    return float(10.0 * np.log10(np.mean(np.abs(np.asarray(rx)) ** 2) + 1e-30))


# ── Per-method analysis ──────────────────────────────────────────────────────

def analyse_chirp(rx, spec, max_delay_s=0.020):
    """LFM chirp: matched-filter each repetition -> impulse response."""
    fs, ref, L = spec["fs"], spec["ref"].astype(np.complex128), spec["n"]
    rx = np.asarray(rx, dtype=np.complex128)
    off = _coarse_sync(rx, ref)
    rx = rx[off:]
    n_reps = len(rx) // L
    W = min(L, max(4, int(round(max_delay_s * fs))))   # cap at one PRI (circular MF wraps)
    margin = int(round(1.0e-3 * fs))               # keep 1 ms before the peak (pre-cursor view)
    Wc = max(8, int(round(0.0015 * fs)))           # +/-1.5 ms pulse-display window
    H = []
    snrs = []
    pulse_acc = np.zeros(2 * Wc)
    for i in range(n_reps):
        rep = rx[i * L:(i + 1) * L]
        mf  = _matched_filter(rep, ref)
        # align so the strongest path is at index `margin`, keep 1 ms before it
        k = int(np.argmax(np.abs(mf)))
        H.append(np.roll(mf, margin - k)[:W])
        mfw = _compressed_pulse(rep, ref, fs, spec["bw"])        # windowed (low sidelobes)
        pulse_acc += np.abs(np.roll(mfw, Wc - k)[:2 * Wc]) ** 2  # centered pulse
        snrs.append(_band_snr(rep, ref))
    delay_axis = (np.arange(W) - margin) / fs       # delays now run from -1 ms
    m = channel_metrics(np.array(H), delay_axis, L / fs,
                        snr_db=float(np.median(snrs)), kind="LFM chirp")
    m["params"] = f"bw={spec['bw']/1e3:.1f} kHz, dur={spec['dur']*1e3:.1f} ms"
    m.update(_mf_quality(pulse_acc / max(n_reps, 1), Wc, fs, spec["bw"],
                         10.0 * np.log10(spec["bw"] * spec["dur"])))   # PG = TBP
    m["power_dbfs"] = _power_dbfs(rx)
    m["power_dbm"]  = m["power_dbfs"] + spec.get("dbm_offset", 0.0)
    return m


def _mf_quality(pulse, Wc, fs, bw, pg_db):
    """Pulse-compression quality: centered pulse (dB) + PG / resolution / SNR / PSLR."""
    pk = pulse.max() + 1e-30
    pulse_db = 10.0 * np.log10(pulse / pk + 1e-12)
    # -3 dB main-lobe full width -> measured resolution (windowing widens it)
    r = Wc
    while r + 1 < len(pulse_db) and pulse_db[r] > -3.0:
        r += 1
    res_us = 2.0 * (r - Wc) / fs * 1e6
    res_bins = max(1, int(round(fs / bw)))
    mask = np.ones(len(pulse), bool)
    mask[Wc - 3 * res_bins: Wc + 3 * res_bins] = False  # exclude the main lobe
    sidelobe = float(pulse_db[mask].max()) if mask.any() else float("nan")
    floor = float(np.median(pulse[mask])) if mask.any() else pk
    return {
        "mf_delay":     ((np.arange(len(pulse)) - Wc) / fs).tolist(),
        "mf_db":        pulse_db.tolist(),
        "mf_pg_db":     float(pg_db),
        "mf_res_us":    res_us,
        "mf_pslr_db":   sidelobe,
        "mf_peaksnr_db": 10.0 * np.log10(pk / (floor + 1e-30)),
    }


def analyse_dsss(rx, spec, max_delay_s=0.020):
    """DSSS/PN: matched-filter each code period -> impulse response (PG = code_len)."""
    fs, ref, L = spec["fs"], spec["ref"].astype(np.complex128), spec["period_n"]
    rx = np.asarray(rx, dtype=np.complex128)
    off = _coarse_sync(rx, ref)
    rx = rx[off:]
    n_reps = len(rx) // L
    W = min(L, max(4, int(round(max_delay_s * fs))))   # cap at one code period (circular MF wraps)
    margin = int(round(1.0e-3 * fs))               # keep 1 ms before the peak (pre-cursor view)
    H, snrs, peaks = [], [], []
    for i in range(n_reps):
        rep = rx[i * L:(i + 1) * L]
        mf  = _matched_filter(rep, ref)
        k   = int(np.argmax(np.abs(mf)))
        H.append(np.roll(mf, margin - k)[:W])      # peak at index `margin`, not 0
        peaks.append(mf[k])                         # complex despread symbol this period
        snrs.append(_band_snr(rep, ref))
    delay_axis = (np.arange(W) - margin) / fs       # delays now run from -1 ms
    m = channel_metrics(np.array(H), delay_axis, L / fs,
                        snr_db=float(np.median(snrs)), kind="DSSS/PN")
    m["params"] = (f"chip_rate={spec['chip_rate']/1e3:.1f} kchip/s, "
                   f"code={spec['code_len']} chips, PG={spec['pg_db']:.1f} dB")
    m["pg_db"] = float(spec["pg_db"])               # processing gain (= code length)
    m["power_dbfs"] = _power_dbfs(rx)
    m["power_dbm"]  = m["power_dbfs"] + spec.get("dbm_offset", 0.0)
    # BPSK despread constellation: normalise the per-period peaks and rotate the
    # cluster onto the vertical (Q) axis so the two BPSK symbols sit at +/-90deg.
    pk = np.array(peaks)
    mean = pk.mean()
    c = (pk / mean * 1j) if abs(mean) > 1e-30 else pk * 1j
    m["dsss_const_i"] = c.real.tolist()
    m["dsss_const_q"] = c.imag.tolist()
    return m


def analyse_ofdm(rx, spec):
    """OFDM: per symbol estimate H[k]=RX[k]/pilot[k] -> impulse response."""
    fs    = spec["fs"]
    n_fft = spec["n_fft"]
    cp    = spec["cp"]
    bins  = spec["bins"]
    qpsk  = spec["qpsk"]
    sym_len = spec["sym_len"]
    n_sym = spec["n_sym"]

    rx = np.asarray(rx, dtype=np.complex128)
    # coarse symbol-boundary sync (CP correlation; only needs to land within the
    # ISI-free part of the CP -- residual timing error is removed by peak-align).
    off = _ofdm_sync(rx, n_fft, cp, sym_len)
    # Blind carrier-frequency-offset removal (measured fresh from THIS capture;
    # the two-radio offset differs every test and drifts all day).  Done before
    # the per-symbol FFT so the loaded subcarriers don't smear into each other.
    rx, cfo_hz = _ofdm_cfo_correct(rx, off, n_fft, cp, sym_len, bins, fs)
    rx = rx[off:]
    n_avail = min(len(rx) // sym_len, 2000)            # process all (pilots loop)

    unused = np.setdiff1d(np.arange(n_fft), bins)      # noise-only carriers
    Hraw, Hus, snrs = [], [], []
    for s in range(n_avail):
        sym = rx[s * sym_len + cp: s * sym_len + cp + n_fft]
        if len(sym) < n_fft:
            break
        R = np.fft.fft(sym) / np.sqrt(len(bins))
        Hf = np.zeros(n_fft, dtype=np.complex128)
        Hf[bins] = R[bins] / qpsk                       # fixed pilots -> channel H
        Hraw.append(np.fft.ifft(Hf))                   # impulse response
        Hus.append(Hf[bins])                           # H at the used subcarriers
        # SNR: used-carrier power vs unused-carrier (noise-only) power
        noise = float(np.mean(np.abs(R[unused]) ** 2))
        sigpn = float(np.mean(np.abs(R[bins]) ** 2))
        snrs.append(10.0 * np.log10(max(sigpn - noise, 1e-30) / (noise + 1e-30)))
    Hraw = np.array(Hraw)

    # Align all symbols by the common (average-PDP) peak -> relative delay axis.
    margin = int(round(0.5e-3 * fs))                   # allow taps up to 0.5 ms early
    k0 = int(np.argmax(np.mean(np.abs(Hraw) ** 2, axis=0)))
    H  = np.roll(Hraw, margin - k0, axis=1)
    W  = min(cp, n_fft)
    delay_axis = (np.arange(W) - margin) / fs
    m = channel_metrics(H[:, :W], delay_axis, sym_len / fs,
                        snr_db=float(np.median(snrs)) if snrs else None,
                        kind="OFDM")
    m["params"] = (f"n_fft={n_fft}, used={spec['n_used']}, cp={cp} "
                   f"({cp/fs*1e3:.1f} ms), df={fs/n_fft:.1f} Hz, bw={spec['bw']/1e3:.1f} kHz")
    m["ofdm_cfo_hz"] = cfo_hz                       # blind per-capture CFO removed
    # Channel frequency response across the used subcarriers (for display):
    # mean shape (flat vs selective) and per-carrier deepest fade.
    if len(Hus):
        mag   = np.abs(np.array(Hus))                   # [n_sym, n_used]
        ref   = mag.mean(0).max() + 1e-30               # peak of mean shape -> 0 dB
        idx   = np.asarray(spec["idx"]); order = np.argsort(idx)
        m["ofdm_freq"]  = (idx[order] * (fs / n_fft)).tolist()              # Hz
        m["ofdm_hf_db"] = (20*np.log10(mag.mean(0)[order]/ref + 1e-9)).tolist()
        m["ofdm_hf_lo"] = (20*np.log10(mag.min(0)[order]/ref + 1e-9)).tolist()
    # data fidelity: per-symbol-aligned QPSK constellation + EVM (Hus = R/qpsk = He)
    if len(Hus) >= 2:
        m.update(_ofdm_fidelity(np.array(Hus), qpsk))
    m["power_dbfs"] = _power_dbfs(rx)
    m["power_dbm"]  = m["power_dbfs"] + spec.get("dbm_offset", 0.0)
    return m


def _qpsk_decode(z):
    """Nearest-QPSK symbol index 0..3 on the pi/4+k*pi/2 grid."""
    return np.mod(np.round((np.angle(z) - np.pi / 4) / (np.pi / 2)), 4).astype(int)


def _ofdm_fidelity(He, qpsk):
    """QPSK constellation + EVM from per-symbol channel estimates He = R/qpsk.

    Each symbol's common phase + timing slope is removed first (a sample slip
    appears as a phase ramp across subcarriers), so a mid-capture timing slip
    doesn't smear the fidelity estimate.
    """
    nsym, nu = He.shape
    kax = np.arange(nu)

    def _align_to(ref):
        # Wrap-free, magnitude-weighted phase+slope fit: the timing slope is the
        # mean phase step between adjacent subcarriers and the common phase is the
        # residual, both via complex sums so deep multipath fades (where the phase
        # is pure noise) self-weight to ~zero.  Avoids the np.unwrap+polyfit blow-up
        # where one noisy fade triggered a spurious 2*pi jump and wrecked a symbol.
        out = np.empty_like(He)
        for s in range(nsym):
            z = He[s] * np.conj(ref)
            m = float(np.angle(np.vdot(z[:-1], z[1:])))        # phase step / bin
            a = float(np.angle(np.sum(z * np.exp(-1j * m * kax))))
            out[s] = He[s] * np.exp(-1j * (a + m * kax))
        return out

    # Build the alignment reference WITHOUT trusting any single (possibly noise-
    # unlucky) symbol.  Bootstrap by removing each symbol's common phase via the
    # cumulative symbol-to-symbol phase difference (an average over all carriers,
    # so no lone symbol dominates), average those to a low-noise reference, then
    # iterate the full phase+slope alignment against the re-averaged reference.
    dphase = np.concatenate(([0.0],
                np.cumsum([np.angle(np.vdot(He[s - 1], He[s])) for s in range(1, nsym)])))
    ref = (He * np.exp(-1j * dphase)[:, None]).mean(0)
    for _ in range(3):
        aligned = _align_to(ref)
        ref = aligned.mean(0)
    H  = aligned.mean(0)
    eq = (aligned / H) * qpsk                      # per-symbol constellation -> qpsk
    eq_mean = eq.mean(0)
    evm1 = float(np.sqrt(np.mean(np.abs(aligned - H) ** 2) / np.mean(np.abs(H) ** 2)))
    evm_avg = evm1 / np.sqrt(max(nsym, 1))
    sent = _qpsk_decode(qpsk)
    ser = float(np.mean(_qpsk_decode(eq) != sent[None, :]))
    flat = eq.ravel()
    step = max(1, flat.size // 3000)               # subsample the cloud for the GUI
    nshow = min(64, nu)
    return {
        "ofdm_const_i": flat.real[::step].tolist(),
        "ofdm_const_q": flat.imag[::step].tolist(),
        "ofdm_cmean_i": eq_mean.real.tolist(),
        "ofdm_cmean_q": eq_mean.imag.tolist(),
        "ofdm_evm1": evm1, "ofdm_evm_avg": evm_avg,
        "ofdm_snr1": -20.0 * np.log10(evm1 + 1e-12),
        "ofdm_snr_avg": -20.0 * np.log10(evm_avg + 1e-12),
        "ofdm_ser": ser, "ofdm_nsym": int(nsym),
        "ofdm_seq_sent": sent[:nshow].tolist(),
        "ofdm_seq_recv": _qpsk_decode(eq_mean)[:nshow].tolist(),
    }


def _ofdm_sync(rx, n_fft, cp, sym_len):
    """Find the symbol start via the cyclic-prefix correlation peak."""
    max_i = len(rx) - (n_fft + cp)
    if max_i <= 0:
        return 0
    L = min(sym_len, max_i)                       # search one symbol period
    metric = np.empty(L)
    for i in range(L):
        metric[i] = np.abs(np.vdot(rx[i:i + cp], rx[i + n_fft:i + n_fft + cp]))
    return int(np.argmax(metric))


def _ofdm_cfo_correct(rx, off, n_fft, cp, sym_len, bins, fs):
    """
    Estimate the carrier-frequency offset BLIND from this capture and remove it,
    returning (derotated rx, estimated CFO in Hz).

    The two radios free-run on independent clocks, so their offset is whatever it
    happens to be on this grab -- a few Hz, different every test, drifting through
    the day -- so we MEASURE it from the signal and never assume a value.

    Two parts:
      * fractional (|df| < half a subcarrier): angle of the cyclic-prefix self-
        correlation (van de Beek estimator).  The CP is a copy of the symbol tail
        n_fft samples earlier, so conj(CP)*tail has phase 2*pi*df*n_fft/fs.
        Averaged over every symbol in the capture to beat down noise.
      * integer-subcarrier: on a hot/cold day the offset can exceed half a
        subcarrier (5.9 Hz here), which the CP angle wraps.  After the fractional
        fix we find where the loaded band actually landed (it is a solid block of
        n_used carriers) and shift back by that many bins.

    Uncorrected, even the ~3 Hz seen on 2026-06-21 collapses the constellation --
    this is exactly what failed OTA.  Matched-filter modes (chirp/DSSS) don't need
    this because an offset only slides their correlation peak.
    """
    N  = len(rx)
    df = fs / n_fft
    idx = np.arange(N)
    n_avail = max(1, (N - off) // sym_len)

    # 1) fractional CFO from the CP self-correlation, summed over all symbols
    acc = 0j
    for s in range(n_avail):
        i = off + s * sym_len
        if i + n_fft + cp > N:
            break
        acc += np.vdot(rx[i:i + cp], rx[i + n_fft:i + n_fft + cp])
    rate = float(np.angle(acc)) / n_fft           # rad/sample = 2*pi*df_frac/fs
    rx = rx * np.exp(-1j * rate * idx)
    cfo = rate * fs / (2.0 * np.pi)

    # 2) integer-subcarrier offset: average symbol power spectrum, find the shift
    #    of the known occupied-carrier mask that best matches where energy landed
    acc2 = np.zeros(n_fft)
    for s in range(min(n_avail, 32)):
        i = off + s * sym_len + cp
        if i + n_fft > N:
            break
        acc2 += np.abs(np.fft.fft(rx[i:i + n_fft])) ** 2
    occ = np.zeros(n_fft)
    occ[bins] = 1.0
    shifts = np.arange(-6, 7)
    scores = [np.sum(acc2 * np.roll(occ, int(g))) for g in shifts]
    g = int(shifts[int(np.argmax(scores))])
    if g:
        rx = rx * np.exp(-1j * (2.0 * np.pi * g * df / fs) * idx)
        cfo += g * df
    return rx, float(cfo)


def analyse_comb(rx, spec):
    """Multi-tone comb: read H at the comb bins -> frequency response -> IR."""
    fs, freqs, n = spec["fs"], spec["freqs"], spec["n"]
    tx_phase = np.exp(-1j * np.asarray(spec["phases"]))   # undo known Schroeder phasing
    rx = np.asarray(rx, dtype=np.complex128)
    # process in blocks of one waveform length to get H(f) over time
    nb = max(1, len(rx) // n)
    fb = np.fft.fftfreq(n, 1.0 / fs)
    bins = np.array([np.argmin(np.abs(fb - f)) for f in freqs])
    Hsnap = []
    for b in range(nb):
        R = np.fft.fft(rx[b * n:(b + 1) * n])
        Hsnap.append(R[bins] * tx_phase)         # true channel H(f) at the tones
    Hsnap = np.array(Hsnap)                       # [nb, n_tones] freq response
    span = freqs[-1] - freqs[0]
    spacing = freqs[1] - freqs[0]
    nt = Hsnap.shape[1]
    # Impulse response PER block, then peak-align to a common tap and stack the
    # POWER (incoherent).  Averaging H coherently across blocks (the old way) let
    # a misaligned grab slide the peak to a random delay and let even a few-Hz
    # carrier offset rotate the per-block phases so the average cancelled to noise.
    # Per-block ifft + |.|^2 averaging (like the matched-filter modes) is robust to
    # both: the grab offset and the offset only attenuate, never randomise.
    Hir = np.fft.ifft(Hsnap, axis=1)              # [nb, nt] impulse response/block
    margin = max(1, int(round(0.001 * span)))     # ~1 ms pre-cursor (1/span bins)
    k0 = int(np.argmax(np.mean(np.abs(Hir) ** 2, axis=0)))
    Hir = np.roll(Hir, margin - k0, axis=1)       # common peak -> index `margin`
    delay_axis = (np.arange(nt) - margin) / span  # delay res = 1/span, from -1 ms
    m = channel_metrics(Hir, delay_axis, n / fs, kind="Comb")
    # Spaced-frequency correlation (the coherence function), averaged over the
    # time snapshots.  Do NOT remove the mean: the flat/coherent part of H(f) is
    # exactly what keeps nearby frequencies correlated, so subtracting it would
    # collapse the curve.  corr[k] = |<H*(f) H(f+k)>| over frequency & snapshots.
    corr = _freq_corr(Hsnap)
    m["coh_bw_direct"]  = _coh_bw_from_corr(corr, spacing)
    m["coh_curve_freq"] = (np.arange(len(corr)) * spacing).tolist()  # Hz lag
    m["coh_curve"]      = corr.tolist()                              # |rho|, 0..1
    m["params"] = (f"{len(freqs)} tones, span={span/1e3:.1f} kHz, "
                   f"spacing={spacing:.0f} Hz")
    m["power_dbfs"] = _power_dbfs(rx)
    m["power_dbm"]  = m["power_dbfs"] + spec.get("dbm_offset", 0.0)
    return m


def _freq_corr(Hsnap):
    """Coherence function |<H*(f)H(f+k)>| vs lag k, averaged over time snapshots."""
    Hsnap = np.atleast_2d(Hsnap)
    nt = Hsnap.shape[1]
    corr = np.array([np.abs((np.conj(Hsnap[:, :nt - k]) * Hsnap[:, k:]).mean())
                     for k in range(nt)])
    return corr / (corr[0] + 1e-30)


def _coh_bw_from_corr(corr, spacing):
    """Coherence BW = freq lag where the correlation falls to 0.5 (interpolated)."""
    for k in range(1, len(corr)):
        if corr[k] < 0.5:
            frac = (corr[k - 1] - 0.5) / (corr[k - 1] - corr[k] + 1e-30)
            return float((k - 1 + frac) * spacing)
    return np.inf


def analyse_two_tone(rx, spec, block_s=0.05):
    """Two tones: track each tone's complex amplitude over time -> fade corr."""
    fs, f1, f2, spacing = spec["fs"], spec["f1"], spec["f2"], spec["spacing"]
    rx = np.asarray(rx, dtype=np.complex128)
    nb = max(8, int(block_s * fs))
    nblocks = len(rx) // nb
    t = np.arange(nb) / fs
    a1 = np.empty(nblocks, dtype=np.complex128)
    a2 = np.empty(nblocks, dtype=np.complex128)
    for b in range(nblocks):
        seg = rx[b * nb:(b + 1) * nb]
        a1[b] = np.vdot(np.exp(1j * 2 * np.pi * f1 * t), seg) / nb
        a2[b] = np.vdot(np.exp(1j * 2 * np.pi * f2 * t), seg) / nb
    # fade correlation between the two tones
    x = a1 - a1.mean()
    y = a2 - a2.mean()
    rho = float(np.abs(np.vdot(x, y)) /
                (np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real) + 1e-30))
    # Doppler from one tone's amplitude time series
    rep = nb / fs
    w = np.hanning(nblocks)
    S = np.fft.fftshift(np.abs(np.fft.fft(a1 * w)) ** 2)
    fa = np.fft.fftshift(np.fft.fftfreq(nblocks, rep))
    sp = S / (S.sum() + 1e-30)
    dmean = np.sum(fa * sp)
    dop = float(np.sqrt(np.sum((fa - dmean) ** 2 * sp)))
    # fade envelopes vs time (dB, shared reference) for the display
    ref = max(np.abs(a1).max(), np.abs(a2).max()) + 1e-30
    return {
        "kind": "Two-tone", "snr_db": None,
        "spacing": spacing, "fade_corr": rho,
        "dop_spread": dop, "coh_time": (1.0 / (2 * np.pi * dop)) if dop > 1e-6 else np.inf,
        "params": f"spacing={spacing:.0f} Hz, blocks={nblocks}",
        "note": ("tones fade together (rho~1) => spacing < coherence BW; "
                 "independent (rho~0) => spacing > coherence BW"),
        "t_axis":   (np.arange(nblocks) * rep).tolist(),
        "tone1_db": (20 * np.log10(np.abs(a1) / ref + 1e-9)).tolist(),
        "tone2_db": (20 * np.log10(np.abs(a2) / ref + 1e-9)).tolist(),
    }


def analyse_cw(rx, spec):
    """
    CW tone diagnostic: measure power / frequency / frequency offset / SNR and
    detect data-path errors (phase discontinuities, amplitude dropouts).  Also
    returns a 100 Hz-RBW spectrum of the capture for display.
    """
    fs, f0 = spec["fs"], spec["freq"]
    rx = np.asarray(rx, dtype=np.complex128)
    N  = len(rx)
    idx = np.arange(N)

    # frequency from the FFT peak with parabolic interpolation (robust to the
    # DC/LO-leakage spike, which sits a fixed offset away from the tone)
    W = np.hanning(N)
    P = np.abs(np.fft.fft(rx * W)) ** 2
    k = int(np.argmax(P))
    a, b2, c = (np.log(P[(k - 1) % N] + 1e-30), np.log(P[k] + 1e-30),
                np.log(P[(k + 1) % N] + 1e-30))
    den = a - 2.0 * b2 + c
    f_meas = (k + (0.5 * (a - c) / den if den != 0 else 0.0)) * fs / N
    if f_meas > fs / 2:
        f_meas -= fs

    # SNR + tone power from the spectrum: tone bins vs the noise floor, excluding
    # both the tone region and the DC region (so LO leakage doesn't skew it)
    tb = max(2, int(round(N * 25.0 / fs)))            # +/-25 Hz tone window
    dcb = max(2, int(round(N * 60.0 / fs)))           # +/-60 Hz DC exclusion
    dist_tone = np.minimum((idx - k) % N, (k - idx) % N)
    dist_dc   = np.minimum(idx, N - idx)
    tone_sel  = dist_tone <= tb
    noise_sel = (dist_tone > tb) & (dist_dc > dcb)
    nfloor  = float(np.median(P[noise_sel])) + 1e-30
    p_tone  = max(float(np.sum(P[tone_sel]) - nfloor * np.sum(tone_sel)), 1e-30)
    snr     = 10.0 * np.log10(p_tone / (nfloor * N))  # tone vs full-band noise
    p_dbfs  = 10.0 * np.log10(np.mean(np.abs(rx) ** 2) + 1e-30)

    # derotate by the measured tone for the phase-discontinuity check
    d = rx * np.exp(-1j * 2.0 * np.pi * f_meas * idx / fs)

    # phase discontinuities: per-sample increment of the derotated tone should be
    # ~constant; outliers = dropped/repeated samples or packet glitches.
    amp   = np.abs(rx)
    a_med = float(np.median(amp)) + 1e-30
    inc   = np.angle(d[1:] * np.conj(d[:-1]))            # phase step, (-pi, pi]
    med   = np.median(inc)
    mad   = np.median(np.abs(inc - med)) * 1.4826 + 1e-12
    thr   = max(0.20, 10.0 * mad)                        # rad
    healthy = (amp[:-1] > 0.3 * a_med) & (amp[1:] > 0.3 * a_med)
    jumps = np.where(healthy & (np.abs(inc - med) > thr))[0]
    drops = np.where(amp < 0.2 * a_med)[0]
    max_jump_deg = float(np.degrees(np.max(np.abs(inc - med)))) if N > 1 else 0.0

    # 100 Hz-RBW Welch spectrum of the capture (for display)
    seg = min(N, max(64, int(round(1.44 * fs / 100.0))))
    w   = np.hanning(seg); wp2 = np.sum(w) ** 2
    acc = np.zeros(seg); cnt = 0
    for s0 in range(0, N - seg + 1, seg // 2):
        acc += np.abs(np.fft.fft(rx[s0:s0 + seg] * w)) ** 2; cnt += 1
    P = np.fft.fftshift(acc / max(cnt, 1))

    return {
        "kind": "CW", "snr_db": float(snr),
        "cw_freq": float(f_meas), "cw_freq_offset": float(f_meas - f0),
        "cw_power_dbfs": float(p_dbfs),
        "cw_power_dbm":  float(p_dbfs + spec.get("dbm_offset", 0.0)),
        "cw_phase_jumps": int(len(jumps)), "cw_max_jump_deg": max_jump_deg,
        "cw_dropouts": int(len(drops)),
        "cw_jump_times": (jumps[:20] / fs).tolist(),
        "cw_spec_f":  np.fft.fftshift(np.fft.fftfreq(seg, 1.0 / fs)).tolist(),
        "cw_spec_db": (10.0 * np.log10(P / wp2 + 1e-24)).tolist(),
        "params": f"f0={f0:.1f} Hz, N={N}, RBW=100 Hz",
    }


# ── Report formatting ────────────────────────────────────────────────────────

def _fmt_hz(x):
    if x == np.inf:
        return ">band"
    return f"{x/1e3:.2f} kHz" if abs(x) >= 1e3 else f"{x:.1f} Hz"


def _fmt_s(x):
    if x == np.inf:
        return ">capture"
    return f"{x*1e3:.3f} ms" if abs(x) < 1.0 else f"{x:.3f} s"


# One-line explanation appended per mode (the only prose in the report).
_MODE_NOTE = {
    "CW":        "freq offset = carrier difference between the two radios; "
                 "data path flags dropped/glitched samples.",
    "LFM chirp": "matched filter adds proc gain to lift the raw signal to the "
                 "detected SNR; carrier-offset tolerant.",
    "DSSS/PN":   "spreading proc gain (= code length) lifts the raw signal to the "
                 "detected SNR; carrier-offset tolerant.",
    "Comb":      "coherence BW from H(f) correlation; smaller = more frequency-"
                 "selective (multipath) fading.",
    "OFDM":      "freq offset (radio-to-radio) measured and removed each capture; "
                 "EVM = residual constellation error.",
    "Two-tone":  "fade corr ~1 => tone spacing inside the coherence BW; ~0 => beyond it.",
}


def _row(label, value):
    return f"{label:<13}: {value}"


def format_report(m, header=None):
    """Render a metrics dict to a compact label/value/units block (terminal + .txt)."""
    kind = m.get("kind", "?")
    L = []
    if header:
        L.append(header.rstrip())
        L.append("")
    L.append(_row("method", kind))

    snr = m.get("snr_db")
    pg  = m.get("mf_pg_db", m.get("pg_db"))
    # Matched-filter modes: the raw -> proc gain -> detected SNR story.
    if kind in ("LFM chirp", "DSSS/PN") and snr is not None and pg is not None:
        L.append(_row("raw SNR",      f"{snr - pg:+.1f} dB"))
        L.append(_row("proc gain",    f"{pg:+.1f} dB"))
        L.append(_row("detected SNR", f"{snr:+.1f} dB"))
    elif snr is not None:
        L.append(_row("SNR", f"{snr:.1f} dB"))

    # Signal power for every mode: dBm primary (= dBFS + offset), dBFS in parens.
    pdbfs = m.get("power_dbfs", m.get("cw_power_dbfs"))
    pdbm  = m.get("power_dbm",  m.get("cw_power_dbm"))
    if pdbfs is not None:
        if pdbm is None:
            pdbm = pdbfs
        L.append(_row("signal power", f"{pdbm:.1f} dBm ({pdbfs:.1f} dBFS)"))

    if kind == "CW":
        L.append(_row("freq offset",  f"{m['cw_freq_offset']:+.2f} Hz"))
        errs = m["cw_phase_jumps"] + m["cw_dropouts"]
        L.append(_row("data path",    "clean" if errs == 0 else f"{errs} errors"))

    if m.get("ofdm_cfo_hz") is not None:
        L.append(_row("freq offset", f"{m['ofdm_cfo_hz']:+.2f} Hz"))
    if m.get("ofdm_evm_avg") is not None:
        L.append(_row("EVM", f"{m['ofdm_evm_avg']*100:.1f} % (avg of {m['ofdm_nsym']})"))

    # Channel metrics (sounding result) -- delay spread, coherence BW, Doppler.
    if "rms_delay" in m:
        L.append(_row("delay spread", _fmt_s(m["rms_delay"])))
    cbw = m.get("coh_bw_direct", m.get("coh_bw"))
    if cbw is not None:
        L.append(_row("coherence BW", _fmt_hz(cbw)))
    if m.get("fade_corr") is not None:
        L.append(_row("fade corr", f"{m['fade_corr']:.2f} @ {m['spacing']:.0f} Hz"))
    if m.get("dop_spread") is not None and ("rms_delay" in m or m.get("fade_corr") is not None):
        L.append(_row("Doppler", f"{m['dop_spread']:.2f} Hz"))

    note = _MODE_NOTE.get(kind)
    if note:
        L.append("")
        L.append("note: " + note)
    return "\n".join(L)
