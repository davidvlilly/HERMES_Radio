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

    sig = pdp >= pk * 10.0 ** (floor_db / 10.0)          # significant taps
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


# ── Per-method analysis ──────────────────────────────────────────────────────

def analyse_chirp(rx, spec, max_delay_s=0.006):
    """LFM chirp: matched-filter each repetition -> impulse response."""
    fs, ref, L = spec["fs"], spec["ref"].astype(np.complex128), spec["n"]
    rx = np.asarray(rx, dtype=np.complex128)
    off = _coarse_sync(rx, ref)
    rx = rx[off:]
    n_reps = len(rx) // L
    W = max(4, int(round(max_delay_s * fs)))
    Wc = max(8, int(round(0.0015 * fs)))           # +/-1.5 ms pulse-display window
    H = []
    snrs = []
    pulse_acc = np.zeros(2 * Wc)
    for i in range(n_reps):
        rep = rx[i * L:(i + 1) * L]
        mf  = _matched_filter(rep, ref)
        # align so the strongest path is at index 0, keep [0, W)
        k = int(np.argmax(np.abs(mf)))
        H.append(np.roll(mf, -k)[:W])
        mfw = _compressed_pulse(rep, ref, fs, spec["bw"])        # windowed (low sidelobes)
        pulse_acc += np.abs(np.roll(mfw, Wc - k)[:2 * Wc]) ** 2  # centered pulse
        snrs.append(_band_snr(rep, ref))
    delay_axis = np.arange(W) / fs
    m = channel_metrics(np.array(H), delay_axis, L / fs,
                        snr_db=float(np.median(snrs)), kind="LFM chirp")
    m["params"] = f"bw={spec['bw']/1e3:.1f} kHz, dur={spec['dur']*1e3:.1f} ms"
    m.update(_mf_quality(pulse_acc / max(n_reps, 1), Wc, fs, spec["bw"],
                         10.0 * np.log10(spec["bw"] * spec["dur"])))   # PG = TBP
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


def analyse_dsss(rx, spec, max_delay_s=0.006):
    """DSSS/PN: matched-filter each code period -> impulse response (PG = code_len)."""
    fs, ref, L = spec["fs"], spec["ref"].astype(np.complex128), spec["period_n"]
    rx = np.asarray(rx, dtype=np.complex128)
    off = _coarse_sync(rx, ref)
    rx = rx[off:]
    n_reps = len(rx) // L
    W = max(4, int(round(max_delay_s * fs)))
    H, snrs = [], []
    for i in range(n_reps):
        rep = rx[i * L:(i + 1) * L]
        mf  = _matched_filter(rep, ref)
        k   = int(np.argmax(np.abs(mf)))
        H.append(np.roll(mf, -k)[:W])
        snrs.append(_band_snr(rep, ref))
    delay_axis = np.arange(W) / fs
    m = channel_metrics(np.array(H), delay_axis, L / fs,
                        snr_db=float(np.median(snrs)), kind="DSSS/PN")
    m["params"] = (f"chip_rate={spec['chip_rate']/1e3:.1f} kchip/s, "
                   f"code={spec['code_len']} chips, PG={spec['pg_db']:.1f} dB")
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
    ref = He[0]
    aligned = np.empty_like(He)
    for s in range(nsym):
        dphi = np.unwrap(np.angle(He[s] * np.conj(ref)))
        b, a = np.polyfit(kax, dphi, 1)
        aligned[s] = He[s] * np.exp(-1j * (a + b * kax))
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
    # IR from the (mean) frequency response across the comb
    Hmean = np.mean(Hsnap, axis=0)
    h = np.fft.ifft(Hmean)
    span = freqs[-1] - freqs[0]
    spacing = freqs[1] - freqs[0]
    delay_axis = np.arange(len(h)) / span         # delay res = 1/span
    W = len(h) // 2
    m = channel_metrics(np.tile(h[:W], (1, 1)), delay_axis[:W], n / fs,
                        kind="Comb")
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


def format_report(m, header=None):
    """Render a metrics dict to a plain-text block for the per-mode .txt file."""
    L = []
    if header:
        L.append(header.rstrip())
        L.append("")
    L.append(f"METHOD          : {m.get('kind','?')}")
    if m.get("params"):
        L.append(f"params          : {m['params']}")
    if m.get("snr_db") is not None:
        L.append(f"in-band SNR     : {m['snr_db']:.1f} dB")
    if m.get("kind") == "CW":
        L.append(f"frequency       : {m['cw_freq']:.2f} Hz   (offset {m['cw_freq_offset']:+.2f} Hz)")
        L.append(f"power           : {m['cw_power_dbfs']:.1f} dBFS  "
                 f"({m.get('cw_power_dbm', m['cw_power_dbfs']):.1f} dBm)")
        L.append(f"phase jumps     : {m['cw_phase_jumps']}  (max {m['cw_max_jump_deg']:.1f} deg)"
                 + ("  <-- DATA-PATH ERRORS" if m['cw_phase_jumps'] else "  (clean)"))
        L.append(f"amplitude drops : {m['cw_dropouts']}"
                 + ("  <-- lost/zero-filled packets" if m['cw_dropouts'] else ""))
        if m.get("cw_jump_times"):
            L.append(f"first jump times: "
                     + ", ".join(f"{t*1e3:.1f}ms" for t in m['cw_jump_times'][:8]))
    if m.get("ofdm_evm1") is not None:
        L.append(f"OFDM EVM        : {m['ofdm_evm1']*100:.1f}% single -> "
                 f"{m['ofdm_evm_avg']*100:.2f}% ({m['ofdm_nsym']}-symbol avg)")
        L.append(f"OFDM eff. SNR   : {m['ofdm_snr1']:.1f} dB -> {m['ofdm_snr_avg']:.1f} dB")
        L.append(f"OFDM symbol err : {m['ofdm_ser']*100:.2f}% (single symbol)")
    if "rms_delay" in m:
        L.append(f"RMS delay spread: {_fmt_s(m['rms_delay'])}")
        L.append(f"delay span      : {_fmt_s(m['delay_span'])}  (mean {_fmt_s(m['mean_delay'])})")
        L.append(f"coherence BW    : {_fmt_hz(m['coh_bw'])}   (1/2pi*delay_spread)")
    if m.get("coh_bw_direct") is not None:
        L.append(f"coherence BW(*) : {_fmt_hz(m['coh_bw_direct'])}   (from H(f) correlation)")
    if m.get("fade_corr") is not None:
        L.append(f"fade corr (rho) : {m['fade_corr']:.3f}  @ spacing {m['spacing']:.0f} Hz")
    if "dop_spread" in m:
        L.append(f"Doppler spread  : {m['dop_spread']:.3f} Hz")
        L.append(f"coherence time  : {_fmt_s(m['coh_time'])}   (1/2pi*Doppler)")
    if m.get("note"):
        L.append(f"note            : {m['note']}")
    return "\n".join(L)
