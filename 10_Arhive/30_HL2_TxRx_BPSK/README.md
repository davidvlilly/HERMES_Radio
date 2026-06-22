# HL2 Tx/Rx — DSSS BPSK (example 2)

A Hermes-Lite 2 toolset that transmits a **spread-spectrum BPSK** signal and
detects it with a **matched filter**, for weak-signal / below-the-noise links
between two stations. Evolved from the CW "spot" tool by replacing the single
tone with a spreading code and the spectrum peak-read with a correlator.

## How it works

- **Sample rate 48 kHz** (HL2 speed bits = 00), ±24 kHz baseband.
- **TX:** a 1023-chip **m-sequence** (deg-10 LFSR, primitive `x¹⁰+x⁷+1`),
  BPSK at DC (`I = ±code, Q = 0`), **1 sample/chip**, repeating every 1023
  samples (21.3 ms). The code is deterministic, so every copy of the software
  transmits and listens for the **same code**.
- **RX (matched filter):** each capture is de-spread against the known code
  (per-period circular correlation, magnitudes summed noncoherently over the
  8 periods in a window). This gives ~**30 dB of processing gain**, so the
  signal can sit well below the noise on the spectrum and still produce a clean
  correlation peak. The correlation also rejects DC/hum (the code is ~zero-mean).
- **Frequency search:** two low-cost oscillators differ by tens to hundreds of
  Hz at 7 MHz, which would smear the correlation. The RX sweeps **±500 Hz in
  10 Hz steps**, de-rotates by each candidate, and keeps the offset that
  maximizes the peak — that value **is** the inter-radio frequency offset, shown
  on screen. (For same-radio loopback the offset is ~0.)

## Display (two panes)

- **Top — spectrum** (dBm, ~17 Hz RBW): shows the wideband spread signal, plus
  the live PEP / PA-current readout.
- **Bottom — matched filter**: correlation magnitude vs code time (one 21.3 ms
  period, 1023 points), in **dB relative to the noise floor** — the peak height
  is the detection SNR. Title shows `peak/noise (dB)` and the measured frequency
  `offset (Hz)`, and flags `DROPS n` in red if any UDP packets were lost.

## Packet-loss handling

The matched filter needs a contiguous time grid. The capture checks the EP6
sequence number and **zero-fills lost packets** so the code phase stays aligned
(a drop then costs ~0.1 dB instead of ~2+ dB and smearing). The socket also uses
a 1 MB receive buffer, and any loss is shown as `DROPS n`.

## Files

| File | Description |
|------|-------------|
| `main_ex1.py` | Main app: HL2 link, 48 kHz config, code generator (`gen_code`), DSSS TX, matched filter + frequency search (`matched_filter`), capture/telemetry, controls. Run: `python main_ex1.py` |
| `spectrum_ex1.py` | Two-pane live display (spectrum + correlation), buttons (PTT / En PA / Overlay), LNA & Freq boxes, TX-power slider, Capture button, PEP readout. |

## Key settings (top of `main_ex1.py`)

- `NCO_FREQ` — carrier / RX frequency (default 7.000 MHz; also live via the **Freq** box).
- `SAMPLE_RATE = 48_000`, `CAPTURE_SIZE = 8192` (8 code periods / window).
- `CODE_LEN = 1023` — m-sequence length (1 sample/chip).
- `FREQ_SEARCH_HZ = 500`, `FREQ_STEP_HZ = 10` — carrier-offset search.
- `TX_AMPLITUDE = 0.8` — digital level; output power set by **En PA** + the TX-power slider.

## Bring-up / test

1. **Loopback first** (same radio): ANT → attenuator → RF3, PTT + En PA on.
   The bottom plot should show one sharp peak at a stable lag, offset ≈ 0 Hz.
2. **Two stations / over the air:** the offset readout shows the real oscillator
   difference; the peak/noise (dB) is your link margin. The ~30 dB processing
   gain should let you detect the code even when nothing is visible on the
   spectrum.
