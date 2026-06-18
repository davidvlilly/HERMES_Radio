# Text Radio 00 — Design & Assumptions

A weak-signal **DBPSK text radio** for the Hermes-Lite 2 (HL2). Two stations
exchange short text messages (or "I'm-alive" beacon bursts) on a fixed
5-second cadence. Built to copy text at roughly **−10 dB SNR** (2.5 kHz ref).

This file records every design choice and every place the original design
notes (`Text_Radio_Design_00.txt`) were adjusted, per the project owner's
guidance ("radio parameters are a guideline — use whatever is best").

---

## 1. Link budget — why 40 baud

What the radio can copy is set by **Eb/N0** (energy per bit), and the matched
filter converts wide-channel SNR into Eb/N0 via *processing gain*:

```
Eb/N0 (dB) = SNR_channel (dB) + 10·log10( B_channel / bit_rate )
```

- DBPSK needs **Eb/N0 ≈ 8 dB** for solid text (BER ~1e-3).
- Reference channel `B_channel = 2500 Hz` (typical SSB passband — the bandwidth
  an operator's "SSB SNR" is measured in).

| Bit rate | Proc. gain | Copies down to | Payload / burst |
|---------:|-----------:|---------------:|----------------:|
|  25 bps  |   20 dB    |   −12 dB       |  ~2 chars       |
| **40 bps** | **18 dB** | **−10 dB** ✅  | **5 chars**     |
|  50 bps  |   17 dB    |   −9 dB        |  ~7 chars       |
| 100 bps  |   14 dB    |   −6 dB        |  ~22 chars      |
| 300 bps  |    9 dB    |   −1 dB        |  ~68 chars      |

**Decision: 40 baud, uncoded.** Hits the −10 dB target with slight margin.
The cost is throughput: a burst carries only **5 chars** (after framing
overhead, §3), so longer messages are fragmented across successive cycles (§4).

> The spectrum **display** is 96 kHz wide for situational awareness. This does
> NOT cost link SNR — the modem's matched filter integrates the noise bandwidth
> down to ≈ the symbol rate regardless of how wide we capture/display.

---

## 2. Waveform

| Parameter        | Value            | Notes |
|------------------|------------------|-------|
| Modulation       | DBPSK (differential) | No Costas loop; tolerant of a static phase offset |
| Baud             | 40 sym/s         | Tsym = 25 ms |
| Bit rate         | 40 bps (1 bit/sym) | No FEC |
| Pedestal / subcarrier | 2000 Hz     | Keeps the signal off DC (avoids LO leakage / 1/f) |
| Pulse shape      | Rectangular NRZ  | Matched rect filter at RX = optimal for AWGN. Splatter from phase jumps is acceptable in a clear ham channel; light shaping is a future improvement. |
| HL2 sample rate  | 96 kHz (speed bits = 01) | ±48 kHz spectrum view |
| Modem baseband   | 2 kHz (decimate 96 k / 48) | 50 samples/symbol |

TX is complex (single-sideband): `tx[n] = d[n] · exp(j·2π·f_ped·n/fs)`, where
`d[n] = ±1` is the current absolute DBPSK phase. HL2 takes I = Re, Q = Im, so
the emitted signal sits at `NCO + 2 kHz`.

---

## 3. Burst frame

Symbols, in order (all 1 bit/symbol):

| Symbols     | Field      | Meaning |
|------------:|------------|---------|
| 1–20        | Preamble   | Fixed ±1 LFSR sequence, good autocorrelation. Used for detection, timing, and carrier-offset estimation. |
| 21–24       | Burst code | 4 bits. 0 = call-continue, 1 = call-end, 2 = detect-continue, 4 = detect-end. |
| 25–32       | Length     | 8 bits = number of ASCII payload chars **in this burst** (0–255; ≤ 5 in practice). |
| 33 …        | Payload    | `length × 8` bits, ASCII, 8 bits each. |
| last 8      | Checksum   | `(code + length + Σ payload) & 0xFF`. Rejects corrupt/partial decodes. |

- **Length 0 ⇒ active-signal beacon** (no payload). End bursts are beacons too.
- Differential encoding starts at the preamble; the last preamble symbol is the
  reference for the first data symbol (continuous DBPSK across the whole burst).
- Bit→phase: bit 1 = 180° flip, bit 0 = no change.
- **Checksum** (added beyond the original spec): without it, a noise-only window
  occasionally decoded as a valid burst — including a false *end-code* that
  dropped the link. The RX requires a matching checksum, AND treats an end-code
  carrying any payload as invalid (genuine end bursts are empty beacons).

Overhead = 20 preamble + 4 code + 8 length + 8 checksum = **40 symbols**.
Max payload = `floor((MAX_BURST_SYMBOLS − 40) / 8)`. With a 2.0 s max burst at
40 baud (80 symbols) that is **5 chars/burst**.

---

## 4. Message fragmentation & reassembly

A full message can't fit in one 5-char burst, so long messages **auto-fragment**:

- Outgoing text is queued with an ASCII **newline (0x0A)** sentinel appended.
- Each transmit cycle sends up to `MAX_PAYLOAD` (5) chars from the head of the
  queue; the buffer drains one fragment per cycle until empty, then beacons
  resume.
- The receiver accumulates payload chars until it sees the newline, then commits
  the completed line. Each received fragment also flags a red `*` so the operator
  sees data arriving before the line completes (§6).
- **No retransmission / ARQ:** each fragment is sent once. A dropped fragment is
  lost (the message arrives incomplete). Retransmission with fragment sequencing
  is the natural next robustness step if needed.

---

## 5. Timing / cadence (half-duplex)

The HL2 is half-duplex. The 5-second cycle is split into two **2.5 s halves**;
each station transmits its (≤2.0 s) burst in its own half and listens the entire
opposite half:

```
        0 ─────────────── 2.5 ─────────────── 5.0
 CALLER:│ TX (≤2.0 s burst) │     RX (listen)    │
 RESP.: │     RX (listen)    │ TX (≤2.0 s burst) │
```

- **Call mode (caller):** transmits at the start of its half, listens the other.
- **Detect mode (responder):** searches for a caller, locks to it, then replies
  **2.5 s after the caller's burst start** (per the notes).
- The ≤2.0 s burst in a 2.5 s half leaves ~0.5 s for the burst to finish + T/R
  settle before the handoff. A burst only *starts* within `BURST_ARM_S` of the
  half's start, so locking mid-half never launches a truncated burst.

**Sample-clock cadence.** The schedule runs off the **EP6 sequence number**
(the HL2's crystal sample clock), not the PC wall-clock — so the schedule, RX
timing, and drift all share one clock and survive ethernet jitter / packet loss
(gaps are zero-filled). On a listen-window open the stale socket backlog is
flushed so the window is anchored to "now". Within a burst the matched filter
re-measures symbol timing + carrier offset every time, so coarse cadence only
has to keep a burst inside the half.

**Drift tracking (responder only).** Each cycle the responder measures the
caller's burst arrival vs. expected; if the error exceeds **±50 ms** it nudges
its cadence clock by **50 ms** (deadband tracker). The caller is the master and
never moves. PC-vs-radio drift is slow, so this rarely fires.

---

## 6. Connection state / display

Message-log markers:

| Marker | Meaning |
|:------:|---------|
| `/`    | beacon **sent** (no text this cycle) |
| `.`    | beacon **received** |
| `#`    | expected burst **missed** |
| red `*`| incoming **message fragment** arriving |
| `> …`  | text you **sent** — blue (full message, echoed once) |
| `    > …` | each **fragment** as it actually transmits — cyan, `<stop>` marks the terminator |
| `< …`  | text **received** — white |
| `<end connection>` | link dropped |

- After a text line, the next notification run is offset by 10 spaces for clarity.
- **3 consecutive misses → `<end connection>` and drop the mode.** A received
  end-code (1 or 4) also drops it; end bursts are sent **twice** for reliability.
- Active mode (CALL / DETECT) in **red** next to the buttons; **TX-active** as a
  red box. Carrier offset and link status in the lower-left status line. PEP
  shown compactly (e.g. `3.1w`) at the right.
- **Spectrum freezes during TX** (holds the last received view) — the RX stream
  during transmit isn't the listened band.

---

## 7. Modules

| File | Role |
|------|------|
| `hl2_transport.py` | HL2 Protocol-1 UDP transport — adapted from the reference (`connect`, `init_radio`, `build_ep2`, `parse_ep6`, `make_cc`, power telemetry, `ep6_seq`). Parameterised for 96 kHz. |
| `modem.py` | DBPSK text modem — framing + checksum, differential encode/decode, modulation, and acquisition/demod (coherent FFT preamble matched filter + carrier-offset search). Pure NumPy. |
| `radio_engine.py` | Orchestration: TX & capture threads, a demod worker thread, the sample-clock scheduler, call/detect state machine, fragmentation/reassembly, drift tracker, optional timing log, and a thread-safe event queue to the GUI. |
| `radio_gui.py` | Tkinter GUI — File/Config menus, spectrum pane (matplotlib), Call/Detect/End buttons + red mode text, TX indicator, LNA box, TX-power slider, Freq box, compact PEP, status line, colored scrolling log + compose + Clear. |
| `sim.py` | No-hardware two-station simulation — a virtual RF medium + fake sockets so two engines talk in-process with noise + carrier offset (`--sim` GUI, `--simtest` headless). |
| `main.py` | Entry point: real radio, `--selftest`, `--sim`, `--simtest`. |

---

## 8. Config (top of `radio_engine.py`)

- `HL2_IP`, `NCO_FREQ` — radio address / carrier (live via the Freq box).
- `SAMPLE_RATE` (96 k), `BAUD` (40), `PEDESTAL_HZ` (2000).
- `CYCLE_S` (5), `HALF_S` (2.5), `TX_SLOT_S` (2.0 max burst), `BURST_ARM_S` (0.3).
- `ACQ_THRESH_DB` (14.5), `MAX_MISS` (3), `TRACK_DEADBAND_S` / `TRACK_STEP_S` (50 ms).
- `TX_AMPLITUDE`, `RX_LNA_GAIN_DB`, `DEFAULT_TX_DRIVE`.
- `TIME_SCALE` (1.0 = real time on hardware; the sim raises it for CPU headroom).
- Acquisition search (`freq_search_hz` ±255, `freq_step_hz` 0.5, `preamble_syms` 20)
  lives in `modem.py`.

## 9. Known limits / to tune on hardware

- **Acquisition** is a coherent FFT preamble matched filter swept over a carrier
  offset (±255 Hz, 0.5 Hz steps). First lock needs a window longer than one cycle
  (~7.5 s) to guarantee it contains a full burst; tune the range to your radios'
  oscillator offset.
- **Detection threshold** (14.5 dB) sits just above the noise-only correlation
  ceiling; re-trim against your real noise floor.
- Burst-completion margins, the arm window, and T/R turnaround are first-cut —
  verify against real PA/relay timing (loopback gives the fixed FIFO latency).
- **No FEC / no ARQ:** relies on the 3-miss logic, the checksum, and operator
  repeats. Add fragment retransmission for guaranteed long-message delivery.
- Rectangular pulses splatter; add raised-cosine phase shaping if ACI matters.
- The in-process sim can't give two radios real-time CPU; it runs at a `>1`
  time scale for headroom. This is a test-rig limit only — one radio at true
  real time on dedicated hardware is not affected.
