# HL2 Text Radio

A weak-signal **DBPSK text radio** for the Hermes-Lite 2. Two stations exchange
short text messages (or "I'm-alive" beacon bursts) on a fixed 5-second cadence,
designed to copy text at roughly **−10 dB SNR** (2.5 kHz reference) — about
13 dB more sensitive than an SSB voice link.

See **[DESIGN.md](DESIGN.md)** for the full design, framing, and link-budget
rationale. The HL2 transport is adapted from the example in `05_Reference/`.

## Run

```
python main.py                          # connect to the HL2, open the GUI
python main.py --ip 169.254.19.221 --freq 7.040
python main.py --selftest               # offline DSP check, no radio needed
python main.py --sim                    # two-station simulation, no radio (two windows)
python main.py --sim --sim-snr -10      # ... at the design SNR
python main.py --simtest                # headless two-station integration test
```

Requires **Python 3.13**, `numpy`, and `matplotlib` (with the Tk backend —
`tkinter` ships with standard Python on Windows).

## Using it

- **Call** — your station drives the link: it transmits at the start of its
  2.5 s half (text if queued, otherwise a beacon) and listens the other half.
- **Detect** — your station searches for a caller, locks to its cadence, and
  replies 2.5 s after each received burst.
- **End** — sends two end-of-connection bursts, then drops the mode.
- Type a message and press **Enter / Send**. Long messages auto-fragment at
  **5 chars per burst** (one fragment per cycle) and reassemble on the far end.
  There's no retransmission, so a dropped fragment is lost — resend if needed.

### Message log — two panes

**LEFT (65%)** shows only **full messages** (one per line); **RIGHT (35%)** is a
per-burst **activity stream** that word-wraps and breaks to a new line only when
an incoming message finishes (or the width rolls over).

| Pane | Symbol | Color | Meaning |
|:----:|:------:|:------|---------|
| left | `> …`  | blue  | full message you **sent** |
| left | `< …`  | white | full message **received** |
| left | `<end connection>` | — | link dropped (4 misses, or partner ended) |
| right| `^`    | white | **empty out** — beacon sent |
| right| `^`    | red   | **message out** — a fragment was sent |
| right| `*`    | white | **empty in** — beacon received |
| right| `[…]`  | cyan  | **message in** — the received fragment's letters |
| right| `#`    | gray  | expected burst **missed** |

### Upper plot — Display menu

- **Sync Corr** (default) — matched-filter **processing-gain** strip chart over
  receive time: blue = peak, orange = average (dB above noise), with a yellow
  dotted **threshold** line. Each synced burst shows as a sharp mark above the
  threshold at the cadence interval = your live detection SNR / link margin. Keeps
  scrolling during TX, leaving a blank gap.
- **Spectrum** — raw spectrum (±48 kHz) at **2.5 kHz RBW**, **antenna-referred**
  (subtracts the LNA gain, so the floor reads true antenna dBm — ~−90 at LNA 30).
  Use it to see **other radios / QRM** on the channel. **Freezes/grays during TX.**

The red **MODE** label shows Call/Detect; the red **TX** box lights on transmit.
**LNA dB**, **TX Pwr**, **Freq**, and **PEP** (`PEP X.X w`) are on the control row;
the carrier **offset** / link status is in the lower-left status line. **Clear**
wipes the message log *and* restarts both plot histories. **Help ▸ Usage Modes /
Symbols** print into the left pane. **End** drops the active mode immediately.

## Files

| File | Role |
|------|------|
| [main.py](main.py) | entry point — real radio, `--selftest`, `--sim`, `--simtest` |
| [radio_engine.py](radio_engine.py) | orchestration, sample-clock scheduler, call/detect state machine, fragmentation, drift tracker, timing log |
| [modem.py](modem.py) | DBPSK text modem — framing + checksum, modulation, acquisition + demod (pure NumPy) |
| [hl2_transport.py](hl2_transport.py) | HL2 Protocol-1 UDP transport (adapted from the reference) |
| [radio_gui.py](radio_gui.py) | Tkinter GUI — Sync-Corr/Spectrum display, controls, two-pane colored log, File/Display/Help menus |
| [sim.py](sim.py) | no-hardware two-station simulation (virtual RF medium + fake sockets) |
| [DESIGN.md](DESIGN.md) | design decisions, assumptions, and the link budget |

## Status

The DSP modem is verified offline (`--selftest` copies cleanly at −10 dB), and
the full two-station exchange — acquisition, lock, drift tracking, bidirectional
beacons, and multi-fragment text — is verified in the simulation (`--simtest`).
The simulation runs at a `>1` time scale (`SIM_TIME_SCALE` in `sim.py`) because
two full radios in one process can't share real-time CPU; that's a test-rig
limit, not a radio one.

Burst-completion / T/R margins, the acquisition search range, and the detection
threshold are first-cut values to **tune against the real radio** — see
DESIGN.md §9.
