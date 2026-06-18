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

### Message-log markers

| Symbol | Color | Meaning |
|:------:|:------|---------|
| `/`    | gray  | beacon **sent** (no text this cycle) |
| `.`    | gray  | beacon **received** |
| `#`    | gray  | expected burst **missed** |
| `*`    | red   | incoming **message fragment** arriving |
| `> …`  | blue  | text you **sent** (full message, echoed once) |
| `    > …` | cyan | each **fragment** as it transmits (`<stop>` = end-of-message) |
| `< …`  | white | text **received** |
| `<end connection>` | — | link dropped (3 misses, or partner ended) |

The red **MODE** label shows Call/Detect; the red **TX** box lights while
transmitting (the **spectrum freezes during TX** — it holds the last received
view). **LNA dB**, **TX Pwr**, **Freq**, and the compact **PEP** (`3.1w`) are on
the control row; the carrier **offset** / link status is in the lower-left
status line. **Clear** wipes the message log.

## Files

| File | Role |
|------|------|
| [main.py](main.py) | entry point — real radio, `--selftest`, `--sim`, `--simtest` |
| [radio_engine.py](radio_engine.py) | orchestration, sample-clock scheduler, call/detect state machine, fragmentation, drift tracker, timing log |
| [modem.py](modem.py) | DBPSK text modem — framing + checksum, modulation, acquisition + demod (pure NumPy) |
| [hl2_transport.py](hl2_transport.py) | HL2 Protocol-1 UDP transport (adapted from the reference) |
| [radio_gui.py](radio_gui.py) | Tkinter GUI — spectrum, controls, colored message log |
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
