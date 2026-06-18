#!/usr/bin/env python3
"""
main.py — HL2 Text Radio entry point.

    python main.py                     # connect to the HL2 and open the GUI
    python main.py --ip 169.254.19.221 --freq 7.040
    python main.py --selftest          # offline DSP check (no radio needed)

See DESIGN.md for the full design and the link-budget rationale.
"""

import argparse
import sys

import radio_engine as eng


def run_selftest():
    """Offline modem loopback: encode -> noisy channel -> decode, at the
    design threshold. Proves the DSP without needing the radio."""
    import numpy as np
    from modem import DbpskModem, CODE_CALL_CONTINUE

    m = DbpskModem(sample_rate=eng.SAMPLE_RATE, baud=eng.BAUD,
                   pedestal_hz=eng.PEDESTAL_HZ)
    fs = m.fs
    msg = b"HELLO"
    iq = m.modulate(CODE_CALL_CONTINUE, msg, amplitude=eng.TX_AMPLITUDE)
    sig = np.concatenate([np.zeros(int(0.3 * fs), complex), iq,
                          np.zeros(int(0.4 * fs), complex)])
    n = np.arange(len(sig))
    sig = sig * np.exp(1j * 2 * np.pi * 37.0 * n / fs)     # carrier offset
    sigp = np.mean(np.abs(iq) ** 2)
    B = 2500.0

    print(f"Modem self-test  ({eng.BAUD} baud, {eng.PEDESTAL_HZ:.0f} Hz pedestal, "
          f"{eng.SAMPLE_RATE//1000} kHz)")
    print(f"  max payload / {eng.TX_SLOT_S:.1f}s burst : {m.max_payload_for_window(eng.TX_SLOT_S)} chars")
    for snr in (-8, -10, -12):
        ok = 0
        for seed in range(30):
            np.random.seed(seed)
            npf = sigp / (10 ** (snr / 10)) * (fs / B)
            rx = sig + (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))) \
                 * np.sqrt(npf / 2)
            _, dec = m.receive(rx)
            if dec and dec["ok"] and dec["payload"] == msg:
                ok += 1
        print(f"  SNR {snr:+3d} dB (2.5 kHz ref): {ok}/30 perfect copies")
    print("Done.")


def run_sim_gui(snr_db):
    """Two-station simulation in two GUI windows; auto-starts Call + Detect."""
    import tkinter as tk
    import sim
    from radio_gui import RadioGUI

    A, B, _medium = sim.build_sim_pair(snr_db=snr_db)
    A.run()
    B.run()

    root = tk.Tk()
    RadioGUI(root, A)
    root.title("HL2 Text Radio — Station A (CALLER)  [SIM]")
    top = tk.Toplevel(root)
    RadioGUI(top, B)
    top.title("HL2 Text Radio — Station B (RESPONDER)  [SIM]")
    try:
        root.geometry("+40+40")
        top.geometry("+1060+40")
    except Exception:
        pass

    print(f"SIM: channel SNR {snr_db:+.0f} dB. Auto-starting Call (A) + Detect (B) …")
    root.after(800, A.start_call)
    root.after(800, B.start_detect)

    def _shutdown():
        A.shutdown(); B.shutdown()
        root.after(200, root.destroy)
    root.protocol("WM_DELETE_WINDOW", _shutdown)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="HL2 Text Radio")
    ap.add_argument("--ip", default=eng.HL2_IP, help="HL2 IP address")
    ap.add_argument("--freq", type=float, default=eng.NCO_FREQ / 1e6,
                    help="carrier frequency in MHz")
    ap.add_argument("--selftest", action="store_true",
                    help="run the offline DSP self-test and exit")
    ap.add_argument("--sim", action="store_true",
                    help="no-hardware two-station simulation (two GUI windows)")
    ap.add_argument("--simtest", action="store_true",
                    help="headless two-station integration test and exit")
    ap.add_argument("--sim-snr", type=float, default=-8.0,
                    help="simulated channel SNR in dB (2.5 kHz ref)")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return
    if args.simtest:
        import sim
        sim.run_simtest(snr_db=args.sim_snr)
        return
    if args.sim:
        run_sim_gui(args.sim_snr)
        return

    import tkinter as tk
    from tkinter import messagebox
    from radio_engine import RadioEngine
    from radio_gui import RadioGUI
    import hl2_transport as hl2

    engine = RadioEngine(ip=args.ip, nco_freq=args.freq * 1e6)
    print(f"Connecting to HL2 at {args.ip} …")
    try:
        engine.connect()
    except Exception as e:                       # noqa: BLE001 — surface to user
        print(f"ERROR: could not connect to the HL2: {e}", file=sys.stderr)
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showerror("HL2 Text Radio",
                                 f"Could not connect to the HL2 at {args.ip}:\n\n{e}\n\n"
                                 "Check the cable, power, IP address and that UDP "
                                 "port 1024 is allowed through the firewall.")
        except Exception:
            pass
        sys.exit(1)

    engine.run()
    print(f"Connected. NCO {engine._nco_hz/1e6:.4f} MHz, {eng.BAUD} baud DBPSK.")

    root = tk.Tk()
    RadioGUI(root, engine)
    try:
        root.mainloop()
    finally:
        engine.shutdown()
        if engine._sock is not None:
            hl2.stop(engine._sock, engine.ip)
        print("Shut down.")


if __name__ == "__main__":
    main()
