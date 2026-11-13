#!/usr/bin/env python3
"""
OpenRPS - Command Line Interface (CLI)

After ``pip install .``:
    open-rps --mode raytracing
    python3 -m open_rps.cli --mode hata --freq 900e6
"""

import argparse
import os
import sys
import warnings

from . import (
    PropagationSimulator,
    Transmitter,
    PlanarReflector,
    MapVisualizer,
    FreeSpacePathLoss,
    TwoRayGroundReflection,
    OkumuraHata,
    Cost231Hata,
)

MAX_GRID_CELLS = 4_000_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenRPS: Open Radiowave Propagation Simulator & Multipath Channel Estimator",
        epilog="Demo scenario: two base stations at 30% and 70% of the area diagonal and, in "
               "raytracing mode, a reflecting wall on the plane x = size/2.",
    )
    parser.add_argument("--mode", choices=["raytracing", "tworay", "hata", "fspl"], default="raytracing",
                        help="Propagation model: 'raytracing' (multipath), 'tworay', 'hata' "
                             "(Okumura-Hata up to 1500 MHz, COST-231 Hata above) or 'fspl'")
    parser.add_argument("--size", type=float, default=1000.0, help="Grid side size in meters (default: 1000m)")
    parser.add_argument("--res", type=float, default=25.0, help="Grid resolution step in meters (default: 25m)")
    parser.add_argument("--freq", type=float, default=2.4e9, help="Carrier frequency in Hz (default: 2.4e9 = 2.4 GHz)")
    parser.add_argument("--power", type=float, default=43.0, help="Tx power in dBm (default: 43 dBm = 20W)")
    parser.add_argument("--tx-height", type=float, default=25.0, help="Height of both base stations in meters (default: 25m)")
    parser.add_argument("--reflector", choices=["on", "off"], default="on",
                        help="Add the reflecting wall (only affects --mode raytracing; default: on)")
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Color theme of the SVG maps")
    parser.add_argument("--outdir", type=str, default="./output",
                        help="Output directory for the CSV, JSON and SVG files (default: ./output)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.size <= 0:
        parser.error("--size must be > 0")
    if args.res <= 0:
        parser.error("--res must be > 0")
    if args.freq <= 0:
        parser.error("--freq must be > 0 (in Hz)")
    if args.tx_height <= 0:
        parser.error("--tx-height must be > 0")
    cells = (int(args.size / args.res) + 1) ** 2
    if cells > MAX_GRID_CELLS:
        parser.error(f"grid too large ({cells} cells > {MAX_GRID_CELLS}); increase --res or reduce --size")

    sim = PropagationSimulator()
    tx0 = Transmitter(tx_id=0, x=args.size * 0.3, y=args.size * 0.3, z=args.tx_height,
                      tx_power_dbm=args.power, frequency_hz=args.freq, antenna_gain_dbi=12.0)
    tx1 = Transmitter(tx_id=1, x=args.size * 0.7, y=args.size * 0.7, z=args.tx_height,
                      tx_power_dbm=args.power - 3.0, frequency_hz=args.freq, antenna_gain_dbi=12.0)
    sim.add_transmitter(tx0)
    sim.add_transmitter(tx1)

    # Reflector for multipath (ray tracing only)
    if args.reflector == "on":
        wall = PlanarReflector(p0=[args.size * 0.5, args.size * 0.5, 0.0], normal=[1.0, 0.0, 0.0], reflectivity=0.5)
        sim.add_reflector(wall)

    if args.mode == "raytracing":
        model, model_name = None, f"ray tracing (reflector {args.reflector})"
    elif args.mode == "tworay":
        model, model_name = TwoRayGroundReflection(), "Two-Ray ground reflection"
    elif args.mode == "hata":
        if args.freq <= 1500e6:
            model, model_name = OkumuraHata("urban_medium"), "Okumura-Hata (urban_medium)"
        else:
            model, model_name = Cost231Hata("urban_medium"), "COST-231 Hata (urban_medium)"
    else:
        model, model_name = FreeSpacePathLoss(), "Free-space path loss"

    print(f"[*] Starting OpenRPS Simulation using model '{args.mode}': {model_name}")
    print(f"[*] Area: {args.size}m x {args.size}m, Resolution: {args.res}m, Frequency: {args.freq/1e6:.1f} MHz")
    if args.mode != "raytracing" and args.reflector == "on":
        print("[*] Note: the reflector is ignored by empirical models (only used with --mode raytracing)")

    area = ((0.0, args.size), (0.0, args.size))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model is None:
            res = sim.run_raytracing_simulation(*area, resolution_m=args.res)
        else:
            res = sim.run_empirical_simulation(model, *area, resolution_m=args.res)
    for message in dict.fromkeys(str(w.message) for w in caught):
        print(f"[!] Warning: {message}")

    print("\n" + MapVisualizer.render_ascii_heatmap(res.max_rx_power_dbm, title="Rx Power Coverage Map", unit="dBm"))

    os.makedirs(args.outdir, exist_ok=True)
    csv_file = os.path.join(args.outdir, "simulation_results.csv")
    json_file = os.path.join(args.outdir, "simulation_summary.json")
    svg_power = os.path.join(args.outdir, "coverage_rx_power.svg")
    svg_server = os.path.join(args.outdir, "coverage_best_server.svg")
    res.export_csv(csv_file)
    res.export_json(json_file)
    MapVisualizer.export_svg(res, svg_power, metric="rx_power", theme=args.theme)
    MapVisualizer.export_svg(res, svg_server, metric="best_server", theme=args.theme)

    print()
    print(f"[+] Exported CSV data:        {csv_file}")
    print(f"[+] Exported JSON summary:    {json_file}")
    print(f"[+] Exported Rx Power SVG:    {svg_power}")
    print(f"[+] Exported Best Server SVG: {svg_server}")
    if args.mode == "raytracing":
        svg_ds = os.path.join(args.outdir, "coverage_delay_spread.svg")
        MapVisualizer.export_svg(res, svg_ds, metric="delay_spread", theme=args.theme)
        print(f"[+] Exported Delay Spread SVG: {svg_ds}")
    print("[*] Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
