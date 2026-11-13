#!/usr/bin/env python3
"""
OpenRPS - Command Line Interface (CLI)

After ``pip install .``:
    open-rps --mode raytracing
    python3 -m open_rps.cli --mode hata --freq 900e6
    open-rps --mode raytracing --coherent --sector --sinr --shadow-sigma 6 --seed 1
    open-rps --mode raytracing --building --wall-loss inf
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
               "raytracing mode, a reflecting wall on the plane x = size/2 (with --building, a "
               "rectangular building east of the first station instead). The walls block nothing "
               "unless --wall-loss is given. With --sector both stations use a 65-degree sector "
               "antenna pointing at the centre of the area.",
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
                        help="Add the reflecting wall, or the building of --building (only affects "
                             "--mode raytracing; default: on)")
    parser.add_argument("--wall-loss", type=float, default=None, metavar="DB",
                        help="Penetration loss of the demo walls in dB: every ray that crosses a wall loses "
                             "DB dB ('inf' = opaque walls, which cast a shadow). Omitted: the walls only "
                             "reflect and block nothing (only affects --mode raytracing)")
    parser.add_argument("--building", action="store_true",
                        help="Replace the infinite wall by a rectangular building made of 4 finite walls "
                             "(0.45-0.60 x size in x, 0.35-0.50 x size in y, twice as tall as the base "
                             "stations); combine it with --wall-loss to see its shadow")
    parser.add_argument("--coherent", action="store_true",
                        help="Add the complex ray amplitudes (fast fading) instead of the ray powers "
                             "(only affects --mode raytracing)")
    parser.add_argument("--sector", action="store_true",
                        help="Use 65-degree 3GPP sector antennas pointing at the centre of the area "
                             "instead of isotropic antennas")
    parser.add_argument("--downtilt", type=float, default=0.0,
                        help="Downtilt of the sector antennas in degrees (only with --sector; default: 0)")
    parser.add_argument("--shadow-sigma", type=float, default=0.0,
                        help="Standard deviation of the log-normal shadow fading in dB (default: 0 = none)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed of the shadow fading, for reproducible maps (default: random)")
    parser.add_argument("--min-distance", type=float, default=1.0,
                        help="Pixels closer than this to an antenna are left without data; 0 disables it (default: 1m)")
    parser.add_argument("--strict", action="store_true",
                        help="With --mode hata: fail instead of extrapolating outside the validity range")
    parser.add_argument("--sinr", action="store_true",
                        help="Also compute the SINR of the best server (extra CSV columns, JSON block and SVG map)")
    parser.add_argument("--noise-figure", type=float, default=7.0,
                        help="Receiver noise figure in dB, used by --sinr (default: 7)")
    parser.add_argument("--bandwidth", type=float, default=20e6,
                        help="Noise bandwidth in Hz, used by --sinr (default: 20e6 = 20 MHz)")
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
    if args.shadow_sigma < 0:
        parser.error("--shadow-sigma must be >= 0")
    if args.min_distance < 0:
        parser.error("--min-distance must be >= 0")
    if args.bandwidth <= 0:
        parser.error("--bandwidth must be > 0 (in Hz)")
    if args.wall_loss is not None and not args.wall_loss >= 0:
        parser.error("--wall-loss must be >= 0 dB (or 'inf' for opaque walls)")
    cells = (int(args.size / args.res) + 1) ** 2
    if cells > MAX_GRID_CELLS:
        parser.error(f"grid too large ({cells} cells > {MAX_GRID_CELLS}); increase --res or reduce --size")

    sim = PropagationSimulator()
    # With --sector each station looks at the centre of the area (Tx0 to the NE, Tx1 to the SW)
    antenna = dict(pattern="sector", downtilt_deg=args.downtilt) if args.sector else {}
    tx0 = Transmitter(tx_id=0, x=args.size * 0.3, y=args.size * 0.3, z=args.tx_height,
                      tx_power_dbm=args.power, frequency_hz=args.freq, antenna_gain_dbi=12.0,
                      azimuth_deg=45.0, **antenna)
    tx1 = Transmitter(tx_id=1, x=args.size * 0.7, y=args.size * 0.7, z=args.tx_height,
                      tx_power_dbm=args.power - 3.0, frequency_hz=args.freq, antenna_gain_dbi=12.0,
                      azimuth_deg=-135.0, **antenna)
    sim.add_transmitter(tx0)
    sim.add_transmitter(tx1)

    # Walls for multipath (ray tracing only).  They obstruct the rays only with --wall-loss.
    walls = []
    if args.reflector == "on" and not args.building:
        walls.append(PlanarReflector(p0=[args.size * 0.5, args.size * 0.5, 0.0], normal=[1.0, 0.0, 0.0],
                                     reflectivity=0.5, transmission_loss_db=args.wall_loss))
    elif args.reflector == "on":
        # Building east of Tx0: 4 vertical faces, each one a finite wall (no roof: it is taller than the antennas)
        bx0, bx1, by0, by1, top = (args.size * 0.45, args.size * 0.60, args.size * 0.35, args.size * 0.50,
                                   2.0 * args.tx_height)
        for name, p0, normal, bounds in (
                ("West face", [bx0, by0, 0.0], [1.0, 0.0, 0.0], (bx0, bx0, by0, by1, 0.0, top)),
                ("East face", [bx1, by0, 0.0], [1.0, 0.0, 0.0], (bx1, bx1, by0, by1, 0.0, top)),
                ("South face", [bx0, by0, 0.0], [0.0, 1.0, 0.0], (bx0, bx1, by0, by0, 0.0, top)),
                ("North face", [bx0, by1, 0.0], [0.0, 1.0, 0.0], (bx0, bx1, by1, by1, 0.0, top))):
            walls.append(PlanarReflector(p0=p0, normal=normal, reflectivity=0.5, name=name, bounds=bounds,
                                         transmission_loss_db=args.wall_loss))
    for wall in walls:
        sim.add_reflector(wall)

    if args.mode == "raytracing":
        summation = ", coherent sum" if args.coherent else ""
        model, model_name = None, f"ray tracing (reflector {args.reflector}{summation})"
    elif args.mode == "tworay":
        model, model_name = TwoRayGroundReflection(), "Two-Ray ground reflection"
    elif args.mode == "hata":
        if args.freq <= 1500e6:
            model, model_name = OkumuraHata("urban_medium", strict=args.strict), "Okumura-Hata (urban_medium)"
        else:
            model, model_name = Cost231Hata("urban_medium", strict=args.strict), "COST-231 Hata (urban_medium)"
    else:
        model, model_name = FreeSpacePathLoss(), "Free-space path loss"

    print(f"[*] Starting OpenRPS Simulation using model '{args.mode}': {model_name}")
    print(f"[*] Area: {args.size}m x {args.size}m, Resolution: {args.res}m, Frequency: {args.freq/1e6:.1f} MHz")
    if args.mode != "raytracing" and args.reflector == "on":
        print("[*] Note: the reflector is ignored by empirical models (only used with --mode raytracing)")
    if args.mode != "raytracing" and args.coherent:
        print("[*] Note: --coherent is ignored by empirical models (only used with --mode raytracing)")
    if args.building or args.wall_loss is not None:
        if args.mode != "raytracing":
            print("[*] Note: --building and --wall-loss are ignored by empirical models (walls neither "
                  "reflect nor block there; only used with --mode raytracing)")
        elif not walls:
            print("[*] Note: --building and --wall-loss have no effect with --reflector off (there are no walls)")
        else:
            what = (f"building of 4 walls, x = {walls[0].bounds[0]:g}-{walls[1].bounds[0]:g} m, "
                    f"y = {walls[2].bounds[2]:g}-{walls[3].bounds[2]:g} m, {walls[0].bounds[5]:g} m tall"
                    if args.building else f"infinite wall on x = {args.size * 0.5:g} m")
            if args.wall_loss is None:
                blocking = "reflects only, blocks nothing (use --wall-loss to make it obstruct)"
            elif args.wall_loss == float("inf"):
                blocking = "opaque (rays that cross a wall are blocked)"
            else:
                blocking = f"{args.wall_loss:g} dB of penetration loss per wall crossed"
            print(f"[*] Walls: {what}; {blocking}")
    if args.mode != "hata" and args.strict:
        print("[*] Note: --strict only applies to --mode hata")
    if args.sector:
        print(f"[*] Antennas: 65-degree sectors pointing at the centre of the area, downtilt {args.downtilt:g} deg")
    if args.shadow_sigma > 0:
        seed_text = "random seed" if args.seed is None else f"seed {args.seed}"
        print(f"[*] Shadow fading: log-normal, sigma = {args.shadow_sigma:g} dB ({seed_text})")

    area = ((0.0, args.size), (0.0, args.size))
    common = dict(resolution_m=args.res, min_distance_m=args.min_distance,
                  shadow_fading_sigma_db=args.shadow_sigma, seed=args.seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if model is None:
                res = sim.run_raytracing_simulation(*area, coherent=args.coherent, **common)
            else:
                res = sim.run_empirical_simulation(model, *area, **common)
        except ValueError as error:
            # e.g. --strict with a Hata model outside its validity range
            print(f"[x] Error: {error}", file=sys.stderr)
            return 1
    for message in dict.fromkeys(str(w.message) for w in caught):
        print(f"[!] Warning: {message}")

    if args.sinr:
        res.compute_sinr(noise_figure_db=args.noise_figure, bandwidth_hz=args.bandwidth)

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
    if args.sinr:
        svg_sinr = os.path.join(args.outdir, "coverage_sinr.svg")
        MapVisualizer.export_svg(res, svg_sinr, metric="sinr", theme=args.theme)
        stats = res.summary()["sinr"]
        print(f"[+] Exported SINR SVG:        {svg_sinr}")
        if stats["mean_sinr_db"] is not None:
            print(f"[*] SINR (noise {stats['noise_power_dbm']:.1f} dBm in {args.bandwidth/1e6:g} MHz, NF {args.noise_figure:g} dB): "
                  f"min {stats['min_sinr_db']:.1f} dB, mean {stats['mean_sinr_db']:.1f} dB, max {stats['max_sinr_db']:.1f} dB")
    if args.mode == "raytracing":
        svg_ds = os.path.join(args.outdir, "coverage_delay_spread.svg")
        MapVisualizer.export_svg(res, svg_ds, metric="delay_spread", theme=args.theme)
        print(f"[+] Exported Delay Spread SVG: {svg_ds}")
    print("[*] Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
