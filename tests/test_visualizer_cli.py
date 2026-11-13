import contextlib
import io
import json
import os
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from open_rps import MapVisualizer, PropagationSimulator, SimulationResult, Transmitter
from open_rps import cli
from open_rps.visualizer import _THEMES, _nice_ticks

SVG = "{http://www.w3.org/2000/svg}"


def empty_result(ny=3, nx=4):
    """A result in which nothing was computed: every metric NaN / -inf, no server."""
    return SimulationResult(
        x_coords=np.arange(nx) * 10.0, y_coords=np.arange(ny) * 10.0,
        rx_power_dbm=np.full((1, ny, nx), -np.inf), path_loss_db=np.full((1, ny, nx), np.nan),
        best_server_id=np.full((ny, nx), -1), max_rx_power_dbm=np.full((ny, nx), -np.inf),
        rms_delay_spread_ns=np.full((ny, nx), np.nan), angular_spread_bs_phi=np.full((ny, nx), np.nan))


def simulate(tx_ids, **kwargs):
    sim = PropagationSimulator()
    for n, tx_id in enumerate(tx_ids):
        sim.add_transmitter(Transmitter(tx_id=tx_id, x=40.0 + 45.0 * n, y=30.0 + 37.0 * (n % 3), z=20.0))
    return sim.run_raytracing_simulation((0.0, 500.0), (0.0, 150.0), resolution_m=10.0, **kwargs)


def export(result, metric, **kwargs):
    """Export to a temporary SVG with warnings turned into errors; return (text, parsed root)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "map.svg")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MapVisualizer.export_svg(result, path, metric=metric, **kwargs)
        with open(path, encoding="utf-8") as f:
            text = f.read()
    return text, ET.fromstring(text)


def texts(root):
    return [el.text for el in root.iter(SVG + "text")]


class SvgTests(unittest.TestCase):
    def test_all_nan_maps_do_not_crash_or_warn(self):
        for metric in ("rx_power", "delay_spread", "best_server"):
            for theme in ("light", "dark"):
                with self.subTest(metric=metric, theme=theme):
                    text, root = export(empty_result(), metric, theme=theme)
                    self.assertNotIn("nan", text.lower().replace("no data", ""))
                    self.assertNotIn("inf", text.lower())
                    self.assertIn("No data", texts(root))

    def test_axes_in_meters_and_equal_aspect(self):
        result = simulate([0, 1])
        text, root = export(result, "rx_power")
        labels = texts(root)
        self.assertIn("x [m]", labels)
        self.assertIn("y [m]", labels)
        for tick in ("0", "100", "200", "300", "400", "500"):
            self.assertIn(tick, labels)
        # Cells are square because both axes share one scale (the grid step is 10 m in x and y)
        cell = next(el for el in root.iter(SVG + "rect") if el.find(SVG + "title") is not None)
        self.assertAlmostEqual(float(cell.get("width")), float(cell.get("height")), delta=0.02)
        self.assertIn("m: ", cell.find(SVG + "title").text)

    def test_continuous_legend_has_end_and_intermediate_values(self):
        result = simulate([0, 1])
        _, root = export(result, "rx_power")
        labels = texts(root)
        valid = result.max_rx_power_dbm[np.isfinite(result.max_rx_power_dbm)]
        self.assertIn(f"{valid.min():.1f}", labels)
        self.assertIn(f"{valid.max():.1f}", labels)
        self.assertIn("dBm", labels)
        intermediate = [float(t) for t in labels
                        if t.lstrip("-").replace(".", "").isdigit() and valid.min() < float(t) < valid.max() and float(t) < 0]
        self.assertGreaterEqual(len(intermediate), 3)
        self.assertEqual(len(list(root.iter(SVG + "linearGradient"))), 1)

    def test_best_server_legend_is_categorical_and_follows_tx_order(self):
        result = simulate([42, 10])               # non-consecutive, unsorted ids
        text, root = export(result, "best_server")
        labels = texts(root)
        self.assertIn("Best server", labels)
        self.assertEqual(labels.count("Tx 42"), 2)        # legend entry + marker label
        self.assertEqual(labels.count("Tx 10"), 2)
        self.assertEqual(len(list(root.iter(SVG + "linearGradient"))), 0)
        palette = _THEMES["light"]["categorical"]
        fills = {el.get("fill") for el in root.iter(SVG + "rect") if el.find(SVG + "title") is not None}
        self.assertEqual(fills, {palette[0], palette[1]})
        # The first transmitter added (id 42) owns the first palette slot
        first = next(el for el in root.iter(SVG + "rect")
                     if el.find(SVG + "title") is not None and el.find(SVG + "title").text.endswith("Tx 42"))
        self.assertEqual(first.get("fill"), palette[0])

    def test_more_transmitters_than_palette_slots(self):
        result = simulate(list(range(100, 111)))  # 11 transmitters, palette has 8 colors
        _, root = export(result, "best_server", theme="dark")
        labels = texts(root)
        self.assertIn("Other (3 Tx)", labels)
        fills = {el.get("fill") for el in root.iter(SVG + "rect") if el.find(SVG + "title") is not None}
        self.assertLessEqual(fills, set(_THEMES["dark"]["categorical"]) | {_THEMES["dark"]["other"]})

    def test_constant_field_and_single_cell(self):
        result = simulate([0])
        result.rms_delay_spread_ns[:] = 0.0
        export(result, "delay_spread")
        sim = PropagationSimulator()
        sim.add_transmitter(Transmitter(tx_id=0, x=5.0, y=5.0, z=10.0))
        export(sim.run_raytracing_simulation((0.0, 0.0), (0.0, 0.0), resolution_m=10.0), "rx_power")

    def test_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MapVisualizer.export_svg(empty_result(), os.path.join(tmp, "x.svg"), metric="snr")
            with self.assertRaises(ValueError):
                MapVisualizer.export_svg(empty_result(), os.path.join(tmp, "x.svg"), theme="sepia")

    def test_nice_ticks(self):
        self.assertEqual(_nice_ticks(0.0, 1000.0, 5), [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
        self.assertEqual(_nice_ticks(-41.5, -11.4, 6), [-40.0, -35.0, -30.0, -25.0, -20.0, -15.0])
        self.assertEqual(_nice_ticks(3.0, 3.0), [3.0])


class AsciiTests(unittest.TestCase):
    def test_all_nan_matrix(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = MapVisualizer.render_ascii_heatmap(np.full((2, 5), np.nan), title="T")
        lines = out.splitlines()
        self.assertIn("no data", lines[0])
        self.assertEqual(lines[2], "|     |")

    def test_shading_and_orientation(self):
        matrix = np.array([[0.0, 5.0, 10.0], [np.nan, -np.inf, np.inf]])
        lines = MapVisualizer.render_ascii_heatmap(matrix, title="T", unit="dB").splitlines()
        self.assertEqual(lines[0], "=== T (0.0 to 10.0 dB) ===")
        self.assertEqual(lines[2], "|   |")       # second matrix row (non-finite) drawn on top
        self.assertEqual(lines[3], "| =@|")       # first matrix row at the bottom: low, mid, high


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_help(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as ctx:
            cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        for option in ("--mode", "--reflector", "--outdir", "--freq", "--coherent", "--sector", "--downtilt",
                       "--shadow-sigma", "--seed", "--min-distance", "--strict", "--sinr", "--noise-figure", "--bandwidth"):
            self.assertIn(option, stdout.getvalue())

    def test_every_mode_writes_its_outputs(self):
        common = ["simulation_results.csv", "simulation_summary.json", "coverage_rx_power.svg", "coverage_best_server.svg"]
        for mode in ("raytracing", "tworay", "hata", "fspl"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                code, out, err = self.run_cli("--mode", mode, "--size", "400", "--res", "50", "--outdir", tmp)
                self.assertEqual(code, 0)
                self.assertEqual(err, "")
                expected = common + (["coverage_delay_spread.svg"] if mode == "raytracing" else [])
                self.assertEqual(sorted(os.listdir(tmp)), sorted(expected))
                with open(os.path.join(tmp, "simulation_summary.json")) as f:
                    summary = json.load(f)
                self.assertEqual(summary["grid"]["nx"], 9)
                self.assertIsInstance(summary["metrics"]["mean_rx_power_dbm"], float)
                for name in expected:
                    if name.endswith(".svg"):
                        ET.parse(os.path.join(tmp, name))

    def test_hata_picks_the_model_by_frequency_and_reports_range_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, _ = self.run_cli("--mode", "hata", "--freq", "900e6", "--tx-height", "30",
                                     "--size", "400", "--res", "100", "--outdir", tmp)
            self.assertIn("Okumura-Hata", out)
            self.assertIn("distance outside 1-20 km", out)       # a 400 m area is closer than 1 km
            self.assertNotIn("frequency", out.split("[!]")[1])
            _, out, _ = self.run_cli("--mode", "hata", "--size", "400", "--res", "100", "--outdir", tmp)
            self.assertIn("COST-231 Hata", out)
            self.assertIn("frequency 2400 MHz outside 1500-2000 MHz", out)

    def test_reflector_switch(self):
        spreads = {}
        for setting in ("on", "off"):
            with tempfile.TemporaryDirectory() as tmp:
                code, _, _ = self.run_cli("--mode", "raytracing", "--reflector", setting,
                                          "--size", "400", "--res", "50", "--outdir", tmp)
                self.assertEqual(code, 0)
                with open(os.path.join(tmp, "simulation_summary.json")) as f:
                    spreads[setting] = json.load(f)["metrics"]["mean_delay_spread_ns"]
        self.assertGreater(spreads["on"], 2 * spreads["off"])

    def read_outputs(self, tmp):
        with open(os.path.join(tmp, "simulation_results.csv")) as f:
            csv_text = f.read()
        with open(os.path.join(tmp, "simulation_summary.json")) as f:
            return csv_text, json.load(f)

    def test_sinr_option_adds_columns_json_block_and_map(self):
        for mode in ("raytracing", "tworay", "hata", "fspl"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                code, out, err = self.run_cli("--mode", mode, "--size", "400", "--res", "50", "--sinr",
                                              "--noise-figure", "5", "--bandwidth", "10e6", "--outdir", tmp)
                self.assertEqual((code, err), (0, ""))
                self.assertIn("coverage_sinr.svg", os.listdir(tmp))
                ET.parse(os.path.join(tmp, "coverage_sinr.svg"))
                csv_text, summary = self.read_outputs(tmp)
                self.assertTrue(csv_text.splitlines()[0].endswith(",sinr_db,interference_plus_noise_dbm"))
                self.assertEqual(summary["sinr"]["noise_figure_db"], 5.0)
                self.assertEqual(summary["sinr"]["bandwidth_hz"], 10e6)
                self.assertIsInstance(summary["sinr"]["mean_sinr_db"], float)
                self.assertIn("SINR (noise -99.0 dBm in 10 MHz, NF 5 dB)", out)
        # Without --sinr nothing changes
        with tempfile.TemporaryDirectory() as tmp:
            self.run_cli("--size", "400", "--res", "50", "--outdir", tmp)
            csv_text, summary = self.read_outputs(tmp)
            self.assertEqual(csv_text.splitlines()[0],
                             "x_m,y_m,best_tx_id,rx_power_dbm,path_loss_db,rms_delay_spread_ns,angular_spread_phi_deg")
            self.assertNotIn("sinr", summary)

    def test_coherent_sector_and_shadow_options(self):
        def run(*options):
            with tempfile.TemporaryDirectory() as tmp:
                code, out, err = self.run_cli("--size", "400", "--res", "20", "--outdir", tmp, *options)
                self.assertEqual((code, err), (0, ""))
                csv_text, summary = self.read_outputs(tmp)
            return out, csv_text, summary["metrics"]

        out, base_csv, base = run()
        self.assertIn("ray tracing (reflector on)", out)          # same banner as 1.1.0 by default
        out, coherent_csv, coherent = run("--coherent")
        self.assertIn("coherent sum", out)
        self.assertNotEqual(coherent_csv, base_csv)
        self.assertLess(coherent["min_rx_power_dbm"], base["min_rx_power_dbm"] - 3.0)      # fades
        # The spreads still use |a_k|^2; the mean only moves because fades change the best server of some pixels
        self.assertAlmostEqual(coherent["mean_delay_spread_ns"], base["mean_delay_spread_ns"], delta=5.0)
        out, _, sector = run("--sector", "--downtilt", "5")
        self.assertIn("65-degree sectors", out)
        self.assertLess(sector["mean_rx_power_dbm"], base["mean_rx_power_dbm"] - 1.0)
        self.assertLessEqual(sector["max_rx_power_dbm"], base["max_rx_power_dbm"])
        # Shadow fading: reproducible with --seed, different with another seed, off by default
        out, shadow_a, _ = run("--shadow-sigma", "8", "--seed", "3")
        self.assertIn("sigma = 8 dB (seed 3)", out)
        _, shadow_b, _ = run("--shadow-sigma", "8", "--seed", "3")
        _, shadow_c, _ = run("--shadow-sigma", "8", "--seed", "4")
        self.assertEqual(shadow_a, shadow_b)
        self.assertNotEqual(shadow_a, shadow_c)
        self.assertNotEqual(shadow_a, base_csv)
        _, seed_only, _ = run("--seed", "3")
        self.assertEqual(seed_only, base_csv)
        # The new options are also accepted by the empirical modes (where --coherent only prints a note)
        out, _, _ = run("--mode", "fspl", "--coherent", "--sector", "--shadow-sigma", "4", "--seed", "1")
        self.assertIn("--coherent is ignored", out)

    def test_min_distance_option(self):
        """A 1.5 m mast on a grid node: that pixel has no data unless --min-distance 0."""
        def nan_rows(*options):
            with tempfile.TemporaryDirectory() as tmp:
                code, _, err = self.run_cli("--mode", "fspl", "--size", "100", "--res", "10", "--tx-height", "1.5",
                                            "--outdir", tmp, *options)
                self.assertEqual((code, err), (0, ""))
                csv_text, _ = self.read_outputs(tmp)
            return [line for line in csv_text.splitlines() if ",nan,nan," in line]

        # Each mast is the only transmitter closer than 1 m to its pixel, so the other one serves it
        self.assertEqual(nan_rows(), [])
        with tempfile.TemporaryDirectory() as tmp:
            self.run_cli("--mode", "fspl", "--size", "100", "--res", "10", "--tx-height", "1.5", "--outdir", tmp)
            near = [line.split(",") for line in self.read_outputs(tmp)[0].splitlines() if line.startswith("30.0,30.0,")][0]
            self.run_cli("--mode", "fspl", "--size", "100", "--res", "10", "--tx-height", "1.5", "--min-distance", "0",
                         "--outdir", tmp)
            raw = [line.split(",") for line in self.read_outputs(tmp)[0].splitlines() if line.startswith("30.0,30.0,")][0]
        self.assertEqual(near[2], "1")                # Tx0 stands on (30, 30): its own pixel is served by Tx1
        self.assertEqual(raw[2], "0")                 # 1.1.0 behaviour: P_rx = P_tx + G at d = 0
        self.assertEqual(float(raw[3]), 55.0)

    def test_strict_hata_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = self.run_cli("--mode", "hata", "--strict", "--size", "400", "--res", "100", "--outdir", tmp)
            self.assertEqual(code, 1)
            self.assertIn("[x] Error: Cost231Hata (strict=True)", err)
            self.assertIn("frequency 2400 MHz outside 1500-2000 MHz", err)
            self.assertEqual(os.listdir(tmp), [])
            code, out, err = self.run_cli("--mode", "fspl", "--strict", "--size", "400", "--res", "100", "--outdir", tmp)
            self.assertEqual(code, 0)
            self.assertIn("--strict only applies to --mode hata", out)

    def test_invalid_arguments_exit_with_usage_error(self):
        for args in (["--res", "0"], ["--size", "-10"], ["--freq", "0"], ["--mode", "magic"],
                     ["--shadow-sigma", "-1"], ["--min-distance", "-1"], ["--bandwidth", "0"], ["--seed", "x"],
                     ["--reflector", "maybe"], ["--size", "100000", "--res", "1"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
                cli.main(args)
            self.assertEqual(ctx.exception.code, 2)
