"""
Tests of OpenRPS 1.7.0: measurements (CSV), comparison with the prediction,
calibration of the scenario, floor plan in the project file, calibration CLI
and API of the graphical interface.
"""

import base64
import copy
import io
import json
import math
import os
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from open_rps.calibrate_cli import main as calibrate_main  # noqa: E402
from open_rps.measurements import (Measurement, MeasurementError, calibrate, compare,  # noqa: E402
                                   predict_points, read_csv)
from open_rps.project import (FloorPlan, Project, ProjectError, TransmitterSpec, example_office,  # noqa: E402
                              example_urban)

# 1 x 1 transparent PNG
PNG = "data:image/png;base64," + base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")).decode()


def office_truth():
    return example_office()


def synthetic(project, n=60, seed=3, noise_db=0.0, per_tx=False):
    """Measurements taken from the prediction of ``project`` itself (plus noise)."""
    rng = np.random.default_rng(seed)
    a = project.area
    pts = np.column_stack([rng.uniform(a.x_min, a.x_max, n), rng.uniform(a.y_min, a.y_max, n),
                           np.full(n, a.rx_height_m)])
    pred = predict_points(project, pts)
    ids = [t.tx_id for t in project.enabled_transmitters()]
    out = []
    for i, p in enumerate(pts):
        if per_tx:
            t = i % len(ids)
            value, tx = pred[t, i], ids[t]
        else:
            row = np.where(np.isnan(pred[:, i]), -np.inf, pred[:, i])
            value, tx = row.max(), None
        if np.isfinite(value):
            out.append(Measurement(x=float(p[0]), y=float(p[1]), rssi_dbm=float(value + rng.normal(0, noise_db)),
                                   tx_id=tx))
    return out


class CsvTests(unittest.TestCase):
    def test_comma_file_with_accents_units_and_optional_columns(self):
        text = ("# medido con el teléfono\n"
                "Punto,X (m),Y (m),Altura,Señal (dBm),AP\n"
                "sala,1.5,2,1.2,-61.5,AP pasillo\n"
                "cocina,10,4,,-70,0\n")
        ms = read_csv(text, [TransmitterSpec(0, name="AP pasillo")])
        self.assertEqual(len(ms), 2)
        self.assertEqual((ms[0].x, ms[0].y, ms[0].z, ms[0].rssi_dbm, ms[0].tx_id, ms[0].label),
                         (1.5, 2.0, 1.2, -61.5, 0, "sala"))
        self.assertIsNone(ms[1].z)
        self.assertEqual(ms[1].tx_id, 0)

    def test_semicolon_and_tab_with_decimal_comma(self):
        for sep in (";", "\t"):
            ms = read_csv(f"x{sep}y{sep}rssi\n1,5{sep}2{sep}-67,25\n3{sep}4,75{sep}−80\n")
            self.assertEqual([(m.x, m.y, m.rssi_dbm) for m in ms], [(1.5, 2.0, -67.25), (3.0, 4.75, -80.0)])

    def test_file_path_and_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.csv")
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write("x,y,dbm\n0,0,-50\n")
            self.assertEqual(read_csv(path)[0].rssi_dbm, -50.0)

    def test_errors_name_the_row_and_column(self):
        with self.assertRaisesRegex(MeasurementError, "faltan las columnas potencia"):
            read_csv("x,y,z\n1,2,3\n")
        with self.assertRaisesRegex(MeasurementError, "fila 3, y"):
            read_csv("x,y,rssi\n1,2,-50\n1,dos,-50\n")
        with self.assertRaisesRegex(MeasurementError, "fila 2: no hay ningún transmisor llamado 'AP techo'"):
            read_csv("x,y,rssi,tx\n1,2,-50,AP techo\n", [TransmitterSpec(0, name="AP pasillo")])
        with self.assertRaisesRegex(MeasurementError, "tx_id 7"):
            read_csv("x,y,rssi,tx\n1,2,-50,7\n", [TransmitterSpec(0)])
        with self.assertRaisesRegex(MeasurementError, "hace falta una fila"):
            read_csv("x,y,rssi\n")


class PredictionTests(unittest.TestCase):
    def assert_matches_grid(self, project):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # Hata outside its range: expected here
            result = project.run()
        xs, ys = np.meshgrid(result.x_coords, result.y_coords)
        pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, project.area.rx_height_m)])
        pred = predict_points(project, pts)
        grid = np.asarray(result.rx_power_dbm).reshape(len(pred), -1)
        np.testing.assert_array_equal(np.isnan(pred), np.isnan(grid))
        np.testing.assert_array_equal(np.isneginf(pred), np.isneginf(grid))
        finite = np.isfinite(grid)
        np.testing.assert_allclose(pred[finite], grid[finite], rtol=0, atol=1e-9)

    def test_raytracing_equals_the_map(self):
        project = example_urban()
        project.area.resolution_m = 25.0
        project.settings.shadow_sigma_db = 0.0
        self.assert_matches_grid(project)

    def test_empirical_modes_and_sector_antenna_equal_the_map(self):
        for mode in ("fspl", "logdistance", "hata", "tworay"):
            project = example_urban()
            project.area.resolution_m = 25.0
            project.settings.mode = mode
            project.transmitters[0].pattern = "sector"
            project.transmitters[0].downtilt_deg = 4.0
            with self.subTest(mode=mode):
                self.assert_matches_grid(project)


class CompareTests(unittest.TestCase):
    def test_sign_statistics_and_statuses(self):
        project = example_urban()
        tx = project.enabled_transmitters()
        exact = synthetic(project, n=20, per_tx=True)
        for m in exact:
            m.rssi_dbm -= 3.0               # measured 3 dB below the prediction: optimistic model, error +3
        near = Measurement(x=tx[0].x, y=tx[0].y, z=tx[0].z, rssi_dbm=-20.0, tx_id=tx[0].tx_id)
        missing = Measurement(x=0.0, y=0.0, rssi_dbm=-80.0, tx_id=99)
        c = compare(project, exact + [near, missing])
        self.assertEqual(c.counts, {"ok": len(exact), "sin señal": 0, "muy cerca": 1, "transmisor ausente": 1})
        self.assertAlmostEqual(c.stats["bias_db"], 3.0, places=9)
        self.assertAlmostEqual(c.stats["rmse_db"], 3.0, places=9)
        self.assertAlmostEqual(c.stats["std_db"], 0.0, places=9)
        self.assertGreater(c.stats["r"], 0.999)
        for p, m in zip(c.points, exact):
            self.assertEqual(p.tx_id, m.tx_id)
        json.dumps(c.to_dict(), allow_nan=False)

    def test_best_server_when_no_transmitter_is_named(self):
        project = example_urban()
        ms = synthetic(project, n=30)
        c = compare(project, ms)
        pts = np.array([[m.x, m.y, project.area.rx_height_m] for m in ms])
        pred = predict_points(project, pts)
        ids = [t.tx_id for t in project.enabled_transmitters()]
        for p, column in zip(c.points, pred.T):
            self.assertEqual(p.tx_id, ids[int(np.nanargmax(column))])
        self.assertLess(c.stats["max_abs_db"], 1e-9)

    def test_no_signal_point(self):
        project = example_urban()
        building = project.buildings[0]
        building.material = "metal"
        x = float(np.mean([v[0] for v in building.footprint]))
        y = float(np.mean([v[1] for v in building.footprint]))
        c = compare(project, [Measurement(x=x, y=y, z=1.0, rssi_dbm=-90.0)])
        self.assertEqual(c.points[0].status, "sin señal")
        self.assertEqual(c.stats["n"], 0)

    def test_without_measurements(self):
        with self.assertRaisesRegex(MeasurementError, "no hay mediciones"):
            compare(example_urban())


class CalibrationTests(unittest.TestCase):
    def test_recovers_material_losses_and_power_without_noise(self):
        truth = office_truth()
        frequency = truth.reference_frequency_hz()
        start = copy.deepcopy(truth)
        materials = start.material_map()
        materials["tablaroca"].loss_db = 9.0
        materials["concreto"].loss_a_db -= 8.0                         # linear law: the offset moves
        start.transmitters[0].tx_power_dbm += 6.0
        start.measurements = synthetic(truth, n=80, seed=4)
        result = calibrate(start)
        got = {p.key: p for p in result.parameters}
        true_loss = {m.id: m.penetration_loss_db(frequency) for m in truth.materials}
        self.assertAlmostEqual(got["tablaroca"].after, true_loss["tablaroca"], delta=0.05)
        self.assertAlmostEqual(got["concreto"].after, true_loss["concreto"], delta=0.3)
        self.assertAlmostEqual(got["*"].after, -6.0, delta=0.05)
        self.assertLess(result.after.stats["rmse_db"], 0.1)
        self.assertGreater(result.before.stats["rmse_db"], 3.0)
        # The linear law keeps its slope; the original project is untouched
        new = result.project.material_map()["concreto"]
        self.assertEqual(new.loss_b_db_per_ghz, truth.material_map()["concreto"].loss_b_db_per_ghz)
        self.assertAlmostEqual(new.penetration_loss_db(frequency), got["concreto"].after, places=9)
        self.assertEqual(start.material_map()["tablaroca"].loss_db, 9.0)
        self.assertEqual(result.project.transmitters[0].tx_power_dbm, start.transmitters[0].tx_power_dbm
                         + got["*"].after)

    def test_noisy_urban_per_transmitter(self):
        truth = example_urban()
        start = copy.deepcopy(truth)
        start.material_map()["ladrillo"].loss_db += 6.0
        start.transmitters[0].tx_power_dbm += 4.0
        start.transmitters[1].tx_power_dbm -= 3.0
        start.measurements = synthetic(truth, n=120, seed=5, noise_db=3.0, per_tx=True)
        result = calibrate(start, materials=["ladrillo", "concreto"])
        got = {p.key: p for p in result.parameters}
        self.assertEqual(set(got), {"ladrillo", "concreto", "0", "1"})
        for key, true in (("ladrillo", 10.0), ("0", -4.0), ("1", 3.0)):
            p = got[key]
            self.assertLess(abs(p.after - true), 3.5 * p.uncertainty + 0.2, (key, p.after, p.uncertainty))
        self.assertLess(result.after.stats["rmse_db"], 3.6)
        self.assertLess(abs(result.after.stats["bias_db"]), 0.5)

    def test_log_distance_exponent(self):
        truth = example_urban()
        truth.settings.mode = "logdistance"
        truth.settings.log_distance_exponent = 3.4
        start = copy.deepcopy(truth)
        start.settings.log_distance_exponent = 2.0
        start.transmitters[0].tx_power_dbm += 5.0
        start.transmitters[1].tx_power_dbm += 5.0
        start.measurements = synthetic(truth, n=50, seed=6)
        result = calibrate(start)
        got = {p.key: p for p in result.parameters}
        self.assertEqual(set(got), {"exponent", "*"})        # no materials outside ray tracing
        self.assertAlmostEqual(got["exponent"].after, 3.4, delta=0.01)
        self.assertAlmostEqual(got["*"].after, -5.0, delta=0.1)
        self.assertAlmostEqual(result.project.settings.log_distance_exponent, got["exponent"].after)

    def test_material_no_ray_crosses_is_left_alone(self):
        truth = office_truth()
        start = copy.deepcopy(truth)
        start.measurements = synthetic(truth, n=30, seed=7)
        start.walls = [w for w in start.walls if w.material != "vidrio"]
        start.buildings = [b for b in start.buildings if b.material != "vidrio"]
        result = calibrate(start, materials=["tablaroca"], tx_power="none")
        self.assertEqual([p.key for p in result.parameters], ["tablaroca"])
        with self.assertRaisesRegex(MeasurementError, "no tiene una pérdida finita"):
            calibrate(start, materials=["metal"])

    def test_parameter_with_too_few_points_is_left_alone(self):
        truth = example_urban()
        start = copy.deepcopy(truth)
        start.transmitters[0].tx_power_dbm += 4.0
        ms = synthetic(truth, n=60, seed=11, per_tx=True)
        first, second = (t.tx_id for t in truth.enabled_transmitters())
        start.measurements = [m for m in ms if m.tx_id == first] + [m for m in ms if m.tx_id == second][:2]
        result = calibrate(start, materials=None)
        got = {p.key: p for p in result.parameters}
        self.assertAlmostEqual(got[str(first)].after, -4.0, delta=0.02)
        self.assertEqual(got[str(second)].after, 0.0)
        self.assertEqual(got[str(second)].points, 2)
        self.assertIsNone(got[str(second)].uncertainty)
        self.assertTrue(any("solo 2 punto(s)" in note for note in result.notes))

    def test_too_few_points(self):
        start = office_truth()
        start.measurements = synthetic(start, n=3, seed=8)
        with self.assertRaisesRegex(MeasurementError, "puntos utilizables"):
            calibrate(start)


class ProjectFileTests(unittest.TestCase):
    def test_round_trip_and_version(self):
        project = example_office()
        self.assertEqual(project.to_dict()["version"], 1)
        self.assertNotIn("floor_plan", project.to_dict())
        project.floor_plan = FloorPlan(image=PNG, pixel_width=800, pixel_height=400, x_m=-2, y_m=-1,
                                       width_m=44.0, opacity=0.5, name="plano.png")
        project.measurements = [Measurement(1.0, 2.0, -60.0, tx_id=0, label="a"), Measurement(3.0, 4.0, -70.5)]
        data = project.to_dict()
        self.assertEqual(data["version"], 2)
        again = Project.from_dict(json.loads(json.dumps(data)))
        self.assertEqual(again.to_dict(), data)
        self.assertAlmostEqual(again.floor_plan.height_m, 22.0)
        self.assertAlmostEqual(again.floor_plan.meters_per_pixel, 0.055)

    def test_validation(self):
        project = example_office()
        for bad, message in ((dict(image="http://x/plano.png"), "data URL"),
                             (dict(image="data:image/bmp;base64,AAAA"), "no soportado"),
                             (dict(opacity=1.5), "opacity"), (dict(pixel_width=0), "pixel_width"),
                             (dict(width_m=0), "width_m")):
            fields = dict(image=PNG, pixel_width=10, pixel_height=10)
            fields.update(bad)
            project.floor_plan = FloorPlan(**fields)
            with self.subTest(bad=bad), self.assertRaisesRegex(ProjectError, message):
                project.validate()
        project.floor_plan = None
        data = project.to_dict()
        data["measurements"] = [{"x": 1, "y": 2, "rssi_dbm": "fuerte"}]
        with self.assertRaisesRegex(ProjectError, "medición 1.rssi_dbm"):
            Project.from_dict(data)
        data["measurements"] = [{"x": 1, "y": 2, "dbm": -50}]
        with self.assertRaisesRegex(ProjectError, "campos desconocidos dbm"):
            Project.from_dict(data)


class CliTests(unittest.TestCase):
    def test_calibrate_and_save(self):
        truth = office_truth()
        start = copy.deepcopy(truth)
        start.material_map()["tablaroca"].loss_db = 9.0
        ms = synthetic(truth, n=40, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            project_path, csv_path = os.path.join(tmp, "p.orps.json"), os.path.join(tmp, "m.csv")
            out_path, errors_path = os.path.join(tmp, "c.orps.json"), os.path.join(tmp, "e.csv")
            start.save(project_path)
            with open(csv_path, "w") as f:
                f.write("x;y;potencia\n" + "".join(f"{m.x};{m.y};{str(m.rssi_dbm).replace('.', ',')}\n" for m in ms))
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = calibrate_main([project_path, csv_path, "-o", out_path, "--errors", errors_path,
                                       "--materials", "tablaroca"])
            self.assertEqual(code, 0, err.getvalue())
            self.assertIn("Tablaroca: 9.00 -> 3.00 dB", out.getvalue())
            calibrated = Project.load(out_path)
            self.assertAlmostEqual(calibrated.material_map()["tablaroca"].loss_db, 3.0, delta=0.02)
            self.assertEqual(len(calibrated.measurements), len(ms))
            with open(errors_path) as f:
                self.assertEqual(len(f.read().strip().splitlines()), len(ms) + 1)
            with redirect_stdout(io.StringIO()) as out2, redirect_stderr(io.StringIO()):
                self.assertEqual(calibrate_main([out_path, "--compare-only"]), 0)
            self.assertIn(f"Comparación: {len(ms)} puntos comparados", out2.getvalue())
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err3:
                self.assertEqual(calibrate_main([project_path, "--compare-only"]), 1)
            self.assertIn("no hay mediciones", err3.getvalue())


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import threading
        from open_rps.gui.server import make_server
        cls.server = make_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, body):
        import urllib.error
        import urllib.request
        request = urllib.request.Request(self.server.url + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_parse_compare_calibrate(self):
        truth = office_truth()
        start = copy.deepcopy(truth)
        start.material_map()["tablaroca"].loss_db = 9.0
        start.floor_plan = FloorPlan(image=PNG, pixel_width=1, pixel_height=1, width_m=40.0)
        ms = synthetic(truth, n=30, seed=10)
        csv_text = "x,y,rssi,tx\n" + "".join(f"{m.x},{m.y},{m.rssi_dbm},AP pasillo\n" for m in ms)
        status, data = self.post("/api/measurements/parse", {"project": start.to_dict(), "csv": csv_text})
        self.assertEqual(status, 200, data)
        self.assertEqual(len(data["measurements"]), len(ms))
        self.assertEqual(data["measurements"][0]["tx_id"], 0)
        project = start.to_dict()
        project["measurements"] = data["measurements"]
        status, data = self.post("/api/compare", {"project": project})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["stats"]["n"], len(ms))
        status, data = self.post("/api/calibrate", {"project": project, "options": {"materials": ["tablaroca"]}})
        self.assertEqual(status, 200, data)
        self.assertNotIn("project", data)
        self.assertEqual([m["id"] for m in data["changes"]["materials"]], ["tablaroca"])
        self.assertAlmostEqual(data["changes"]["materials"][0]["loss_db"], 3.0, delta=0.02)
        self.assertEqual(data["changes"]["settings"], {})
        status, data = self.post("/api/measurements/parse", {"project": project, "csv": "x,y\n1,2\n"})
        self.assertEqual(status, 400)
        self.assertIn("faltan las columnas", data["error"])


if __name__ == "__main__":
    unittest.main()
