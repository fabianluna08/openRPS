import csv
import json
import math
import os
import tempfile
import unittest
import warnings

import numpy as np

from open_rps import (
    FreeSpacePathLoss,
    LogDistancePathLoss,
    OkumuraHata,
    PlanarReflector,
    PropagationModel,
    PropagationSimulator,
    Transmitter,
    TwoRayGroundReflection,
)

AREA = ((0.0, 400.0), (0.0, 300.0))


def build_sim(tx_ids=(0, 1), reflector=True):
    sim = PropagationSimulator()
    sim.add_transmitter(Transmitter(tx_id=tx_ids[0], x=100.0, y=100.0, z=25.0, tx_power_dbm=43.0,
                                    frequency_hz=2.4e9, antenna_gain_dbi=12.0))
    sim.add_transmitter(Transmitter(tx_id=tx_ids[1], x=300.0, y=200.0, z=20.0, tx_power_dbm=40.0,
                                    frequency_hz=1.8e9, antenna_gain_dbi=10.0))
    if reflector:
        sim.add_reflector(PlanarReflector(p0=[200.0, 150.0, 0.0], normal=[1.0, 0.2, 0.0], reflectivity=0.6))
    return sim


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class RayTracingSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = build_sim()
        cls.result = cls.sim.run_raytracing_simulation(*AREA, resolution_m=20.0, rx_height_m=1.5)

    def test_shapes_and_coordinates(self):
        r = self.result
        self.assertEqual(len(r.x_coords), 21)
        self.assertEqual(len(r.y_coords), 16)
        self.assertEqual(r.rx_power_dbm.shape, (2, 16, 21))
        self.assertEqual(r.path_loss_db.shape, (2, 16, 21))
        for grid in (r.best_server_id, r.max_rx_power_dbm, r.rms_delay_spread_ns, r.angular_spread_bs_phi):
            self.assertEqual(grid.shape, (16, 21))
        self.assertEqual(r.tx_ids, [0, 1])

    def test_values_are_physically_consistent(self):
        r = self.result
        self.assertTrue(np.isfinite(r.rx_power_dbm).all())
        self.assertTrue(np.isfinite(r.rms_delay_spread_ns).all())
        self.assertTrue((r.rms_delay_spread_ns >= 0).all())
        self.assertTrue((r.angular_spread_bs_phi >= 0).all() and (r.angular_spread_bs_phi <= 180).all())
        np.testing.assert_allclose(r.max_rx_power_dbm, r.rx_power_dbm.max(axis=0))
        np.testing.assert_allclose(r.rx_power_dbm[0] + r.path_loss_db[0], 43.0)
        np.testing.assert_allclose(r.rx_power_dbm[1] + r.path_loss_db[1], 40.0)
        # Each transmitter serves its own surroundings; the strongest pixel is next to a transmitter
        self.assertEqual(r.best_server_id[5, 5], 0)       # (100, 100)
        self.assertEqual(r.best_server_id[10, 15], 1)     # (300, 200)
        j, i = np.unravel_index(np.argmax(r.max_rx_power_dbm), r.max_rx_power_dbm.shape)
        self.assertIn((r.x_coords[i], r.y_coords[j]), [(100.0, 100.0), (300.0, 200.0)])
        # Never more than direct + ground + wall arriving with the full direct-ray power each
        d0 = np.sqrt((r.x_coords[None, :] - 100.0) ** 2 + (r.y_coords[:, None] - 100.0) ** 2 + 23.5 ** 2)
        direct_dbm = 43.0 + 12.0 - FreeSpacePathLoss().path_loss(d0, 2.4e9)
        self.assertTrue((r.rx_power_dbm[0] >= direct_dbm - 1e-9).all())
        self.assertTrue((r.rx_power_dbm[0] <= direct_dbm + 10 * math.log10(3.0)).all())

    def test_vectorized_matches_pointwise_reference(self):
        reference = self.sim.run_raytracing_simulation(*AREA, resolution_m=20.0, rx_height_m=1.5, vectorized=False)
        r = self.result
        np.testing.assert_array_equal(r.best_server_id, reference.best_server_id)
        np.testing.assert_array_equal(r.best_server_index, reference.best_server_index)
        for name in ("rx_power_dbm", "path_loss_db", "max_rx_power_dbm", "rms_delay_spread_ns", "angular_spread_bs_phi"):
            np.testing.assert_allclose(getattr(r, name), getattr(reference, name), rtol=1e-9, atol=1e-7,
                                       equal_nan=True, err_msg=name)

    def test_chunked_evaluation_gives_the_same_result(self):
        from open_rps import simulation
        original = simulation._MAX_CHUNK_POINTS
        simulation._MAX_CHUNK_POINTS = 50          # forces several row blocks
        try:
            chunked = self.sim.run_raytracing_simulation(*AREA, resolution_m=20.0, rx_height_m=1.5)
        finally:
            simulation._MAX_CHUNK_POINTS = original
        for name in ("rx_power_dbm", "rms_delay_spread_ns", "angular_spread_bs_phi", "best_server_id"):
            np.testing.assert_array_equal(getattr(chunked, name), getattr(self.result, name), err_msg=name)

    def test_reflector_adds_delay_spread(self):
        without = build_sim(reflector=False).run_raytracing_simulation(*AREA, resolution_m=20.0)
        self.assertGreater(np.mean(self.result.rms_delay_spread_ns), np.mean(without.rms_delay_spread_ns) + 1.0)
        self.assertTrue((self.result.max_rx_power_dbm >= without.max_rx_power_dbm - 1e-9).all())

    def test_export_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path, csv_path = os.path.join(tmp, "s.json"), os.path.join(tmp, "s.csv")
            self.result.export_json(json_path)
            self.result.export_csv(csv_path)
            with open(json_path) as f:
                summary = json.load(f)
            rows = read_csv(csv_path)
        self.assertEqual(summary["grid"], {"x_min": 0.0, "x_max": 400.0, "y_min": 0.0, "y_max": 300.0, "nx": 21, "ny": 16})
        self.assertAlmostEqual(summary["metrics"]["max_rx_power_dbm"], float(self.result.max_rx_power_dbm.max()))
        self.assertAlmostEqual(summary["metrics"]["mean_delay_spread_ns"], float(self.result.rms_delay_spread_ns.mean()))
        self.assertEqual(len(rows), 16 * 21)
        self.assertEqual((rows[0]["x_m"], rows[0]["y_m"]), ("0.0", "0.0"))
        self.assertEqual((rows[-1]["x_m"], rows[-1]["y_m"]), ("400.0", "300.0"))


class NonConsecutiveTxIdTests(unittest.TestCase):
    """Regression: export_csv indexed path_loss_db with the tx_id instead of the transmitter position."""

    def check(self, result, power_by_id):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            result.export_csv(path)          # raised IndexError in 1.0.0
            rows = read_csv(path)
        self.assertEqual(set(int(r["best_tx_id"]) for r in rows), {10, 42})
        self.assertEqual(result.tx_ids, [10, 42])
        nx = len(result.x_coords)
        for n, row in enumerate(rows):
            j, i = divmod(n, nx)
            tx_id = int(row["best_tx_id"])
            position = result.tx_ids.index(tx_id)
            self.assertEqual(result.best_server_index[j, i], position)
            self.assertAlmostEqual(float(row["path_loss_db"]), result.path_loss_db[position, j, i], delta=0.006)
            # P_rx + PL = P_tx of the *best* transmitter
            self.assertAlmostEqual(float(row["rx_power_dbm"]) + float(row["path_loss_db"]), power_by_id[tx_id], delta=0.011)

    def test_raytracing(self):
        result = build_sim(tx_ids=(10, 42)).run_raytracing_simulation(*AREA, resolution_m=25.0)
        self.check(result, {10: 43.0, 42: 40.0})

    def test_empirical(self):
        result = build_sim(tx_ids=(10, 42)).run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=25.0)
        self.check(result, {10: 43.0, 42: 40.0})

    def test_ids_in_reverse_order_are_not_swapped(self):
        """With ids (1, 0) the old code silently reported the path loss of the *other* transmitter."""
        result = build_sim(tx_ids=(1, 0)).run_raytracing_simulation(*AREA, resolution_m=25.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            result.export_csv(path)
            rows = read_csv(path)
        powers = {1: 43.0, 0: 40.0}
        for row in rows:
            self.assertAlmostEqual(float(row["rx_power_dbm"]) + float(row["path_loss_db"]),
                                   powers[int(row["best_tx_id"])], delta=0.011)

    def test_duplicate_and_negative_ids_are_rejected(self):
        sim = build_sim()
        with self.assertRaises(ValueError):
            sim.add_transmitter(Transmitter(tx_id=1, x=0.0, y=0.0, z=10.0))
        with self.assertRaises(ValueError):
            sim.add_transmitter(Transmitter(tx_id=-1, x=0.0, y=0.0, z=10.0))


class EmpiricalSimulationTests(unittest.TestCase):
    def test_fspl_grid_uses_3d_distance_and_antenna_gain(self):
        sim = build_sim()
        r = sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=20.0, rx_height_m=1.5)
        fspl = FreeSpacePathLoss()
        for (j, i) in ((0, 0), (5, 5), (15, 20), (7, 13)):
            d = math.dist((r.x_coords[i], r.y_coords[j], 1.5), (100.0, 100.0, 25.0))
            self.assertAlmostEqual(r.rx_power_dbm[0, j, i], 43.0 + 12.0 - fspl.path_loss(d, 2.4e9), places=9)
            self.assertAlmostEqual(r.path_loss_db[0, j, i], fspl.path_loss(d, 2.4e9) - 12.0, places=9)
        np.testing.assert_allclose(r.max_rx_power_dbm, r.rx_power_dbm.max(axis=0))
        # No multipath information in empirical mode
        self.assertTrue(np.isnan(r.rms_delay_spread_ns).all() and np.isnan(r.angular_spread_bs_phi).all())

    def test_vectorized_grid_matches_scalar_model_calls(self):
        sim = build_sim()
        models = [(TwoRayGroundReflection(), True), (OkumuraHata("suburban"), True), (LogDistancePathLoss(3.2), False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for model, ground in models:
                r = sim.run_empirical_simulation(model, *AREA, resolution_m=20.0, rx_height_m=1.5)
                self.assertTrue(np.isfinite(r.path_loss_db).all())
                for t, tx in enumerate(sim.transmitters):
                    for (j, i) in ((0, 0), (5, 5), (10, 15), (15, 20), (3, 17)):
                        d = math.hypot(r.x_coords[i] - tx.x, r.y_coords[j] - tx.y)
                        if not ground:
                            d = math.hypot(d, 1.5 - tx.z)
                        expected = model.path_loss(d, tx.frequency_hz, ht_m=tx.z, hr_m=1.5, tx_gain_dbi=tx.antenna_gain_dbi)
                        self.assertAlmostEqual(r.path_loss_db[t, j, i], expected, places=9, msg=type(model).__name__)

    def test_pixel_under_the_mast_is_not_a_zero_loss_hot_spot(self):
        r = build_sim().run_empirical_simulation(TwoRayGroundReflection(), *AREA, resolution_m=20.0)
        self.assertEqual((r.x_coords[5], r.y_coords[5]), (100.0, 100.0))
        self.assertGreater(r.path_loss_db[0, 5, 5], 30.0)
        self.assertLess(r.rx_power_dbm[0, 5, 5], 10.0)

    def test_user_defined_scalar_model_still_works(self):
        class LegacyModel(PropagationModel):
            """Written for OpenRPS 1.0: overrides path_loss and only understands floats."""
            def path_loss(self, distance_m, frequency_hz, **kwargs):
                return 40.0 + 30.0 * math.log10(max(1.0, distance_m))

        r = build_sim().run_empirical_simulation(LegacyModel(), *AREA, resolution_m=50.0, rx_height_m=1.5)
        d = math.dist((r.x_coords[8], r.y_coords[6], 1.5), (100.0, 100.0, 25.0))
        self.assertAlmostEqual(r.path_loss_db[0, 6, 8], 40.0 + 30.0 * math.log10(d), places=9)
        self.assertAlmostEqual(r.rx_power_dbm[0, 6, 8], 43.0 - r.path_loss_db[0, 6, 8], places=9)

    def test_json_export_is_strict_json_without_multipath_metrics(self):
        r = build_sim().run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=50.0)

        def reject(constant):
            raise AssertionError(f"non-standard JSON constant {constant}")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            with warnings.catch_warnings():
                warnings.simplefilter("error")       # e.g. numpy "All-NaN slice"
                r.export_json(path)
            with open(path) as f:
                summary = json.loads(f.read(), parse_constant=reject)
        self.assertIsNone(summary["metrics"]["mean_delay_spread_ns"])
        self.assertIsNone(summary["metrics"]["max_delay_spread_ns"])
        self.assertIsInstance(summary["metrics"]["mean_rx_power_dbm"], float)


class EdgeCaseTests(unittest.TestCase):
    def test_invalid_grid_arguments(self):
        sim = build_sim()
        with self.assertRaises(ValueError):
            sim.run_raytracing_simulation(*AREA, resolution_m=0.0)
        with self.assertRaises(ValueError):
            sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=-5.0)
        with self.assertRaises(ValueError):
            sim.run_raytracing_simulation((100.0, 0.0), (0.0, 100.0), resolution_m=10.0)

    def test_no_transmitters(self):
        sim = PropagationSimulator()
        for result in (sim.run_raytracing_simulation(*AREA, resolution_m=100.0),
                       sim.run_raytracing_simulation(*AREA, resolution_m=100.0, vectorized=False),
                       sim.run_empirical_simulation(FreeSpacePathLoss(), *AREA, resolution_m=100.0)):
            self.assertEqual(result.rx_power_dbm.shape, (0, 4, 5))
            self.assertTrue((result.best_server_id == -1).all())
            self.assertTrue(np.isneginf(result.max_rx_power_dbm).all())
            with tempfile.TemporaryDirectory() as tmp:
                result.export_csv(os.path.join(tmp, "e.csv"))
                result.export_json(os.path.join(tmp, "e.json"))
                with open(os.path.join(tmp, "e.json")) as f:
                    self.assertIsNone(json.load(f)["metrics"]["max_rx_power_dbm"])

    def test_fine_grid_coordinates_keep_their_precision_in_csv(self):
        sim = PropagationSimulator()
        sim.add_transmitter(Transmitter(tx_id=0, x=1.0, y=1.0, z=2.5, frequency_hz=5.2e9))
        result = sim.run_raytracing_simulation((0.0, 1.0), (0.0, 0.5), resolution_m=0.25)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fine.csv")
            result.export_csv(path)
            rows = read_csv(path)
        self.assertEqual([r["x_m"] for r in rows[:5]], ["0.0", "0.25", "0.5", "0.75", "1.0"])
