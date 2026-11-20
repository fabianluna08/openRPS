"""
Tests of OpenRPS 1.9.0: wall thickness (slab model of ITU-R P.2040), the multi-wall
indoor model, antenna presets and power in mW, and the restaurant of the Acrylic
Wi-Fi practice.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_rps.measurements import Measurement, calibrate, predict_points
from open_rps.project import (ANTENNA_PRESETS, Area, Project, ProjectError, Settings, TransmitterSpec, Wall,
                              default_materials, example_restaurant, mw_to_dbm, restaurant_floor_plan,
                              restaurant_px_to_m, slab_loss_db)

STATIC = os.path.join(os.path.dirname(__file__), "..", "src", "open_rps", "gui", "static", "js")
F = 2.4e9


def transfer_matrix_loss_db(eps_r, sigma, f, d):
    """Independent check: characteristic matrix of a single layer between two half-spaces of air."""
    n = np.sqrt(complex(eps_r, -sigma / (2 * math.pi * f * 8.8541878128e-12)))
    delta = 2 * math.pi * f / 299792458.0 * n * d
    m = np.array([[np.cos(delta), 1j * np.sin(delta) / n], [1j * n * np.sin(delta), np.cos(delta)]])
    t = 2.0 / (m[0, 0] + m[0, 1] + m[1, 0] + m[1, 1])
    return -20 * math.log10(abs(t))


def materials():
    return {m.id: m for m in default_materials()}


class SlabTests(unittest.TestCase):
    def test_against_transfer_matrix(self):
        mats = materials()
        for mid, d in [("concreto", 0.15), ("concreto", 0.4), ("vidrio", 0.003), ("ladrillo", 0.25),
                       ("tablaroca", 0.0125), ("madera", 0.05)]:
            m = mats[mid]
            for f in (900e6, 2.4e9, 5.5e9):
                self.assertAlmostEqual(slab_loss_db(m.eps_r, m.conductivity(f), f, d),
                                       transfer_matrix_loss_db(m.eps_r, m.conductivity(f), f, d), places=9, msg=(mid, d, f))

    def test_limits(self):
        self.assertEqual(slab_loss_db(5.24, 0.09, F, 0.0), 0.0)
        self.assertLess(slab_loss_db(5.24, 0.09, F, 1e-6), 1e-3)                  # thin: transparent
        lam = 299792458.0 / F
        self.assertAlmostEqual(slab_loss_db(4.0, 0.0, F, lam / 4.0), 0.0, places=9)      # half wave inside (n = 2)
        # quarter wave inside, lossless: |T|^2 = 4 n^2 / (1 + n^2)^2
        self.assertAlmostEqual(slab_loss_db(4.0, 0.0, F, lam / 8.0), 10 * math.log10(25 / 16), places=9)
        # thick and lossy: attenuation inside plus the two interfaces
        eta = complex(5.24, -0.09 / (2 * math.pi * F * 8.8541878128e-12))
        n = np.sqrt(eta)
        alpha = -2 * math.pi * F / 299792458.0 * n.imag
        r = (1 - n) / (1 + n)
        expected = 8.685889638 * alpha * 2.0 - 20 * math.log10(abs(1 - r * r))
        self.assertAlmostEqual(slab_loss_db(5.24, 0.09, F, 2.0), expected, places=4)

    def test_values_of_the_practice(self):
        mats = materials()
        f = 2.412e9
        concrete = slab_loss_db(mats["concreto"].eps_r, mats["concreto"].conductivity(f), f, 0.15)
        glass = slab_loss_db(mats["vidrio"].eps_r, mats["vidrio"].conductivity(f), f, 0.003)
        self.assertAlmostEqual(concrete, 11.43, delta=0.02)
        self.assertAlmostEqual(glass, 0.67, delta=0.02)
        self.assertGreater(slab_loss_db(mats["concreto"].eps_r, mats["concreto"].conductivity(f), f, 0.30), 2 * concrete - 3)


class WallThicknessTests(unittest.TestCase):
    def test_loss_of_a_wall(self):
        mats = materials()
        wall = Wall("w", "", [0, 0], [10, 0], 0, 3, "concreto", 0.15)
        wall.validate(mats)
        f = 2.4e9
        slab = slab_loss_db(mats["concreto"].eps_r, mats["concreto"].conductivity(f), f, 0.15)
        self.assertAlmostEqual(wall.penetration_loss_db(mats["concreto"], f), slab)
        self.assertAlmostEqual(wall.reflector(mats["concreto"], f).transmission_loss_db, slab)
        self.assertTrue(all(abs(e.shadow_loss_db - slab) < 1e-12 for e in wall.edges(mats["concreto"], f)))
        # without thickness: the law of the material, as before
        plain = Wall("w", "", [0, 0], [10, 0], 0, 3, "concreto")
        self.assertAlmostEqual(plain.penetration_loss_db(mats["concreto"], f), mats["concreto"].penetration_loss_db(f))
        # materials that do not obstruct or are opaque ignore the thickness
        self.assertIsNone(Wall("w", "", [0, 0], [1, 0], 0, 3, "ideal", 0.2).penetration_loss_db(mats["ideal"], f))
        self.assertTrue(math.isinf(Wall("w", "", [0, 0], [1, 0], 0, 3, "metal", 0.2).penetration_loss_db(mats["metal"], f)))

    def test_validation(self):
        mats = materials()
        for bad in (0.0, -0.1, 6.0, "x", float("nan")):
            with self.assertRaises(ProjectError, msg=bad):
                Wall("w", "", [0, 0], [10, 0], 0, 3, "concreto", bad).validate(mats)

    def test_direct_ray_pays_the_slab(self):
        project = Project(area=Area(0, 20, 0, 10, 1.0, 1.5),
                          settings=Settings(mode="raytracing", diffraction=False, sinr=False),
                          transmitters=[TransmitterSpec(0, "Tx", 2.0, 5.0, 1.5, 20.0, 2.4e9, 0.0)],
                          walls=[Wall("w1", "Muro", [10.0, -50.0], [10.0, 60.0], 0.0, 30.0, "concreto", 0.15)])
        project.validate()
        probe = project.probe((15.0, 5.0)).to_dict()
        direct = [r for r in probe["transmitters"][0]["rays"] if r["kind"] == "direct"][0]
        m = project.material_map()["concreto"]
        self.assertAlmostEqual(direct["obstruction_loss_db"], slab_loss_db(m.eps_r, m.conductivity(2.4e9), 2.4e9, 0.15))

    def test_file_format(self):
        project = example_restaurant()
        data = project.to_dict()
        self.assertEqual(data["version"], 4)
        again = Project.from_dict(json.loads(json.dumps(data)))
        self.assertEqual([w.thickness_m for w in again.walls], [w.thickness_m for w in project.walls])
        # without thickness the key is not written and the version does not change
        for w in project.walls:
            w.thickness_m = None
        data = project.to_dict()
        self.assertTrue(all("thickness_m" not in w for w in data["walls"]))
        self.assertEqual(data["version"], 3)

    def test_calibration_leaves_walls_with_thickness(self):
        project = Project(area=Area(0, 40, 0, 20, 2.0, 1.5), settings=Settings(mode="raytracing", diffraction=False),
                          transmitters=[TransmitterSpec(0, "Tx", 5.0, 10.0, 2.0, 20.0, 2.4e9, 0.0)],
                          walls=[Wall("w1", "Muro", [15.0, -10.0], [15.0, 30.0], 0.0, 3.0, "concreto", 0.15)])
        project.validate()
        pts = [[x, y, 1.5] for x in (20.0, 25.0, 30.0, 35.0) for y in (4.0, 10.0, 16.0)]
        power = predict_points(project, pts)[0]
        project.measurements = [Measurement(p[0], p[1], float(v) - 2.0, p[2]) for p, v in zip(pts, power)]
        result = calibrate(project)
        concrete = [p for p in result.parameters if p.kind == "material"][0]
        self.assertEqual(concrete.points, 0)
        self.assertEqual(concrete.after, concrete.before)


class MultiwallTests(unittest.TestCase):
    def project(self, mode="multiwall"):
        p = Project(area=Area(0, 30, 0, 20, 1.0, 1.0), settings=Settings(mode=mode, log_distance_exponent=3.0, sinr=False),
                    transmitters=[TransmitterSpec(0, "AP", 5.0, 10.0, 2.5, 15.0, 2.412e9, 2.0)],
                    walls=[Wall("w1", "Concreto", [12.0, -5.0], [12.0, 25.0], 0.0, 3.0, "concreto", 0.15),
                           Wall("w2", "Vidrio", [20.0, -5.0], [20.0, 25.0], 0.0, 3.0, "vidrio", 0.003)])
        return p.validate()

    def test_map_is_log_distance_plus_walls(self):
        mw = self.project().run()
        ld = self.project("logdistance").run()
        m = self.project().material_map()
        concrete = slab_loss_db(m["concreto"].eps_r, m["concreto"].conductivity(2.412e9), 2.412e9, 0.15)
        glass = slab_loss_db(m["vidrio"].eps_r, m["vidrio"].conductivity(2.412e9), 2.412e9, 0.003)
        x = mw.x_coords
        diff = ld.rx_power_dbm[0] - mw.rx_power_dbm[0]
        np.testing.assert_allclose(diff[:, x < 12], 0.0, atol=1e-9)
        np.testing.assert_allclose(diff[:, (x > 12) & (x < 20)], concrete, atol=1e-9)
        np.testing.assert_allclose(diff[:, x > 20], concrete + glass, atol=1e-9)

    def test_probe_and_points_match_the_map(self):
        project = self.project()
        result = project.run()
        for i, j in [(3, 4), (15, 10), (25, 18), (29, 0)]:
            x, y = result.x_coords[i], result.y_coords[j]
            probe = project.probe((x, y)).to_dict()["transmitters"][0]
            self.assertAlmostEqual(probe["rx_power_dbm"], float(result.rx_power_dbm[0, j, i]), places=9)
            self.assertEqual(len(probe["walls"]), int(x > 12) + int(x > 20))
            self.assertAlmostEqual(float(predict_points(project, [[x, y, 1.0]])[0, 0]), float(result.rx_power_dbm[0, j, i]),
                                   places=9)

    def test_opaque_wall(self):
        project = self.project()
        project.walls[0].material = "metal"
        result = project.run()
        self.assertTrue(np.all(np.isneginf(result.rx_power_dbm[0][:, result.x_coords > 12])))

    def test_calibration_fits_materials_and_exponent(self):
        truth = self.project()
        truth.walls[0].thickness_m = None
        truth.material_map()["concreto"].penetration = "fixed"
        truth.material_map()["concreto"].loss_db = 18.0
        truth.settings.log_distance_exponent = 2.7
        pts = [[x, y, 1.0] for x in np.arange(2.0, 30.0, 2.0) for y in (3.0, 8.0, 13.0, 18.0)]
        power = predict_points(truth, pts)[0]
        model = self.project()
        model.walls[0].thickness_m = None
        model.material_map()["concreto"].penetration = "fixed"
        model.measurements = [Measurement(p[0], p[1], float(v), p[2], tx_id=0) for p, v in zip(pts, power)]
        result = calibrate(model, materials=["concreto"], tx_power="none")
        fitted = {p.kind: p.after for p in result.parameters}
        self.assertAlmostEqual(fitted["material"], 18.0, delta=0.01)
        self.assertAlmostEqual(fitted["exponent"], 2.7, delta=0.001)


class PresetTests(unittest.TestCase):
    def test_mw(self):
        self.assertAlmostEqual(mw_to_dbm(1.0), 0.0)
        self.assertAlmostEqual(mw_to_dbm(100.0), 20.0)
        self.assertAlmostEqual(mw_to_dbm(9.0), 9.5424250944, places=9)

    def test_presets_are_consistent(self):
        for pid, preset in ANTENNA_PRESETS.items():
            tx = TransmitterSpec(0, "t", **{k: v for k, v in preset.items() if k != "label"})
            tx.validate()
            if preset["pattern"] == "sector":
                # gain ~ 0.7 * 41253 / (az * el), within 0.5 dB
                expected = 10 * math.log10(0.7 * 41253 / (preset["beamwidth_az_deg"] * preset["beamwidth_el_deg"]))
                self.assertAlmostEqual(preset["antenna_gain_dbi"], expected, delta=0.5, msg=pid)

    def test_same_presets_in_the_interface(self):
        with open(os.path.join(STATIC, "physics.js"), encoding="utf-8") as f:
            text = f.read()
        for pid, preset in ANTENNA_PRESETS.items():
            line = [l for l in text.splitlines() if l.strip().startswith(f"{pid}:")]
            self.assertEqual(len(line), 1, pid)
            self.assertIn(f'"{preset["label"]}"', line[0])
            for key in ("antenna_gain_dbi", "beamwidth_az_deg", "beamwidth_el_deg"):
                if key in preset:
                    self.assertIn(f"{key}: {preset[key]:.1f}", line[0], (pid, key))

    @unittest.skipUnless(shutil.which("node"), "node no está instalado")
    def test_same_slab_in_the_interface(self):
        mats = materials()
        cases = [("concreto", 0.15, 2.412e9), ("vidrio", 0.003, 2.412e9), ("ladrillo", 0.25, 5.5e9), ("tablaroca", 0.0125, 900e6)]
        args = [[mats[m].eps_r, mats[m].conductivity(f), f, d] for m, d, f in cases]
        url = "file://" + os.path.abspath(os.path.join(STATIC, "physics.js"))
        script = (f"import('{url}').then(m => console.log(JSON.stringify("
                  f"{json.dumps(args)}.map(a => m.slabLossDb(...a)))))")
        out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        for js, a in zip(json.loads(out.stdout), args):
            self.assertAlmostEqual(js, slab_loss_db(*a), places=9)


class RestaurantTests(unittest.TestCase):
    def test_example(self):
        p = example_restaurant()
        self.assertEqual(len(p.walls), 33)
        self.assertEqual(p.settings.mode, "multiwall")
        glass = [w for w in p.walls if w.material == "vidrio"]
        self.assertEqual(len(glass), 7)                                # 2 windows + 5 sides of the terrace
        self.assertTrue(all(w.thickness_m == 0.003 for w in glass))
        self.assertTrue(all(w.thickness_m == 0.15 for w in p.walls if w.material == "concreto"))
        expected = [(9.0, "omni_2_2", 0.0, 1), (30.0, "yagi_13", 0.0, 6), (40.0, "sector_6", 180.0, 11)]
        for tx, (mw, preset, azimuth, channel) in zip(p.transmitters, expected):
            self.assertAlmostEqual(10 ** (tx.tx_power_dbm / 10), mw, places=9)
            self.assertEqual(tx.pattern, ANTENNA_PRESETS[preset]["pattern"])
            self.assertEqual(tx.antenna_gain_dbi, ANTENNA_PRESETS[preset]["antenna_gain_dbi"])
            self.assertEqual((tx.azimuth_deg, tx.wifi_band, tx.wifi_channel, tx.wifi_width_mhz), (azimuth, "2.4", channel, 20))
        # the calibration of the practice: 100 m between the top corners of the medium salon
        a, b = restaurant_px_to_m(313, 255), restaurant_px_to_m(512, 255)
        self.assertEqual(a, [130.0, 115.0])
        self.assertAlmostEqual(math.dist(a, b), 100.0, places=3)

    def test_floor_plan_matches_the_walls(self):
        plan = restaurant_floor_plan("data:image/jpeg;base64,AAAA")
        scale = plan.width_m / plan.pixel_width
        for u, v in [(313, 255), (60, 390), (512, 151)]:
            x = plan.x_m + (u + 0.5) * scale
            y = plan.y_m + (plan.pixel_height - (v + 0.5)) * scale
            np.testing.assert_allclose([x, y], restaurant_px_to_m(u, v), atol=1e-3)

    def test_file_in_examples_is_up_to_date(self):
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "restaurante_polanco.orps.json")
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), example_restaurant().to_dict())

    def test_coverage(self):
        p = example_restaurant()
        p.area.resolution_m = 2.0
        r = p.run()
        best = np.nanmax(r.rx_power_dbm, axis=0)
        x, y = np.meshgrid(r.x_coords, r.y_coords)
        # each AP serves its own salon
        (x0, y1), (x1, y0) = restaurant_px_to_m(119, 151), restaurant_px_to_m(313, 289)
        large = (x > x0) & (x < x1) & (y > y0) & (y < y1)
        self.assertGreater(np.mean(best[large] >= -75), 0.95)
        self.assertEqual(np.bincount(r.best_server_indices()[large].ravel()).argmax(), 0)
        self.assertAlmostEqual(r.summary()["wifi"]["mean_contention"], 0.0)      # channels 1, 6 and 11


if __name__ == "__main__":
    unittest.main()
