"""Tests of OpenRPS 1.4.0: ray geometry, polygon roofs (``PlanarReflector.footprint``) and project files."""
import json
import math
import os
import tempfile
import unittest
import warnings

import numpy as np

from open_rps import ChannelImpulseResponse, MultipathRay, PlanarReflector, PropagationSimulator, RayTracer, Transmitter
from open_rps.project import (
    Area,
    Building,
    EXAMPLES,
    Material,
    Project,
    ProjectError,
    Settings,
    TransmitterSpec,
    Wall,
    default_materials,
    example_office,
    example_urban,
    vertical_wall,
)
from open_rps.raytracer import _in_polygon, _in_polygon_scalar

INF = float("inf")
L_SHAPE = [[0.0, 0.0], [40.0, 0.0], [40.0, 15.0], [15.0, 15.0], [15.0, 40.0], [0.0, 40.0]]
RESULT_GRIDS = ("rx_power_dbm", "path_loss_db", "best_server_id", "best_server_index", "max_rx_power_dbm",
                "rms_delay_spread_ns", "angular_spread_bs_phi")


def make_tx(x=0.0, y=0.0, z=10.0, tx_id=0, **kwargs):
    kwargs = {"tx_power_dbm": 30.0, "frequency_hz": 2.4e9, "antenna_gain_dbi": 6.0, **kwargs}
    return Transmitter(tx_id=tx_id, x=x, y=y, z=z, **kwargs)


def roof(polygon, z, loss=INF, **kwargs):
    return PlanarReflector(p0=[polygon[0][0], polygon[0][1], z], normal=[0.0, 0.0, 1.0], footprint=polygon,
                           transmission_loss_db=loss, **kwargs)


def assert_same_result(test, a, b, tol=1e-9):
    """Same best server, same non-finite pixels and finite values within ``tol`` (as in test_v1_3)."""
    np.testing.assert_array_equal(a.best_server_id, b.best_server_id)
    for name in RESULT_GRIDS:
        x, y = np.asarray(getattr(a, name), dtype=float), np.asarray(getattr(b, name), dtype=float)
        np.testing.assert_array_equal(np.isnan(x), np.isnan(y), err_msg=name)
        np.testing.assert_array_equal(np.isinf(x), np.isinf(y), err_msg=name)
        finite = np.isfinite(x)
        if finite.any():
            test.assertLess(float(np.max(np.abs(x[finite] - y[finite]))), tol, name)


class RayGeometryTests(unittest.TestCase):
    def setUp(self):
        self.tx = make_tx(10.0, 20.0, 15.0)
        self.tracer = RayTracer()
        self.tracer.add_reflector(PlanarReflector(p0=[50.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0], name="Muro"))
        self.rx = (30.0, 45.0, 1.5)
        self.cir = self.tracer.trace_paths(self.tx, self.rx)

    def test_kinds_and_paths(self):
        kinds = [r.kind for r in self.cir.rays]
        self.assertEqual(kinds, ["direct", "ground", "wall"])
        tx_p, rx_p = (10.0, 20.0, 15.0), self.rx
        for ray in self.cir.rays:
            self.assertEqual(ray.path[0], tx_p)
            self.assertEqual(ray.path[-1], rx_p)
        self.assertEqual(len(self.cir.rays[0].path), 2)
        self.assertEqual(self.cir.rays[0].reflector_index, -1)
        self.assertEqual(self.cir.rays[1].reflector_index, -1)
        self.assertEqual(self.cir.rays[2].reflector_index, 0)

    def test_bounce_points_and_path_length(self):
        ground, wall = self.cir.rays[1], self.cir.rays[2]
        self.assertAlmostEqual(ground.path[1][2], 0.0, places=12)
        self.assertAlmostEqual(wall.path[1][0], 50.0, places=12)
        for ray in self.cir.rays:
            length = sum(math.dist(p, q) for p, q in zip(ray.path, ray.path[1:]))
            self.assertAlmostEqual(length / 299_792_458.0 * 1e9, ray.tau_ns, places=9)

    def test_specular_reflection_law(self):
        # At the wall the angle of incidence equals the angle of reflection
        tx, hit, rx = (np.array(p) for p in self.cir.rays[2].path)
        n = np.array([1.0, 0.0, 0.0])
        incoming, outgoing = (hit - tx) / np.linalg.norm(hit - tx), (rx - hit) / np.linalg.norm(rx - hit)
        np.testing.assert_allclose(outgoing, incoming - 2 * np.dot(incoming, n) * n, atol=1e-12)

    def test_geometry_does_not_change_equality_or_repr(self):
        ray = self.cir.rays[0]
        bare = MultipathRay(ray.tx_id, ray.magnitude, ray.tau_ns, ray.tx_phi_deg, ray.tx_theta_deg,
                            ray.rx_phi_deg, ray.rx_theta_deg, ray.obstruction_loss_db)
        self.assertEqual(ray, bare)
        self.assertEqual(repr(ray), repr(bare))
        self.assertNotIn("path=", repr(ray))
        self.assertNotIn("kind=", repr(ray))
        self.assertEqual(bare.kind, "")
        self.assertIsNone(bare.path)

    def test_blocked_rays_keep_their_geometry(self):
        self.tracer.add_reflector(PlanarReflector(p0=[20.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0],
                                                  bounds=(20, 20, 0, 100, 0, 100), transmission_loss_db=INF))
        cir = self.tracer.trace_paths(self.tx, self.rx)
        direct = cir.rays[0]
        self.assertEqual(direct.power_linear, 0.0)
        self.assertEqual(direct.path, ((10.0, 20.0, 15.0), self.rx))


class ReceiverOnAWallPlaneTests(unittest.TestCase):
    """Fix of 1.4.0: a point on the plane of a wall (within 1 um) gets no reflection from it, in both tracers."""

    def test_oblique_wall_through_grid_points(self):
        tracer = RayTracer()
        # Plane x + 0.2 y = 230: it goes exactly through many points of a 5 m grid
        tracer.add_reflector(PlanarReflector(p0=[200.0, 150.0, 0.0], normal=[1.0, 0.2, 0.0], reflectivity=0.6))
        tx = make_tx(100.0, 100.0, 25.0)
        xs, ys = np.arange(0.0, 400.01, 5.0), np.arange(0.0, 300.01, 5.0)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx, gy, np.full(gx.shape, 1.5)], axis=-1)
        on_plane = np.abs(gx + 0.2 * gy - 230.0) < 1e-9
        self.assertGreater(int(on_plane.sum()), 10)
        field = tracer.trace_points(tx, pts)
        self.assertFalse(field.valid[2][on_plane].any())
        self.assertTrue(field.valid[2][(gx + 0.2 * gy < 229.0)].all())    # Tx side: reflection everywhere
        for j, i in np.argwhere(on_plane):
            cir = tracer.trace_paths(tx, tuple(pts[j, i]))
            self.assertEqual([r.kind for r in cir.rays], ["direct", "ground"])

    def test_transmitter_on_the_plane_has_no_reflection(self):
        tracer = RayTracer()
        tracer.add_reflector(PlanarReflector(p0=[0.0, 0.0, 0.0], normal=[3.0, 4.0, 0.0]))
        tx = make_tx(4.0, -3.0, 10.0)                                     # on the plane 3x + 4y = 0
        field = tracer.trace_points(tx, np.array([[10.0, 10.0, 1.5], [-10.0, -10.0, 1.5]]))
        self.assertFalse(field.valid[2].any())
        self.assertNotIn("wall", [r.kind for r in tracer.trace_paths(tx, (10.0, 10.0, 1.5)).rays])


class PolygonTests(unittest.TestCase):
    def test_vectorized_and_scalar_point_in_polygon_agree(self):
        rng = np.random.default_rng(4)
        polygon = np.array(L_SHAPE)
        # Random points, plus vertices, edge midpoints and points just off the edges
        pts = list(rng.uniform(-5, 45, size=(2000, 2)))
        for i in range(len(L_SHAPE)):
            a, b = polygon[i], polygon[(i + 1) % len(L_SHAPE)]
            pts += [a, (a + b) / 2, (a + b) / 2 + 1e-7, (a + b) / 2 - 1e-7]
        pts = np.array(pts)
        vec = _in_polygon(pts[:, 0], pts[:, 1], polygon)
        ref = np.array([_in_polygon_scalar(float(x), float(y), L_SHAPE) for x, y in pts])
        np.testing.assert_array_equal(vec, ref)

    def test_closed_patch_and_notch(self):
        polygon = np.array(L_SHAPE)
        inside = _in_polygon(np.array([5.0, 30.0, 30.0, 40.0, 15.0, 27.5]), np.array([5.0, 5.0, 30.0, 7.0, 27.0, 15.0]),
                             polygon)
        # interior, interior, notch (outside), on an edge, on the inner edge, on the inner edge
        np.testing.assert_array_equal(inside, [True, True, False, True, True, True])


class FootprintTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):          # vertical plane
            PlanarReflector(p0=[0, 0, 0], normal=[1, 0, 0], footprint=L_SHAPE)
        with self.assertRaises(ValueError):          # too few vertices
            roof([[0, 0], [1, 0]], 5.0)
        with self.assertRaises(ValueError):          # zero area
            roof([[0, 0], [1, 1], [2, 2]], 5.0)
        with self.assertRaises(ValueError):          # not finite
            roof([[0, 0], [1, 0], [float("nan"), 1]], 5.0)

    def test_bounds_from_footprint_and_closing_vertex(self):
        r = roof(L_SHAPE + [L_SHAPE[0]], 12.0)
        self.assertEqual(r.bounds, (0.0, 40.0, 0.0, 40.0, 12.0, 12.0))
        self.assertEqual(len(r.footprint), len(L_SHAPE))
        explicit = roof(L_SHAPE, 12.0, bounds=(0, 20, 0, 40, 12, 12))
        self.assertEqual(explicit.bounds, (0.0, 20.0, 0.0, 40.0, 12.0, 12.0))

    def test_contains_and_crosses_respect_the_polygon(self):
        r = roof(L_SHAPE, 12.0)
        np.testing.assert_array_equal(r.contains(np.array([[5, 5, 12], [30, 30, 12], [30, 5, 12]], dtype=float)),
                                      [True, False, True])
        # Vertical segments through the roof: inside the L, in the notch, outside
        tops = np.array([[5, 5, 20], [30, 30, 20], [50, 5, 20]], dtype=float)
        bottoms = tops * [1, 1, 0]
        np.testing.assert_array_equal(r.crosses(tops, bottoms), [True, False, False])

    def test_roof_reflection_only_over_the_polygon(self):
        tracer = RayTracer()
        tracer.add_reflector(roof(L_SHAPE, 10.0, loss=None, reflectivity=0.8))
        tx = make_tx(5.0, 5.0, 30.0)
        # Receiver above the roof: the reflection point is the midpoint (z = 10) when both are at the same height
        over_l = tracer.trace_paths(tx, (5.0, 35.0, 30.0))
        over_notch = tracer.trace_paths(tx, (55.0, 55.0, 30.0))   # hit point (30, 30): in the notch
        self.assertIn("wall", [r.kind for r in over_l.rays])
        self.assertNotIn("wall", [r.kind for r in over_notch.rays])

    def test_opaque_roof_blocks_exactly_the_rays_that_cross_the_polygon(self):
        tracer = RayTracer()
        tracer.add_reflector(roof(L_SHAPE, 10.0))
        tx = make_tx(-30.0, -30.0, 60.0)
        xs = np.arange(-5.0, 45.01, 1.25)
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx, gy, np.full(gx.shape, 1.5)], axis=-1)
        field = tracer.trace_points(tx, pts)
        # Independent reference: point where the direct ray meets the plane z = 10, tested against the L
        t = (60.0 - 10.0) / (60.0 - 1.5)
        cx, cy = -30.0 + t * (pts[..., 0] + 30.0), -30.0 + t * (pts[..., 1] + 30.0)
        in_l = ((cx >= 0) & (cx <= 40) & (cy >= 0) & (cy <= 15)) | ((cx >= 0) & (cx <= 15) & (cy >= 0) & (cy <= 40))
        near_edge = np.zeros(cx.shape, dtype=bool)
        for value, axis in ((0, cx), (40, cx), (15, cx), (0, cy), (15, cy), (40, cy)):
            near_edge |= np.abs(axis - value) < 1e-6
        blocked = np.isinf(field.obstruction_loss_db[0])
        np.testing.assert_array_equal(blocked[~near_edge], in_l[~near_edge])
        self.assertGreater(int(in_l.sum()), 100)
        self.assertGreater(int((~in_l).sum()), 100)
        # The notch is not covered: some pixels in it keep the direct ray, others are shadowed by the arms
        notch = (pts[..., 0] > 15.5) & (pts[..., 1] > 15.5) & (pts[..., 0] < 40) & (pts[..., 1] < 40)
        self.assertTrue((~blocked & notch).any() and (blocked & notch).any())

    def test_vectorized_equals_pointwise_with_polygon_roofs(self):
        sim = PropagationSimulator()
        sim.add_transmitter(make_tx(-20.0, 55.0, 25.0, tx_id=0))
        sim.add_transmitter(make_tx(60.0, -10.0, 8.0, tx_id=3, pattern="sector", azimuth_deg=135.0))
        building = Building("b", "L", L_SHAPE, 12.0, "concreto")
        building.validate({m.id: m for m in default_materials()})
        material = Material("m", reflection="fresnel", penetration="fixed", loss_db=7.0)
        for wall in building.reflectors(material, 2.4e9):
            sim.add_reflector(wall)
        oblique = [[45.0, 20.0], [60.0, 35.0], [50.0, 55.0], [38.0, 48.0]]
        b2 = Building("c", "oblicuo", oblique, 20.0, "concreto")
        b2.validate({m.id: m for m in default_materials()})
        for wall in b2.reflectors(Material("o", penetration="opaque"), 2.4e9):
            sim.add_reflector(wall)
        area = ((-10.0, 70.0), (-10.0, 65.0))
        for coherent in (False, True):
            fast = sim.run_raytracing_simulation(*area, resolution_m=2.5, coherent=coherent)
            slow = sim.run_raytracing_simulation(*area, resolution_m=2.5, coherent=coherent, vectorized=False)
            assert_same_result(self, fast, slow)

    def test_ray_field_matches_cir_with_roofs(self):
        tracer = RayTracer()
        tracer.add_reflector(roof(L_SHAPE, 10.0, loss=6.0))
        for wall in Building("b", "L", L_SHAPE, 10.0).reflectors(Material("m", penetration="fixed", loss_db=4.0),
                                                                  2.4e9):
            tracer.add_reflector(wall)
        tx = make_tx(-15.0, 20.0, 30.0)
        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(-10, 50, 300), rng.uniform(-10, 50, 300), np.full(300, 1.5)])
        field = tracer.trace_points(tx, pts)
        for k, p in enumerate(pts):
            cir = tracer.trace_paths(tx, tuple(p))
            np.testing.assert_allclose(field.total_power_linear()[k], cir.total_received_power_linear(),
                                       rtol=1e-12, atol=0.0)
            np.testing.assert_array_equal(np.sort(field.obstruction_loss_db[field.valid[:, k], k]),
                                          np.sort([r.obstruction_loss_db for r in cir.rays]))


class MaterialTests(unittest.TestCase):
    def test_itu_p2040_values(self):
        materials = {m.id: m for m in default_materials()}
        # ITU-R P.2040-3, Table 3, evaluated at 9 GHz (values published by MathWorks' implementation)
        for mid, sigma in (("concreto", 0.25766), ("ladrillo", 0.033826), ("tablaroca", 0.066978),
                           ("madera", 0.049528), ("vidrio", 0.068299)):
            self.assertAlmostEqual(materials[mid].conductivity(9e9), sigma, delta=sigma * 2e-3)

    def test_penetration_models(self):
        m = Material("x")
        m.penetration = "none"
        self.assertIsNone(m.penetration_loss_db(2.4e9))
        m.penetration = "opaque"
        self.assertEqual(m.penetration_loss_db(2.4e9), INF)
        m.penetration, m.loss_db = "fixed", 12.0
        self.assertEqual(m.penetration_loss_db(28e9), 12.0)
        m.penetration, m.loss_a_db, m.loss_b_db_per_ghz = "frequency", 5.0, 4.0     # 3GPP concrete
        self.assertAlmostEqual(m.penetration_loss_db(2.4e9), 14.6)
        self.assertAlmostEqual(m.penetration_loss_db(28e9), 117.0)

    def test_reflector_kwargs(self):
        constant = Material("c", reflection="constant", reflectivity=0.3, penetration="none")
        self.assertEqual(constant.reflector_kwargs(1e9), {"transmission_loss_db": None, "reflectivity": 0.3})
        fresnel = Material("f", eps_r=4.0, sigma_c=0.01, sigma_d=1.0, penetration="opaque")
        kwargs = fresnel.reflector_kwargs(2e9)
        self.assertEqual(kwargs["eps_r"], 4.0)
        self.assertAlmostEqual(kwargs["sigma"], 0.02)
        self.assertEqual(kwargs["transmission_loss_db"], INF)


class ProjectTests(unittest.TestCase):
    def test_examples_round_trip_and_run(self):
        for name, factory in EXAMPLES.items():
            project = factory()
            data = project.to_dict()
            json.dumps(data, allow_nan=False)
            again = Project.from_dict(json.loads(json.dumps(data)))
            self.assertEqual(again.to_dict(), project.validate().to_dict(), name)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            office = example_office()
            office.area.resolution_m = 1.0
            result, report = office.run_with_report()
        self.assertEqual(report["warnings"], [])
        self.assertIsNotNone(result.sinr_db)
        self.assertTrue(np.isfinite(result.max_rx_power_dbm).any())

    def test_save_and_load(self):
        project = example_urban()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.orps.json")
            project.save(path)
            self.assertEqual(Project.load(path).to_dict(), project.to_dict())

    def test_building_becomes_faces_and_roof(self):
        project = Project(transmitters=[TransmitterSpec(0, x=-50, y=-50)],
                          buildings=[Building("b", "Casa", L_SHAPE, 9.0, "concreto")])
        sim = project.build_simulator()
        walls = sim.ray_tracer.reflectors
        self.assertEqual(len(walls), len(L_SHAPE) + 1)
        self.assertEqual(walls[-1].name, "Casa · techo")
        np.testing.assert_array_equal(walls[-1].footprint, np.array(L_SHAPE))
        for wall in walls[:-1]:
            self.assertEqual(wall.bounds[4:], (0.0, 9.0))
            self.assertAlmostEqual(abs(wall.normal[2]), 0.0)
        # concrete at 2.4 GHz: 3GPP 5 + 4 * 2.4 dB
        self.assertAlmostEqual(walls[0].transmission_loss_db, 14.6)

    def test_oblique_vertical_wall_is_exact(self):
        wall = vertical_wall((0.0, 0.0), (30.0, 40.0), 0.0, 5.0, "oblicuo", transmission_loss_db=INF)
        # Segments crossing the line of the wall inside and outside the segment (0,0)-(30,40)
        inside = wall.crosses(np.array([10.0, 20.0, 1.0]), np.array([20.0, 20.0, 1.0]))   # crosses at x = 15
        beyond = wall.crosses(np.array([35.0, 50.0, 1.0]), np.array([40.0, 50.0, 1.0]))   # x = 37.5, beyond the end
        above = wall.crosses(np.array([10.0, 20.0, 6.0]), np.array([20.0, 20.0, 6.0]))
        self.assertTrue(bool(inside))
        self.assertFalse(bool(beyond))
        self.assertFalse(bool(above))

    def test_disabled_transmitters_and_reference_frequency(self):
        project = Project(transmitters=[TransmitterSpec(0, frequency_hz=3.5e9, enabled=False),
                                        TransmitterSpec(1, frequency_hz=5e9)])
        self.assertEqual([t.tx_id for t in project.build_simulator().transmitters], [1])
        self.assertEqual(project.reference_frequency_hz(), 5e9)
        project.settings.reference_frequency_hz = 1e9
        self.assertEqual(project.reference_frequency_hz(), 1e9)

    def test_invalid_projects(self):
        base = example_urban().to_dict()
        cases = [
            ("area", {**base["area"], "resolution_m": 0}),
            ("area", {**base["area"], "x_max": -1}),
            ("settings", {**base["settings"], "mode": "magia"}),
            ("buildings", [{**base["buildings"][0], "material": "kryptonita"}]),
            ("buildings", [{**base["buildings"][0], "footprint": [[0, 0], [10, 10], [10, 0], [0, 10]]}]),
            ("transmitters", [base["transmitters"][0], base["transmitters"][0]]),
            ("walls", [{**base["walls"][0], "end": base["walls"][0]["start"]}]),
            ("materials", [{"id": "x", "penetration": "a veces"}]),
            ("area", {**base["area"], "resolutoin_m": 3}),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ProjectError):
                Project.from_dict({**base, key: value})
        with self.assertRaises(ProjectError):
            Project.from_dict({**base, "format": "otra-cosa"})
        with self.assertRaises(ProjectError):
            Project.from_dict({**base, "version": 99})
        with self.assertRaises(ProjectError):
            Project(area=Area(0, 10_000, 0, 10_000, 1.0)).validate()

    def test_probe_matches_the_grid(self):
        project = example_urban()
        project.area.resolution_m = 20.0
        result = project.run()
        sim = project.build_simulator()
        for j, i in ((3, 4), (10, 20), (15, 25)):
            x, y = float(result.x_coords[i]), float(result.y_coords[j])
            probe = project.probe((x, y), sim).to_dict()
            for k, entry in enumerate(probe["transmitters"]):
                expected = result.rx_power_dbm[k, j, i]
                got = entry["rx_power_dbm"]
                if np.isfinite(expected):
                    self.assertAlmostEqual(got, float(expected), places=9)
                else:
                    self.assertEqual(got, "-inf")
                total = sum(10 ** (r["power_dbm"] / 10) for r in entry["rays"] if r["power_dbm"] != "-inf")
                if total > 0:
                    self.assertAlmostEqual(10 * math.log10(total), float(expected), places=9)

    def test_probe_empirical_matches_the_grid(self):
        project = example_urban()
        project.settings.mode = "fspl"
        project.area.resolution_m = 25.0
        result = project.run()
        x, y = float(result.x_coords[5]), float(result.y_coords[7])
        probe = project.probe((x, y)).to_dict()
        for k, entry in enumerate(probe["transmitters"]):
            self.assertAlmostEqual(entry["rx_power_dbm"], float(result.rx_power_dbm[k, 7, 5]), places=9)
            self.assertEqual(entry["rays"], [])

    def test_every_mode_runs(self):
        for mode in ("raytracing", "fspl", "tworay", "logdistance", "hata"):
            project = example_urban()
            project.area.resolution_m = 40.0
            project.settings.mode = mode
            result, report = project.run_with_report()
            self.assertEqual(report["mode"], mode)
            self.assertEqual(result.max_rx_power_dbm.shape, (11, 16))
            if mode == "hata":
                self.assertTrue(report["warnings"])     # 2.4 GHz and < 1 km: outside the range of Hata


if __name__ == "__main__":
    unittest.main()
