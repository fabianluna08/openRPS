"""Tests of OpenRPS 1.6.0: double diffraction and rays that combine a reflection with a diffraction."""
import math
import unittest
import unittest.mock
import warnings

import numpy as np

from open_rps import PlanarReflector, RayTracer, Transmitter
from open_rps.diffraction import knife_edge_field
from open_rps.project import Area, Building, Material, Project, TransmitterSpec, Wall, example_urban
from open_rps.raytracer import DiffractionEdge

INF = float("inf")
LAMBDA = 299_792_458.0 / 2.4e9
K = 2 * math.pi / LAMBDA
Z0 = 1000.0                      # geometry lifted far above the ground
RESULT_GRIDS = ("rx_power_dbm", "path_loss_db", "best_server_id", "max_rx_power_dbm", "rms_delay_spread_ns",
                "angular_spread_bs_phi")


def tx_at(x, y, z, tx_id=0):
    return Transmitter(tx_id=tx_id, x=x, y=y, z=z, tx_power_dbm=30.0, frequency_hz=2.4e9, antenna_gain_dbi=0.0)


def free_space(tx, rx):
    return math.sqrt(tx.power_mw) * LAMBDA / (4 * math.pi * math.dist((tx.x, tx.y, tx.z), rx))


def two_screens(h1, h2, a, b, polarization):
    """Two wide opaque screens (x = a and x = a + b) with diffracting top edges at heights Z0 + h1 and Z0 + h2."""
    tracer = RayTracer(max_diffractions=2)
    for x, h in ((a, h1), (a + b, h2)):
        tracer.add_reflector(PlanarReflector(p0=[x, 0, 0], normal=[1, 0, 0], bounds=(x, x, -1e4, 1e4, 0, Z0 + h),
                                             reflectivity=1.0, transmission_loss_db=INF))
        tracer.add_edge(DiffractionEdge(start=[x, -1e4, Z0 + h], end=[x, 1e4, Z0 + h], face_direction=[0, 0, -1],
                                        face_normal=[-1, 0, 0], wedge_n=2.0, polarization=polarization))
    return tracer


def fresnel_kirchhoff_two_edges(zs, h1, h2, zr, a, b, c):
    """
    Exact paraxial Fresnel-Kirchhoff reference for two knife edges: the field on the
    plane of the second screen is the knife-edge field of the first one (closed form),
    then the integral over the opening of the second screen is computed numerically.
    Relative to free space.
    """
    d1 = a + b
    z2 = np.arange(h2, h2 + 900.0, 0.002)
    nu1 = (h1 - (zs + (z2 - zs) * a / d1)) * np.sqrt(2 / LAMBDA * (1 / a + 1 / b))
    field2 = np.exp(-1j * K * (z2 - zs) ** 2 / (2 * d1)) * knife_edge_field(nu1)
    kernel = np.exp(-1j * K * (zr - z2) ** 2 / (2 * c))
    taper = np.ones_like(z2)
    tail = len(z2) // 4
    taper[-tail:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, tail)))
    numerator = np.trapezoid(field2 * kernel * taper, z2)
    denominator = np.sqrt(2 * np.pi / (1j * K * (1 / d1 + 1 / c))) * np.exp(-1j * K * (zr - zs) ** 2 / (2 * (d1 + c)))
    return abs(numerator / denominator)


def by_kind(cir, kinds):
    return sum((r.magnitude for r in cir.rays if r.kind in kinds), 0j)


class TwoKnifeEdgeTests(unittest.TestCase):
    """Only the doubly diffracted ray reaches the receiver: compare it with the exact two-edge field."""

    GEOMETRIES = ((15.0, 10.0, 0.0), (12.0, 12.0, -4.0), (10.0, 18.0, -2.0), (20.0, 8.0, 3.0))

    def test_cascaded_utd_matches_fresnel_kirchhoff(self):
        a = b = c = 200.0
        tx = tx_at(0.0, 0.0, Z0)
        for h1, h2, zr in self.GEOMETRIES:
            reference = 20 * math.log10(fresnel_kirchhoff_two_edges(0.0, h1, h2, zr, a, b, c))
            rx = (a + b + c, 0.0, Z0 + zr)
            levels = []
            for polarization in ("soft", "hard"):
                cir = two_screens(h1, h2, a, b, polarization).trace_paths(tx, rx)
                self.assertEqual(sum(1 for r in cir.rays if r.kind == "double-diffraction"), 1)
                field = by_kind(cir, ("direct", "diffraction", "double-diffraction"))
                levels.append(20 * math.log10(abs(field) / free_space(tx, rx)))
                self.assertLess(abs(levels[-1] - reference), 1.0, (h1, h2, zr, polarization))
            # Soft and hard fall on both sides of the Kirchhoff (polarization-free) result
            self.assertLess(abs(sum(levels) / 2 - reference), 0.3, (h1, h2, zr))

    def test_without_double_diffraction_nothing_arrives(self):
        tracer = two_screens(15.0, 10.0, 200.0, 200.0, "hard")
        tracer.max_diffractions = 1
        cir = tracer.trace_paths(tx_at(0.0, 0.0, Z0), (600.0, 0.0, Z0))
        self.assertEqual(abs(by_kind(cir, ("direct", "diffraction"))), 0.0)

    def test_path_goes_over_both_edges(self):
        cir = two_screens(15.0, 10.0, 200.0, 200.0, "hard").trace_paths(tx_at(0.0, 0.0, Z0), (600.0, 0.0, Z0))
        ray = [r for r in cir.rays if r.kind == "double-diffraction"][0]
        self.assertEqual(ray.interactions, (("edge", 0), ("edge", 1)))
        q1, q2 = ray.path[1], ray.path[2]
        self.assertAlmostEqual(q1[0], 200.0, places=9)
        self.assertAlmostEqual(q1[2], Z0 + 15.0, places=9)
        self.assertAlmostEqual(q2[2], Z0 + 10.0, places=9)
        length = sum(math.dist(p, q) for p, q in zip(ray.path, ray.path[1:]))
        self.assertAlmostEqual(length / 299_792_458.0 * 1e9, ray.tau_ns, places=9)


class DoubleKellerTests(unittest.TestCase):
    def test_newton_on_skew_edges_satisfies_keller_at_both_edges(self):
        e1 = DiffractionEdge(start=[0, 0, 0], end=[0, 0, 30], face_direction=[1, 0, 0], face_normal=[0, -1, 0],
                             wedge_n=1.5)
        e2 = DiffractionEdge(start=[10, -20, 12], end=[40, 25, 12], face_direction=[0, 0, -1], face_normal=[1, 0, 0])
        tx = np.array([-15.0, -10.0, 4.0])
        rx = np.array([[60.0, 10.0, 1.5], [45.0, 30.0, 6.0]])
        _, r_t, _ = e1.cylindrical(tx)
        s_r, r_r, _ = e2.cylindrical(rx)
        t1, t2, ok = RayTracer._newton_double_keller(e1, e2, tx, rx, float(r_t), s_r, r_r)
        self.assertTrue(ok.all())
        for k in range(len(rx)):
            q1, q2 = e1.start + t1[k] * e1.axis, e2.start + t2[k] * e2.axis
            for edge, before, at, after in ((e1, tx, q1, q2), (e2, q1, q2, rx[k])):
                cos_in = np.dot((at - before) / np.linalg.norm(at - before), edge.axis)
                cos_out = np.dot((after - at) / np.linalg.norm(after - at), edge.axis)
                self.assertAlmostEqual(cos_in, cos_out, places=9)       # equal angles with the edge

    def test_closed_form_for_parallel_edges_equals_newton(self):
        e1 = DiffractionEdge(start=[0, 0, 0], end=[0, 0, 30], face_direction=[1, 0, 0], face_normal=[0, -1, 0],
                             wedge_n=1.5)
        e2 = DiffractionEdge(start=[20, 5, 30], end=[20, 5, 0], face_direction=[0, 1, 0], face_normal=[1, 0, 0],
                             wedge_n=1.5)
        tx = np.array([-15.0, -10.0, 4.0])
        rx = np.array([[40.0, 20.0, 1.5], [30.0, 25.0, 9.0], [50.0, 5.0, 2.0]])
        _, r_t, _ = e1.cylindrical(tx)
        s_r, r_r, _ = e2.cylindrical(rx)
        closed = RayTracer._parallel_double_keller(e1, e2, tx, rx)
        newton = RayTracer._newton_double_keller(e1, e2, tx, rx, float(r_t), s_r, r_r)
        np.testing.assert_allclose(closed[0], newton[0], atol=1e-7)
        np.testing.assert_allclose(closed[1], newton[1], atol=1e-7)


class ReciprocityTests(unittest.TestCase):
    """With perfectly conducting edges the new rays are reciprocal; reflection-diffraction and
    diffraction-reflection are computed by different code, so swapping Tx and Rx checks both."""

    def corner_scene(self):
        tracer = RayTracer(mixed_interactions=True, max_diffractions=2)
        tracer.add_reflector(PlanarReflector(p0=[0, 30, 0], normal=[0, 1, 0], bounds=(-100, 100, 30, 30, 0, 20),
                                             eps_r=5.0, sigma=0.05, transmission_loss_db=INF))
        tracer.add_reflector(PlanarReflector(p0=[20, 0, 0], normal=[1, 0, 0], bounds=(20, 20, -50, 10, 0, 15),
                                             reflectivity=1.0, transmission_loss_db=INF))
        tracer.add_edge(DiffractionEdge(start=[20, 10, 0], end=[20, 10, 15], face_direction=[0, -1, 0],
                                        face_normal=[1, 0, 0], wedge_n=2.0, polarization="soft"))
        tracer.add_edge(DiffractionEdge(start=[20, 10, 15], end=[20, -50, 15], face_direction=[0, 0, -1],
                                        face_normal=[1, 0, 0], wedge_n=2.0, polarization="hard"))
        return tracer

    def assert_reciprocal(self, tracer, a, b, kinds):
        forward = tracer.trace_paths(tx_at(*a), b).rays
        backward = {r.interactions: r for r in tracer.trace_paths(tx_at(*b, tx_id=1), a).rays}
        # The same rays exist in both directions (the shadow-zone filter is symmetric)
        self.assertEqual(sorted(tuple(reversed(r.interactions)) for r in forward if r.kind in kinds),
                         sorted(i for i, r in backward.items() if r.kind in kinds))
        checked = 0
        for ray in forward:
            if ray.kind not in kinds:
                continue
            other = backward[tuple(reversed(ray.interactions))]

            self.assertAlmostEqual(abs(ray.magnitude - other.magnitude) / abs(ray.magnitude), 0.0, delta=1e-12)
            self.assertAlmostEqual(ray.tau_ns, other.tau_ns, places=9)
            checked += 1
        return checked

    def test_reflection_and_diffraction(self):
        checked = self.assert_reciprocal(self.corner_scene(), (0.0, 5.0, 8.0), (45.0, -5.0, 2.0),
                                         ("reflection-diffraction", "diffraction-reflection"))
        self.assertGreaterEqual(checked, 3)

    def test_double_diffraction(self):
        tracer = two_screens(15.0, 10.0, 200.0, 200.0, "hard")
        self.assertEqual(self.assert_reciprocal(tracer, (0.0, 0.0, Z0), (600.0, 3.0, Z0 - 2.0),
                                                ("double-diffraction",)), 1)
        # Skew edges (Newton): Tx at (-15, 5) puts part of the second edge in the shadow of the first
        tracer = RayTracer(max_diffractions=2)
        tracer.add_edge(DiffractionEdge(start=[0, 0, 0], end=[0, 0, 30], face_direction=[1, 0, 0],
                                        face_normal=[0, -1, 0], wedge_n=1.5, polarization="soft"))
        tracer.add_edge(DiffractionEdge(start=[10, -20, 12], end=[40, 25, 12], face_direction=[0, 0, -1],
                                        face_normal=[1, 0, 0], polarization="hard"))
        self.assertTrue(tracer._active_higher_order(tx_at(-15.0, 5.0, 4.0)))
        found = 0
        for rx in ((40.0, -15.0, 1.5), (50.0, -5.0, 3.0), (30.0, -25.0, 1.0), (60.0, 0.0, 2.0)):
            found += self.assert_reciprocal(tracer, (-15.0, 5.0, 4.0), rx, ("double-diffraction",))
        self.assertGreaterEqual(found, 1)

    def test_reflection_law_and_keller_law(self):
        cir = self.corner_scene().trace_paths(tx_at(0.0, 5.0, 8.0), (45.0, -5.0, 2.0))
        for ray in cir.rays:
            if ray.kind not in ("reflection-diffraction", "diffraction-reflection"):
                continue
            points = [np.array(p) for p in ray.path]
            for k, (kind, index) in enumerate(ray.interactions, start=1):
                before, at, after = points[k - 1], points[k], points[k + 1]
                inc, out = (at - before) / np.linalg.norm(at - before), (after - at) / np.linalg.norm(after - at)
                if kind == "edge":
                    axis = np.array([0, 0, 1.0]) if index == 0 else np.array([0, -1.0, 0])
                    self.assertAlmostEqual(np.dot(inc, axis), np.dot(out, axis), places=9)
                else:
                    n = np.array([0, 0, 1.0]) if kind == "ground" else np.array([0, 1.0, 0])
                    np.testing.assert_allclose(out, inc - 2 * np.dot(inc, n) * n, atol=1e-9)


class ShadowBehindTallBuildingTests(unittest.TestCase):
    """The case that motivated 1.6.0: a tall, wide building that no single edge can see past."""

    def project(self):
        return Project(transmitters=[TransmitterSpec(0, x=-30.0, y=20.0, z=8.0)],
                       buildings=[Building("b", "Torre", [[0, 0], [20, 0], [20, 40], [0, 40]], 40.0, "metal")],
                       area=Area(-40.0, 80.0, -20.0, 60.0, 4.0, 1.5))

    def test_double_diffraction_fills_it(self):
        project = self.project()
        single = project.run()
        project.settings.max_diffractions = 2
        double = project.run()
        x, y = np.meshgrid(single.x_coords, single.y_coords)
        behind = np.isneginf(single.max_rx_power_dbm) & (x > 20.5)
        self.assertGreater(int(behind.sum()), 100)
        self.assertTrue(np.isfinite(double.max_rx_power_dbm[behind]).all())
        inside = (x > 0.5) & (x < 19.5) & (y > 0.5) & (y < 39.5)
        self.assertTrue(np.isneginf(double.max_rx_power_dbm[inside]).all())
        self.assertLess(float(np.max(double.max_rx_power_dbm[behind])), float(np.median(double.max_rx_power_dbm[x < -35])))

    def test_no_round_off_rays_along_a_facade(self):
        # Around two corners of the same facade the leg between them grazes it: for soft polarization the
        # UTD coefficient is exactly 0 there, so those rays are left out (not kept as ~-700 dBm residues)
        project = self.project()
        project.settings.max_diffractions = 2
        sim = project.build_simulator()
        rays = [r for r in sim.ray_tracer.trace_paths(sim.transmitters[0], (45.0, 20.0, 1.5)).rays
                if r.kind == "double-diffraction" and r.power_linear > 0]
        self.assertTrue(rays)
        for ray in rays:
            self.assertGreater(10 * math.log10(ray.power_linear), -200.0, ray.interactions)
        corners = [e for e, edge in enumerate(sim.ray_tracer.edges) if edge.polarization == "soft"]
        self.assertFalse([r for r in rays if all(i[1] in corners for i in r.interactions)])

    def test_lossy_roof_is_not_cancelled(self):
        opaque_concrete = Material("hormigon", "Concreto opaco", eps_r=5.24, sigma_c=0.0462, sigma_d=0.7822,
                                   penetration="opaque")
        project = self.project()
        project.materials.append(opaque_concrete)
        project.buildings[0].material = "hormigon"
        project.settings.max_diffractions = 2
        probe = project.probe((40.0, 20.0)).to_dict()
        rays = [r for r in probe["transmitters"][0]["rays"] if r["kind"] == "double-diffraction" and not r["blocked"]]
        self.assertTrue(any("techo" in r["label"] for r in rays))
        best = max(r["power_dbm"] for r in rays)
        self.assertGreater(best, -150.0)


class EngineTests(unittest.TestCase):
    def scene(self, **options):
        project = Project(
            transmitters=[TransmitterSpec(0, x=-15.0, y=30.0, z=12.0),
                          TransmitterSpec(3, x=70.0, y=-5.0, z=6.0, pattern="sector", azimuth_deg=150.0)],
            buildings=[Building("b1", "L", [[0, 0], [30, 0], [30, 12], [12, 12], [12, 30], [0, 30]], 10.0, "metal"),
                       Building("b2", "oblicuo", [[40, 15], [55, 28], [46, 44], [36, 38]], 16.0, "concreto")],
            walls=[Wall("w1", "barda", [5.0, 45.0], [35.0, 50.0], 0.0, 3.0, "ladrillo")],
        )
        for key, value in options.items():
            setattr(project.settings, key, value)
        return project.build_simulator()

    def test_defaults_add_nothing(self):
        tracer = RayTracer()
        self.assertEqual((tracer.max_diffractions, tracer.mixed_interactions), (1, False))
        sim = self.scene()
        kinds = {kind for kind, _ in sim.ray_tracer.candidate_rays()}
        self.assertEqual(kinds, {"direct", "ground", "wall", "diffraction"})
        with self.assertRaises(ValueError):
            RayTracer(max_diffractions=3)
        with self.assertRaises(ValueError):
            RayTracer(mixed_interactions="yes")

    def test_vectorized_equals_pointwise(self):
        sim = self.scene(max_diffractions=2, mixed_interactions=True, max_reflections=2)
        area = ((-10.0, 70.0), (-10.0, 60.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fast = sim.run_raytracing_simulation(*area, resolution_m=10.0)
            slow = sim.run_raytracing_simulation(*area, resolution_m=10.0, vectorized=False)
        np.testing.assert_array_equal(fast.best_server_id, slow.best_server_id)
        # A receiver exactly on the face of a soft wedge gets D = 0 there, computed as a cancellation that
        # leaves round-off of ~1e-20 (here, a pixel on a wall of the metal building at -380 dBm); that
        # residue depends on how numpy splits the array, so fields below -300 dBm count as zero
        power = np.asarray(fast.rx_power_dbm, float)
        zero = np.isfinite(power) & (power < -300.0)
        self.assertLess(int(zero.sum()), 3)
        for name in RESULT_GRIDS:
            x, y = np.asarray(getattr(fast, name), float), np.asarray(getattr(slow, name), float)
            if x.shape == zero.shape:
                x, y = x[~zero], y[~zero]
            np.testing.assert_array_equal(np.isfinite(x), np.isfinite(y), err_msg=name)
            finite = np.isfinite(x)
            self.assertLess(float(np.max(np.abs(x[finite] - y[finite]))), 1e-9, name)

    def test_compact_mode_gives_the_same_rays(self):
        sim = self.scene(max_diffractions=2, mixed_interactions=True)
        tracer, tx = sim.ray_tracer, sim.transmitters[0]
        xs, ys = np.meshgrid(np.linspace(-10, 70, 17), np.linspace(-10, 60, 15))
        pts = np.stack([xs, ys, np.full(xs.shape, 1.5)], axis=-1)
        full = tracer.trace_points(tx, pts)
        compact = tracer.trace_points(tx, pts, compact=True)
        self.assertEqual(full.labels, tracer.candidate_rays())
        self.assertLess(len(compact.labels), len(full.labels))
        rows = [full.labels.index(label) for label in compact.labels]
        np.testing.assert_array_equal(full.amplitude[rows], compact.amplitude)
        dropped = np.setdiff1d(np.arange(len(full.labels)), rows)
        self.assertFalse(full.valid[dropped].any())                 # nothing that exists was left out
        np.testing.assert_array_equal(full.power_linear.sum(axis=0), compact.power_linear.sum(axis=0))
        cir = tracer.trace_paths(tx, (50.0, 50.0, 1.5))
        field = tracer.trace_points(tx, np.array([[50.0, 50.0, 1.5]]), compact=True)
        self.assertEqual([compact_label for compact_label, ok in zip(field.labels, field.valid[:, 0]) if ok],
                         [(r.kind, r.interactions) for r in cir.rays])

    def test_transition_zone(self):
        zone = RayTracer._transition_zone
        n, kl = 1.5, K * 20.0
        phi_in = 1.0
        # Deep in the lit region, away from every boundary: out; in the shadow and at the boundaries: in
        self.assertFalse(bool(zone(phi_in, 3.5, n, kl)))
        self.assertTrue(bool(zone(phi_in, phi_in + math.pi + 0.5, n, kl)))
        for boundary in (phi_in + math.pi, math.pi - phi_in, (2 * n - 1) * math.pi - phi_in):
            self.assertTrue(bool(zone(phi_in, boundary + 0.05, n, kl)))
        # Symmetric in the two angles (keeps the rays reciprocal)
        rng = np.random.default_rng(3)
        a, b = rng.uniform(0, n * math.pi, 500), rng.uniform(0, n * math.pi, 500)
        np.testing.assert_array_equal(zone(a, b, n, kl), zone(b, a, n, kl))

    def test_angular_range_holds_every_point_of_the_hull(self):
        rng = np.random.default_rng(11)
        two_pi = 2 * math.pi
        checked = 0
        for _ in range(300):
            axis = rng.normal(size=3)
            face = np.cross(axis, rng.normal(size=3))
            start = rng.uniform(-5, 5, 3)
            edge = DiffractionEdge(start=start, end=start + 10 * axis / np.linalg.norm(axis),
                                   face_direction=face, face_normal=np.cross(axis, face), wedge_n=1.5)
            points = rng.uniform(-20, 20, (int(rng.integers(2, 6)), 3))
            intervals = RayTracer._angular_range(edge, points)
            if intervals is None:
                continue
            weights = rng.dirichlet(np.ones(len(points)), 200)
            _, _, phi = edge.cylindrical(weights @ points)
            inside = np.zeros(len(phi), dtype=bool)
            for lo, hi in intervals:
                self.assertTrue(0.0 <= lo <= hi <= two_pi)
                inside |= (phi >= lo) & (phi <= hi)
            self.assertTrue(inside.all())
            checked += 1
        self.assertGreater(checked, 150)

    def test_pruning_changes_nothing(self):
        # The pair and per-receiver tests before Newton's method and the UTD are necessary
        # conditions: turning them off gives the same rays
        sim = self.scene(max_diffractions=2, mixed_interactions=True)
        tracer = sim.ray_tracer
        xs, ys = np.meshgrid(np.linspace(-10, 70, 33), np.linspace(-10, 60, 29))
        pts = np.stack([xs, ys, np.full(xs.shape, 1.5)], axis=-1)
        fast = [tracer.trace_points(tx, pts) for tx in sim.transmitters]
        everything = staticmethod(lambda *a: True if a[2] is None else np.ones(np.shape(a[2]), dtype=bool))
        fits = staticmethod(lambda edge, s_p, *a: np.ones(np.shape(s_p), dtype=bool) if np.ndim(s_p) else True)
        with unittest.mock.patch.object(RayTracer, "_zone_reachable", everything), \
                unittest.mock.patch.object(RayTracer, "_keller_fits", fits), \
                unittest.mock.patch.object(RayTracer, "_dr_pair_possible", lambda *a: True):
            tracer._active_cache = (None, None)
            slow = [tracer.trace_points(tx, pts) for tx in sim.transmitters]
        tracer._active_cache = (None, None)
        for a, b in zip(fast, slow):
            self.assertEqual(a.labels, b.labels)
            np.testing.assert_array_equal(a.valid, b.valid)
            np.testing.assert_allclose(a.amplitude, b.amplitude, rtol=1e-12, atol=0)

    def test_project_settings_and_report(self):
        project = example_urban()
        project.area.resolution_m = 40.0
        project.settings.max_diffractions = 2
        project.settings.mixed_interactions = True
        result, report = project.run_with_report()
        self.assertEqual((report["max_diffractions"], report["mixed_interactions"]), (2, True))
        self.assertEqual(len(report["active_rays_per_tx"]), 2)
        self.assertLess(max(report["active_rays_per_tx"]), report["candidate_rays"])
        project.settings.diffraction = False
        self.assertEqual(project.build_simulator().ray_tracer.max_diffractions, 1)
        self.assertFalse(project.build_simulator().ray_tracer.mixed_interactions)
        data = example_urban().to_dict()
        data["settings"]["max_diffractions"] = 0
        with self.assertRaises(ValueError):
            Project.from_dict(data)

    def test_probe_labels(self):
        project = example_urban()
        project.settings.max_diffractions = 2
        project.settings.mixed_interactions = True
        probe = project.probe((592.0, 110.0)).to_dict()
        labels = {r["label"].split(":")[0] for t in probe["transmitters"] for r in t["rays"]}
        self.assertIn("Doble difracción", labels)
        self.assertTrue({"Reflexión + difracción", "Difracción + reflexión"} & labels)


if __name__ == "__main__":
    unittest.main()
