"""Tests of OpenRPS 1.5.0: UTD diffraction by edges (``DiffractionEdge``) and double reflections."""
import math
import unittest
import warnings

import numpy as np

from open_rps import PlanarReflector, PropagationSimulator, RayTracer, Transmitter
from open_rps.diffraction import fresnel_integrals, fresnel_tail, knife_edge_field, utd_coefficient, utd_transition
from open_rps.project import Area, Building, Material, Project, TransmitterSpec, Wall, example_office, example_urban
from open_rps.raytracer import DiffractionEdge

INF = float("inf")
C_LIGHT = 299_792_458.0
LAMBDA = C_LIGHT / 2.4e9
RESULT_GRIDS = ("rx_power_dbm", "path_loss_db", "best_server_id", "best_server_index", "max_rx_power_dbm",
                "rms_delay_spread_ns", "angular_spread_bs_phi")

# Reference values computed with mpmath (40 digits)
FRESNEL = [(0.3, 0.2994009760520472, 0.014116998006576583), (1.0, 0.7798934003768229, 0.43825914739035476),
           (1.4999, 0.44535357300621886, 0.6975432066559988), (1.5, 0.4452611760398215, 0.6975049600820931),
           (2.0, 0.48825340607534073, 0.34341567836369824), (3.7, 0.5419456621544874, 0.5749803498874729),
           (10.0, 0.49989869420551575, 0.46816997858488224)]
TRANSITION = [(1e-06, complex(0.0012533128853340696, 0.0012513153906290114)),
              (0.05, complex(0.26922506265477175, 0.1939726181680934)),
              (0.7, complex(0.7439503623137119, 0.2548566234879513)),
              (1.7671458676442586, complex(0.8945674916394606, 0.18291444500888365)),
              (3.0, complex(0.9472422587410706, 0.13257826183062646)),
              (25.0, complex(0.9988161809409175, 0.019882866355392576)),
              (10000.0, complex(0.9999999925000007, 4.9999998125000294e-05))]


def tx_at(x, y, z, tx_id=0, **kwargs):
    kwargs = {"tx_power_dbm": 30.0, "frequency_hz": 2.4e9, "antenna_gain_dbi": 0.0, **kwargs}
    return Transmitter(tx_id=tx_id, x=x, y=y, z=z, **kwargs)


def free_space_amplitude(tx, rx):
    return math.sqrt(tx.power_mw) * LAMBDA / (4 * math.pi * math.dist((tx.x, tx.y, tx.z), rx))


def screen_tracer(polarization, height=1000.0):
    """Wide opaque screen on the plane x = 0 with its top edge at z = height (a knife edge)."""
    tracer = RayTracer()
    tracer.add_reflector(PlanarReflector(p0=[0, 0, 0], normal=[1, 0, 0], bounds=(0, 0, -1e4, 1e4, 0, height),
                                         reflectivity=1.0, transmission_loss_db=INF))
    tracer.add_edge(DiffractionEdge(start=[0, -1e4, height], end=[0, 1e4, height], face_direction=[0, 0, -1],
                                    face_normal=[-1, 0, 0], wedge_n=2.0, polarization=polarization))
    return tracer


def square_building_tracer(loss=INF, eps_r=5.24, sigma=0.09):
    """Square building 0..40 x 0..40 x 30 m and the diffracting vertical edge of its corner (40, 40)."""
    tracer = RayTracer()
    square = [(0, 0), (40, 0), (40, 40), (0, 40)]
    for i in range(4):
        (x0, y0), (x1, y1) = square[i], square[(i + 1) % 4]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        tracer.add_reflector(PlanarReflector(p0=[x0, y0, 0], normal=[dy / length, -dx / length, 0],
                                             bounds=(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), 0, 30),
                                             eps_r=eps_r, sigma=sigma, transmission_loss_db=loss))
    tracer.add_edge(DiffractionEdge(start=[40, 40, 0], end=[40, 40, 30], face_direction=[-1, 0, 0], face_normal=[0, 1, 0],
                                    wedge_n=1.5, polarization="soft", eps_r=eps_r, sigma=sigma,
                                    shadow_loss_db=INF if loss == INF else 2 * loss))
    return tracer


def field_of(cir, kinds):
    return sum((r.magnitude for r in cir.rays if r.kind in kinds), 0j)


class SpecialFunctionTests(unittest.TestCase):
    def test_fresnel_integrals(self):
        for t, c, s in FRESNEL:
            got_c, got_s = fresnel_integrals(t)
            self.assertAlmostEqual(got_c, c, delta=1e-13)
            self.assertAlmostEqual(got_s, s, delta=1e-13)
            got_c, got_s = fresnel_integrals(-t)                       # odd functions
            self.assertAlmostEqual(got_c, -c, delta=1e-13)
        c, s = fresnel_integrals(np.array([[0.3, 2.0], [10.0, 0.0]]))
        self.assertEqual(c.shape, (2, 2))
        self.assertEqual(c[1, 1], 0.0)

    def test_tail_has_no_cancellation(self):
        for t, c, s in FRESNEL:
            self.assertAlmostEqual(abs(fresnel_tail(t) - complex(0.5 - c, -(0.5 - s))), 0.0, delta=1e-13)
        self.assertAlmostEqual(fresnel_tail(0.0), 0.5 - 0.5j, delta=1e-15)
        # For large t the tail ~ exp(-j pi t^2 / 2) / (j pi t): its modulus is exact even though 1/2 - C is tiny
        self.assertAlmostEqual(abs(fresnel_tail(300.0)) * math.pi * 300.0, 1.0, delta=1e-9)
        with self.assertRaises(ValueError):
            fresnel_tail(-1.0)

    def test_transition_function(self):
        for x, value in TRANSITION:
            self.assertAlmostEqual(abs(utd_transition(x) - value) / abs(value), 0.0, delta=1e-12)
        self.assertEqual(utd_transition(0.0), 0.0)
        small = 1e-9
        self.assertAlmostEqual(abs(utd_transition(small) / (math.sqrt(math.pi * small) * np.exp(0.25j * math.pi)) - 1), 0, delta=1e-4)
        self.assertEqual(utd_transition(np.array([[1.0, 2.0, 3.0]])).shape, (1, 3))
        with self.assertRaises(ValueError):
            utd_transition(-0.1)

    def test_knife_edge(self):
        self.assertAlmostEqual(abs(knife_edge_field(0.0)), 0.5, delta=1e-15)          # -6.02 dB on the shadow boundary
        self.assertAlmostEqual(-20 * math.log10(abs(knife_edge_field(1.0))), 13.86, delta=0.01)
        self.assertAlmostEqual(abs(knife_edge_field(-30.0)), 1.0, delta=0.02)          # far in the lit region
        for nu in (5.0, 10.0, 20.0):                                                    # ITU-R P.526: J ~ 13 + 20 log10(nu)
            self.assertAlmostEqual(-20 * math.log10(abs(knife_edge_field(nu))), 13 + 20 * math.log10(nu), delta=0.1)

    def test_coefficient_singularities_are_finite(self):
        # Exactly on the incident and reflection shadow boundaries the poles of the cotangents cancel
        k = 2 * math.pi / LAMBDA
        for phi_p, phi in ((0.5, 0.5 + math.pi), (0.5, math.pi - 0.5), (1.0, 2 * 1.5 * math.pi - math.pi - 1.0)):
            d = utd_coefficient(phi, phi_p, 1.5, k, 30.0, 1.0)
            self.assertTrue(np.isfinite(d))
            near = utd_coefficient(phi + 1e-4, phi_p, 1.5, k, 30.0, 1.0)
            far = utd_coefficient(phi - 1e-4, phi_p, 1.5, k, 30.0, 1.0)
            self.assertLess(min(abs(d - near), abs(d - far)), 0.05 * abs(d) + 1e-6)


class KnifeEdgeTests(unittest.TestCase):
    """A thin opaque screen: UTD (direct + diffracted ray) against the exact Fresnel-Kirchhoff knife edge."""

    def test_both_polarizations_match_itu_r_p526(self):
        height = 1000.0
        tx = tx_at(-500.0, 0.0, height)
        for polarization in ("soft", "hard"):
            tracer = screen_tracer(polarization, height)
            worst = 0.0
            for z_rx in np.linspace(height + 12.0, height - 25.0, 25):
                rx = (500.0, 0.0, float(z_rx))
                cir = tracer.trace_paths(tx, rx)
                utd_db = 20 * math.log10(abs(field_of(cir, ("direct", "diffraction"))) / free_space_amplitude(tx, rx))
                clearance = height - (tx.z + (z_rx - tx.z) / 2)
                d = math.hypot(500.0, (z_rx - tx.z) / 2)
                nu = clearance * math.sqrt(2 / LAMBDA * (2 / d))
                worst = max(worst, abs(utd_db - 20 * math.log10(abs(knife_edge_field(nu)))))
            self.assertLess(worst, 0.35, polarization)

    def test_direct_ray_is_blocked_only_in_the_shadow(self):
        tracer = screen_tracer("soft")
        tx = tx_at(-500.0, 0.0, 1000.0)
        lit = tracer.trace_paths(tx, (500.0, 0.0, 1005.0))
        shadow = tracer.trace_paths(tx, (500.0, 0.0, 995.0))
        self.assertGreater(lit.rays[0].power_linear, 0.0)
        self.assertEqual(shadow.rays[0].power_linear, 0.0)
        diffracted = [r for r in shadow.rays if r.kind == "diffraction"][0]
        self.assertAlmostEqual(diffracted.path[1][2], 1000.0, places=9)                # on the edge
        self.assertAlmostEqual(diffracted.path[1][0], 0.0, places=9)
        self.assertEqual(diffracted.interactions, (("edge", 0),))


class ShadowBoundaryTests(unittest.TestCase):
    """The defining property of the UTD: the total field is continuous across the shadow boundary."""

    def sweep(self, tracer, kinds=("direct", "diffraction")):
        tx = tx_at(100.0, 20.0, 10.0)
        xs = np.linspace(-90.0, -70.0, 201)        # crosses the boundary of the corner (40, 40) at x = -80, y = 80
        out = []
        for x in xs:
            rx = (float(x), 80.0, 10.0)
            cir = tracer.trace_paths(tx, rx)
            out.append(abs(field_of(cir, kinds)) / free_space_amplitude(tx, rx))
        with np.errstate(divide="ignore"):
            return xs, 20 * np.log10(np.array(out))

    def test_opaque_corner(self):
        xs, total = self.sweep(square_building_tracer())
        self.assertLess(float(np.max(np.abs(np.diff(total)))), 0.3)
        self.assertAlmostEqual(float(total[np.argmin(np.abs(xs + 80))]), -6.0, delta=0.8)
        self.assertLess(float(total[0]), -10.0)                    # deep shadow
        self.assertGreater(float(total[-1]), -1.5)                 # lit region: close to free space
        _, direct = self.sweep(square_building_tracer(), kinds=("direct",))
        self.assertTrue(np.isneginf(direct[0]))                    # without diffraction the shadow is empty

    def test_semi_transparent_corner(self):
        xs, total = self.sweep(square_building_tracer(loss=10.0))
        self.assertLess(float(np.max(np.abs(np.diff(total)))), 0.3)
        _, direct = self.sweep(square_building_tracer(loss=10.0), kinds=("direct",))
        self.assertAlmostEqual(float(np.max(np.abs(np.diff(direct)))), 20.0, delta=0.5)   # 2 walls x 10 dB
        self.assertTrue(np.isfinite(direct).all())

    def test_reciprocity_of_a_perfectly_conducting_wedge(self):
        tracer = RayTracer()
        tracer.add_edge(DiffractionEdge(start=[40, 40, 0], end=[40, 40, 30], face_direction=[-1, 0, 0],
                                        face_normal=[0, 1, 0], wedge_n=1.5, polarization="soft"))
        a, b = (100.0, 20.0, 10.0), (-60.0, 75.0, 4.0)
        d_ab = [r for r in tracer.trace_paths(tx_at(*a), b).rays if r.kind == "diffraction"][0].magnitude
        d_ba = [r for r in tracer.trace_paths(tx_at(*b), a).rays if r.kind == "diffraction"][0].magnitude
        self.assertAlmostEqual(abs(d_ab - d_ba) / abs(d_ab), 0.0, delta=1e-12)

    def test_no_ray_from_inside_the_wedge_or_off_the_edge(self):
        tracer = square_building_tracer()
        tx = tx_at(100.0, 20.0, 10.0)
        inside = tracer.trace_paths(tx, (20.0, 20.0, 1.5))                   # inside the building
        self.assertNotIn("diffraction", [r.kind for r in inside.rays])
        above = tracer.trace_paths(tx_at(100.0, 20.0, 80.0), (-80.0, 80.0, 70.0))   # Keller point above the edge
        self.assertNotIn("diffraction", [r.kind for r in above.rays])


class DoubleReflectionTests(unittest.TestCase):
    def corridor(self):
        tracer = RayTracer(max_reflections=2)
        tracer.add_reflector(PlanarReflector(p0=[0, 0, 0], normal=[0, 1, 0], reflectivity=0.7))
        tracer.add_reflector(PlanarReflector(p0=[0, 10, 0], normal=[0, 1, 0], reflectivity=0.7))
        return tracer

    def test_corridor_geometry(self):
        tracer = self.corridor()
        tx, rx = tx_at(0.0, 2.0, 1.5), (20.0, 3.0, 1.5)
        rays = {r.interactions: r for r in tracer.trace_paths(tx, rx).rays if r.kind == "double"}
        for pair in ((("wall", 0), ("wall", 1)), (("wall", 1), ("wall", 0)), (("ground", -1), ("wall", 1)),
                     (("wall", 0), ("ground", -1))):
            self.assertIn(pair, rays)
        # wall 0 (y = 0) then wall 1 (y = 10): image of Tx at y = -2, then at y = 22
        ray = rays[(("wall", 0), ("wall", 1))]
        self.assertAlmostEqual(ray.tau_ns * C_LIGHT * 1e-9, math.dist((0, 22, 1.5), rx), places=9)
        p1, p2 = np.array(ray.path[1]), np.array(ray.path[2])
        self.assertAlmostEqual(p1[1], 0.0, places=12)
        self.assertAlmostEqual(p2[1], 10.0, places=12)
        for a, b, c in ((ray.path[0], p1, p2), (p1, p2, ray.path[3])):          # specular law at both bounces
            a, b, c = np.array(a), np.array(b), np.array(c)
            inc, out = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
            n = np.array([0, 1, 0])
            np.testing.assert_allclose(out, inc - 2 * np.dot(inc, n) * n, atol=1e-12)
        expected = math.sqrt(tx.power_mw) * LAMBDA / (4 * math.pi * math.dist((0, 22, 1.5), rx)) * 0.7 * 0.7
        self.assertAlmostEqual(abs(ray.magnitude), expected, delta=1e-12 * expected)

    def test_wall_then_ground(self):
        tracer = self.corridor()
        tx, rx = tx_at(0.0, 2.0, 1.5), (20.0, 3.0, 1.5)
        ray = [r for r in tracer.trace_paths(tx, rx).rays if r.interactions == (("wall", 0), ("ground", -1))][0]
        self.assertAlmostEqual(ray.path[1][1], 0.0, places=12)
        self.assertAlmostEqual(ray.path[2][2], 0.0, places=12)
        self.assertAlmostEqual(ray.tau_ns * C_LIGHT * 1e-9, math.dist((0, -2, -1.5), rx), places=9)

    def test_finite_walls_and_obstruction(self):
        tracer = RayTracer(max_reflections=2)
        tracer.add_reflector(PlanarReflector(p0=[0, 0, 0], normal=[0, 1, 0], reflectivity=0.7, bounds=(0, 6, 0, 0, 0, 5)))
        tracer.add_reflector(PlanarReflector(p0=[0, 10, 0], normal=[0, 1, 0], reflectivity=0.7))
        tx, rx = tx_at(0.0, 2.0, 1.5), (20.0, 3.0, 1.5)
        kinds = [r.interactions for r in tracer.trace_paths(tx, rx).rays]
        self.assertIn((("wall", 0), ("wall", 1)), kinds)          # its first bounce, x = 2.1, is on the patch...
        tracer.reflectors[0].bounds = (3, 6, 0, 0, 0, 5)
        kinds = [r.interactions for r in tracer.trace_paths(tx, rx).rays]
        self.assertNotIn((("wall", 0), ("wall", 1)), kinds)       # ...and off it now
        # An opaque wall across the corridor blocks the double reflections that cross it
        tracer.reflectors[0].bounds = None
        tracer.add_reflector(PlanarReflector(p0=[10, 0, 0], normal=[1, 0, 0], bounds=(10, 10, 0, 10, 0, 5),
                                             transmission_loss_db=INF))
        for ray in tracer.trace_paths(tx, rx).rays:
            if ray.kind == "double":
                self.assertEqual(ray.power_linear, 0.0)

    def test_pruned_pairs_never_had_a_path(self):
        tracer = RayTracer(max_reflections=2)
        tracer.add_reflector(PlanarReflector(p0=[0, 0, 0], normal=[0, 1, 0], reflectivity=0.7, bounds=(0, 50, 0, 0, 0, 5)))
        tracer.add_reflector(PlanarReflector(p0=[0, -5, 0], normal=[0, 1, 0], reflectivity=0.7, bounds=(0, 50, -5, -5, 0, 5)))
        tx = tx_at(10.0, 3.0, 2.0)                                  # in front of wall 0; wall 1 is behind it
        pts = np.column_stack([np.linspace(0, 50, 60), np.full(60, 4.0), np.full(60, 1.5)])
        field = tracer.trace_points(tx, pts)
        order = tracer.candidate_rays()
        k = order.index(("double", (("wall", 0), ("wall", 1))))
        self.assertFalse(field.valid[k].any())

    def test_validation(self):
        with self.assertRaises(ValueError):
            RayTracer(max_reflections=3)
        with self.assertRaises(ValueError):
            RayTracer().max_reflections = True


class EdgeValidationTests(unittest.TestCase):
    def test_invalid_edges(self):
        ok = dict(start=[0, 0, 0], end=[0, 0, 10], face_direction=[1, 0, 0], face_normal=[0, 1, 0])
        DiffractionEdge(**ok)
        for bad in (dict(end=[0, 0, 0]), dict(face_direction=[0, 0, 1]), dict(face_normal=[1, 0, 0]),
                    dict(wedge_n=1.0), dict(wedge_n=2.5), dict(polarization="circular"), dict(reflectivity=1.5),
                    dict(eps_r=0.5), dict(shadow_loss_db=-1.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DiffractionEdge(**{**ok, **bad})

    def test_basis_is_orthonormalized(self):
        edge = DiffractionEdge(start=[0, 0, 0], end=[0, 0, 10], face_direction=[2, 0, 5], face_normal=[1, 3, 0])
        np.testing.assert_allclose(edge.face_direction, [1, 0, 0])
        np.testing.assert_allclose(edge.face_normal, [0, 1, 0])
        s, r, phi = edge.cylindrical(np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
        np.testing.assert_allclose(s, [4, 1, 0, 0])
        np.testing.assert_allclose(r, [1, 2, 1, 1])
        np.testing.assert_allclose(phi, [0, math.pi / 2, math.pi, 1.5 * math.pi])
        self.assertEqual(DiffractionEdge(**{**dict(start=[0, 0, 0], end=[0, 0, 1], face_direction=[1, 0, 0],
                                                   face_normal=[0, 1, 0]), "shadow_loss_db": None}).isb_scale, 0.0)


class EngineConsistencyTests(unittest.TestCase):
    def scene(self):
        project = Project(
            transmitters=[TransmitterSpec(0, x=-15.0, y=30.0, z=12.0, frequency_hz=2.4e9),
                          TransmitterSpec(3, x=70.0, y=-5.0, z=6.0, pattern="sector", azimuth_deg=150.0)],
            buildings=[Building("b1", "L", [[0, 0], [30, 0], [30, 12], [12, 12], [12, 30], [0, 30]], 10.0, "concreto"),
                       Building("b2", "oblicuo", [[40, 15], [55, 28], [46, 44], [36, 38]], 16.0, "ladrillo")],
            walls=[Wall("w1", "barda", [5.0, 45.0], [35.0, 50.0], 0.0, 3.0, "vidrio")],
        )
        project.settings.max_reflections = 2
        return project.build_simulator()

    def test_candidate_rays_follow_the_axis_of_trace_points(self):
        sim = self.scene()
        tracer = sim.ray_tracer
        order = tracer.candidate_rays()
        n = len(tracer.reflectors)
        self.assertEqual(len(order), 2 + n + len(tracer.edges) + (n + 1) * n)
        field = tracer.trace_points(sim.transmitters[0], np.array([[20.0, 40.0, 1.5]]))
        self.assertEqual(field.amplitude.shape[0], len(order))
        cir = tracer.trace_paths(sim.transmitters[0], (20.0, 40.0, 1.5))
        existing = [order[k] for k in np.flatnonzero(field.valid[:, 0])]
        self.assertEqual(existing, [(r.kind, r.interactions) for r in cir.rays])

    def test_vectorized_equals_pointwise(self):
        sim = self.scene()
        area = ((-10.0, 70.0), (-10.0, 60.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for coherent in (False, True):
                fast = sim.run_raytracing_simulation(*area, resolution_m=5.0, coherent=coherent)
                slow = sim.run_raytracing_simulation(*area, resolution_m=5.0, coherent=coherent, vectorized=False)
                np.testing.assert_array_equal(fast.best_server_id, slow.best_server_id)
                for name in RESULT_GRIDS:
                    x, y = np.asarray(getattr(fast, name), float), np.asarray(getattr(slow, name), float)
                    np.testing.assert_array_equal(np.isnan(x), np.isnan(y), err_msg=name)
                    np.testing.assert_array_equal(np.isinf(x), np.isinf(y), err_msg=name)
                    finite = np.isfinite(x)
                    self.assertLess(float(np.max(np.abs(x[finite] - y[finite]))), 1e-9, name)

    def test_small_chunks_give_the_same_grid(self):
        from open_rps import simulation
        sim = self.scene()
        area = ((-10.0, 70.0), (-10.0, 60.0))
        whole = sim.run_raytracing_simulation(*area, resolution_m=4.0)
        saved = simulation._MAX_RAY_POINTS
        try:
            simulation._MAX_RAY_POINTS = 5000
            chunked = sim.run_raytracing_simulation(*area, resolution_m=4.0)
        finally:
            simulation._MAX_RAY_POINTS = saved
        np.testing.assert_array_equal(whole.best_server_id, chunked.best_server_id)
        for name in RESULT_GRIDS:
            # numpy may round a transcendental function one ulp differently on arrays of other lengths
            np.testing.assert_allclose(getattr(whole, name), getattr(chunked, name), rtol=1e-12, atol=0,
                                       err_msg=name)

    def test_diffraction_fills_the_shadow_of_an_opaque_building(self):
        # A low opaque building and a station above its roof: the edge of the roof lights up the shadow
        project = Project(transmitters=[TransmitterSpec(0, x=-30.0, y=20.0, z=30.0)],
                          buildings=[Building("b", "Bodega", [[0, 0], [20, 0], [20, 40], [0, 40]], 10.0, "metal")],
                          area=Area(-40.0, 80.0, -20.0, 60.0, 4.0, 1.5))
        project.settings.diffraction = False
        without = project.run()
        project.settings.diffraction = True
        with_utd = project.run()
        x, y = np.meshgrid(without.x_coords, without.y_coords)
        behind = np.isneginf(without.max_rx_power_dbm) & (x > 20.5)
        self.assertGreater(int(behind.sum()), 20)                                      # 1.4: an empty shadow
        self.assertTrue(np.isfinite(with_utd.max_rx_power_dbm[behind]).all())          # 1.5: diffracted rays
        inside = (x > 0.5) & (x < 19.5) & (y > 0.5) & (y < 39.5)
        self.assertTrue(np.isneginf(with_utd.max_rx_power_dbm[inside]).all())         # nothing gets in
        lit = x < -35
        np.testing.assert_allclose(with_utd.max_rx_power_dbm[lit], without.max_rx_power_dbm[lit], atol=1.0)


class ProjectEdgeTests(unittest.TestCase):
    def test_building_edges(self):
        building = Building("b", "L", [[0, 0], [30, 0], [30, 12], [12, 12], [12, 30], [0, 30]], 10.0, "concreto")
        material = Material("m", penetration="fixed", loss_db=7.0)
        edges = building.edges(material, 2.4e9)
        corners = [e for e in edges if "esquina" in e.name]
        roofs = [e for e in edges if "techo" in e.name]
        self.assertEqual(len(corners), 5)                         # the reflex corner (12, 12) does not diffract
        self.assertNotIn("L · esquina 4", [e.name for e in corners])
        self.assertEqual(len(roofs), 6)
        for e in corners:
            self.assertAlmostEqual(e.wedge_n, 1.5)
            self.assertEqual(e.polarization, "soft")
            self.assertEqual(e.shadow_loss_db, 14.0)                 # two walls of 7 dB
        for e in roofs:
            self.assertEqual(e.polarization, "hard")
            np.testing.assert_allclose(e.face_normal, [0, 0, 1])
        # A clockwise footprint gives the same edges (same corners, same wedges)
        reversed_building = Building("r", "L", building.footprint[::-1], 10.0, "concreto")
        again = reversed_building.edges(material, 2.4e9)
        key = lambda e: (tuple(np.round(e.start, 9)), tuple(np.round(e.end, 9)), round(e.wedge_n, 9))
        self.assertEqual(sorted(map(key, again)), sorted(map(key, edges)))
        no_roof = Building("n", "N", building.footprint, 10.0, "concreto", roof=False)
        self.assertEqual(len(no_roof.edges(material, 2.4e9)), 5)

    def test_wall_edges_and_joints(self):
        project = example_office()
        sim = project.build_simulator()
        names = [e.name for e in sim.ray_tracer.edges]
        self.assertEqual(sum("borde superior" in n for n in names), len(project.walls))
        self.assertIn("Pasillo sur · extremo (final)", names)        # free end at the door (17, 8)
        self.assertNotIn("Pasillo sur · extremo (inicio)", names)     # it starts on the west facade
        self.assertNotIn("División sur 1 · extremo (inicio)", names)
        project.settings.diffraction = False
        self.assertEqual(project.build_simulator().ray_tracer.edges, [])

    def test_settings_and_report(self):
        project = example_urban()
        project.area.resolution_m = 40.0
        result, report = project.run_with_report()
        self.assertEqual(report["edges"], 62)
        self.assertEqual(report["max_reflections"], 1)
        data = project.to_dict()
        data["settings"]["max_reflections"] = 3
        with self.assertRaises(ValueError):
            Project.from_dict(data)
        project.settings.max_reflections = 2
        result2, report2 = project.run_with_report()
        self.assertEqual(report2["candidate_rays"], 2 + 38 + 62 + 39 * 38)
        # More rays can only add power (incoherent sum)
        both = np.isfinite(result.max_rx_power_dbm) & np.isfinite(result2.max_rx_power_dbm)
        self.assertTrue(np.all(result2.max_rx_power_dbm[both] >= result.max_rx_power_dbm[both] - 1e-9))

    def test_probe_labels(self):
        project = example_urban()
        project.settings.max_reflections = 2
        probe = project.probe((300.0, 190.0)).to_dict()
        labels = [r["label"] for t in probe["transmitters"] for r in t["rays"]]
        self.assertTrue(any(label.startswith("Difracción: ") for label in labels))
        self.assertTrue(any(label.startswith("Doble reflexión: ") for label in labels))
        diffracted = [r for t in probe["transmitters"] for r in t["rays"] if r["kind"] == "diffraction"]
        self.assertTrue(all(len(r["path"]) == 3 for r in diffracted))


if __name__ == "__main__":
    unittest.main()
