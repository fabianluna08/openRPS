"""
Tests of OpenRPS 1.8.0: Wi-Fi channels, spectral overlap (adjacent-channel
interference), MCS and PHY rate, Wi-Fi maps, projects, probe, channel plan and API.
"""

import copy
import json
import math
import os
import sys
import threading
import unittest
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from open_rps.channel_plan import allowed_blocks, plan_channels  # noqa: E402
from open_rps.project import Project, ProjectError, TransmitterSpec, example_office, example_urban  # noqa: E402
from open_rps.simulation import thermal_noise_dbm  # noqa: E402
from open_rps.wifi import (ChannelSpec, WifiError, best_mcs, channel_blocks, channel_center_hz,  # noqa: E402
                           mcs_table, overlap_factor, phy_rate_mbps, required_sinr_db, spectral_mask_db,
                           valid_channels)


def ch(band, channel, width=20):
    return ChannelSpec(channel_center_hz(band, channel, width), width * 1e6)


def numeric_overlap(tx: ChannelSpec, rx: ChannelSpec, step_hz=1e3):
    """Independent check: the mask sampled every kHz and summed (rectangle rule)."""
    f = np.arange(tx.center_hz - 2 * tx.width_hz, tx.center_hz + 2 * tx.width_hz, step_hz) + step_hz / 2
    psd = 10.0 ** (spectral_mask_db(f - tx.center_hz, tx.width_hz) / 10.0)
    inside = (f >= rx.low_hz) & (f <= rx.high_hz)
    return psd[inside].sum() / psd.sum()


def wifi_office(channels=(1, 3), band="2.4", width=20, mode="fspl", positions=((5, 10), (35, 10))):
    p = example_office()
    p.settings.mode = mode
    p.settings.shadow_sigma_db = 0.0
    p.transmitters = [TransmitterSpec(k, f"AP{k + 1}", x, y, 3.0, 20.0, 2.4e9, 3.0, wifi_band=band,
                                      wifi_channel=c, wifi_width_mhz=width)
                      for k, (c, (x, y)) in enumerate(zip(channels, positions))]
    return p.validate()


class ChannelTests(unittest.TestCase):
    def test_centre_frequencies(self):
        cases = [("2.4", 1, 20, 2412), ("2.4", 6, 20, 2437), ("2.4", 13, 20, 2472), ("5", 36, 20, 5180),
                 ("5", 165, 20, 5825), ("5", 40, 40, 5190), ("5", 44, 80, 5210), ("5", 64, 160, 5250),
                 ("5", 149, 80, 5775), ("5", 100, 160, 5570), ("6", 1, 20, 5955), ("6", 5, 80, 5985),
                 ("6", 33, 160, 6185), ("6", 233, 20, 7115)]
        for band, channel, width, mhz in cases:
            with self.subTest(band=band, channel=channel, width=width):
                self.assertAlmostEqual(channel_center_hz(band, channel, width), mhz * 1e6)

    def test_channel_counts(self):
        expected = {("2.4", 20): 13, ("5", 20): 28, ("5", 40): 28, ("5", 80): 28, ("5", 160): 24,
                    ("6", 20): 59, ("6", 40): 58, ("6", 80): 56, ("6", 160): 56}
        for (band, width), count in expected.items():
            self.assertEqual(len(valid_channels(band, width)), count, (band, width))
        self.assertEqual([c for _, c in channel_blocks("6", 160)], [15, 47, 79, 111, 143, 175, 207])

    def test_invalid(self):
        for args, message in ((("5", 38, 20), "no existe"), (("2.4", 1, 40), "solo se admiten canales de 20"),
                              (("7", 1, 20), "desconocida"), (("5", 36, 30), "no válido"), (("6", 233, 40), "no existe")):
            with self.subTest(args=args), self.assertRaisesRegex(WifiError, message):
                channel_center_hz(*args)


class SpectralTests(unittest.TestCase):
    def test_mask_breakpoints(self):
        for width in (20e6, 40e6, 80e6, 160e6):
            w = width / 1e6
            levels = spectral_mask_db(np.array([0, w / 2 - 1, w / 2 + 1, w, 1.5 * w]) * 1e6, width)
            np.testing.assert_allclose(levels, [0, 0, -20, -28, -40])
            self.assertEqual(spectral_mask_db(-0.7 * w * 1e6, width), spectral_mask_db(0.7 * w * 1e6, width))
            self.assertEqual(spectral_mask_db(1.6 * w * 1e6, width), -np.inf)

    def test_overlap_against_numeric_integration(self):
        pairs = [(ch("2.4", 1), ch("2.4", c)) for c in range(1, 8)] + [
            (ch("5", 36, 80), ch("5", 36, 20)), (ch("5", 36, 20), ch("5", 44, 80)), (ch("5", 36, 20), ch("5", 40)),
            (ch("5", 36, 40), ch("5", 44, 20)), (ChannelSpec(2.420e9, 10e6, wifi=False), ch("2.4", 1)),
            (ch("2.4", 1), ChannelSpec(2.427e9, 10e6, wifi=False))]
        for tx, rx in pairs:
            with self.subTest(tx=tx, rx=rx):
                exact = overlap_factor(tx, rx)
                if tx.wifi:
                    self.assertAlmostEqual(exact, numeric_overlap(tx, rx), delta=2e-4 * max(exact, 1e-3))
                else:     # flat 10 MHz around 2420 MHz: [2415, 2425], 7 MHz of it inside channel 1 [2402, 2422]
                    self.assertAlmostEqual(exact, 0.7, places=12)

    def test_overlap_values(self):
        db = lambda a, b: 10 * math.log10(overlap_factor(ch("2.4", a), ch("2.4", b)))
        values = [db(1, c) for c in range(1, 7)]
        self.assertTrue(all(x > y for x, y in zip(values, values[1:])), values)   # decreases with the offset
        self.assertAlmostEqual(values[0], -0.04, delta=0.01)       # co-channel: the mask leaks a little
        self.assertAlmostEqual(values[2], -3.01, delta=0.01)       # 10 MHz apart: half of the channel
        self.assertAlmostEqual(values[5], -29.95, delta=0.05)      # channels 1 and 6
        self.assertEqual(overlap_factor(ch("2.4", 1), ch("2.4", 11)), 0.0)
        self.assertAlmostEqual(overlap_factor(ch("2.4", 1), ch("2.4", 3)), overlap_factor(ch("2.4", 3), ch("2.4", 1)))
        # A wide interferer leaves only the part that falls in a narrow receiver; a narrow one, almost all
        self.assertAlmostEqual(10 * math.log10(overlap_factor(ch("5", 36, 80), ch("5", 36, 20))), -6.11, delta=0.01)
        self.assertGreater(overlap_factor(ch("5", 36, 20), ch("5", 36, 80)), 0.99)


class RateTests(unittest.TestCase):
    def test_published_rates(self):
        # IEEE 802.11 tables (Mbit/s)
        cases = [("n", 7, 20, 1, 0.8, 65.0), ("n", 7, 40, 1, 0.4, 150.0), ("n", 0, 20, 1, 0.8, 6.5),
                 ("n", 7, 40, 4, 0.4, 600.0), ("ac", 9, 80, 1, 0.8, 390.0), ("ac", 9, 80, 1, 0.4, 433.3),
                 ("ac", 9, 160, 2, 0.4, 1733.3), ("ac", 8, 20, 1, 0.8, 78.0), ("ac", 9, 20, 3, 0.8, 260.0),
                 ("ax", 11, 20, 1, 0.8, 143.4), ("ax", 11, 80, 2, 0.8, 1201.0), ("ax", 11, 160, 2, 0.8, 2402.0),
                 ("ax", 0, 20, 1, 3.2, 7.3), ("ax", 7, 40, 1, 1.6, 162.5)]
        for standard, mcs, width, nss, gi, rate in cases:
            with self.subTest(standard=standard, mcs=mcs, width=width, nss=nss, gi=gi):
                self.assertAlmostEqual(phy_rate_mbps(standard, mcs, width, nss, gi), rate, delta=0.06)

    def test_excluded_combinations(self):
        for width, mcs, nss in ((20, 9, 1), (20, 9, 2), (80, 6, 3), (80, 9, 6), (160, 9, 3)):
            self.assertIsNone(phy_rate_mbps("ac", mcs, width, nss), (width, mcs, nss))
        self.assertIsNotNone(phy_rate_mbps("ac", 9, 20, 6))
        self.assertIsNone(phy_rate_mbps("n", 8, 20))
        self.assertEqual([r["mcs"] for r in mcs_table("ac", 20, 1)], list(range(9)))
        for args in (("n", 0, 80), ("ax", 0, 20, 9), ("ax", 0, 20, 1, 0.4), ("n", 0, 20, 1, 1.6), ("x", 0, 20)):
            with self.assertRaises(WifiError):
                phy_rate_mbps(*args)

    def test_thresholds_and_best_mcs(self):
        req = [required_sinr_db(m) for m in range(12)]
        self.assertAlmostEqual(req[0], -82 + 173.975 - 73.0103 - 10, places=3)   # sensitivity - (kTB + 10 dB)
        self.assertTrue(all(a < b for a, b in zip(req, req[1:])))
        sinr = np.array([np.nan, -5.0, req[0], req[3] - 0.01, req[3], 60.0])
        mcs, rate = best_mcs(sinr, "ax", 20, 2)
        np.testing.assert_array_equal(mcs, [np.nan, -1, 0, 2, 3, 11])
        self.assertEqual(rate[1], 0.0)
        self.assertAlmostEqual(rate[-1], phy_rate_mbps("ax", 11, 20, 2))
        # 802.11ac at 20 MHz with one stream skips MCS 9: at its SINR it stays at MCS 8
        mcs, _ = best_mcs(np.array([req[9] + 1]), "ac", 20, 1)
        self.assertEqual(mcs[0], 8)


class EngineTests(unittest.TestCase):
    def test_project_without_wifi_keeps_the_old_sinr(self):
        p = example_office()
        r = p.run()
        self.assertIsNone(r.tx_channels)
        self.assertIsNone(r.wifi_rate_mbps)
        self.assertNotIn("wifi", r.summary())

    def test_sinr_by_hand(self):
        for channels, width, band in (((1, 3), 20, "2.4"), ((1, 1), 20, "2.4"), ((1, 11), 20, "2.4"),
                                      ((36, 36), 80, "5"), ((36, 44), 20, "5")):
            p = wifi_office(channels, band, width)
            r = p.run()
            specs = [ch(band, c, width) for c in channels]
            noise = thermal_noise_dbm(width * 1e6, p.settings.noise_figure_db)
            for j, i in ((10, 10), (40, 40), (20, 150), (45, 88)):
                power = r.rx_power_dbm[:, j, i]
                if not np.all(np.isfinite(power)):
                    continue
                s = int(np.argmax(power))
                o = 1 - s
                f = overlap_factor(specs[o], specs[s])
                total = 10 ** (noise / 10) + f * 10 ** (power[o] / 10)
                with self.subTest(channels=channels, width=width, pixel=(j, i)):
                    self.assertAlmostEqual(r.sinr_db[j, i], power[s] - 10 * math.log10(total), places=9)
                    m, rate = best_mcs(r.sinr_db[j, i], "ax", width, 2)
                    self.assertEqual(r.wifi_mcs[j, i], m)
                    self.assertAlmostEqual(r.wifi_rate_mbps[j, i], rate)
                    shared = (channels[0] == channels[1] or abs(channels[0] - channels[1]) < 4) and band == "2.4" \
                        or channels[0] == channels[1]
                    self.assertEqual(r.wifi_contention[j, i], float(shared and power[o] >= -82.0))

    def test_mixed_with_a_transmitter_without_channel(self):
        p = wifi_office((1, 6))
        p.transmitters[1].wifi_channel = None
        p.transmitters[1].frequency_hz = 2.417e9          # flat 20 MHz around 2417: overlaps channel 1
        p.validate()
        r = p.run()
        j, i = 10, 10
        power = r.rx_power_dbm[:, j, i]
        f = overlap_factor(ChannelSpec(2.417e9, 20e6, wifi=False), ch("2.4", 1))
        self.assertAlmostEqual(f, 15 / 20)
        noise = thermal_noise_dbm(20e6, p.settings.noise_figure_db)
        expected = power[0] - 10 * math.log10(10 ** (noise / 10) + f * 10 ** (power[1] / 10))
        self.assertAlmostEqual(r.sinr_db[j, i], expected, places=9)
        served_by_plain = r.best_server_indices() == 1
        self.assertTrue(np.all(np.isnan(r.wifi_rate_mbps[served_by_plain])))

    def test_probe_matches_the_map(self):
        p = wifi_office((1, 4), mode="raytracing")
        p.area.resolution_m = 4.0
        r = p.run()
        for j, i in ((2, 2), (3, 7), (4, 9)):
            x, y = float(r.x_coords[i]), float(r.y_coords[j])
            w = p.probe((x, y)).to_dict()["wifi"]
            with self.subTest(point=(x, y)):
                self.assertAlmostEqual(w["sinr_db"], r.sinr_db[j, i], places=6)
                self.assertEqual(w["mcs"], r.wifi_mcs[j, i])
                self.assertEqual(w["contention"], r.wifi_contention[j, i])
                self.assertEqual(len(w["interferers"]), 1)
                self.assertAlmostEqual(w["interferers"][0]["overlap_db"],
                                       10 * math.log10(overlap_factor(ch("2.4", [1, 4][1 - [0, 1].index(
                                           p.transmitters.index(next(t for t in p.transmitters
                                                                     if t.tx_id == w["serving_tx_id"])))]),
                                           ch("2.4", [1, 4][[t.tx_id for t in p.transmitters].index(
                                               w["serving_tx_id"])]))), places=9)

    def test_csv_and_summary(self):
        import tempfile
        p = wifi_office((1, 6))
        r = p.run()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.csv")
            r.export_csv(path)
            with open(path) as f:
                header = f.readline().strip().split(",")
            self.assertEqual(header[-3:], ["wifi_mcs", "wifi_rate_mbps", "wifi_contention"])
        w = r.summary()["wifi"]
        self.assertEqual(w["standards"], ["ax", "ax"])
        self.assertGreater(w["mean_rate_mbps"], 0)


class ProjectTests(unittest.TestCase):
    def test_frequency_follows_the_channel_and_file_versions(self):
        p = example_office()
        self.assertEqual(p.to_dict()["version"], 1)
        self.assertNotIn("wifi_channel", p.to_dict()["transmitters"][0])
        self.assertNotIn("wifi_standard", p.to_dict()["settings"])
        tx = p.transmitters[0]
        tx.wifi_band, tx.wifi_channel, tx.wifi_width_mhz = "5", 44, 80
        p.validate()
        self.assertEqual(tx.frequency_hz, 5210e6)
        data = p.to_dict()
        self.assertEqual(data["version"], 3)
        self.assertEqual(data["transmitters"][0]["wifi_channel"], 44)
        again = Project.from_dict(json.loads(json.dumps(data)))
        self.assertEqual(again.to_dict(), data)
        tx.wifi_channel = None
        p.validate()
        self.assertIsNone(tx.wifi_band)
        self.assertEqual(p.to_dict()["version"], 1)
        p.settings.wifi_standard = "ac"
        self.assertEqual(p.to_dict()["version"], 3)          # a non-default Wi-Fi setting is kept

    def test_validation(self):
        cases = [(dict(wifi_band="6", wifi_channel=5), {}, "6 GHz requiere"),
                 (dict(wifi_band="5", wifi_channel=36, wifi_width_mhz=80), {"wifi_standard": "n"}, "no admite canales de 80"),
                 (dict(wifi_band="5", wifi_channel=38), {}, "no existe"),
                 (dict(wifi_band="2.4", wifi_channel=1, wifi_width_mhz=40), {}, "solo se admiten"),
                 (dict(wifi_band="5", wifi_channel=36, wifi_streams=9), {}, "wifi_streams"),
                 (dict(wifi_band="5", wifi_channel=36), {"wifi_guard_interval_us": 0.4}, "0.8 o 1.6 o 3.2"),
                 (dict(wifi_band="5", wifi_channel=36), {"wifi_client_streams": 0}, "wifi_client_streams")]
        for tx_fields, settings, message in cases:
            p = example_office()
            for k, v in tx_fields.items():
                setattr(p.transmitters[0], k, v)
            for k, v in settings.items():
                setattr(p.settings, k, v)
            if "wifi_standard" in settings or "6" == tx_fields.get("wifi_band"):
                p.settings.wifi_standard = settings.get("wifi_standard", "ac")
            with self.subTest(fields=tx_fields, settings=settings), self.assertRaisesRegex(ProjectError, message):
                p.validate()

    def test_ac_on_24ghz_uses_n_rates(self):
        p = wifi_office((1, 11))
        p.settings.wifi_standard = "ac"
        p.settings.wifi_guard_interval_us = 0.4
        p.validate()
        r = p.run()
        self.assertEqual(r.wifi_settings["standards"], ["n", "n"])
        top = np.nanmax(r.wifi_rate_mbps)
        self.assertAlmostEqual(top, phy_rate_mbps("n", 7, 20, 2, 0.4))


class PlanTests(unittest.TestCase):
    def four_on_one(self):
        p = wifi_office((1, 1, 1, 1), positions=((6, 4), (16, 14), (26, 4), (36, 14)), mode="raytracing")
        p.area.resolution_m = 1.0
        return p

    def test_spreads_co_channel_aps(self):
        p = self.four_on_one()
        plan = plan_channels(p)
        after = [a["after"] for a in plan.assignments]
        self.assertTrue(set(after) <= {1, 6, 11})
        self.assertEqual(len(set(after)), 3)
        # Neighbours (AP1-AP2, AP2-AP3, AP3-AP4) end on different channels
        for a, b in zip(after, after[1:]):
            self.assertNotEqual(a, b)
        self.assertGreater(plan.after["mean_capacity_bps_hz"], plan.before["mean_capacity_bps_hz"] + 3)
        self.assertGreater(plan.after["mean_rate_mbps"], 3 * plan.before["mean_rate_mbps"])
        self.assertEqual([a["after"] for a in plan_channels(p).assignments], after)    # deterministic
        # The suggested plan, applied and simulated again, is as good as the search said (same band)
        q = copy.deepcopy(p)
        for t, a in zip(q.transmitters, plan.assignments):
            t.wifi_channel = a["after"]
        q.validate()
        self.assertAlmostEqual(np.nanmedian(q.run().sinr_db), plan.after["median_sinr_db"], delta=0.5)

    def test_allowed_channels(self):
        five = [b[0] for b, _ in allowed_blocks("5", 20)]
        self.assertEqual(five, [36, 40, 44, 48, 149, 153, 157, 161, 165])
        self.assertIn(52, [b[0] for b, _ in allowed_blocks("5", 20, include_dfs=True)])
        self.assertIn(169, [b[0] for b, _ in allowed_blocks("5", 20, include_unii4=True)])
        self.assertEqual(allowed_blocks("5", 160), [])
        self.assertEqual([b[0] for b, _ in allowed_blocks("2.4", 20)], [1, 6, 11])
        p = wifi_office((36, 36), band="5", width=160)
        plan = plan_channels(p)
        self.assertEqual([a["after"] for a in plan.assignments], [36, 36])
        self.assertTrue(any("no hay canales permitidos" in n for n in plan.notes))

    def test_without_wifi(self):
        with self.assertRaisesRegex(WifiError, "ningún transmisor"):
            plan_channels(example_office())


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from open_rps.gui.server import make_server
        cls.server = make_server(port=0)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, body):
        request = urllib.request.Request(self.server.url + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def test_maps_and_plan(self):
        p = wifi_office((1, 1, 1), positions=((5, 10), (20, 10), (35, 10)))
        data = self.post("/api/simulate", {"project": p.to_dict()})
        self.assertTrue({"wifi_rate", "wifi_mcs", "wifi_contention"} <= set(data["metrics"]))
        plan = self.post("/api/wifi/plan", {"project": p.to_dict(), "result_id": data["result_id"]})
        self.assertEqual(len(plan["assignments"]), 3)
        self.assertEqual(len({a["after"] for a in plan["assignments"]}), 3)
        probe = self.post("/api/probe", {"project": p.to_dict(), "point": [12.0, 8.0]})
        self.assertIn("wifi", probe)


if __name__ == "__main__":
    unittest.main()
