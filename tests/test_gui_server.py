"""Tests of the local server of the graphical interface (open_rps.gui.server, new in 1.4.0)."""
import base64
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

from open_rps.gui import build_parser
from open_rps.gui.server import STATIC_DIR, make_server
from open_rps.project import example_urban


def decode(grid):
    values = np.frombuffer(base64.b64decode(grid["data"]), dtype="<f4")
    return values.reshape(grid["shape"])


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = cls.server.url.rstrip("/")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return response.status, response.headers, response.read()

    def post(self, path, body, headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method="POST",
                                         headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def small_project(self):
        project = example_urban()
        project.area.resolution_m = 20.0
        return project.to_dict()

    def test_static_files(self):
        status, headers, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"js/main.js", body)
        for path in ("/js/main.js", "/css/app.css", "/vendor/three.bundle.min.js"):
            status, headers, _ = self.get(path)
            self.assertEqual(status, 200, path)
        self.assertIn("javascript", self.get("/js/main.js")[1]["Content-Type"])
        for js in os.listdir(os.path.join(STATIC_DIR, "js")):
            self.assertEqual(self.get(f"/js/{js}")[0], 200, js)

    def test_no_path_traversal(self):
        for path in ("/../server.py", "/%2e%2e/server.py", "/js/../../server.py", "/..%2f..%2fproject.py", "/nope.js"):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as ctx:
                self.get(path)
            self.assertEqual(ctx.exception.code, 404)

    def test_info_project_and_examples(self):
        info = json.loads(self.get("/api/info")[2])
        self.assertIn("raytracing", info["modes"])
        self.assertTrue(any(m["id"] == "concreto" for m in info["materials"]))
        self.assertEqual({e["id"] for e in info["examples"]}, {"urbano", "oficina", "wifi", "restaurante", "antenas"})
        project = json.loads(self.get("/api/project")[2])
        self.assertEqual(project["project"]["format"], "openrps-project")
        self.assertIsNone(project["path"])
        office = json.loads(self.get("/api/example/oficina")[2])["project"]
        self.assertEqual(office["name"], "Oficina (interior)")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/example/marte")
        self.assertEqual(ctx.exception.code, 404)

    def test_validate(self):
        status, data = self.post("/api/validate", {"project": {"name": "x", "transmitters": [{"tx_id": 0}]}})
        self.assertEqual(status, 200)
        self.assertEqual(data["project"]["settings"]["mode"], "raytracing")     # defaults filled in
        status, data = self.post("/api/validate", {"project": {"area": {"resolution_m": -1}}})
        self.assertEqual(status, 400)
        self.assertIn("resolution_m", data["error"])

    def test_simulate_payload(self):
        status, data = self.post("/api/simulate", {"project": self.small_project()})
        self.assertEqual(status, 200, data)
        grid = data["grid"]
        self.assertEqual((grid["ny"], grid["nx"]), (21, 31))
        self.assertEqual(grid["dx"], 20.0)
        for name in ("rx_power", "path_loss", "best_server", "delay_spread", "angular_spread", "sinr", "interference"):
            self.assertEqual(decode(data["metrics"][name]).shape, (21, 31), name)
        power = decode(data["metrics"]["rx_power"])
        self.assertTrue(np.isfinite(power).any())
        self.assertTrue(np.isneginf(power).any() or np.isnan(power).any())      # interiors of opaque buildings
        self.assertEqual(len(data["per_tx_rx_power"]), 2)
        self.assertEqual(data["tx_ids"], [0, 1])
        self.assertEqual(data["report"]["reflectors"], 38)
        # Same numbers as running the project directly (float32 rounding)
        result = example_urban()
        result.area.resolution_m = 20.0
        direct = result.run()
        np.testing.assert_allclose(power, direct.max_rx_power_dbm.astype(np.float32), rtol=0, atol=0)
        # Exports of the cached result
        rid = data["result_id"]
        status, headers, body = self.get(f"/api/export/{rid}/csv")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"x_m,y_m,best_tx_id"))
        self.assertIn("attachment", headers["Content-Disposition"])
        summary = json.loads(self.get(f"/api/export/{rid}/json")[2])
        self.assertIn("sinr", summary)
        for metric in ("rx_power", "best_server", "delay_spread", "sinr", "interference"):
            self.assertIn(b"<svg", self.get(f"/api/export/{rid}/svg-{metric}")[2][:200], metric)
        for bad in (f"/api/export/{rid}/svg-magia", "/api/export/nope/csv"):
            with self.subTest(bad=bad), self.assertRaises(urllib.error.HTTPError):
                self.get(bad)

    def test_simulate_errors(self):
        project = self.small_project()
        project["buildings"][0]["material"] = "kryptonita"
        status, data = self.post("/api/simulate", {"project": project})
        self.assertEqual(status, 400)
        self.assertIn("kryptonita", data["error"])
        self.assertEqual(self.post("/api/simulate", b"{no es json")[0], 400)
        self.assertEqual(self.post("/api/simulate", {"otra": 1})[0], 400)
        self.assertEqual(self.post("/api/nada", {"project": project})[0], 404)

    def test_foreign_origin_is_rejected(self):
        status, data = self.post("/api/simulate", {"project": self.small_project()},
                                 headers={"Origin": "https://example.com"})
        self.assertEqual(status, 403)
        status, _ = self.post("/api/validate", {"project": self.small_project()}, headers={"Origin": self.base})
        self.assertEqual(status, 200)

    def test_probe(self):
        status, data = self.post("/api/probe", {"project": self.small_project(), "point": [300.0, 190.0]})
        self.assertEqual(status, 200, data)
        self.assertEqual(data["point"], [300.0, 190.0, 1.5])
        self.assertEqual(len(data["transmitters"]), 2)
        rays = data["transmitters"][0]["rays"]
        self.assertGreaterEqual(len(rays), 2)
        self.assertEqual(rays[0]["kind"], "direct")
        self.assertEqual(rays[0]["path"][-1], [300.0, 190.0, 1.5])
        self.assertEqual(self.post("/api/probe", {"project": self.small_project()})[0], 400)
        self.assertEqual(self.post("/api/probe", {"project": self.small_project(), "point": [1]})[0], 400)

    def test_initial_project_from_file(self):
        project = example_urban()
        project.name = "Desde archivo"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.orps.json")
            project.save(path)
            server = make_server("127.0.0.1", 0, project_path=path)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(server.url + "api/project", timeout=10) as response:
                    data = json.loads(response.read())
            finally:
                server.shutdown()
                server.server_close()
        self.assertEqual(data["project"]["name"], "Desde archivo")
        self.assertEqual(data["path"], path)

    def test_command_line_parser(self):
        args = build_parser().parse_args(["mi.orps.json", "--port", "0", "--no-browser"])
        self.assertEqual((args.project, args.port, args.no_browser, args.host), ("mi.orps.json", 0, True, "127.0.0.1"))


if __name__ == "__main__":
    unittest.main()
