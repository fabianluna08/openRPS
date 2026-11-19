"""
OpenRPS - Graphical interface: local web server

Serves the single-page interface (``static/``) and a small JSON API that runs the
simulations with the same engine as the library.  Standard library only
(``http.server``); it listens on 127.0.0.1 by default, so only this computer can
reach it.

API (every body is JSON):
    GET  /api/info                    version, simulation modes, default materials, examples
    GET  /api/project                 project given with --project, or the urban example
    GET  /api/example/<name>          one of the examples (``open_rps.project.EXAMPLES``)
    POST /api/validate  {project}     normalized project, or 400 with the error
    POST /api/simulate  {project}     grids of the result (float32, base64) + summary + report
    POST /api/probe     {project, point: [x, y] | [x, y, z]}   rays and CIR at one point
    GET  /api/export/<result_id>/<what>   csv | json | svg-<metric> of a cached result
"""

import base64
import io
import json
import math
import os
import posixpath
import tempfile
import threading
import traceback
import uuid
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlsplit

import numpy as np

from .. import __version__
from ..project import (EXAMPLES, HATA_ENVIRONMENTS, PENETRATION_MODELS, REFLECTION_MODELS, SIMULATION_MODES,
                       Project, ProjectError, default_materials)
from ..visualizer import MapVisualizer

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_CACHED_RESULTS = 4
# Per-transmitter power grids are sent only while their total size stays reasonable
MAX_PER_TX_VALUES = 4_000_000
SVG_METRICS = ("rx_power", "best_server", "delay_spread", "sinr", "interference")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json",
                 ".png": "image/png", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8",
                 ".md": "text/plain; charset=utf-8"}


def encode_grid(values) -> Dict[str, Any]:
    """[Ny, Nx] grid as little-endian float32 in base64 (NaN and +-inf are kept)."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    return {"shape": list(array.shape), "dtype": "float32", "data": base64.b64encode(array.tobytes()).decode("ascii")}


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def result_payload(result, report: Dict[str, Any], result_id: str) -> Dict[str, Any]:
    """Everything the interface needs to draw the maps of a ``SimulationResult``."""
    metrics = {
        "rx_power": encode_grid(result.max_rx_power_dbm),
        "path_loss": encode_grid(result.best_server_path_loss_db()),
        "best_server": encode_grid(result.best_server_id),
        "delay_spread": encode_grid(result.rms_delay_spread_ns),
        "angular_spread": encode_grid(result.angular_spread_bs_phi),
    }
    if result.sinr_db is not None:
        metrics["sinr"] = encode_grid(result.sinr_db)
        metrics["interference"] = encode_grid(result.interference_plus_noise_dbm)
    per_tx = None
    if result.rx_power_dbm.size <= MAX_PER_TX_VALUES:
        per_tx = [encode_grid(grid) for grid in result.rx_power_dbm]
    x, y = np.asarray(result.x_coords, dtype=float), np.asarray(result.y_coords, dtype=float)
    return {
        "result_id": result_id,
        "grid": {"x0": float(x[0]), "y0": float(y[0]), "nx": int(len(x)), "ny": int(len(y)),
                 "dx": float(x[1] - x[0]) if len(x) > 1 else 0.0, "dy": float(y[1] - y[0]) if len(y) > 1 else 0.0},
        "tx_ids": [int(t) for t in (result.tx_ids or [])],
        "metrics": metrics,
        "per_tx_rx_power": per_tx,
        "summary": result.summary(),
        "report": {k: (_finite_or_none(v) if isinstance(v, float) else v) for k, v in report.items()},
    }


class ResultCache:
    """The last few results, for the exports (thread safe)."""

    def __init__(self, size: int = MAX_CACHED_RESULTS):
        self.size = size
        self._items: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, result) -> str:
        key = uuid.uuid4().hex
        with self._lock:
            self._items[key] = result
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return key

    def get(self, key: str):
        with self._lock:
            return self._items.get(key)


class OpenRPSRequestHandler(BaseHTTPRequestHandler):
    server_version = f"OpenRPS/{__version__}"
    protocol_version = "HTTP/1.1"

    # ----- plumbing
    def log_message(self, format, *args):        # noqa: A002 (signature of the base class)
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def _send(self, status: int, body: bytes, content_type: str, extra_headers: Optional[Dict[str, str]] = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, data: Any):
        body = json.dumps(data, allow_nan=False, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str):
        self._json(status, {"error": message})

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ProjectError("la petición no tiene cuerpo JSON")
        if length > MAX_BODY_BYTES:
            raise ProjectError("la petición es demasiado grande")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProjectError(f"JSON inválido: {error}") from None

    def _same_origin(self) -> bool:
        """Reject requests sent by other web sites (the API is only for the page served here)."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host", "")
        return urlsplit(origin).netloc == host

    # ----- verbs
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path.startswith("/api/"):
                self._api_get(path)
            else:
                self._static(path)
        except Exception:                                    # pragma: no cover - last resort
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, traceback.format_exc(limit=1).strip().splitlines()[-1])

    def do_POST(self):
        path = unquote(urlsplit(self.path).path)
        if not self._same_origin():
            self._error(HTTPStatus.FORBIDDEN, "origen no permitido")
            return
        try:
            body = self._read_json()
            if path == "/api/validate":
                self._json(HTTPStatus.OK, {"project": self._project(body).to_dict()})
            elif path == "/api/simulate":
                project = self._project(body)
                result, report = project.run_with_report()
                key = self.server.results.put(result)
                self._json(HTTPStatus.OK, result_payload(result, report, key))
            elif path == "/api/probe":
                project = self._project(body)
                point = body.get("point") if isinstance(body, dict) else None
                if not isinstance(point, (list, tuple)):
                    raise ProjectError("probe: falta el punto [x, y]")
                self._json(HTTPStatus.OK, project.probe(point).to_dict())
            else:
                self._error(HTTPStatus.NOT_FOUND, f"ruta desconocida {path}")
        except ProjectError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except ValueError as error:            # e.g. a model asked for impossible parameters
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception:                      # pragma: no cover - bug: report it to the page
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, traceback.format_exc(limit=2).strip().splitlines()[-1])

    # ----- API
    @staticmethod
    def _project(body) -> Project:
        if not isinstance(body, dict) or "project" not in body:
            raise ProjectError("falta el proyecto en la petición")
        return Project.from_dict(body["project"])

    def _api_get(self, path: str):
        if path == "/api/info":
            self._json(HTTPStatus.OK, {
                "version": __version__, "modes": list(SIMULATION_MODES), "hata_environments": list(HATA_ENVIRONMENTS),
                "reflection_models": list(REFLECTION_MODELS), "penetration_models": list(PENETRATION_MODELS),
                "materials": Project(materials=default_materials()).to_dict()["materials"],
                "examples": [{"id": key, "name": factory().name} for key, factory in EXAMPLES.items()],
                "svg_metrics": list(SVG_METRICS),
            })
        elif path == "/api/project":
            project = self.server.initial_project or EXAMPLES["urbano"]()
            self._json(HTTPStatus.OK, {"project": project.to_dict(), "path": self.server.initial_project_path})
        elif path.startswith("/api/example/"):
            factory = EXAMPLES.get(path[len("/api/example/"):])
            if factory is None:
                self._error(HTTPStatus.NOT_FOUND, "ejemplo desconocido")
            else:
                self._json(HTTPStatus.OK, {"project": factory().to_dict()})
        elif path.startswith("/api/export/"):
            self._export(path[len("/api/export/"):])
        else:
            self._error(HTTPStatus.NOT_FOUND, f"ruta desconocida {path}")

    def _export(self, rest: str):
        key, _, what = rest.partition("/")
        result = self.server.results.get(key)
        if result is None:
            self._error(HTTPStatus.NOT_FOUND, "resultado no disponible: vuelve a ejecutar la simulación")
            return
        with tempfile.TemporaryDirectory() as tmp:
            if what == "csv":
                path, mime, name = os.path.join(tmp, "r.csv"), "text/csv; charset=utf-8", "simulation_results.csv"
                result.export_csv(path)
            elif what == "json":
                path, mime, name = os.path.join(tmp, "r.json"), "application/json", "simulation_summary.json"
                result.export_json(path)
            elif what.startswith("svg-") and what[4:] in SVG_METRICS:
                metric = what[4:]
                if metric in ("sinr", "interference") and result.sinr_db is None:
                    self._error(HTTPStatus.BAD_REQUEST, "este resultado no tiene SINR (actívalo en la configuración)")
                    return
                path, mime, name = os.path.join(tmp, "r.svg"), "image/svg+xml", f"coverage_{metric}.svg"
                MapVisualizer.export_svg(result, path, metric=metric)
            else:
                self._error(HTTPStatus.NOT_FOUND, f"exportación desconocida {what!r}")
                return
            with open(path, "rb") as f:
                body = f.read()
        self._send(HTTPStatus.OK, body, mime, {"Content-Disposition": f'attachment; filename="{name}"'})

    # ----- static files
    def _static(self, path: str):
        if path in ("", "/"):
            path = "/index.html"
        relative = posixpath.normpath(path).lstrip("/")
        if relative.startswith("..") or "\\" in relative:
            self._error(HTTPStatus.NOT_FOUND, "no encontrado")
            return
        full = os.path.realpath(os.path.join(STATIC_DIR, *relative.split("/")))
        if not full.startswith(os.path.realpath(STATIC_DIR) + os.sep) or not os.path.isfile(full):
            self._error(HTTPStatus.NOT_FOUND, "no encontrado")
            return
        with open(full, "rb") as f:
            body = f.read()
        ext = os.path.splitext(full)[1].lower()
        self._send(HTTPStatus.OK, body, CONTENT_TYPES.get(ext, "application/octet-stream"))


class OpenRPSServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: Tuple[str, int], initial_project: Optional[Project] = None,
                 initial_project_path: Optional[str] = None, verbose: bool = False):
        super().__init__(address, OpenRPSRequestHandler)
        self.results = ResultCache()
        self.initial_project = initial_project
        self.initial_project_path = initial_project_path
        self.verbose = verbose

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        host = "127.0.0.1" if host in ("", "0.0.0.0") else host
        return f"http://{host}:{port}/"


def make_server(host: str = "127.0.0.1", port: int = 8765, project_path: Optional[str] = None,
                verbose: bool = False) -> OpenRPSServer:
    """Server ready to ``serve_forever()``; ``port=0`` picks a free port."""
    project = Project.load(project_path) if project_path else None
    return OpenRPSServer((host, port), project, project_path, verbose)
