"""
OpenRPS - Graphical interface

``open-rps-gui`` (or ``python3 -m open_rps.gui``) starts a local web server and
opens the interface in the default browser: 2D editor of the scenario, 3D view
with the rays, coverage maps and channel impulse response at any point.  It
needs nothing beyond numpy and a modern browser; the 3D view uses a copy of
three.js (MIT license) shipped in ``static/vendor``.
"""

import argparse
import sys
import threading
import webbrowser

from .server import OpenRPSServer, make_server

__all__ = ["OpenRPSServer", "make_server", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-rps-gui",
        description="OpenRPS: interfaz gráfica (editor 2D, vista 3D, mapas de cobertura y respuesta al impulso)",
    )
    parser.add_argument("project", nargs="?", default=None,
                        help="Proyecto .orps.json que se abre al iniciar (por defecto, el ejemplo urbano)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Dirección en la que escucha el servidor (por defecto 127.0.0.1: solo este equipo)")
    parser.add_argument("--port", type=int, default=8765, help="Puerto (por defecto 8765; 0 = uno libre)")
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automáticamente")
    parser.add_argument("--verbose", action="store_true", help="Mostrar cada petición en la consola")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        server = make_server(args.host, args.port, args.project, args.verbose)
    except OSError as error:
        print(f"[x] No se pudo abrir el puerto {args.port}: {error}. Prueba con --port 0", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"[x] No se pudo leer el proyecto: {error}", file=sys.stderr)
        return 1
    print(f"[*] OpenRPS GUI en {server.url}  (Ctrl+C para salir)")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("[!] Aviso: el servidor es accesible desde otros equipos de la red")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(server.url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Cerrando OpenRPS GUI")
    finally:
        server.server_close()
    return 0
