#!/usr/bin/env python3
"""
OpenRPS - compare a project with measurements and calibrate it (new in 1.7.0)

    open-rps-calibrate proyecto.orps.json mediciones.csv
    open-rps-calibrate proyecto.orps.json mediciones.csv -o calibrado.orps.json
    open-rps-calibrate proyecto.orps.json --compare-only --errors errores.csv
    python3 -m open_rps.calibrate_cli proyecto.orps.json mediciones.csv --materials concreto,tablaroca

Without a CSV, the measurements stored in the project are used.
"""

import argparse
import csv
import sys

from .measurements import MeasurementError, calibrate, compare, read_csv
from .project import Project, ProjectError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-rps-calibrate",
        description="Compara la predicción de un proyecto de OpenRPS con mediciones y ajusta las pérdidas de "
                    "los materiales, la potencia de los transmisores y, en log-distancia, el exponente.")
    parser.add_argument("project", help="proyecto .orps.json")
    parser.add_argument("csv", nargs="?", help="mediciones (CSV con columnas x, y, potencia y, opcionales, z, tx "
                                               "y etiqueta); sin él se usan las guardadas en el proyecto")
    parser.add_argument("-o", "--output", help="guardar aquí el proyecto calibrado (con las mediciones)")
    parser.add_argument("--compare-only", action="store_true", help="solo comparar, sin calibrar")
    parser.add_argument("--materials", default="auto",
                        help="materiales a ajustar: 'auto' (los que usan los edificios y muros), 'none' o una "
                             "lista de ids separados por comas")
    parser.add_argument("--tx-power", choices=["auto", "per_tx", "common", "none"], default="auto",
                        help="ajuste de potencia: uno por transmisor, uno común o ninguno (auto: por transmisor "
                             "si todas las mediciones dicen de qué transmisor son)")
    parser.add_argument("--no-exponent", action="store_true", help="no ajustar el exponente (modo log-distancia)")
    parser.add_argument("--errors", metavar="CSV", help="guardar el error de cada punto en este CSV")
    return parser


def _print_stats(title, stats, counts):
    print(f"{title}: {stats['n']} puntos comparados", end="")
    skipped = {k: v for k, v in counts.items() if k != "ok" and v}
    print(" (" + ", ".join(f"{v} {k}" for k, v in skipped.items()) + ")" if skipped else "")
    if stats["n"]:
        r = "—" if stats["r"] is None else f"{stats['r']:.3f}"
        print(f"  sesgo {stats['bias_db']:+.2f} dB   desviación {stats['std_db']:.2f} dB   "
              f"RMSE {stats['rmse_db']:.2f} dB   error máximo {stats['max_abs_db']:.1f} dB   r = {r}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project = Project.load(args.project)
        if args.csv:
            project.measurements = read_csv(args.csv, project.transmitters)
        if args.compare_only:
            comparison = compare(project)
            _print_stats("Comparación", comparison.stats, comparison.counts)
            points = comparison.points
        else:
            materials = None if args.materials == "none" else "auto" if args.materials == "auto" \
                else [m.strip() for m in args.materials.split(",") if m.strip()]
            print("Calibrando (cada iteración vuelve a trazar los rayos en los puntos medidos)...", file=sys.stderr)
            result = calibrate(project, materials=materials, tx_power=args.tx_power,
                               exponent=not args.no_exponent)
            _print_stats("Antes", result.before.stats, result.before.counts)
            _print_stats("Después", result.after.stats, result.after.counts)
            print(f"Parámetros ({result.iterations} iteraciones, {result.evaluations} evaluaciones del modelo):")
            for p in result.parameters:
                unc = "" if p.uncertainty is None else f" ± {p.uncertainty:.2f}"
                if p.kind == "tx_power":
                    print(f"  {p.name}: {p.after:+.2f} dB{unc}  ({p.points} puntos)")
                else:
                    unit = " dB" if p.unit == "dB" else ""
                    print(f"  {p.name}: {p.before:.2f} -> {p.after:.2f}{unit}{unc}  ({p.points} puntos)")
            for note in result.notes:
                print(f"  Nota: {note}")
            points = result.after.points
            if args.output:
                result.project.save(args.output)
                print(f"Proyecto calibrado guardado en {args.output}")
        if args.errors:
            with open(args.errors, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z", "tx_id", "medido_dbm", "predicho_dbm", "error_db", "estado",
                                 "etiqueta"])
                for p in points:
                    writer.writerow([p.x, p.y, p.z, "" if p.tx_id is None else p.tx_id, p.measured_dbm,
                                     "" if p.predicted_dbm != p.predicted_dbm else round(p.predicted_dbm, 3),
                                     "" if p.error_db != p.error_db else round(p.error_db, 3), p.status, p.label])
            print(f"Errores por punto guardados en {args.errors}")
    except (ProjectError, MeasurementError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
