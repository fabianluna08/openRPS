"""
OpenRPS - Visualization Engine
Zero-dependency heatmap renderer: ANSI terminal heatmaps and standalone SVG graphics.
"""

import math
from typing import List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np

from .simulation import SimulationResult

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Single-hue sequential ramp (light -> dark blue) used for magnitudes.
_BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# Chart chrome and colors per theme.  ``categorical`` is a fixed-order palette: slot i
# always belongs to the i-th transmitter, and it is never cycled (transmitters beyond
# the 8th share the neutral ``other`` color and are told apart by their map labels).
_THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "axis": "#c3c2b7", "nodata": "#e1e0d9", "other": "#898781",
        "ramp": _BLUE_RAMP,                      # low = light (recedes into the surface)
        "categorical": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "axis": "#383835", "nodata": "#2c2c2a", "other": "#898781",
        "ramp": _BLUE_RAMP[::-1],                # low = dark (recedes into the surface)
        "categorical": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}

# Above this number of cells the per-cell hover tooltips are dropped to bound the file size
_MAX_TOOLTIP_CELLS = 40_000


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _ramp_color(ramp: Sequence[str], norm: float) -> str:
    """Color at position norm (0..1) of a ramp, linearly interpolated between its stops."""
    norm = min(1.0, max(0.0, norm))
    pos = norm * (len(ramp) - 1)
    lo = min(int(pos), len(ramp) - 2)
    frac = pos - lo
    a, b = _hex_to_rgb(ramp[lo]), _hex_to_rgb(ramp[lo + 1])
    return "#" + "".join(f"{round(a[c] + (b[c] - a[c]) * frac):02x}" for c in range(3))


def _nice_ticks(lo: float, hi: float, target: int = 6) -> List[float]:
    """Round tick values (1/2/5 x 10^n) covering [lo, hi]."""
    span = hi - lo
    if not (math.isfinite(span) and span > 0.0):
        return [lo]
    raw = span / max(1, target)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    # Candidate step whose number of intervals is closest to the target
    step = min((m * magnitude for m in (1.0, 2.0, 5.0, 10.0)), key=lambda s: abs(span / s - target))
    ticks = []
    n = math.ceil(lo / step - 1e-9)
    while n * step <= hi + 1e-9 * span:
        ticks.append(round(n * step, 10))
        n += 1
    return ticks


def _fmt(value: float) -> str:
    text = f"{value:.6g}"
    return "0" if text in ("-0", "0") else text


def _finite(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    return data[np.isfinite(data)]


class MapVisualizer:
    """Renders coverage and metric maps into terminal text or vector SVG files."""

    SHADES = " .:-=+*#%@"

    @staticmethod
    def render_ascii_heatmap(
        matrix: np.ndarray,
        title: str = "Coverage Heatmap",
        val_range: Optional[Tuple[float, float]] = None,
        unit: str = "dBm"
    ) -> str:
        """
        Renders a 2D numpy array into a readable ASCII text representation
        (first row of the matrix at the bottom, like a map with y pointing up).
        Non-finite cells (NaN, +/-inf) are drawn as blanks.
        """
        matrix = np.asarray(matrix, dtype=float)
        ny, nx = matrix.shape
        valid = _finite(matrix)
        if val_range is not None:
            min_v, max_v = val_range
        elif valid.size > 0:
            min_v, max_v = float(np.min(valid)), float(np.max(valid))
        else:
            min_v, max_v = 0.0, 1.0

        shades = MapVisualizer.SHADES
        lines = []
        if valid.size > 0:
            lines.append(f"=== {title} ({min_v:.1f} to {max_v:.1f} {unit}) ===")
        else:
            lines.append(f"=== {title} (no data) ===")
        lines.append("+" + "-" * nx + "+")

        for j in range(ny - 1, -1, -1): # Top-down
            row = []
            for i in range(nx):
                val = matrix[j, i]
                if not np.isfinite(val):
                    char = " "
                else:
                    norm = (val - min_v) / max(1e-5, (max_v - min_v))
                    norm = min(1.0, max(0.0, norm))
                    idx = int(norm * (len(shades) - 1))
                    char = shades[idx]
                row.append(char)
            lines.append("|" + "".join(row) + "|")

        lines.append("+" + "-" * nx + "+")
        lines.append(f"    scale: '{shades}'  (low -> high)")
        return "\n".join(lines)

    @staticmethod
    def export_svg(
        result: SimulationResult,
        output_svg_path: str,
        metric: str = "rx_power",
        width_px: int = 800,
        height_px: int = 600,
        theme: str = "light"
    ):
        """
        Exports a standalone SVG map of the coverage result, drawn to scale
        (equal aspect ratio) with axes in meters, transmitter markers and a legend.
        width_px / height_px are the maximum canvas size: the image is cropped to
        the drawing when the map's aspect ratio leaves part of it unused.
        metric options: 'rx_power', 'delay_spread' (continuous color scale with
        intermediate values) or 'best_server' (one color per transmitter).
        theme: 'light' or 'dark'.
        Cells without data (NaN, -inf, no server) are left in a neutral color;
        a result with no data at all produces a valid, empty map.
        """
        if theme not in _THEMES:
            raise ValueError(f"theme must be one of {sorted(_THEMES)}, got {theme!r}")
        colors = _THEMES[theme]

        categorical = metric == "best_server"
        if metric == "rx_power":
            data = np.asarray(result.max_rx_power_dbm, dtype=float)
            valid = _finite(data)
            min_v = float(np.min(valid)) if valid.size else 0.0
            max_v = float(np.max(valid)) if valid.size else 1.0
            title = "Received Power (P_r) Coverage Map"
            unit = "dBm"
        elif metric == "delay_spread":
            data = np.asarray(result.rms_delay_spread_ns, dtype=float)
            valid = _finite(data)
            min_v = 0.0
            max_v = float(np.max(valid)) if valid.size else 1.0
            title = "RMS Delay Spread (sigma_tau) Map"
            unit = "ns"
        elif categorical:
            data = np.asarray(result.best_server_id)
            title = "Best Serving Cell Map"
            unit = ""
        else:
            raise ValueError(f"metric must be 'rx_power', 'delay_spread' or 'best_server', got {metric!r}")

        ny, nx = data.shape
        x_coords = np.asarray(result.x_coords, dtype=float)
        y_coords = np.asarray(result.y_coords, dtype=float)

        # --- Geometry: cells are centered on the grid coordinates; equal scale on both axes ---
        dx = float(x_coords[1] - x_coords[0]) if nx > 1 else (float(y_coords[1] - y_coords[0]) if ny > 1 else 1.0)
        dy = float(y_coords[1] - y_coords[0]) if ny > 1 else dx
        x0, x1 = float(x_coords[0]) - dx / 2.0, float(x_coords[-1]) + dx / 2.0
        y0, y1 = float(y_coords[0]) - dy / 2.0, float(y_coords[-1]) + dy / 2.0

        margin_left, margin_top, margin_bottom, legend_w = 72.0, 52.0, 56.0, 150.0
        avail_w = max(50.0, width_px - margin_left - legend_w)
        avail_h = max(50.0, height_px - margin_top - margin_bottom)
        scale = min(avail_w / (x1 - x0), avail_h / (y1 - y0))
        plot_w, plot_h = (x1 - x0) * scale, (y1 - y0) * scale
        plot_x, plot_y = margin_left, margin_top

        def px(x_m: float) -> float:
            return plot_x + (x_m - x0) * scale

        def py(y_m: float) -> float:
            return plot_y + plot_h - (y_m - y0) * scale

        svg = [
            f'<text x="{plot_x:.1f}" y="30" fill="{colors["ink"]}" font-size="17" font-weight="600">{escape(title)}</text>',
            f'<rect x="{plot_x:.1f}" y="{plot_y:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" fill="{colors["nodata"]}"/>',
        ]

        # --- Cell colors ---
        if result.tx_ids is not None:
            tx_ids = [int(t) for t in result.tx_ids]
        elif categorical:       # result built by hand without metadata: use the ids found in the map
            tx_ids = sorted(int(v) for v in np.unique(data) if v >= 0)
        else:
            tx_ids = []
        palette = colors["categorical"]

        def server_color(tx_id: int) -> str:
            slot = tx_ids.index(tx_id) if tx_id in tx_ids else len(palette)
            return palette[slot] if slot < len(palette) else colors["other"]

        if categorical:
            lookup = {tx_id: server_color(tx_id) for tx_id in tx_ids}
            has_data = np.isin(data, tx_ids) if tx_ids else np.zeros(data.shape, dtype=bool)
        else:
            if max_v - min_v < 1e-12:      # constant field: center it on the scale
                min_v, max_v = min_v - 0.5, max_v + 0.5
            has_data = np.isfinite(data)

        # --- Cells (one rect each; pixel edges shared between neighbours, no anti-aliasing seams) ---
        with_tooltips = nx * ny <= _MAX_TOOLTIP_CELLS
        x_edges = [px(x0 + i * dx) for i in range(nx + 1)]
        y_edges = [py(y0 + j * dy) for j in range(ny + 1)]
        svg.append('<g shape-rendering="crispEdges">')
        for j in range(ny):
            top, height = y_edges[j + 1], y_edges[j] - y_edges[j + 1]
            for i in range(nx):
                if not has_data[j, i]:
                    continue
                val = data[j, i]
                if categorical:
                    color = lookup[int(val)]
                    label = f"Tx {int(val)}"
                else:
                    color = _ramp_color(colors["ramp"], (float(val) - min_v) / (max_v - min_v))
                    label = f"{float(val):.1f} {unit}"
                rect = (f'<rect x="{x_edges[i]:.2f}" y="{top:.2f}" width="{x_edges[i + 1] - x_edges[i]:.2f}" '
                        f'height="{height:.2f}" fill="{color}"')
                if with_tooltips:
                    svg.append(f'{rect}><title>x={_fmt(x_coords[i])} m, y={_fmt(y_coords[j])} m: {label}</title></rect>')
                else:
                    svg.append(rect + '/>')
        svg.append('</g>')

        if not has_data.any():
            svg.append(f'<text x="{plot_x + plot_w / 2:.1f}" y="{plot_y + plot_h / 2:.1f}" text-anchor="middle" '
                       f'fill="{colors["ink2"]}" font-size="14">No data</text>')

        # --- Axes in meters ---
        axis = colors["axis"]
        bottom = plot_y + plot_h
        svg.append(f'<path d="M{plot_x:.1f} {plot_y:.1f}V{bottom:.1f}H{plot_x + plot_w:.1f}" fill="none" stroke="{axis}" stroke-width="1"/>')
        for tick in _nice_ticks(x0, x1, target=max(2, int(plot_w // 80))):
            svg.append(f'<path d="M{px(tick):.1f} {bottom:.1f}v5" stroke="{axis}" stroke-width="1"/>')
            svg.append(f'<text x="{px(tick):.1f}" y="{bottom + 18:.1f}" text-anchor="middle" fill="{colors["muted"]}" font-size="11">{_fmt(tick)}</text>')
        for tick in _nice_ticks(y0, y1, target=max(2, int(plot_h // 60))):
            svg.append(f'<path d="M{plot_x:.1f} {py(tick):.1f}h-5" stroke="{axis}" stroke-width="1"/>')
            svg.append(f'<text x="{plot_x - 9:.1f}" y="{py(tick) + 4:.1f}" text-anchor="end" fill="{colors["muted"]}" font-size="11">{_fmt(tick)}</text>')
        svg.append(f'<text x="{plot_x + plot_w / 2:.1f}" y="{bottom + 40:.1f}" text-anchor="middle" fill="{colors["ink2"]}" font-size="12">x [m]</text>')
        svg.append(f'<text transform="translate({plot_x - 52:.1f} {plot_y + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" fill="{colors["ink2"]}" font-size="12">y [m]</text>')

        # --- Transmitter markers, labelled with their tx_id ---
        if result.tx_positions is not None and result.tx_ids is not None:
            markers = [(int(tx_id), px(pos[0]), py(pos[1])) for tx_id, pos in zip(result.tx_ids, result.tx_positions)
                       if x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1]
            for _, cx, cy in markers:
                svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{colors["ink"]}" stroke="{colors["surface"]}" stroke-width="2"/>')
            placed = []   # label boxes (left, top, right, bottom) already used
            for tx_id, cx, cy in markers:
                text = f"Tx {tx_id}"
                width = 7.5 * len(text)             # rough text width at 12px
                # Candidate corners around the marker; take the first free one inside the map
                candidates = []
                for side, dy_text in ((1, -7.0), (-1, -7.0), (1, 17.0), (-1, 17.0)):
                    left = cx + 9 if side > 0 else cx - 9 - width
                    box = (left, cy + dy_text - 11, left + width, cy + dy_text + 3)
                    inside = box[0] >= plot_x and box[2] <= plot_x + plot_w
                    clear = (all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed) and
                             all(not (box[0] - 7 < mx < box[2] + 7 and box[1] - 7 < my < box[3] + 7) for _, mx, my in markers))
                    candidates.append((not (inside and clear), not inside, side, dy_text, box))
                _, _, side, dy_text, box = min(candidates, key=lambda c: c[:2])
                placed.append(box)
                svg.append(f'<text x="{cx + 9 * side:.1f}" y="{cy + dy_text:.1f}" text-anchor="{"start" if side > 0 else "end"}" '
                           f'fill="{colors["ink"]}" stroke="{colors["surface"]}" stroke-width="3" paint-order="stroke" '
                           f'font-size="12" font-weight="600">{text}</text>')

        # --- Legend (right of the map) ---
        legend_x = plot_x + plot_w + 28
        legend_bottom = plot_y
        if categorical:
            svg.append(f'<text x="{legend_x:.1f}" y="{plot_y + 4:.1f}" fill="{colors["ink2"]}" font-size="12">Best server</text>')
            entries = [(lookup[tx_id], f"Tx {tx_id}") for tx_id in tx_ids[:len(palette)]]
            if len(tx_ids) > len(palette):
                entries.append((colors["other"], f"Other ({len(tx_ids) - len(palette)} Tx)"))
            if not has_data.all():
                entries.append((colors["nodata"], "No server"))
            for row, (color, label) in enumerate(entries):
                y_row = plot_y + 18 + row * 20
                svg.append(f'<rect x="{legend_x:.1f}" y="{y_row:.1f}" width="12" height="12" rx="2" fill="{color}"/>')
                svg.append(f'<text x="{legend_x + 20:.1f}" y="{y_row + 10:.1f}" fill="{colors["ink2"]}" font-size="12">{escape(label)}</text>')
            legend_bottom = plot_y + 18 + len(entries) * 20
        elif has_data.any():
            bar_w = 14.0
            bar_y = plot_y + 18
            bar_h = min(max(plot_h - 18, 120.0), 280.0)
            ramp = colors["ramp"]
            stops = "".join(f'<stop offset="{100.0 * s / (len(ramp) - 1):.2f}%" stop-color="{c}"/>' for s, c in enumerate(ramp))
            svg.append(f'<defs><linearGradient id="scale" x1="0" y1="1" x2="0" y2="0">{stops}</linearGradient></defs>')
            svg.append(f'<text x="{legend_x:.1f}" y="{plot_y + 4:.1f}" fill="{colors["ink2"]}" font-size="12">{escape(unit)}</text>')
            svg.append(f'<rect x="{legend_x:.1f}" y="{bar_y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="url(#scale)"/>')

            def bar_pos(value: float) -> float:
                return bar_y + bar_h * (1.0 - (value - min_v) / (max_v - min_v))

            # End values exactly, plus round intermediate values that do not collide with them
            labels = [(min_v, f"{min_v:.1f}"), (max_v, f"{max_v:.1f}")]
            for tick in _nice_ticks(min_v, max_v, target=6):
                if all(abs(bar_pos(tick) - bar_pos(v)) >= 14.0 for v, _ in labels[:2]):
                    labels.append((tick, _fmt(tick)))
            for value, text in labels:
                y_tick = bar_pos(value)
                svg.append(f'<path d="M{legend_x + bar_w:.1f} {y_tick:.1f}h4" stroke="{axis}" stroke-width="1"/>')
                svg.append(f'<text x="{legend_x + bar_w + 8:.1f}" y="{y_tick + 4:.1f}" fill="{colors["muted"]}" font-size="11">{text}</text>')
            legend_bottom = bar_y + bar_h + 6
            if not has_data.all():
                y_row = bar_y + bar_h + 14
                svg.append(f'<rect x="{legend_x:.1f}" y="{y_row:.1f}" width="12" height="12" rx="2" fill="{colors["nodata"]}"/>')
                svg.append(f'<text x="{legend_x + 20:.1f}" y="{y_row + 10:.1f}" fill="{colors["ink2"]}" font-size="12">No data</text>')
                legend_bottom = y_row + 12

        # --- Canvas: width_px x height_px is the maximum size; crop it to the drawing ---
        canvas_w = math.ceil(min(width_px, max(380.0, plot_x + plot_w + legend_w)))
        canvas_h = math.ceil(min(height_px, max(plot_y + plot_h + margin_bottom, legend_bottom + 16.0)))
        svg[:0] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_w} {canvas_h}" '
            f'width="{canvas_w}" height="{canvas_h}" font-family=\'{_FONT}\'>',
            f'<rect width="100%" height="100%" fill="{colors["surface"]}"/>',
        ]
        svg.append('</svg>')

        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg))
