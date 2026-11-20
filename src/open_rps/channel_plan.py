"""
Suggestion of Wi-Fi channels for the APs of a project (new in 1.8.0).

Each AP keeps its band and width; the search chooses, for each one, the block of
channels it occupies so that the mean spectral efficiency of the area served by
Wi-Fi is the highest:

    mean over pixels of  min(log2(1 + SINR), log2(1 + SINR of the highest MCS))

with the SINR of ``SimulationResult.sinr_map`` (spectral overlap of every
transmitter, noise of the serving channel).  The cap keeps the search from
chasing SINR that no MCS uses.  The received powers are the ones of one
simulation: changing channel inside a band changes the frequency a little (up to
about 1 dB of path loss between the ends of the 5 GHz band), which the search
ignores; after applying the plan, simulate again to see the exact result.

Search: coordinate descent (each AP in turn takes the channel that best improves
the objective, until none changes) from the current channels and from several
random starts, keeping the best; on a sample of at most ``max_pixels`` pixels,
and the final numbers on all of them.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .simulation import thermal_noise_dbm
from .wifi import (DFS_CHANNELS_5GHZ, UNII4_CHANNELS_5GHZ, ChannelSpec, WifiError, best_mcs, channel_blocks,
                   channel_center_hz, max_mcs, overlap_factor, required_sinr_db)

__all__ = ["plan_channels", "ChannelPlan", "allowed_blocks"]


def allowed_blocks(band: str, width_mhz: int, include_dfs: bool = False, include_unii4: bool = False):
    """Blocks of channels (``wifi.channel_blocks``) that the plan may use."""
    out = []
    for block, centre in channel_blocks(band, width_mhz):
        if band == "5":
            if not include_dfs and any(c in DFS_CHANNELS_5GHZ for c in block):
                continue
            if not include_unii4 and any(c in UNII4_CHANNELS_5GHZ for c in block):
                continue
        if band == "2.4" and block[0] not in (1, 6, 11):
            continue                # the three channels that do not overlap in 2.4 GHz
        out.append((block, centre))
    return out


@dataclass
class ChannelPlan:
    assignments: List[Dict[str, Any]]      # per Wi-Fi AP: tx_id, name, band, width, before, after
    before: Dict[str, Any]                 # metrics with the current channels
    after: Dict[str, Any]                  # metrics with the suggested channels
    evaluations: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"assignments": self.assignments, "before": self.before, "after": self.after,
                "evaluations": self.evaluations, "notes": list(self.notes)}


def _metrics(sinr: np.ndarray, rate: np.ndarray, capacity: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(sinr)
    if not finite.any():
        return {"pixels": 0}
    s = sinr[finite]
    return {"pixels": int(finite.sum()), "mean_capacity_bps_hz": float(np.mean(capacity[finite])),
            "median_sinr_db": float(np.median(s)), "fraction_sinr_20db": float(np.mean(s >= 20.0)),
            "mean_rate_mbps": float(np.mean(rate[finite])), "fraction_connected": float(np.mean(rate[finite] > 0))}


def plan_channels(project, result=None, include_dfs: bool = False, include_unii4: bool = False,
                  restarts: int = 8, seed: int = 0, max_pixels: int = 20000) -> ChannelPlan:
    """
    Suggest a channel for every enabled AP with a Wi-Fi channel (see the module
    docstring).  ``result``: a ``SimulationResult`` of this same project (else it is
    simulated).  The project is not changed.

    In 2.4 GHz only channels 1, 6 and 11 are used; in 5 GHz, without
    ``include_dfs`` the blocks with DFS channels (52-144, which need radar detection)
    are left out, and without ``include_unii4`` those with channels 169-177.  Check
    which channels your country allows.
    """
    sim = project.build_simulator()
    wifi = project.wifi_setup(sim)
    if wifi is None:
        raise WifiError("ningún transmisor activo tiene un canal Wi-Fi")
    result = result if result is not None else project.run(sim)
    s = project.settings
    channels: List[ChannelSpec] = list(wifi["channels"])
    standards: List[Optional[str]] = wifi["standards"]
    specs = {t.tx_id: t for t in project.transmitters}
    n = len(channels)
    power_dbm = np.asarray(result.rx_power_dbm, dtype=float).reshape(n, -1)
    index = result.best_server_indices().reshape(-1)
    wifi_tx = np.array([standards[k] is not None for k in range(n)])
    pixels = np.flatnonzero((index >= 0) & wifi_tx[np.clip(index, 0, n - 1)]
                            & np.all(~np.isnan(power_dbm), axis=0))
    if not pixels.size:
        raise WifiError("ningún píxel del área lo sirve un AP con canal Wi-Fi")
    power_mw = 10.0 ** (power_dbm[:, pixels] / 10.0)
    serving = index[pixels]
    noise = np.array([10.0 ** (thermal_noise_dbm(c.width_hz, s.noise_figure_db) / 10.0) for c in channels])
    caps = np.array([math.log2(1.0 + 10.0 ** (required_sinr_db(max_mcs(st)) / 10.0)) if st else 0.0
                     for st in standards])
    rng = np.random.default_rng(seed)
    sample = np.arange(len(pixels)) if len(pixels) <= max_pixels else np.sort(rng.choice(len(pixels), max_pixels,
                                                                                         replace=False))
    notes: List[str] = []

    # Candidates of each AP: (primary channel, ChannelSpec); fixed transmitters have only their current one
    candidates: List[List[Tuple[Optional[int], ChannelSpec]]] = []
    current: List[int] = []
    for k, tx in enumerate(sim.transmitters):
        spec = specs[tx.tx_id]
        if standards[k] is None:
            candidates.append([(None, channels[k])])
            current.append(0)
            continue
        blocks = allowed_blocks(spec.wifi_band, spec.wifi_width_mhz, include_dfs, include_unii4)
        options = []
        for block, _ in blocks:
            primary = spec.wifi_channel if spec.wifi_channel in block else block[0]
            options.append((primary, ChannelSpec(channel_center_hz(spec.wifi_band, primary, spec.wifi_width_mhz),
                                                 spec.wifi_width_mhz * 1e6)))
        here = [i for i, (_, c) in enumerate(options) if c == channels[k]]
        if here:
            current.append(here[0])
        else:
            options.append((spec.wifi_channel, channels[k]))
            current.append(len(options) - 1)
            if not blocks:
                notes.append(f"{spec.name or spec.tx_id}: no hay canales permitidos de {spec.wifi_width_mhz} MHz en "
                             f"{spec.wifi_band} GHz con estas opciones; se deja el actual")
        candidates.append(options)

    factor_cache: Dict[Tuple[ChannelSpec, ChannelSpec], float] = {}

    def factor(interferer: ChannelSpec, receiver: ChannelSpec) -> float:
        key = (interferer, receiver)
        if key not in factor_cache:
            factor_cache[key] = overlap_factor(interferer, receiver)
        return factor_cache[key]

    evaluations = 0

    def sinr_of(choice: List[int], cols) -> np.ndarray:
        spec = [candidates[k][choice[k]][1] for k in range(n)]
        F = np.array([[0.0 if j == i else factor(spec[j], spec[i]) for j in range(n)] for i in range(n)])
        srv = serving[cols]
        p = power_mw[:, cols]
        interference = np.einsum("pj,jp->p", F[srv], p)
        signal = p[srv, np.arange(len(cols))]
        return 10.0 * np.log10(signal / (interference + noise[srv]))

    def objective(choice: List[int]) -> float:
        nonlocal evaluations
        evaluations += 1
        sinr = sinr_of(choice, sample)
        cap = caps[serving[sample]]
        return float(np.mean(np.minimum(np.log2(1.0 + 10.0 ** (sinr / 10.0)), cap)))

    def descend(choice: List[int]) -> Tuple[List[int], float]:
        best = objective(choice)
        for _ in range(50):
            changed = False
            for k in range(n):
                if len(candidates[k]) < 2:
                    continue
                keep = choice[k]
                for option in range(len(candidates[k])):
                    if option == keep:
                        continue
                    choice[k] = option
                    value = objective(choice)
                    if value > best + 1e-12:
                        best, keep, changed = value, option, True
                choice[k] = keep
            if not changed:
                break
        return choice, best

    best_choice, best_value = descend(list(current))
    for _ in range(max(0, restarts)):
        start = [int(rng.integers(len(c))) for c in candidates]
        choice, value = descend(start)
        if value > best_value + 1e-12:
            best_choice, best_value = choice, value

    def report(choice: List[int]) -> Dict[str, Any]:
        cols = np.arange(len(pixels))
        sinr = sinr_of(choice, cols)
        rate = np.zeros(len(cols))
        capacity = np.minimum(np.log2(1.0 + 10.0 ** (sinr / 10.0)), caps[serving])
        for k in range(n):
            sel = serving == k
            if sel.any() and standards[k] is not None:
                _, rate[sel] = best_mcs(sinr[sel], standards[k], int(round(channels[k].width_hz / 1e6)),
                                        wifi["streams"][k], s.wifi_guard_interval_us)
        return _metrics(sinr, rate, capacity)

    assignments = []
    for k, tx in enumerate(sim.transmitters):
        if standards[k] is None:
            continue
        spec = specs[tx.tx_id]
        assignments.append({"tx_id": tx.tx_id, "name": spec.name or f"Tx {tx.tx_id}", "band": spec.wifi_band,
                            "width_mhz": spec.wifi_width_mhz, "before": spec.wifi_channel,
                            "after": candidates[k][best_choice[k]][0]})
    return ChannelPlan(assignments=assignments, before=report(current), after=report(best_choice),
                       evaluations=evaluations, notes=notes)
