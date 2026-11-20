"""
Wi-Fi (IEEE 802.11) layer of OpenRPS, new in 1.8.0.

* Channels of the 2.4, 5 and 6 GHz bands and their centre frequencies, for 20,
  40, 80 and 160 MHz (``channel_center_hz``, ``valid_channels``).
* Spectral overlap between two channels (``overlap_factor``): the fraction of the
  power of a transmitter that falls inside the band of a receiver.  The transmitter
  radiates with the OFDM transmit spectral mask of 802.11 (0 dBr up to W/2 - 1 MHz,
  -20 dBr at W/2 + 1, -28 dBr at W, -40 dBr at 1.5 W, linear in dB in between and
  nothing beyond 1.5 W); the receiver filter is ideal over its own channel.  The
  mask is the most a compliant radio may emit, so the adjacent-channel
  interference is an upper bound.
* Modulation and coding (MCS) that a SINR allows, and the PHY rate of 802.11n (HT),
  802.11ac (VHT) and 802.11ax (HE) from the parameters of the OFDM symbol
  (``required_sinr_db``, ``phy_rate_mbps``, ``best_mcs``).  The SINR needed by each
  MCS is the minimum sensitivity of the standard minus the noise the standard
  assumes for it (kTB with a 10 dB noise figure), so it includes the 5 dB of
  implementation margin of the standard.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["BANDS", "STANDARDS", "ChannelSpec", "channel_center_hz", "valid_channels", "channel_blocks",
           "spectral_mask_db", "overlap_factor", "overlap_matrix", "bands_overlap", "required_sinr_db",
           "phy_rate_mbps", "best_mcs", "mcs_table", "max_mcs", "max_streams", "effective_standard", "WifiError", "DFS_CHANNELS_5GHZ",
           "UNII4_CHANNELS_5GHZ"]


class WifiError(ValueError):
    """Invalid Wi-Fi channel, width or standard (message in Spanish, for the user)."""


# --------------------------------------------------------------------------- channels

BANDS = ("2.4", "5", "6")
WIDTHS_MHZ = (20, 40, 80, 160)
_BASE_MHZ = {"2.4": 2407.0, "5": 5000.0, "6": 5950.0}
CHANNELS_24 = tuple(range(1, 14))
CHANNELS_5 = (36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
              149, 153, 157, 161, 165, 169, 173, 177)
CHANNELS_6 = tuple(range(1, 234, 4))
DFS_CHANNELS_5GHZ = frozenset(range(52, 145, 4))          # UNII-2A and UNII-2C: need radar detection
UNII4_CHANNELS_5GHZ = frozenset((169, 173, 177))          # 5850-5895 MHz: allowed only in some countries

# 5 GHz bonded channels: blocks of 20 MHz channels and the centre channel of each block
_BLOCKS_5 = {
    40: [((36, 40), 38), ((44, 48), 46), ((52, 56), 54), ((60, 64), 62), ((100, 104), 102), ((108, 112), 110),
         ((116, 120), 118), ((124, 128), 126), ((132, 136), 134), ((140, 144), 142), ((149, 153), 151),
         ((157, 161), 159), ((165, 169), 167), ((173, 177), 175)],
    80: [((36, 48), 42), ((52, 64), 58), ((100, 112), 106), ((116, 128), 122), ((132, 144), 138),
         ((149, 161), 155), ((165, 177), 171)],
    160: [((36, 64), 50), ((100, 128), 114), ((149, 177), 163)],
}


def channel_blocks(band: str, width_mhz: int) -> List[Tuple[Tuple[int, ...], int]]:
    """
    Channels that can be used with that width, grouped by the band they occupy:
    a list of (20 MHz channels of the block, centre channel).  Any channel of a
    block can be the primary one; the occupied band is the same.
    """
    band, width = _check_band_width(band, width_mhz)
    if width == 20:
        channels = {"2.4": CHANNELS_24, "5": CHANNELS_5, "6": CHANNELS_6}[band]
        return [((c,), c) for c in channels]
    if band == "5":
        return [(tuple(range(lo, hi + 1, 4)), centre) for (lo, hi), centre in _BLOCKS_5[width]]
    # 6 GHz: blocks of n = W/20 channels starting at 1, 1 + 4n, ...; the centre is 2(n-1) above the first
    n = width // 20
    out = []
    for first in range(1, 234, 4 * n):
        last = first + 4 * (n - 1)
        if last <= 233:
            out.append((tuple(range(first, last + 1, 4)), first + 2 * (n - 1)))
    return out


def _check_band_width(band, width_mhz) -> Tuple[str, int]:
    band = str(band)
    if band not in BANDS:
        raise WifiError(f"banda Wi-Fi {band!r} desconocida (2.4, 5 o 6 GHz)")
    if isinstance(width_mhz, bool) or width_mhz not in WIDTHS_MHZ:
        raise WifiError(f"ancho de canal {width_mhz!r} MHz no válido (20, 40, 80 o 160)")
    if band == "2.4" and width_mhz != 20:
        raise WifiError("en 2.4 GHz solo se admiten canales de 20 MHz (los de 40 MHz se enciman con todos)")
    return band, int(width_mhz)


def valid_channels(band: str, width_mhz: int) -> List[int]:
    """Primary channels allowed with that width (every 20 MHz channel of every block)."""
    return [c for block, _ in channel_blocks(band, width_mhz) for c in block]


def channel_center_hz(band: str, channel: int, width_mhz: int = 20) -> float:
    """Centre frequency (Hz) of the band occupied by a channel whose primary 20 MHz channel is ``channel``."""
    for block, centre in channel_blocks(band, width_mhz):
        if channel in block:
            return (_BASE_MHZ[str(band)] + 5.0 * centre) * 1e6
    raise WifiError(f"el canal {channel!r} no existe en {band} GHz con {width_mhz} MHz "
                    f"(válidos: {', '.join(map(str, valid_channels(band, width_mhz)))})")


@dataclass(frozen=True)
class ChannelSpec:
    """Band occupied by a transmitter: centre and width; ``wifi`` selects the 802.11 mask (else a flat spectrum)."""
    center_hz: float
    width_hz: float
    wifi: bool = True

    @property
    def low_hz(self) -> float:
        return self.center_hz - self.width_hz / 2.0

    @property
    def high_hz(self) -> float:
        return self.center_hz + self.width_hz / 2.0


def bands_overlap(a: ChannelSpec, b: ChannelSpec) -> bool:
    """Whether the nominal bands of two channels intersect (they share spectrum: co-channel or overlapping)."""
    return a.low_hz < b.high_hz - 1.0 and b.low_hz < a.high_hz - 1.0


# --------------------------------------------------------------------------- spectral mask

def _mask_points(width_mhz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Breakpoints (offset from the centre in MHz, level in dBr) of the OFDM transmit mask, offset >= 0."""
    w = float(width_mhz)
    return (np.array([0.0, w / 2.0 - 1.0, w / 2.0 + 1.0, w, 1.5 * w]),
            np.array([0.0, 0.0, -20.0, -28.0, -40.0]))


def spectral_mask_db(offset_hz, width_hz) -> np.ndarray:
    """Level (dBr) of the 802.11 OFDM transmit mask at ``offset_hz`` from the centre; -inf beyond 1.5 W."""
    x, level = _mask_points(width_hz / 1e6)
    off = np.abs(np.asarray(offset_hz, dtype=float)) / 1e6
    return np.where(off <= x[-1], np.interp(off, x, level), -np.inf)


def _psd_integral(spec: ChannelSpec, lo_hz: float, hi_hz: float) -> float:
    """
    Power of transmitter ``spec`` (relative to a 0 dBr density, in MHz) between
    ``lo_hz`` and ``hi_hz``: exact integral of the mask, which is linear in dB on
    each segment (so exponential in power).
    """
    if hi_hz <= lo_hz:
        return 0.0
    a, b = (lo_hz - spec.center_hz) / 1e6, (hi_hz - spec.center_hz) / 1e6
    if not spec.wifi:                     # flat spectrum over the nominal band
        half = spec.width_hz / 2e6
        return max(0.0, min(b, half) - max(a, -half))
    x, level = _mask_points(spec.width_hz / 1e6)
    xs = np.concatenate([-x[::-1], x[1:]])
    ls = np.concatenate([level[::-1], level[1:]])
    total = 0.0
    for k in range(len(xs) - 1):
        x0, x1 = xs[k], xs[k + 1]
        u0, u1 = max(a, x0), min(b, x1)
        if u1 <= u0:
            continue
        slope = (ls[k + 1] - ls[k]) / (x1 - x0)
        l0 = ls[k] + slope * (u0 - x0)
        l1 = ls[k] + slope * (u1 - x0)
        if abs(slope) < 1e-12:
            total += (u1 - u0) * 10.0 ** (l0 / 10.0)
        else:
            total += (10.0 ** (l1 / 10.0) - 10.0 ** (l0 / 10.0)) / (slope * math.log(10.0) / 10.0)
    return total


def overlap_factor(interferer: ChannelSpec, receiver: ChannelSpec) -> float:
    """
    Fraction (linear, 0..1) of the power radiated by ``interferer`` that falls inside
    the band of ``receiver`` (ideal filter over its nominal width).  Co-channel
    transmitters of the same width give about 0.97 (the mask leaks a little outside
    the nominal band); partially overlapping 2.4 GHz channels give between that and
    about -30 dB, and channels far apart give 0.
    """
    total = _psd_integral(interferer, interferer.center_hz - 2.0 * interferer.width_hz,
                          interferer.center_hz + 2.0 * interferer.width_hz)
    inside = _psd_integral(interferer, receiver.low_hz, receiver.high_hz)
    return inside / total if total > 0 else 0.0


def overlap_matrix(channels: Sequence[ChannelSpec]) -> np.ndarray:
    """F[s, j]: fraction of the power of transmitter j inside the band of a receiver served by s (F[s, s] = 0)."""
    n = len(channels)
    out = np.zeros((n, n))
    cache: Dict[Tuple[ChannelSpec, ChannelSpec], float] = {}
    for s in range(n):
        for j in range(n):
            if s != j:
                key = (channels[j], channels[s])
                if key not in cache:
                    cache[key] = overlap_factor(channels[j], channels[s])
                out[s, j] = cache[key]
    return out


# --------------------------------------------------------------------------- MCS and PHY rate

STANDARDS = {
    # name: (label, maximum MCS, data subcarriers per width, useful symbol (us), guard intervals (us), max streams)
    "n": ("802.11n (Wi-Fi 4)", 7, {20: 52, 40: 108}, 3.2, (0.8, 0.4), 4),
    "ac": ("802.11ac (Wi-Fi 5)", 9, {20: 52, 40: 108, 80: 234, 160: 468}, 3.2, (0.8, 0.4), 8),
    "ax": ("802.11ax (Wi-Fi 6/6E)", 11, {20: 234, 40: 468, 80: 980, 160: 1960}, 12.8, (0.8, 1.6, 3.2), 8),
}
# (bits per subcarrier, coding rate) of MCS 0..11
_MODULATION = [(1, 1 / 2), (2, 1 / 2), (2, 3 / 4), (4, 1 / 2), (4, 3 / 4), (6, 2 / 3), (6, 3 / 4), (6, 5 / 6),
               (8, 3 / 4), (8, 5 / 6), (10, 3 / 4), (10, 5 / 6)]
_MODULATION_NAMES = ["BPSK 1/2", "QPSK 1/2", "QPSK 3/4", "16-QAM 1/2", "16-QAM 3/4", "64-QAM 2/3", "64-QAM 3/4",
                     "64-QAM 5/6", "256-QAM 3/4", "256-QAM 5/6", "1024-QAM 3/4", "1024-QAM 5/6"]
# Minimum receiver sensitivity at 20 MHz (dBm) of 802.11n/ac/ax for MCS 0..11; +3 dB per doubling of the width
MIN_SENSITIVITY_20MHZ_DBM = (-82, -79, -77, -74, -70, -66, -65, -64, -59, -57, -54, -52)
# Noise that the standard assumes for those sensitivities at 20 MHz: kTB + 10 dB of noise figure
_STANDARD_NOISE_20MHZ_DBM = -173.975 + 10.0 * math.log10(20e6) + 10.0
# VHT combinations that the standard excludes ((width, MCS) -> numbers of streams)
_VHT_INVALID = {(20, 9): {1, 2, 4, 5, 7, 8}, (80, 6): {3, 7}, (80, 9): {6}, (160, 9): {3}}


def required_sinr_db(mcs: int) -> float:
    """SINR (dB) that MCS ``mcs`` needs: IEEE minimum sensitivity minus the noise the standard assumes."""
    return MIN_SENSITIVITY_20MHZ_DBM[mcs] - _STANDARD_NOISE_20MHZ_DBM


def _check_standard(standard: str, width_mhz: int, streams: int, gi_us: float):
    if standard not in STANDARDS:
        raise WifiError(f"estándar Wi-Fi {standard!r} desconocido (n, ac o ax)")
    label, _, subcarriers, _, gis, max_streams = STANDARDS[standard]
    if width_mhz not in subcarriers:
        raise WifiError(f"{label} no admite canales de {width_mhz} MHz")
    if isinstance(streams, bool) or not isinstance(streams, int) or not 1 <= streams <= max_streams:
        raise WifiError(f"{label}: el número de flujos espaciales debe estar entre 1 y {max_streams}")
    if not any(abs(gi_us - g) < 1e-9 for g in gis):
        raise WifiError(f"{label}: el intervalo de guarda debe ser {' o '.join(f'{g:g}' for g in gis)} µs")


def phy_rate_mbps(standard: str, mcs: int, width_mhz: int, streams: int = 1, gi_us: float = 0.8) -> Optional[float]:
    """
    PHY data rate (Mbit/s): data subcarriers x bits per subcarrier x coding rate x
    streams / symbol duration.  None for a combination that the standard excludes.
    """
    _check_standard(standard, width_mhz, streams, gi_us)
    _, top, subcarriers, t_useful, _, _ = STANDARDS[standard]
    if not 0 <= mcs <= top:
        return None
    if standard == "ac" and streams in _VHT_INVALID.get((width_mhz, mcs), ()):
        return None
    bits, rate = _MODULATION[mcs]
    return subcarriers[width_mhz] * bits * rate * streams / (t_useful + gi_us)


def effective_standard(standard: str, band: str) -> str:
    """Standard that an AP of that generation uses in that band (Wi-Fi 5 works as Wi-Fi 4 in 2.4 GHz)."""
    if str(band) == "2.4" and standard == "ac":
        return "n"
    return standard


def max_streams(standard: str) -> int:
    return STANDARDS[standard][5]


def max_mcs(standard: str) -> int:
    return STANDARDS[standard][1]


def mcs_table(standard: str, width_mhz: int, streams: int = 1, gi_us: float = 0.8) -> List[Dict]:
    """MCS of a configuration: index, modulation, SINR required (dB) and rate (Mbit/s) of the valid ones."""
    rows = []
    for mcs in range(max_mcs(standard) + 1):
        rate = phy_rate_mbps(standard, mcs, width_mhz, streams, gi_us)
        if rate is not None:
            rows.append({"mcs": mcs, "modulation": _MODULATION_NAMES[mcs], "sinr_db": required_sinr_db(mcs),
                         "rate_mbps": rate})
    return rows


def best_mcs(sinr_db, standard: str, width_mhz: int, streams: int = 1, gi_us: float = 0.8):
    """
    Highest valid MCS that each SINR allows, and its PHY rate: arrays like
    ``sinr_db`` (MCS -1 and rate 0 below MCS 0; NaN where the SINR is NaN).
    """
    sinr = np.asarray(sinr_db, dtype=float)
    mcs = np.full(sinr.shape, -1.0)
    rate = np.zeros(sinr.shape)
    for row in mcs_table(standard, width_mhz, streams, gi_us):
        ok = sinr >= row["sinr_db"]
        mcs = np.where(ok, row["mcs"], mcs)
        rate = np.where(ok, row["rate_mbps"], rate)
    nan = np.isnan(sinr)
    return np.where(nan, np.nan, mcs), np.where(nan, np.nan, rate)
