"""
OpenRPS - Open Radiowave Propagation Simulator
An open-source radio propagation modeling and multipath channel simulation library.
"""

from .models import (
    PropagationModel,
    FreeSpacePathLoss,
    TwoRayGroundReflection,
    OkumuraHata,
    Cost231Hata,
    LogDistancePathLoss,
    ModelRangeWarning,
    fresnel_reflection_coefficient,
)
from .multipath import (
    MultipathRay,
    ChannelImpulseResponse,
)
from .raytracer import (
    Transmitter,
    PlanarReflector,
    DiffractionEdge,
    RayTracer,
    RayField,
)
from .simulation import (
    PropagationSimulator,
    SimulationResult,
    thermal_noise_dbm,
)
from .visualizer import (
    MapVisualizer,
)
from .measurements import (
    Measurement,
    MeasurementError,
    read_csv as read_measurements_csv,
    compare as compare_measurements,
    calibrate,
)
from .wifi import (
    ChannelSpec,
    WifiError,
    channel_center_hz,
    valid_channels,
    overlap_factor,
    phy_rate_mbps,
    required_sinr_db,
    best_mcs,
)
from .channel_plan import ChannelPlan, plan_channels

__version__ = "1.8.0"
