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
    RayTracer,
    RayField,
)
from .simulation import (
    PropagationSimulator,
    SimulationResult,
)
from .visualizer import (
    MapVisualizer,
)

__version__ = "1.1.0"
