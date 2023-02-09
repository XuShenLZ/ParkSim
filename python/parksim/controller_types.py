from dataclasses import dataclass, field
import numpy as np

from parksim.pytypes import PythonMsg


@dataclass
class StanleyParams(PythonMsg):
    """
    Parameters for Stanley controller
    """

    k: float = field(default=0.5)  # control gain
    Kp: float = field(default=1.0)  # speed proportional gain
    Kp_braking: float = field(default=5.0)  # braking gain
    dt: float = field(default=0.1)  # [s] time difference


@dataclass
class MPCParams(PythonMsg):
    """
    Parameters for MPC controller
    """

    N: int = field(default=10)
    dt: float = field(default=0.1)

    Q: np.ndarray = field(default=np.array([1, 1, 0, 0]))
    R: np.ndarray = field(default=np.array([0, 0]))

    obs_buffer_size: int = field(default=6)

    static_distance: float = field(
        default=0.01
    )  # distance to static obstacles (e.g. parked cars)
    static_radius: float = field(
        default=10
    )  # radius another static obstacle must be in before we include it in collision avoidance
