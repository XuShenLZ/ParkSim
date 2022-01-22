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
    Kp_braking: float = field(default=5.0) # braking gain
    dt: float = field(default=0.1) # [s] time difference