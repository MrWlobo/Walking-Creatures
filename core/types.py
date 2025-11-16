from dataclasses import dataclass
import numpy.typing as npt
import numpy as np

@dataclass
class RunResult:
    """Holds relevant results from a single simulation run."""
    time: int
    final_position: npt.NDArray[np.float64]