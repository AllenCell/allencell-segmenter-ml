from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ImageData:
    dim_x: int
    dim_y: int
    dim_z: int
    channels: int
    np_data: np.ndarray
    path: Path
