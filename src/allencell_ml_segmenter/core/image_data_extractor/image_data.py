from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Optional


@dataclass
class ImageData:
    dim_x: Optional[int]
    dim_y: Optional[int]
    dim_z: Optional[int]
    channels: Optional[int]
    np_data: Optional[np.ndarray]
    path: Path
