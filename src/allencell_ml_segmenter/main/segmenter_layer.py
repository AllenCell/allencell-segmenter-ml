from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Optional


@dataclass
class SegmenterLayer:
    name: str


@dataclass
class ShapesLayer(SegmenterLayer):
    data: np.ndarray

@dataclass
class Source(SegmenterLayer):
    path: Optional[Path] = None

@dataclass
class ImageLayer(SegmenterLayer):
    path: Optional[Path] = None
    data: Optional[np.ndarray] = None
    source: Optional[Source] = Source


@dataclass
class LabelsLayer(SegmenterLayer):
    pass
