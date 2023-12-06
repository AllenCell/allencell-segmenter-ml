from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurationRecord:
    raw_file: Path
    seg1: Path
    seg2: Path
    excluding_mask: str
    merging_mask: str
    base_image_index: str
    to_use: bool
