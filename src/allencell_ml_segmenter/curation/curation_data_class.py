from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CurationRecord:
    raw_file: Path
    seg1: Path
    seg2: Optional[Path]
    excluding_mask: str
    merging_mask: str
    base_image_index: str
    to_use: bool
