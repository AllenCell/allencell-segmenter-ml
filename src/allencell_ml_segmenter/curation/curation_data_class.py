from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurationRecord:
    raw_file: Path
    seg1: Path
    excluding_mask: Path
    to_use: bool
