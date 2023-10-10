from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurationRecord:
    raw_file: Path
    seg1: Path
    to_use: bool
