from pathlib import Path


class CytoDlConfig():
    def __init__(self, path: Path):
        self.path = path

    def get_path(self) -> Path:
        return self.path