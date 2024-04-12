from abc import ABC, abstractmethod
from pathlib import Path


class IExtractorFactory(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self, img_path: Path):
        pass
