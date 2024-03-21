from abc import ABC, abstractmethod
from pathlib import Path
from allencell_ml_segmenter.core.image_data_extractor import ImageData


class IImageDataExtractor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract_image_data(
        self, img_path: Path, dims: bool = True, np_data: bool = True
    ) -> ImageData:
        pass

    @classmethod
    @abstractmethod
    def global_instance(cls):
        pass
