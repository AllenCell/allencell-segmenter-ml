from abc import ABC, abstractmethod
from .i_curation_image_loader import ICurationImageLoader


class ICurationImageLoaderFactory(ABC):
    @abstractmethod
    def create(self) -> ICurationImageLoader:
        pass
