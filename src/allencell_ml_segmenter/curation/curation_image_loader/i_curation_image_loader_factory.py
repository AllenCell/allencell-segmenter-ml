from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
from .i_curation_image_loader import ICurationImageLoader


class ICurationImageLoaderFactory(ABC):
    @abstractmethod
    def create(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]] = None,
    ) -> ICurationImageLoader:
        pass
