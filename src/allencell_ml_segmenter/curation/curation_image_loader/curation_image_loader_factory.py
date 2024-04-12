from .i_curation_image_loader import ICurationImageLoader
from .i_curation_image_loader_factory import ICurationImageLoaderFactory
from .curation_image_loader import CurationImageLoader
from typing import Optional, List
from pathlib import Path


class CurationImageLoaderFactory(ICurationImageLoaderFactory):
    def create(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]] = None,
    ) -> ICurationImageLoader:
        return CurationImageLoader(raw_images, seg1_images, seg2_images)
