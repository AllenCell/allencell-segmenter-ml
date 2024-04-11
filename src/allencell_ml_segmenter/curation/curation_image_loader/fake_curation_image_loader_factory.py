from .i_curation_image_loader import ICurationImageLoader
from .i_curation_image_loader_factory import ICurationImageLoaderFactory
from .fake_curation_image_loader import FakeCurationImageLoader
from typing import Optional, List
from pathlib import Path


class FakeCurationImageLoaderFactory(ICurationImageLoaderFactory):
    def create(
        self,
        raw_images: List[Path],
        seg1_images: List[Path],
        seg2_images: Optional[List[Path]] = None,
    ) -> ICurationImageLoader:
        return FakeCurationImageLoader(raw_images, seg1_images, seg2_images)
