from pathlib import Path

import allencell_ml_segmenter
from allencell_ml_segmenter.core.directories import Directories


def test_get_assets_dir():
    """
    Test that the assets directory is returned correctly.
    """
    # ARRANGE
    assets_path: Path = Path(allencell_ml_segmenter.__file__).parent / "assets"

    # ACT/ASSERT
    assert Directories.get_assets_dir() == assets_path


def test_get_style_dir():
    """
    Test that the styles directory is returned correctly.
    """
    # ARRANGE
    styles_path: Path = Path(allencell_ml_segmenter.__file__).parent / "styles"

    # ACT/ASSERT
    assert Directories.get_style_dir() == styles_path
