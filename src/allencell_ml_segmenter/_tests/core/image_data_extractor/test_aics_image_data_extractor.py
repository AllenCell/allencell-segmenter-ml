from pathlib import Path

import numpy as np
import pytest
from aicsimageio import AICSImage

import allencell_ml_segmenter
from allencell_ml_segmenter.core.image_data_extractor import (
    AICSImageDataExtractor,
)


def test_extract_image_data_all() -> None:
    # Arrange
    extractor = AICSImageDataExtractor.global_instance()
    t1_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
        / "test_3_channels.tiff"
    )
    expected_img_data = AICSImage(t1_path).data

    # Act
    image_data = extractor.extract_image_data(t1_path, dims=True, np_data=True)
    assert image_data.channels == 3
    assert image_data.dim_x == 2
    assert image_data.dim_y == 2
    assert image_data.dim_z == 2
    assert np.array_equal(image_data.np_data, expected_img_data)
    assert image_data.path == t1_path

def test_extract_image_data_dims_only() -> None:
    # Arrange
    extractor = AICSImageDataExtractor.global_instance()
    t1_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
        / "test_3_channels.tiff"
    )

    # Act
    image_data = extractor.extract_image_data(
        t1_path, dims=True, np_data=False
    )
    assert image_data.channels == 3
    assert image_data.dim_x == 2
    assert image_data.dim_y == 2
    assert image_data.dim_z == 2
    assert image_data.np_data is None
    assert image_data.path == t1_path
