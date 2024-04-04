import pytest
from pathlib import Path
import allencell_ml_segmenter
from allencell_ml_segmenter.core.channel_extraction import *


def test_extract_channels_from_image():
    img: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
        / "test_3_channels.tiff"
    )
    assert extract_channels_from_image(img) == 3


def test_get_img_path_from_csv():
    csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
        / "test_csv.csv"
    )
    img_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
        / "test_3_channels.tiff"
    )
    assert get_img_path_from_csv(csv_path).samefile(img_path)
