import pytest
import allencell_ml_segmenter
from allencell_ml_segmenter.core.channel_extraction import *


def test_get_img_path_from_csv():
    csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
        / "train.csv"
    )
    img_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
        / "test_3_channels.tiff"
    )
    assert get_img_path_from_csv(csv_path).samefile(img_path)
