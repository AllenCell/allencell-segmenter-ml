from pathlib import Path

import allencell_ml_segmenter
from allencell_ml_segmenter.utils.file_utils import FileUtils
import pytest

def test_get_all_files_in_dir() -> None:
    # arrange
    folder: Path = (
            Path(allencell_ml_segmenter.__file__).parent
            / "_tests"
            / "test_files"
            / "img_folder"
    )

    # act
    all_files = FileUtils.get_all_files_in_dir(folder)

    # assert
    assert len(all_files) == 2
    assert all_files[0].name == "t1.tiff"
    assert all_files[1].name == "t2.tiff"
def test_get_all_files_in_dir_ignore_hidden_files() -> None:
    # arrange
    folder: Path = (
            Path(allencell_ml_segmenter.__file__).parent
            / "_tests"
            / "test_files"
            / "img_folder_with_hidden_files"
    )

    # act
    all_files = FileUtils.get_all_files_in_dir(folder, ignore_hidden=True)

    # assert
    assert len(all_files) == 1
    assert all_files[0].name =="t1.tiff"


