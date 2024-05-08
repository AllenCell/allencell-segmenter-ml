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
    all_files = FileUtils.get_all_files_in_dir_ignore_hidden(folder)

    # assert
    assert len(all_files) == 3
    assert all_files[0].name == "t1.tiff"
    assert all_files[1].name == "t2.tiff"
    assert all_files[2].name == "t3.tiff"


def test_get_all_files_in_dir_with_hidden_files() -> None:
    # arrange
    folder: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "img_folder_with_hidden_files"
    )

    # act
    all_files = FileUtils.get_all_files_in_dir_ignore_hidden(folder)

    # assert
    assert len(all_files) == 1
    assert all_files[0].name == "t1.tiff"


def test_get_img_path_from_folder():
    folder: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "img_folder"
    )
    t1: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "img_folder"
        / "t1.tiff"
    )
    t2: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "img_folder"
        / "t2.tiff"
    )
    t3: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "img_folder"
        / "t3.tiff"
    )
    img: Path = FileUtils.get_img_path_from_folder(folder)
    assert img.samefile(t1) or img.samefile(t2) or img.samefile(t3)


def test_get_img_path_from_folder_hidden_files():
    folder: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "img_folder_with_hidden_files"
    )
    t1: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "img_folder_with_hidden_files"
        / "t1.tiff"
    )
    img: Path = FileUtils.get_img_path_from_folder(folder)
    assert img.samefile(t1)
