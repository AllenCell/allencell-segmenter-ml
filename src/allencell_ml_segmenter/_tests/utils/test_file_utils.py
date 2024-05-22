from pathlib import Path

import allencell_ml_segmenter
from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord

import pytest
from unittest.mock import patch, mock_open, MagicMock
import builtins
from typing import List, Set, Tuple
import numpy as np


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


@patch("builtins.open", new_callable=mock_open)
def test_write_curation_record_writes_to_disk(open_mock: MagicMock):
    """
    Verify that a call to the public write_curation_record will result in train, test, val csvs
    being written to disk, and excluding/merging masks being written to disk
    """
    fake_curation_record: List[CurationRecord] = [
        CurationRecord(
            Path("raw_1"),
            Path("seg1_1"),
            Path("seg2_1"),
            np.asarray([[[1, 2], [3, 4], [5, 6]]]),
            None,
            "seg1",
            True,
        ),
        CurationRecord(
            Path("raw_2"),
            Path("seg1_2"),
            Path("seg2_2"),
            None,
            None,
            "seg1",
            True,
        ),
        CurationRecord(
            Path("raw_3"),
            Path("seg1_3"),
            Path("seg2_3"),
            None,
            None,
            "seg1",
            True,
        ),
        CurationRecord(
            Path("raw_4"),
            Path("seg1_4"),
            Path("seg2_4"),
            None,
            None,
            "seg1",
            False,
        ),
        CurationRecord(
            Path("raw_5"),
            Path("seg1_5"),
            Path("seg2_5"),
            None,
            np.asarray([[[1, 2], [3, 4], [5, 6]]]),
            "seg1",
            True,
        ),
    ]
    fake_csv_path: Path = Path("fakecsv")
    fake_mask_path: Path = Path("fakemask")
    FileUtils.write_curation_record(
        fake_curation_record, fake_csv_path, fake_mask_path
    )
    print(open_mock.mock_calls)

    # Note: these assertions rely on naming conventions, somewhat brittle
    open_mock.assert_any_call(fake_csv_path / "train.csv", "w")
    open_mock.assert_any_call(fake_csv_path / "test.csv", "w")
    open_mock.assert_any_call(fake_csv_path / "val.csv", "w")

    # for some reason, np.save(path: Path) ends up calling open with a str instead of a path
    open_mock.assert_any_call(
        f"{str(fake_mask_path)}/excluding_masks/excluding_mask_raw_1.npy", "wb"
    )
    open_mock.assert_any_call(
        f"{str(fake_mask_path)}/merging_masks/merging_mask_raw_5.npy", "wb"
    )


def _get_curation_lists_from_mock_calls(
    mock_calls,
) -> Tuple[List[CurationRecord], List[CurationRecord], List[CurationRecord]]:
    train: List[CurationRecord] = None
    test: List[CurationRecord] = None
    val: List[CurationRecord] = None

    for c in mock_calls:
        if "train.csv" in str(c.args[1]):
            train = c.args[0]
        elif "test.csv" in str(c.args[1]):
            test = c.args[0]
        elif "val.csv" in str(c.args[1]):
            val = c.args[0]

    return train, test, val


def _append_default_record(cr: List[CurationRecord], to_use=True) -> None:
    cr.append(
        CurationRecord(
            Path(f"raw_{len(cr)}"),
            Path(f"seg1_{len(cr)}"),
            Path(f"seg2_{len(cr)}"),
            None,
            None,
            "seg1",
            to_use,
        )
    )


def test_write_curation_record_split_sizes():
    fake_csv_path: Path = Path("fakecsv")
    fake_mask_path: Path = Path("fakemask")

    fake_curation_record: List[CurationRecord] = []
    _append_default_record(fake_curation_record)

    with patch(
        "allencell_ml_segmenter.utils.file_utils.FileUtils._write_curation_csv"
    ) as write_csv_mock:
        FileUtils.write_curation_record(
            fake_curation_record, fake_csv_path, fake_mask_path
        )
        train, test, val = _get_curation_lists_from_mock_calls(
            write_csv_mock.mock_calls
        )
        assert len(train) == 1
        assert len(test) == 0
        assert len(val) == 0

    _append_default_record(fake_curation_record)
    with patch(
        "allencell_ml_segmenter.utils.file_utils.FileUtils._write_curation_csv"
    ) as write_csv_mock:
        FileUtils.write_curation_record(
            fake_curation_record, fake_csv_path, fake_mask_path
        )
        train, test, val = _get_curation_lists_from_mock_calls(
            write_csv_mock.mock_calls
        )
        assert len(train) == 2
        assert len(test) == 0
        assert len(val) == 0

    _append_default_record(fake_curation_record)
    with patch(
        "allencell_ml_segmenter.utils.file_utils.FileUtils._write_curation_csv"
    ) as write_csv_mock:
        FileUtils.write_curation_record(
            fake_curation_record, fake_csv_path, fake_mask_path
        )
        train, test, val = _get_curation_lists_from_mock_calls(
            write_csv_mock.mock_calls
        )
        assert len(train) == 1
        assert len(test) == 2
        assert len(val) == 2

    for _ in range(97):
        _append_default_record(fake_curation_record)
    with patch(
        "allencell_ml_segmenter.utils.file_utils.FileUtils._write_curation_csv"
    ) as write_csv_mock:
        FileUtils.write_curation_record(
            fake_curation_record, fake_csv_path, fake_mask_path
        )
        train, test, val = _get_curation_lists_from_mock_calls(
            write_csv_mock.mock_calls
        )
        assert len(train) == 90
        assert len(test) == 10
        assert len(val) == 10

    for _ in range(1200):
        _append_default_record(fake_curation_record)
    with patch(
        "allencell_ml_segmenter.utils.file_utils.FileUtils._write_curation_csv"
    ) as write_csv_mock:
        FileUtils.write_curation_record(
            fake_curation_record, fake_csv_path, fake_mask_path
        )
        train, test, val = _get_curation_lists_from_mock_calls(
            write_csv_mock.mock_calls
        )
        assert len(train) == 1200
        assert len(test) == 100
        assert len(val) == 100


def test_write_curation_record_includes_only_selected_images():
    fake_csv_path: Path = Path("fakecsv")
    fake_mask_path: Path = Path("fakemask")

    fake_curation_record: List[CurationRecord] = []
    for _ in range(20):
        _append_default_record(fake_curation_record)
    for _ in range(10):
        _append_default_record(fake_curation_record, to_use=False)

    with patch(
        "allencell_ml_segmenter.utils.file_utils.FileUtils._write_curation_csv"
    ) as write_csv_mock:
        FileUtils.write_curation_record(
            fake_curation_record, fake_csv_path, fake_mask_path
        )
        train, test, val = _get_curation_lists_from_mock_calls(
            write_csv_mock.mock_calls
        )
        assert len(train) == 18
        assert len(test) == 2
        assert len(val) == 2


def test_write_curation_record_does_not_include_duplicates():
    fake_csv_path: Path = Path("fakecsv")
    fake_mask_path: Path = Path("fakemask")

    fake_curation_record: List[CurationRecord] = []
    for _ in range(20):
        _append_default_record(fake_curation_record)

    with patch(
        "allencell_ml_segmenter.utils.file_utils.FileUtils._write_curation_csv"
    ) as write_csv_mock:
        FileUtils.write_curation_record(
            fake_curation_record, fake_csv_path, fake_mask_path
        )
        train, test, val = _get_curation_lists_from_mock_calls(
            write_csv_mock.mock_calls
        )
        unique_paths: Set[Path] = set()
        for record in train:
            assert record.raw_file not in unique_paths
            unique_paths.add(record.raw_file)
        for record in val:
            assert record.raw_file not in unique_paths
            unique_paths.add(record.raw_file)


def test_write_curation_record_test_val_same():
    fake_csv_path: Path = Path("fakecsv")
    fake_mask_path: Path = Path("fakemask")

    fake_curation_record: List[CurationRecord] = []
    for _ in range(20):
        _append_default_record(fake_curation_record)

    with patch(
        "allencell_ml_segmenter.utils.file_utils.FileUtils._write_curation_csv"
    ) as write_csv_mock:
        FileUtils.write_curation_record(
            fake_curation_record, fake_csv_path, fake_mask_path
        )
        train, test, val = _get_curation_lists_from_mock_calls(
            write_csv_mock.mock_calls
        )
        unique_paths: Set[Path] = set()
        for record in test:
            unique_paths.add(record.raw_file)
        for record in val:
            assert record.raw_file in unique_paths
