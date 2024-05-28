from pathlib import Path

import allencell_ml_segmenter
from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.file_writer import (
    IFileWriter,
    FakeFileWriter,
)
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord

from unittest.mock import patch, mock_open, MagicMock
from typing import List, Set, Tuple
import numpy as np
import platform


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


FAKE_CSV_PATH: Path = Path("fakecsv").resolve()
FAKE_MASK_PATH: Path = Path("fakemask").resolve()
EXP_TRAIN_PATH: Path = FAKE_CSV_PATH / "train.csv"
EXP_TEST_PATH: Path = FAKE_CSV_PATH / "test.csv"
EXP_VAL_PATH: Path = FAKE_CSV_PATH / "val.csv"


def _append_default_record(
    cr: List[CurationRecord], e_mask=None, m_mask=None, to_use=True
) -> None:
    """
    Given a list of records, appends a new record with file names determined
    by the length of the existing curation record. :param to_use: controls
    the to_use flag for the record.
    """
    cr.append(
        CurationRecord(
            Path(f"raw_{len(cr)}"),
            Path(f"seg1_{len(cr)}"),
            Path(f"seg2_{len(cr)}"),
            e_mask,
            m_mask,
            "seg1",
            to_use,
        )
    )


def test_write_curation_record_closes_files():
    fake_writer: IFileWriter = FakeFileWriter.global_instance()
    f_utils: FileUtils = FileUtils(fake_writer)
    fake_curation_record: List[CurationRecord] = []
    for _ in range(10):
        _append_default_record(fake_curation_record)

    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    assert not fake_writer.csv_state[EXP_TRAIN_PATH]["open"]
    assert not fake_writer.csv_state[EXP_TEST_PATH]["open"]
    assert not fake_writer.csv_state[EXP_VAL_PATH]["open"]


def test_write_curation_record_writes_mask_to_disk():
    """
    Verify that a call to the public write_curation_record will result in excluding/merging masks being written to disk
    """
    fake_writer: IFileWriter = FakeFileWriter.global_instance()
    f_utils: FileUtils = FileUtils(fake_writer)
    fake_curation_record: List[CurationRecord] = []
    _append_default_record(
        fake_curation_record, e_mask=np.asarray([[[1, 2], [3, 4], [5, 7]]])
    )
    _append_default_record(
        fake_curation_record, m_mask=np.asarray([[[1, 2], [3, 4], [5, 6]]])
    )
    _append_default_record(
        fake_curation_record,
        e_mask=np.asarray([[[8, 2], [3, 4], [5, 6]]]),
        m_mask=np.asarray([[[9, 2], [3, 4], [5, 6]]]),
    )

    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    assert len(fake_writer.np_save_state) == 4
    exp_excl_path: Path = (
        FAKE_MASK_PATH / "excluding_masks" / "excluding_mask_raw_0.npy"
    )
    assert exp_excl_path in fake_writer.np_save_state
    assert np.array_equal(
        np.asarray([[[1, 2], [3, 4], [5, 7]]]),
        fake_writer.np_save_state[exp_excl_path],
    )

    exp_merg_path: Path = (
        FAKE_MASK_PATH / "merging_masks" / "merging_mask_raw_1.npy"
    )
    assert exp_merg_path in fake_writer.np_save_state
    assert np.array_equal(
        np.asarray([[[1, 2], [3, 4], [5, 6]]]),
        fake_writer.np_save_state[exp_merg_path],
    )

    exp_excl_path = (
        FAKE_MASK_PATH / "excluding_masks" / "excluding_mask_raw_2.npy"
    )
    exp_merg_path = FAKE_MASK_PATH / "merging_masks" / "merging_mask_raw_2.npy"
    assert exp_excl_path in fake_writer.np_save_state
    assert np.array_equal(
        np.asarray([[[8, 2], [3, 4], [5, 6]]]),
        fake_writer.np_save_state[exp_excl_path],
    )
    assert exp_merg_path in fake_writer.np_save_state
    assert np.array_equal(
        np.asarray([[[9, 2], [3, 4], [5, 6]]]),
        fake_writer.np_save_state[exp_merg_path],
    )


def test_write_curation_record_split_sizes():
    """
    Test that the split sizes are as expected for curation records
    of different lengths.
    """
    fake_writer: IFileWriter = FakeFileWriter.global_instance()
    f_utils: FileUtils = FileUtils(fake_writer)

    fake_curation_record: List[CurationRecord] = []
    _append_default_record(fake_curation_record)

    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    # one header row, one content row
    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 2
    # one header row
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 1
    # one header row
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 1

    _append_default_record(fake_curation_record)
    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 3
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 1
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 1

    _append_default_record(fake_curation_record)
    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 2
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 3
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 3

    for _ in range(97):
        _append_default_record(fake_curation_record)
    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 91
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 11
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 11

    for _ in range(1200):
        _append_default_record(fake_curation_record)
    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 1201
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 101
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 101


def test_write_curation_record_includes_only_selected_images():
    """
    Test that images marked as not to_use are not included in the records
    to be written.
    """
    fake_writer: IFileWriter = FakeFileWriter.global_instance()
    f_utils: FileUtils = FileUtils(fake_writer)

    fake_curation_record: List[CurationRecord] = []
    for _ in range(20):
        _append_default_record(fake_curation_record)
    # expect that these additional 10 images will not add to the total number of images
    # since they are marked as not to_use
    for _ in range(10):
        _append_default_record(fake_curation_record, to_use=False)

    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 19
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 3
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 3


def test_write_curation_record_does_not_include_duplicates():
    """
    Test that the records in train and val are mutually exclusive. And that there
    are no duplicates within train or val. Uses raw image path as an id for records.
    """
    fake_writer: IFileWriter = FakeFileWriter.global_instance()
    f_utils: FileUtils = FileUtils(fake_writer)

    fake_curation_record: List[CurationRecord] = []
    for _ in range(20):
        _append_default_record(fake_curation_record)

    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )
    unique_rows: Set[str] = set()
    for row in fake_writer.csv_state[EXP_TRAIN_PATH]["rows"][1:]:
        row_minus_index: str = ",".join(row[1:])
        assert row_minus_index not in unique_rows
        unique_rows.add(row_minus_index)
    for row in fake_writer.csv_state[EXP_VAL_PATH]["rows"][1:]:
        row_minus_index: str = ",".join(row[1:])
        assert row_minus_index not in unique_rows
        unique_rows.add(row_minus_index)


def test_write_curation_record_test_val_same():
    """
    Test that the records in val and test are the same.
    Uses raw image path as an id for records.
    """
    fake_writer: IFileWriter = FakeFileWriter.global_instance()
    f_utils: FileUtils = FileUtils(fake_writer)

    fake_curation_record: List[CurationRecord] = []
    for _ in range(20):
        _append_default_record(fake_curation_record)

    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )
    unique_rows: Set[str] = set()
    for row in fake_writer.csv_state[EXP_TEST_PATH]["rows"][1:]:
        row_minus_index: str = ",".join(row[1:])
        unique_rows.add(row_minus_index)
    for row in fake_writer.csv_state[EXP_VAL_PATH]["rows"][1:]:
        row_minus_index: str = ",".join(row[1:])
        assert row_minus_index in unique_rows
