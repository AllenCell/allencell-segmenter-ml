from pathlib import Path
import pytest

import allencell_ml_segmenter
from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.file_writer import (
    IFileWriter,
    FakeFileWriter,
)
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord

from typing import List, Set
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
    assert len(all_files) == 5
    assert all_files[0].name == "t1.tiff"
    assert all_files[1].name == "t2.tiff"
    assert all_files[2].name == "t3.tiff"
    assert all_files[3].name == "t4.tiff"
    assert all_files[4].name == "t5.tiff"


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
    img: Path = FileUtils.get_img_path_from_folder(folder)
    assert any(
        [
            img.samefile(exist_img)
            for exist_img in [folder / f"t{i + 1}.tiff" for i in range(5)]
        ]
    )


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


def _generate_default_records(num_records: int) -> List[CurationRecord]:
    cr: List[CurationRecord] = []
    for i in range(num_records):
        cr.append(
            CurationRecord(
                Path(f"raw_{i}"),
                Path(f"seg1_{i}"),
                Path(f"seg2_{i}"),
                None,
                None,
                "seg1",
                True,
            )
        )
    return cr


def test_write_curation_record_closes_files():
    # Arrange
    fake_writer: IFileWriter = FakeFileWriter()
    f_utils: FileUtils = FileUtils(fake_writer)
    fake_curation_record: List[CurationRecord] = _generate_default_records(10)

    # Act
    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    # Assert
    assert not fake_writer.csv_state[EXP_TRAIN_PATH]["open"]
    assert not fake_writer.csv_state[EXP_TEST_PATH]["open"]
    assert not fake_writer.csv_state[EXP_VAL_PATH]["open"]


def test_write_curation_record_writes_mask_to_disk():
    """
    Verify that a call to the public write_curation_record will result in excluding/merging masks being written to disk
    """
    # Arrange
    fake_writer: IFileWriter = FakeFileWriter()
    f_utils: FileUtils = FileUtils(fake_writer)
    fake_curation_record: List[CurationRecord] = _generate_default_records(4)
    masks: List[np.ndarray] = [
        np.asarray([[[1, 2], [3, 4], [5, 6]]]),
        np.asarray([[[2, 2], [3, 4], [5, 6]]]),
        np.asarray([[[3, 2], [3, 4], [5, 6]]]),
        np.asarray([[[4, 2], [3, 4], [5, 6]]]),
    ]
    # first record has just an excluding mask, second just a merging mask, third both
    fake_curation_record[0].excluding_mask = masks[0]
    fake_curation_record[1].merging_mask = masks[1]
    fake_curation_record[2].excluding_mask = masks[2]
    fake_curation_record[2].merging_mask = masks[3]

    expected_save_paths: List[Path] = [
        FAKE_MASK_PATH / "excluding_masks" / "excluding_mask_raw_0.npy",
        FAKE_MASK_PATH / "merging_masks" / "merging_mask_raw_1.npy",
        FAKE_MASK_PATH / "excluding_masks" / "excluding_mask_raw_2.npy",
        FAKE_MASK_PATH / "merging_masks" / "merging_mask_raw_2.npy",
    ]

    # Act
    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    # Assert
    assert len(fake_writer.np_save_state) == len(expected_save_paths)
    # verify that the expected mask was saved to the expected save path
    for mask, save_path in zip(masks, expected_save_paths):
        assert save_path in fake_writer.np_save_state
        assert np.array_equal(mask, fake_writer.np_save_state[save_path])


def test_write_curation_record_split_sizes():
    """
    Test that the split sizes are as expected for curation records
    of different lengths.
    """
    # Arrange
    fake_writer: IFileWriter = FakeFileWriter()
    f_utils: FileUtils = FileUtils(fake_writer)

    # Act / Assert
    with pytest.raises(RuntimeError) as e:
        f_utils.write_curation_record(
            _generate_default_records(1), FAKE_CSV_PATH, FAKE_MASK_PATH
        )

    with pytest.raises(RuntimeError) as e:
        f_utils.write_curation_record(
            _generate_default_records(3), FAKE_CSV_PATH, FAKE_MASK_PATH
        )

    with pytest.raises(RuntimeError) as e:
        records: list[CurationRecord] = _generate_default_records(4)
        records[0].to_use = False
        f_utils.write_curation_record(records, FAKE_CSV_PATH, FAKE_MASK_PATH)

    # Act
    f_utils.write_curation_record(
        _generate_default_records(4), FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    # Assert
    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 3
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 3
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 3

    # Act
    f_utils.write_curation_record(
        _generate_default_records(100), FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    # Assert
    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 91
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 11
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 11

    # Act
    f_utils.write_curation_record(
        _generate_default_records(1300), FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    # Assert
    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 1201
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 101
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 101


def test_write_curation_record_includes_only_selected_images():
    """
    Test that images marked as not to_use are not included in the records
    to be written.
    """
    # Arrange
    fake_writer: IFileWriter = FakeFileWriter()
    f_utils: FileUtils = FileUtils(fake_writer)

    fake_records: List[CurationRecord] = _generate_default_records(40)

    # mark half of the records as not to use
    for i in range(0, len(fake_records), 2):
        fake_records[i].to_use = False

    # Act
    f_utils.write_curation_record(fake_records, FAKE_CSV_PATH, FAKE_MASK_PATH)

    # Assert
    # there are 20 records to use, so we expect 18 for training 2 shared between test/val (plus header rows)
    assert len(fake_writer.csv_state[EXP_TRAIN_PATH]["rows"]) == 19
    assert len(fake_writer.csv_state[EXP_TEST_PATH]["rows"]) == 3
    assert len(fake_writer.csv_state[EXP_VAL_PATH]["rows"]) == 3


def test_write_curation_record_does_not_include_duplicates():
    """
    Test that the records in train and val are mutually exclusive. And that there
    are no duplicates within train or val.
    """
    # Arrange
    fake_writer: IFileWriter = FakeFileWriter()
    f_utils: FileUtils = FileUtils(fake_writer)

    fake_curation_record: List[CurationRecord] = _generate_default_records(20)

    # Act
    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    # Assert
    # expect that the rows in the training and val csv are mutually exclusive
    row_minus_index = lambda row: ",".join(row[1:])
    unique_rows: Set[str] = set()
    for row in fake_writer.csv_state[EXP_TRAIN_PATH]["rows"][1:]:
        assert row_minus_index(row) not in unique_rows
        unique_rows.add(row_minus_index(row))
    for row in fake_writer.csv_state[EXP_VAL_PATH]["rows"][1:]:
        assert row_minus_index(row) not in unique_rows
        unique_rows.add(row_minus_index(row))


def test_write_curation_record_test_val_same():
    """
    Test that the records in val and test are the same.
    """
    # Arrange
    fake_writer: IFileWriter = FakeFileWriter()
    f_utils: FileUtils = FileUtils(fake_writer)

    fake_curation_record: List[CurationRecord] = _generate_default_records(20)

    # Act
    f_utils.write_curation_record(
        fake_curation_record, FAKE_CSV_PATH, FAKE_MASK_PATH
    )

    # Assert
    # expect that rows in val and test are the same
    row_minus_index = lambda row: ",".join(row[1:])
    unique_rows: Set[str] = set()
    for row in fake_writer.csv_state[EXP_TEST_PATH]["rows"][1:]:
        unique_rows.add(row_minus_index(row))
    for row in fake_writer.csv_state[EXP_VAL_PATH]["rows"][1:]:
        assert row_minus_index(row) in unique_rows


def test_count_images_in_csv_folder_one_csv() -> None:
    # Arrange
    folder: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
    )

    # Act / Assert
    # expect one since there is only one unique raw file in this directory
    assert FileUtils.count_images_in_csv_folder(folder) == 1


def test_count_images_in_csv_folder_multiple_csv() -> None:
    """
    A more realistic test of the count function on a folder with a train, test, and val csv
    """
    # Arrange
    folder: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "multiple_csv"
    )

    # Act / Assert
    # expect 6 since train has 3 unique, test has 3 unique, val is a copy of test
    assert FileUtils.count_images_in_csv_folder(folder) == 6


def test_get_min_loss_from_csv() -> None:
    """
    Ensure the function returns the expected minimum loss from a well-formatted
    metrics csv file.
    """
    # Arrange
    csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
        / "metrics.csv"
    )

    # Act / Assert
    assert FileUtils.get_min_loss_from_csv(csv_path) == 0.9382843971252441


def test_get_min_loss_from_csv_invalid_csv() -> None:
    """
    Ensure the function returns none when the CSV doesn't contain the expected
    loss column.
    """
    # Arrange
    csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
        / "test_csv.csv"
    )

    # Act / Assert
    assert FileUtils.get_min_loss_from_csv(csv_path) is None


def test_get_min_loss_from_csv_empty_loss_col() -> None:
    """
    Ensure the function returns none when the CSV's loss column is empty.
    """
    # Arrange
    csv_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
        / "metrics_empty_loss.csv"
    )

    # Act / Assert
    assert FileUtils.get_min_loss_from_csv(csv_path) is None
