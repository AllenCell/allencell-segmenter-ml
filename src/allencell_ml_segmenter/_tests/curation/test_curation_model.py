from typing import List

import pytest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.curation.curation_model import CurationModel, CurationView
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import FakeExperimentsModel
from allencell_ml_segmenter.curation.curation_image_loader import FakeCurationImageLoaderFactory


@pytest.fixture
def curation_model() -> CurationModel:
    return CurationModel(FakeExperimentsModel(), FakeCurationImageLoaderFactory())


def test_set_raw_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    dir_set_slot: Mock = Mock()
    curation_model.raw_directory_set.connect(dir_set_slot)

    # Act
    curation_model.set_raw_directory(directory)

    # Assert
    assert curation_model.get_raw_directory() == directory
    dir_set_slot.assert_called_once()


def test_set_seg1_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    dir_set_slot: Mock = Mock()
    curation_model.seg1_directory_set.connect(dir_set_slot)

    # Act
    curation_model.set_seg1_directory(directory)

    # Assert
    assert curation_model.get_seg1_directory() == directory
    dir_set_slot.assert_called_once()


def test_set_seg2_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    dir_set_slot: Mock = Mock()
    curation_model.seg2_directory_set.connect(dir_set_slot)

    # Act
    curation_model.set_seg2_directory(directory)

    # Assert
    assert curation_model.get_seg2_directory() == directory
    dir_set_slot.assert_called_once()


def test_set_raw_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 0

    # Act
    curation_model.set_raw_channel(channel)

    # Assert
    assert curation_model.get_raw_channel() == channel


def test_set_seg1_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 1

    # Act
    curation_model.set_seg1_channel(channel)

    # Assert
    assert curation_model.get_seg1_channel() == channel


def test_set_seg2_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 2

    # Act
    curation_model.set_seg2_channel(channel)

    # Assert
    assert curation_model.get_seg2_channel() == channel


def test_set_current_view_to_main_view(curation_model: CurationModel) -> None:
    # Arrange

    # necessary for init of image loader
    curation_model.set_raw_directory_paths([Path("r1"), Path("r2"), Path("r3")])
    curation_model.set_seg1_directory_paths([Path("s11"), Path("s12"), Path("s13")])
    curation_model.set_seg2_directory_paths([Path("s21"), Path("s22"), Path("s23")])

    view_changed_slot: Mock = Mock()
    curation_model.current_view_changed.connect(view_changed_slot)

    first_image_ready_slot: Mock = Mock()
    curation_model.first_image_data_ready.connect(first_image_ready_slot)

    next_image_ready_slot: Mock = Mock()
    curation_model.next_image_data_ready.connect(next_image_ready_slot)

    # Act
    curation_model.set_current_view(CurationView.MAIN_VIEW)

    # Assert
    assert curation_model.get_current_view() == CurationView.MAIN_VIEW

    view_changed_slot.assert_called_once()
    first_image_ready_slot.assert_called_once()
    next_image_ready_slot.assert_called_once() # should be called since > 1 image in dirs

    assert curation_model.get_raw_image_data().path == Path("r1")
    assert curation_model.get_seg1_image_data().path == Path("s11")
    assert curation_model.get_seg2_image_data().path == Path("s21")


# TODO: if we end up needing to change back from main to input view, add tests for that
# here

def test_set_merging_mask(curation_model: CurationModel) -> None:
    # Arrange
    mask: np.ndarray = np.asarray([[1, 2], [3, 4]])
    # Act
    curation_model.set_merging_mask(mask)
    # Assert
    assert np.array_equal(curation_model.get_merging_mask(), mask)

def test_set_excluding_mask(curation_model: CurationModel) -> None:
    # Arrange
    mask: np.ndarray = np.asarray([[8, 2], [3, 5]])
    # Act
    curation_model.set_excluding_mask(mask)
    # Assert
    assert np.array_equal(curation_model.get_excluding_mask(), mask)

def test_set_base_image(curation_model: CurationModel) -> None:
    # Act
    curation_model.set_base_image("seg2")

    # Assert
    assert curation_model.get_base_image() == "seg2"

def test_set_use_image(curation_model: CurationModel) -> None:
    # Act
    curation_model.set_use_image(False)

    # Assert
    assert not curation_model.get_use_image()

def test_set_raw_image_channel_count(curation_model: CurationModel) -> None:
    # Arrange
    channel_count_set_slot: Mock = Mock()
    curation_model.raw_image_channel_count_set.connect(channel_count_set_slot)

    # Act
    curation_model.set_raw_image_channel_count(4)

    # Assert
    assert curation_model.get_raw_image_channel_count() == 4
    channel_count_set_slot.assert_called_once()

def test_set_seg1_image_channel_count(curation_model: CurationModel) -> None:
    # Arrange
    channel_count_set_slot: Mock = Mock()
    curation_model.seg1_image_channel_count_set.connect(channel_count_set_slot)

    # Act
    curation_model.set_seg1_image_channel_count(5)

    # Assert
    assert curation_model.get_seg1_image_channel_count() == 5
    channel_count_set_slot.assert_called_once()

def test_set_seg2_image_channel_count(curation_model: CurationModel) -> None:
    # Arrange
    channel_count_set_slot: Mock = Mock()
    curation_model.seg2_image_channel_count_set.connect(channel_count_set_slot)

    # Act
    curation_model.set_seg2_image_channel_count(6)

    # Assert
    assert curation_model.get_seg2_image_channel_count() == 6
    channel_count_set_slot.assert_called_once()

def test_set_curation_saved_to_disk(curation_model: CurationModel) -> None:
    saved_slot: Mock = Mock()
    curation_model.saved_to_disk.connect(saved_slot)

    curation_model.set_curation_record_saved_to_disk(False)

    assert not curation_model.get_curation_record_saved_to_disk()
    saved_slot.assert_not_called()

    curation_model.set_curation_record_saved_to_disk(True)

    assert curation_model.get_curation_record_saved_to_disk()
    saved_slot.assert_called_once()

def test_save_curation_record_with_seg2(curation_model: CurationModel) -> None:
    # Arrange
    # necessary for init of image loader
    curation_model.set_raw_directory_paths([Path("r1"), Path("r2"), Path("r3")])
    curation_model.set_seg1_directory_paths([Path("s11"), Path("s12"), Path("s13")])
    curation_model.set_seg2_directory_paths([Path("s21"), Path("s22"), Path("s23")])
    curation_model.set_current_view(CurationView.MAIN_VIEW)

    # Act
    curation_model.set_base_image("seg1")
    curation_model.set_merging_mask(np.asarray([[1, 2], [3, 4]]))
    curation_model.set_excluding_mask(np.asarray([[2, 3], [4, 5]]))
    curation_model.set_use_image(True)
    curation_model.save_curr_curation_record()

    # Assert
    assert len(curation_model.get_curation_record()) == 1
    record: CurationRecord = curation_model.get_curation_record()[-1]
    assert record.raw_file == Path("r1")
    assert record.seg1 == Path("s11")
    assert record.seg2 == Path("s21")
    assert record.base_image_index == "seg1"
    assert record.to_use
    assert np.array_equal(record.merging_mask, np.asarray([[1, 2], [3, 4]]))
    assert np.array_equal(record.excluding_mask, np.asarray([[2, 3], [4, 5]]))

    # Act
    curation_model.next_image()
    curation_model.set_base_image("seg2")
    curation_model.set_merging_mask(np.asarray([[3, 4], [5, 6]]))
    curation_model.set_excluding_mask(np.asarray([[4, 5], [6, 7]]))
    curation_model.set_use_image(False)
    curation_model.save_curr_curation_record()

    # Assert
    assert len(curation_model.get_curation_record()) == 2
    record = curation_model.get_curation_record()[-1]
    assert record.raw_file == Path("r2")
    assert record.seg1 == Path("s12")
    assert record.seg2 == Path("s22")
    assert record.base_image_index == "seg2"
    assert not record.to_use
    assert np.array_equal(record.merging_mask, np.asarray([[3, 4], [5, 6]]))
    assert np.array_equal(record.excluding_mask, np.asarray([[4, 5], [6, 7]]))

    # Act
    curation_model.next_image()
    # curation_model.set_base_image("seg2") no base image -> expect default seg1
    # curation_model.set_merging_mask(np.asarray([[3, 4], [5, 6]]))
    curation_model.set_excluding_mask(np.asarray([[5, 6], [7, 8]]))
    # curation_model.set_use_image(True) expect default True
    curation_model.save_curr_curation_record()

    # Assert
    assert len(curation_model.get_curation_record()) == 3
    record = curation_model.get_curation_record()[-1]
    assert record.raw_file == Path("r3")
    assert record.seg1 == Path("s13")
    assert record.seg2 == Path("s23")
    assert record.base_image_index == "seg1"
    assert record.to_use
    assert record.merging_mask is None
    assert np.array_equal(record.excluding_mask, np.asarray([[5, 6], [7, 8]]))

def test_save_curation_record_without_seg2(curation_model: CurationModel) -> None:
    # Arrange
    # necessary for init of image loader
    curation_model.set_raw_directory_paths([Path("r1"), Path("r2"), Path("r3")])
    curation_model.set_seg1_directory_paths([Path("s11"), Path("s12"), Path("s13")])
    curation_model.set_current_view(CurationView.MAIN_VIEW)

    # Act
    curation_model.set_excluding_mask(np.asarray([[2, 3], [4, 5]]))
    curation_model.set_use_image(True)
    curation_model.save_curr_curation_record()

    # Assert
    assert len(curation_model.get_curation_record()) == 1
    record: CurationRecord = curation_model.get_curation_record()[-1]
    assert record.raw_file == Path("r1")
    assert record.seg1 == Path("s11")
    assert record.seg2 is None
    assert record.base_image_index == "seg1"
    assert record.to_use
    assert record.merging_mask is None
    assert np.array_equal(record.excluding_mask, np.asarray([[2, 3], [4, 5]]))

    # Act
    curation_model.next_image()
    curation_model.set_excluding_mask(np.asarray([[4, 5], [6, 7]]))
    curation_model.set_use_image(False)
    curation_model.save_curr_curation_record()

    # Assert
    assert len(curation_model.get_curation_record()) == 2
    record = curation_model.get_curation_record()[-1]
    assert record.raw_file == Path("r2")
    assert record.seg1 == Path("s12")
    assert record.seg2 is None
    assert record.base_image_index == "seg1"
    assert not record.to_use
    assert record.merging_mask is None
    assert np.array_equal(record.excluding_mask, np.asarray([[4, 5], [6, 7]]))

    # Act
    curation_model.next_image()
    curation_model.set_base_image("seg2") # no seg2 image -> expect default seg1
    curation_model.set_merging_mask(np.asarray([[3, 4], [5, 6]])) # no seg2 image -> expect default None
    curation_model.set_excluding_mask(np.asarray([[5, 6], [7, 8]]))
    # curation_model.set_use_image(True) expect default True
    curation_model.save_curr_curation_record()

    # Assert
    assert len(curation_model.get_curation_record()) == 3
    record = curation_model.get_curation_record()[-1]
    assert record.raw_file == Path("r3")
    assert record.seg1 == Path("s13")
    assert record.seg2 is None
    assert record.base_image_index == "seg1"
    assert record.to_use
    assert record.merging_mask is None
    assert np.array_equal(record.excluding_mask, np.asarray([[5, 6], [7, 8]]))

def test_state_resets_on_next(curation_model: CurationModel) -> None:
    # Arrange
    # necessary for init of image loader
    curation_model.set_raw_directory_paths([Path("r1"), Path("r2"), Path("r3")])
    curation_model.set_seg1_directory_paths([Path("s11"), Path("s12"), Path("s13")])
    curation_model.set_seg2_directory_paths([Path("s21"), Path("s22"), Path("s23")])
    curation_model.set_current_view(CurationView.MAIN_VIEW)

    # Act
    curation_model.set_base_image("seg2")
    curation_model.set_merging_mask(np.asarray([[1, 2], [3, 4]]))
    curation_model.set_excluding_mask(np.asarray([[2, 3], [4, 5]]))
    curation_model.set_use_image(False)
    curation_model.next_image()

    # Assert
    assert curation_model.get_base_image() == "seg1"
    assert curation_model.get_merging_mask() == None
    assert curation_model.get_excluding_mask() == None
    assert curation_model.get_use_image() == True

def test_save_curation_record_overwrite(curation_model: CurationModel) -> None:
    # Arrange
    # necessary for init of image loader
    curation_model.set_raw_directory_paths([Path("r1"), Path("r2"), Path("r3")])
    curation_model.set_seg1_directory_paths([Path("s11"), Path("s12"), Path("s13")])
    curation_model.set_seg2_directory_paths([Path("s21"), Path("s22"), Path("s23")])
    curation_model.set_current_view(CurationView.MAIN_VIEW)

    # Act
    curation_model.set_base_image("seg1")
    curation_model.set_merging_mask(np.asarray([[1, 2], [3, 4]]))
    curation_model.set_excluding_mask(np.asarray([[2, 3], [4, 5]]))
    curation_model.set_use_image(True)
    curation_model.save_curr_curation_record()

    # Act
    curation_model.set_base_image("seg2")
    curation_model.set_merging_mask(np.asarray([[3, 4], [5, 6]]))
    curation_model.set_excluding_mask(np.asarray([[4, 5], [6, 7]]))
    curation_model.set_use_image(False)
    curation_model.save_curr_curation_record()

    # Assert
    assert len(curation_model.get_curation_record()) == 1
    record = curation_model.get_curation_record()[-1]
    assert record.raw_file == Path("r1")
    assert record.seg1 == Path("s11")
    assert record.seg2 == Path("s21")
    assert record.base_image_index == "seg2"
    assert not record.to_use
    assert np.array_equal(record.merging_mask, np.asarray([[3, 4], [5, 6]]))
    assert np.array_equal(record.excluding_mask, np.asarray([[4, 5], [6, 7]]))

def test_save_curation_record_to_disk(curation_model: CurationModel) -> None:
    # Arrange
    save_requested_slot: Mock = Mock()
    curation_model.save_to_disk_requested.connect(save_requested_slot)

    # Act
    curation_model.save_curr_curation_record_to_disk()

    # Assert
    save_requested_slot.assert_called_once()

