from typing import List

import pytest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

from allencell_ml_segmenter.core.event import Event
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









