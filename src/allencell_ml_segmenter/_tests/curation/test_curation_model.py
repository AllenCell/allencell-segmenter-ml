import pytest
from pathlib import Path
from unittest.mock import Mock
from aicsimageio import AICSImage

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.curation_model import CurationModel
from unittest.mock import patch


@pytest.fixture
def curation_model():
    return CurationModel()


def test_set_raw_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")

    # Act
    curation_model.set_raw_directory(directory)

    # Assert
    assert curation_model.get_raw_directory() == directory


def test_get_raw_directory(curation_model: CurationModel):
    # Arrange
    test_path: Path = Path("test_raw_path")
    model = CurationModel(raw_path=test_path)

    # Act / Assert
    assert model.get_raw_directory() == test_path


def test_set_seg1_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    # Act
    curation_model.set_seg1_directory(directory)

    # Assert
    assert curation_model.get_seg1_directory() == directory


def test_get_seg1_directory():
    # Arrange
    test_path: Path = Path("test_seg1_path")
    model = CurationModel(seg1_path=test_path)

    # Act / Assert
    assert model.get_seg1_directory() == test_path


def test_set_seg2_directory(curation_model: CurationModel):
    # Arrange
    directory: Path = Path("fake_path")

    # Act
    curation_model.set_seg2_directory(directory)

    # Assert
    assert curation_model.get_seg2_directory() == directory


def test_get_seg2_directory(curation_model: CurationModel):
    # Arrange
    test_path: Path = Path("test_seg2_path")
    model = CurationModel(seg2_path=test_path)

    # Act / Assert
    assert model.get_seg2_directory() == test_path


def test_set_raw_channel(curation_model: CurationModel):
    # Arrange
    channel: int = 0

    # Act
    curation_model.set_raw_channel(channel)

    # Assert
    assert curation_model.get_raw_channel() == channel


def test_set_seg1_channel(curation_model: CurationModel):
    # Arrange
    channel: int = 1

    # Act
    curation_model.set_seg1_channel(channel)

    # Assert
    assert curation_model.get_seg1_channel() == channel


def test_set_seg2_channel(curation_model: CurationModel):
    # Arrange
    channel: int = 2

    # Act
    curation_model.set_seg2_channel(channel)

    # Assert
    assert curation_model.get_seg2_channel() == channel


def test_set_view(curation_model: CurationModel):
    # Arrange
    with patch.object(CurationModel, "dispatch") as dispatch_mock:
        # Act
        curation_model.set_view()

        # Assert
        dispatch_mock.assert_called_once_with(
            Event.PROCESS_CURATION_INPUT_STARTED
        )



