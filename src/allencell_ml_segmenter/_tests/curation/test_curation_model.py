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
    with patch.object(CurationModel, "get_total_num_channels", return_value=3):
        with patch.object(CurationModel, "dispatch") as dispatch_mock:

            # Act
            curation_model.set_raw_directory(directory)

            # Assert
            assert curation_model._raw_image_channel_count == 3
            assert curation_model.get_raw_directory() == directory
            assert dispatch_mock.called_with("ACTION_CURATION_RAW_SELECTED")


def test_get_raw_directory(curation_model: CurationModel):
    # Arrange
    test_path: Path = Path("test_raw_path")
    model = CurationModel(raw_path=test_path)

    # Act / Assert
    assert model.get_raw_directory() == test_path


def test_set_seg1_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    with patch.object(CurationModel, "get_total_num_channels", return_value=4):
        with patch.object(CurationModel, "dispatch") as dispatch_mock:

            # Act
            curation_model.set_seg1_directory(directory)

            # Assert
            assert curation_model._seg1_image_channel_count == 4
            assert curation_model.get_seg1_directory() == directory
            assert dispatch_mock.called_with("ACTION_CURATION_SEG1_SELECTED")


def test_get_seg1_directory():
    # Arrange
    test_path: Path = Path("test_seg1_path")
    model = CurationModel(seg1_path=test_path)

    # Act / Assert
    assert model.get_seg1_directory() == test_path


def test_set_seg2_directory(curation_model: CurationModel):
    # Arrange
    directory: Path = Path("fake_path")
    with patch.object(CurationModel, "get_total_num_channels", return_value=5):
        with patch.object(CurationModel, "dispatch") as dispatch_mock:

            # Act
            curation_model.set_seg2_directory(directory)

            # Assert
            assert curation_model._seg2_image_channel_count == 5
            assert curation_model.get_seg2_directory() == directory
            assert dispatch_mock.called_with("ACTION_CURATION_SEG2_SELECTED")


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


def test_get_total_num_channels_raw(curation_model: CurationModel):
    # Arrange
    directory: Path = Path("/path/to/raw_directory")
    with patch.object(CurationModel, "get_total_num_channels", return_value=3):
        curation_model.set_raw_directory(directory)

        # Act
        num_channels = curation_model.get_total_num_channels_raw()

        # Assert
        assert num_channels == 3


def test_get_total_num_channels_seg1(curation_model: CurationModel):
    # Arrange
    directory: Path = Path("/path/to/seg1_directory")
    with patch.object(CurationModel, "get_total_num_channels", return_value=3):
        curation_model.set_seg1_directory(directory)

        # Act
        num_channels: int = curation_model.get_total_num_channels_seg1()

        # Assert
        assert num_channels == 3


def test_get_total_num_channels_seg2(curation_model: CurationModel):
    # Arrange
    directory: Path = Path("/path/to/seg2_directory")
    with patch.object(CurationModel, "get_total_num_channels", return_value=4):
        curation_model.set_seg2_directory(directory)

        # Act
        num_channels = curation_model.get_total_num_channels_seg2()

        # Assert
        assert num_channels == 4


def test_get_total_num_channels(curation_model: CurationModel):
    # Arrange
    directory = Path("/path/to/raw_directory")
    with patch.object(CurationModel, "get_total_num_channels", return_value=6):

        # Act
        num_channels = curation_model.get_total_num_channels(directory)

        # Assert
        assert num_channels == 6
