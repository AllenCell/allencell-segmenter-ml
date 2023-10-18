from pathlib import Path
from unittest.mock import Mock

import pytest

import napari

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import CurationService


@pytest.fixture
def curation_service() -> CurationService:
    return CurationService(curation_model=Mock(spec=CurationModel), viewer= Mock(spec=napari.Viewer))

def test_remove_all_images_from_viewer_layers(curation_service: CurationService) -> None:
    # Arrange
    curation_service._viewer.layers: Mock = Mock()

    # Act
    curation_service.remove_all_images_from_viewer_layers()

    # Assert
    curation_service._viewer.layers.clear.assert_called_once()

def test_enable_shape_selection_viewer(curation_service: CurationService) -> None:
    curation_service.enable_shape_selection_viewer()

    curation_service._viewer.add_shapes.assert_called_once()

def test_select_directory_raw(curation_service: CurationService) -> None:
    # Arrange
    curation_service.get_total_num_channels_of_images_in_path: Mock = Mock(return_value=3)

    # Act
    curation_service.select_directory_raw(Path("test_path"))

    # Assert
    curation_service._curation_model.set_raw_directory.assert_called_once_with(Path("test_path"))
    curation_service._curation_model.set_total_num_channels_raw.assert_called_once_with(3)
    curation_service._curation_model.dispatch.assert_called_once_with(Event.ACTION_CURATION_RAW_SELECTED)


def test_select_directory_seg1(curation_service: CurationService) -> None:
    # Arrange
    curation_service.get_total_num_channels_of_images_in_path: Mock = Mock(return_value=4)

    # Act
    curation_service.select_directory_seg1(Path("test_path"))

    # Assert
    curation_service._curation_model.set_seg1_directory.assert_called_once_with(Path("test_path"))
    curation_service._curation_model.set_total_num_channels_seg1.assert_called_once_with(4)
    curation_service._curation_model.dispatch.assert_called_once_with(Event.ACTION_CURATION_SEG1_SELECTED)

def test_select_directory_seg2(curation_service: CurationService) -> None:
    # Arrange
    curation_service.get_total_num_channels_of_images_in_path: Mock = Mock(return_value=2)

    # Act
    curation_service.select_directory_seg2(Path("test_path"))

    # Assert
    curation_service._curation_model.set_seg2_directory.assert_called_once_with(Path("test_path"))
    curation_service._curation_model.set_total_num_channels_seg2.assert_called_once_with(2)
    curation_service._curation_model.dispatch.assert_called_once_with(Event.ACTION_CURATION_SEG2_SELECTED)
