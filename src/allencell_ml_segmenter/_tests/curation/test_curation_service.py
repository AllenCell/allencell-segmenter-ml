import builtins
import csv
from pathlib import Path, PurePath
from typing import List
from unittest.mock import Mock, mock_open, patch, call

import pytest

import napari

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import CurationService, SelectionMode
from allencell_ml_segmenter.main.viewer import Viewer


@pytest.fixture
def curation_service() -> CurationService:
    return CurationService(
        curation_model=Mock(spec=CurationModel),
        viewer=Mock(spec=Viewer),
    )


def test_get_raw_images_list(curation_service: CurationService):
    # Arrange
    curation_service._curation_model.get_raw_directory: Mock = Mock(
        return_value=Path(__file__).parent / "curation_tests"
    )
    curation_service._get_files_list_from_path = Mock()
    # Act
    curation_service.get_raw_images_list()
    # Assert
    curation_service._get_files_list_from_path.assert_called_once_with(
        Path(__file__).parent / "curation_tests"
    )


def test_get_raw_images_list_invalid_path(curation_service: CurationService):
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_service._curation_model.get_raw_directory: Mock = Mock(
        return_value=None
    )
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.get_raw_images_list()


def test_get_seg1_images_list(curation_service: CurationService):
    # Arrange
    curation_service._curation_model.get_seg1_directory: Mock = Mock(
        return_value=Path(__file__).parent / "curation_tests"
    )
    curation_service._get_files_list_from_path = Mock()
    curation_service.get_seg1_images_list()

    # Act/Assert
    curation_service._get_files_list_from_path.assert_called_once_with(
        Path(__file__).parent / "curation_tests"
    )


def test_get_seg1_images_list_invalid_path(curation_service: CurationService):
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_service._curation_model.get_seg1_directory: Mock = Mock(
        return_value=None
    )
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.get_seg1_images_list()


def test_get_seg2_images_list(curation_service: CurationService):
    # Arrange
    curation_service._curation_model.get_seg2_directory: Mock = Mock(
        return_value=Path(__file__).parent / "curation_tests"
    )
    curation_service._get_files_list_from_path = Mock()
    # Act
    curation_service.get_seg2_images_list()
    # Assert
    curation_service._get_files_list_from_path.assert_called_once_with(
        Path(__file__).parent / "curation_tests"
    )


def test_get_seg2_images_list_invalid_path(curation_service: CurationService):
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_service._curation_model.get_seg2_directory: Mock = Mock(
        return_value=None
    )
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.get_seg2_images_list()


def test_get_files_list_from_path(curation_service: CurationService) -> None:
    paths = curation_service._get_files_list_from_path(
        Path(__file__).parent / "curation_tests"
    )

    assert len(paths) == 2
    assert (
        paths[0] == Path(__file__).parent / "curation_tests" / "file1.ome.tiff"
    )


def test_remove_all_images_from_viewer_layers(
    curation_service: CurationService,
) -> None:
    # Act
    curation_service.remove_all_images_from_viewer_layers()

    # Assert
    curation_service._viewer.clear_layers.assert_called_once()


def test_enable_shape_selection_viewer(
    curation_service: CurationService,
) -> None:
    # Arrange
    curation_service._curation_model.excluding_mask_shape_layers = list()

    curation_service.enable_shape_selection_viewer(mode=SelectionMode.EXCLUDING)

    curation_service._viewer.add_shapes.assert_called_once()
    assert len(curation_service._curation_model.excluding_mask_shape_layers) == 1



def test_select_directory_raw(curation_service: CurationService) -> None:
    # Arrange
    curation_service.get_total_num_channels_of_images_in_path: Mock = Mock(
        return_value=3
    )

    # Act
    curation_service.select_directory_raw(Path("test_path"))

    # Assert
    curation_service._curation_model.set_raw_directory.assert_called_once_with(
        Path("test_path")
    )
    curation_service._curation_model.set_total_num_channels_raw.assert_called_once_with(
        3
    )
    curation_service._curation_model.dispatch.assert_called_once_with(
        Event.ACTION_CURATION_RAW_SELECTED
    )


def test_select_directory_seg1(curation_service: CurationService) -> None:
    # Arrange
    curation_service.get_total_num_channels_of_images_in_path: Mock = Mock(
        return_value=4
    )

    # Act
    curation_service.select_directory_seg1(Path("test_path"))

    # Assert
    curation_service._curation_model.set_seg1_directory.assert_called_once_with(
        Path("test_path")
    )
    curation_service._curation_model.set_total_num_channels_seg1.assert_called_once_with(
        4
    )
    curation_service._curation_model.dispatch.assert_called_once_with(
        Event.ACTION_CURATION_SEG1_SELECTED
    )


def test_select_directory_seg2(curation_service: CurationService) -> None:
    # Arrange
    curation_service.get_total_num_channels_of_images_in_path: Mock = Mock(
        return_value=2
    )

    # Act
    curation_service.select_directory_seg2(Path("test_path"))

    # Assert
    curation_service._curation_model.set_seg2_directory.assert_called_once_with(
        Path("test_path")
    )
    curation_service._curation_model.set_total_num_channels_seg2.assert_called_once_with(
        2
    )
    curation_service._curation_model.dispatch.assert_called_once_with(
        Event.ACTION_CURATION_SEG2_SELECTED
    )


def test_write_curation_record(curation_service: CurationService) -> None:
    # Arrange
    curation_record: List[CurationRecord] = [
        CurationRecord(to_use=True, raw_file="raw1", seg1="seg1", excluding_mask=""),
        CurationRecord(to_use=True, raw_file="raw2", seg1="seg2", excluding_mask="excluding_mask_file"),
    ]
    # Mock open, Path, and csv.writer
    with patch(
        "allencell_ml_segmenter.curation.curation_service.open", mock_open()
    ) as mock_file, patch(
        "allencell_ml_segmenter.curation.curation_service.Path"
    ) as mock_path, patch(
        "allencell_ml_segmenter.curation.curation_service.csv.writer"
    ) as mock_writer:
        mock_path.return_value.parents = [Path("/parent")]

        # Act
        curation_service.write_curation_record(
            curation_record, Path(__file__).parent / "curation_tests"
        )

        # Assert
        mock_file.assert_called_with(
            Path(__file__).parent / "curation_tests", "w"
        )
        assert call().writerow(["", "raw", "seg", "mask"]) in mock_writer.mock_calls
        assert call().writerow(["0", "raw1", "seg1", ""]) in mock_writer.mock_calls
        assert call().writerow(["1", "raw2", "seg2", "excluding_mask_file"]) in mock_writer.mock_calls
