import builtins
import csv
from pathlib import Path, PurePath
from typing import List
from unittest.mock import Mock, mock_open, patch, call

import numpy as np
import pytest

import napari
from napari.layers import Shapes

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import (
    CurationService,
    SelectionMode,
)
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.viewer import Viewer


@pytest.fixture
def curation_service() -> CurationService:
    return CurationService(
        curation_model=Mock(spec=CurationModel),
        viewer=Mock(spec=Viewer),
    )


def test_build_raw_images_list(curation_service: CurationService) -> None:
    # Arrange
    curation_service._curation_model.get_raw_directory = Mock(
        return_value=Path(__file__).parent / "curation_tests"
    )
    curation_service._get_files_list_from_path = Mock()
    # Act
    curation_service.build_raw_images_list()
    # Assert
    curation_service._get_files_list_from_path.assert_called_once_with(
        Path(__file__).parent / "curation_tests"
    )


def test_build_raw_images_list_invalid_path(
    curation_service: CurationService,
) -> None:
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_service._curation_model.get_raw_directory = Mock(
        return_value=None
    )
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.build_raw_images_list()


def test_build_seg1_images_list(curation_service: CurationService) -> None:
    # Arrange
    curation_service._curation_model.get_seg1_directory = Mock(
        return_value=Path(__file__).parent / "curation_tests"
    )
    curation_service._get_files_list_from_path = Mock()
    curation_service.build_seg1_images_list()

    # Act/Assert
    curation_service._get_files_list_from_path.assert_called_once_with(
        Path(__file__).parent / "curation_tests"
    )


def test_build_seg1_images_list_invalid_path(
    curation_service: CurationService,
) -> None:
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_service._curation_model.get_seg1_directory = Mock(
        return_value=None
    )
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.build_seg1_images_list()


def test_build_seg2_images_list(curation_service: CurationService) -> None:
    # Arrange
    curation_service._curation_model.get_seg2_directory = Mock(
        return_value=Path(__file__).parent / "curation_tests"
    )
    curation_service._get_files_list_from_path = Mock()
    # Act
    curation_service.build_seg2_images_list()
    # Assert
    curation_service._get_files_list_from_path.assert_called_once_with(
        Path(__file__).parent / "curation_tests"
    )


def test_build_seg2_images_list_invalid_path(
    curation_service: CurationService,
) -> None:
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_service._curation_model.get_seg2_directory: Mock = Mock(
        return_value=None
    )
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.build_seg2_images_list()


def test_get_files_list_from_path(curation_service: CurationService) -> None:
    paths: List[Path] = curation_service._get_files_list_from_path(
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


def test_add_image_to_viewer(curation_service: CurationService) -> None:
    # Act
    mock_array = np.ndarray([1, 1, 2])
    curation_service.add_image_to_viewer(mock_array, title="hello")

    curation_service._viewer.add_image.assert_called_once_with(
        mock_array, name="hello"
    )


def test_enable_shape_selection_viewer_merging(
    curation_service: CurationService,
) -> None:
    # Arrange
    curation_service.enable_shape_selection_viewer(
        mode=SelectionMode.EXCLUDING
    )

    curation_service._viewer.add_shapes.assert_called_once_with(
        "Excluding Mask"
    )
    curation_service._curation_model.append_excluding_mask_shape_layer.assert_called_once()


def test_enable_shape_selection_viewer_merging(
    curation_service: CurationService,
) -> None:
    # Arrange
    curation_service.enable_shape_selection_viewer(mode=SelectionMode.MERGING)

    curation_service._viewer.add_shapes.assert_called_once_with(
        name="Merging Mask"
    )
    curation_service._curation_model.append_merging_mask_shape_layer.assert_called_once()


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
        CurationRecord(
            to_use=True,
            raw_file="raw1",
            seg1="seg1",
            seg2="seg2",
            excluding_mask="",
            merging_mask="",
            base_image_index="seg1",
        ),
        CurationRecord(
            to_use=True,
            raw_file="raw2",
            seg1="seg3",
            seg2="seg4",
            excluding_mask="excluding_mask_2",
            merging_mask="merging_mask_2",
            base_image_index="seg1",
        ),
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
        assert (
            call().writerow(
                [
                    "",
                    "raw",
                    "seg1",
                    "seg2",
                    "excluding_mask",
                    "merging_mask",
                    "merging_col",
                ]
            )
            in mock_writer.mock_calls
        )
        assert (
            call().writerow(["0", "raw1", "seg1", "seg2", "", "", "seg1"])
            in mock_writer.mock_calls
        )
        assert (
            call().writerow(
                [
                    "1",
                    "raw2",
                    "seg3",
                    "seg4",
                    "excluding_mask_2",
                    "merging_mask_2",
                    "seg1",
                ]
            )
            in mock_writer.mock_calls
        )


@pytest.mark.parametrize(
    "use_this_image, expected_result", [(True, True), (False, False)]
)
def test_update_curation_record(
    curation_service: CurationService,
    use_this_image: str,
    expected_result: bool,
) -> None:
    # Arrange
    curation_service._curation_model.get_current_excluding_mask_path.return_value = (
        "excluding_mask_path"
    )
    curation_service._curation_model.get_current_merging_mask_path.return_value = (
        "merging_mask_path"
    )
    curation_service._curation_model.get_current_raw_image.return_value = (
        "raw_image_path"
    )
    curation_service._curation_model.get_current_seg1_image.return_value = (
        "seg1_image_path"
    )
    curation_service._curation_model.get_current_seg2_image.return_value = (
        "seg2_image_path"
    )
    curation_service._curation_model.get_merging_mask_base_layer.return_value = (
        "seg2"
    )
    curation_service._curation_model.get_curation_index.return_value = 1

    # Act
    curation_service.update_curation_record(use_image=True)

    # Assert
    # Ensure last record in curation_record is the one we just added
    assert curation_service._curation_model.append_curation_record.called_once_with(
        CurationRecord(
            "raw_image_path",
            "seg1_image_path",
            "seg2_image_path",
            "excluding_mask_path",
            "merging_mask_path",
            "seg2",
            expected_result,
        )
    )
    assert (
        curation_service._curation_model.set_curation_index.called_once_with(2)
    )


def test_finished_shape_selection_excluding(curation_service) -> None:
    # Arrange
    curation_service._curation_model.get_excluding_mask_shape_layers = Mock()
    shapes: Shapes = Shapes()
    curation_service._curation_model.get_excluding_mask_shape_layers.return_value = [
        shapes
    ]

    # Act/Assert
    curation_service.finished_shape_selection(
        selection_mode=SelectionMode.EXCLUDING
    )
    assert shapes.mode == "pan_zoom"


def test_finished_shape_selection_merging(curation_service) -> None:
    # Arrange
    curation_service._curation_model.get_merging_mask_shape_layers = Mock()
    shapes: Shapes = Shapes()
    curation_service._curation_model.get_merging_mask_shape_layers.return_value = [
        shapes
    ]

    # Act/Assert
    curation_service.finished_shape_selection(
        selection_mode=SelectionMode.MERGING
    )
    assert shapes.mode == "pan_zoom"


def test_clear_merging_mask_layers_all(
    curation_service: CurationService,
) -> None:
    # Arrange
    shapes_layers: List[Shapes] = [
        Shapes(name="merging_layer"),
        Shapes(name="merging_layer2"),
        Shapes(name="merging_layer3"),
    ]

    curation_service._curation_model.get_merging_mask_shape_layers.return_value = (
        shapes_layers
    )
    curation_service._viewer.viewer = Mock()

    # act
    curation_service.clear_merging_mask_layers_all()

    # Assert
    curation_service._curation_model.merging_mask_shape_layers = []


def test_clear_excluding_mask_layers_all(curation_service) -> None:
    # Arrange
    shapes_layers: List[Shapes] = [
        Shapes(name="merging_layer"),
        Shapes(name="merging_layer2"),
        Shapes(name="merging_layer3"),
    ]

    curation_service._curation_model.get_excluding_mask_shape_layers.return_value = (
        shapes_layers
    )
    curation_service._viewer.viewer = Mock()

    # act
    curation_service.clear_excluding_mask_layers_all()

    # Assert
    curation_service._curation_model.excluding_mask_shape_layers = []


def test_next_image_no_seg2(curation_service: CurationService) -> None:
    # Arrange
    curation_service.update_curation_record = Mock()
    curation_service.remove_all_images_from_viewer_layers = Mock()
    curation_service.add_image_to_viewer_from_path = Mock()
    curation_service._curation_model.image_available.return_value = True
    raw_path = Path("raw_test")
    curation_service._curation_model.get_current_raw_image.return_value = (
        raw_path
    )
    curation_service._curation_model.get_current_seg1_image.return_value = (
        raw_path
    )
    curation_service._curation_model.get_seg2_images.return_value = None

    # Act
    curation_service.next_image(use_image=True)

    # Assert
    curation_service.update_curation_record.assert_called_once_with(True)
    curation_service.remove_all_images_from_viewer_layers.assert_called_once()
    curation_service._curation_model.set_current_merging_mask_path.assert_called_once()
    curation_service._curation_model.set_current_merging_mask_path.assert_called_once()
    curation_service._curation_model.set_current_loaded_images.assert_called_with(
        (raw_path, raw_path, None)
    )
    curation_service._curation_model.dispatch.assert_called_once_with(
        Event.PROCESS_CURATION_NEXT_IMAGE
    )


def test_next_image_with_seg2(curation_service: CurationService) -> None:
    # Arrange
    curation_service.update_curation_record = Mock()
    curation_service.remove_all_images_from_viewer_layers = Mock()
    curation_service.add_image_to_viewer_from_path = Mock()
    curation_service._curation_model.image_available.return_value = True
    raw_path: Path = Path("raw_test")
    seg1_path: Path = Path("seg1_test")
    seg2_path: Path = Path("seg2_test")
    curation_service._curation_model.get_current_raw_image.return_value = (
        raw_path
    )
    curation_service._curation_model.get_current_seg1_image.return_value = (
        seg1_path
    )
    curation_service._curation_model.get_current_seg2_image.return_value = (
        seg2_path
    )
    curation_service._curation_model.set_seg2_images([seg2_path])

    # Act
    curation_service.next_image(use_image=True)

    # Assert
    curation_service.update_curation_record.assert_called_once_with(True)
    curation_service.remove_all_images_from_viewer_layers.assert_called_once()
    curation_service._curation_model.set_current_merging_mask_path.assert_called_once()
    curation_service._curation_model.set_current_merging_mask_path.assert_called_once()
    curation_service._curation_model.set_current_loaded_images.assert_called_once_with(
        (raw_path, seg1_path, seg2_path)
    )
    curation_service._curation_model.dispatch.assert_called_once_with(
        Event.PROCESS_CURATION_NEXT_IMAGE
    )


def test_next_image_finished(curation_service: CurationService) -> None:
    # Arrange
    curation_service.update_curation_record = Mock()
    curation_service._curation_model.image_available.return_value = False
    curation_service.write_curation_record = Mock()
    curation_service._curation_model.experiments_model = Mock(
        spec=ExperimentsModel
    )
    curation_service._curation_model.experiments_model.get_user_experiments_path.return_value = Path(
        "path"
    )
    curation_service._curation_model.experiments_model.get_experiment_name.return_value = Path(
        "test_exp"
    )

    # Act
    curation_service.next_image(use_image=True)

    # Assert
    curation_service.write_curation_record.assert_called_once()
