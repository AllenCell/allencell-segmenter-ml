from pathlib import Path
from typing import List
from unittest.mock import Mock, mock_open, patch, call

import numpy as np
import pytest
from napari.layers import Shapes

from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
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
def curation_model() -> CurationModel:
    return Mock(spec=CurationModel)


@pytest.fixture
def viewer() -> Viewer:
    return Mock(spec=Viewer)


@pytest.fixture
def curation_service(
    curation_model: CurationModel, viewer: Viewer
) -> CurationService:
    return CurationService(
        curation_model=curation_model,
        viewer=viewer,
    )


def test_build_raw_images_list(
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    curation_model.get_raw_directory = Mock(
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
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_model.get_raw_directory = Mock(return_value=None)
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.build_raw_images_list()


def test_build_seg1_images_list(
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    curation_model.get_seg1_directory = Mock(
        return_value=Path(__file__).parent / "curation_tests"
    )
    curation_service._get_files_list_from_path = Mock()
    curation_service.build_seg1_images_list()

    # Act/Assert
    curation_service._get_files_list_from_path.assert_called_once_with(
        Path(__file__).parent / "curation_tests"
    )


def test_build_seg1_images_list_invalid_path(
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_model.get_seg1_directory = Mock(return_value=None)
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.build_seg1_images_list()


def test_build_seg2_images_list(
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    curation_model.get_seg2_directory = Mock(
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
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    # There is no raw direcotry set in the model- getter returns None
    curation_model.get_seg2_directory = Mock(return_value=None)
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


def test_remove_all_images_from_viewer_layers() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_service_fake_viewer: CurationService = CurationService(Mock(spec=CurationModel), fake_viewer)

    # Act
    curation_service_fake_viewer.remove_all_images_from_viewer_layers()

    # Assert
    assert fake_viewer.layers_cleared_count == 1


def test_add_image_to_viewer() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_service_fake_viewer: CurationService = CurationService(Mock(spec=CurationModel), fake_viewer)

    # Act
    mock_array: np.ndarray = np.ndarray([1, 1, 2])
    curation_service_fake_viewer.add_image_to_viewer(mock_array, title="hello")

    assert np.array_equal(fake_viewer.images_added["hello"], mock_array)



def test_enable_shape_selection_viewer_merging() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_model = CurationModel()
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    curation_model.subscribe(
        Event.ACTION_CURATION_DRAW_EXCLUDING,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e)
    )
    curation_service_fake_viewer: CurationService = CurationService(curation_model, fake_viewer)

    # Arrange
    curation_service_fake_viewer.enable_shape_selection_viewer(
        mode=SelectionMode.EXCLUDING
    )

    assert "Excluding_mask" in fake_viewer.shapes_layers_added
    assert curation_model.get_excluding_mask_shape_layers()[0].name == "Excluding_mask"
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_DRAW_EXCLUDING)


def test_enable_shape_selection_viewer_merging() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_model = CurationModel()
    fake_subscriber = FakeSubscriber()
    curation_model.subscribe(
        Event.ACTION_CURATION_DRAW_MERGING,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e)
    )
    curation_service_fake_viewer: CurationService = CurationService(curation_model, fake_viewer)

    # Act
    curation_service_fake_viewer.enable_shape_selection_viewer(mode=SelectionMode.MERGING)

    assert "Merging Mask" in fake_viewer.shapes_layers_added
    assert curation_model.get_merging_mask_shape_layers()[0].name == "Merging Mask"
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_DRAW_MERGING)


def test_select_directory_raw() -> None:
    # Arrange
    model: CurationModel = CurationModel()
    curation_service: CurationService = CurationService(
        model, viewer=Mock(spec=Viewer)
    )
    curation_service.get_total_num_channels_of_images_in_path: Mock = Mock(
        return_value=3
    )
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    model.subscribe(
        Event.ACTION_CURATION_RAW_SELECTED,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )

    # Act
    curation_service.select_directory_raw(Path("test_path"))

    # Assert
    assert model.get_raw_directory() == Path("test_path")
    assert model.get_total_num_channels_raw() == 3
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_RAW_SELECTED)


def test_select_directory_seg1() -> None:
    # Arrange
    model: CurationModel = CurationModel()
    test_service_with_model: CurationService = CurationService(
        model, viewer=Mock(spec=Viewer)
    )
    test_service_with_model.get_total_num_channels_of_images_in_path: Mock = (
        Mock(return_value=4)
    )
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    model.subscribe(
        Event.ACTION_CURATION_SEG1_SELECTED,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )

    # Act
    test_service_with_model.select_directory_seg1(Path("test_path"))

    # Assert
    assert model.get_seg1_directory() == Path("test_path")
    assert model.get_total_num_channels_seg1() == 4
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_SEG1_SELECTED)


def test_select_directory_seg2() -> None:
    # Arrange
    curation_model = CurationModel()

    curation_service = CurationService(
        curation_model=curation_model, viewer=Mock(spec=Viewer)
    )
    curation_service.get_total_num_channels_of_images_in_path: Mock = Mock(
        return_value=2
    )
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    curation_model.subscribe(
        Event.ACTION_CURATION_SEG2_SELECTED,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )

    # Act
    curation_service.select_directory_seg2(Path("test_path"))

    # Assert
    assert curation_model.get_seg2_directory() == Path("test_path")
    assert curation_model.get_total_num_channels_seg2() == 2
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_SEG2_SELECTED)


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
    curation_model: CurationModel,
    use_this_image: str,
    expected_result: bool,
) -> None:
    # Arrange
    curation_model.get_current_excluding_mask_path.return_value = (
        "excluding_mask_path"
    )
    curation_model.get_current_merging_mask_path.return_value = (
        "merging_mask_path"
    )
    curation_model.get_current_raw_image.return_value = "raw_image_path"
    curation_model.get_current_seg1_image.return_value = "seg1_image_path"
    curation_model.get_current_seg2_image.return_value = "seg2_image_path"
    curation_model.get_merging_mask_base_layer.return_value = "seg2"
    curation_model.get_curation_index.return_value = 1

    # Act
    curation_service.update_curation_record(use_image=True)

    # Assert
    # Ensure last record in curation_record is the one we just added
    assert curation_model.append_curation_record.called_once_with(
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
    assert curation_model.set_curation_index.called_once_with(2)


def test_finished_shape_selection_excluding(
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    curation_model.get_excluding_mask_shape_layers = Mock()
    shapes: Shapes = Shapes()
    curation_model.get_excluding_mask_shape_layers.return_value = [shapes]

    # Act/Assert
    curation_service.finished_shape_selection(
        selection_mode=SelectionMode.EXCLUDING
    )
    assert shapes.mode == "pan_zoom"


def test_finished_shape_selection_merging(
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    curation_model.get_merging_mask_shape_layers = Mock()
    shapes: Shapes = Shapes()
    curation_model.get_merging_mask_shape_layers.return_value = [shapes]

    # Act/Assert
    curation_service.finished_shape_selection(
        selection_mode=SelectionMode.MERGING
    )
    assert shapes.mode == "pan_zoom"


def test_clear_merging_mask_layers_all(curation_model: CurationModel) -> None:
    fake_viewer: FakeViewer = FakeViewer()
    test_service_with_viewer: CurationService = CurationService(
        curation_model, viewer=fake_viewer
    )
    # Arrange
    shapes_layers: List[Shapes] = [
        Shapes(name="merging_layer"),
        Shapes(name="merging_layer2"),
        Shapes(name="merging_layer3"),
    ]

    curation_model.get_merging_mask_shape_layers.return_value = shapes_layers
    test_service_with_viewer._viewer.viewer = fake_viewer

    # act
    test_service_with_viewer.clear_merging_mask_layers_all()

    # Assert
    curation_model.merging_mask_shape_layers = []
    fake_viewer.layers.is_removed(shapes_layers[0])
    fake_viewer.layers.is_removed(shapes_layers[1])
    fake_viewer.layers.is_removed(shapes_layers[2])


def test_clear_excluding_mask_layers_all(
    curation_service: CurationService,
    curation_model: CurationModel,
    viewer: Viewer,
) -> None:
    # Arrange
    shapes_layers: List[Shapes] = [
        Shapes(name="merging_layer"),
        Shapes(name="merging_layer2"),
        Shapes(name="merging_layer3"),
    ]

    curation_model.get_excluding_mask_shape_layers.return_value = shapes_layers
    viewer.viewer = Mock()

    # act
    curation_service.clear_excluding_mask_layers_all()

    # Assert
    curation_model.excluding_mask_shape_layers = []


def test_next_image_no_seg2() -> None:
    # Arrange
    model: CurationModel = CurationModel()
    viewer: FakeViewer = FakeViewer()
    test_service_with_model: CurationService = CurationService(
        curation_model=model, viewer=viewer
    )
    model.image_available = Mock(return_value=True)
    raw_path: Mock = Mock(spec=Path)
    model.get_current_raw_image = Mock(return_vƒalue=raw_path)
    model.get_current_seg1_image = Mock(return_value=raw_path)
    model.get_save_masks_path = Mock(return_value=Path("fake_path_save_mask"))
    model.get_seg2_images = Mock(return_value=None)
    model.is_user_experiment_selected = Mock(return_value=False)

    test_service_with_model.add_image_to_viewer_from_path = Mock()
    test_service_with_model.update_curation_record = Mock()
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    model.subscribe(
        Event.PROCESS_CURATION_NEXT_IMAGE,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )

    # Act
    test_service_with_model.next_image(use_image=True)

    # Assert
    test_service_with_model.update_curation_record.assert_called_once_with(
        True
    )
    assert viewer.layers_cleared_count == 1
    assert model.get_current_merging_mask_path() == None
    assert model.get_current_excluding_mask_path() == None
    assert model.get_current_loaded_images() == (raw_path, raw_path, None)
    assert fake_subscriber.was_handled(Event.PROCESS_CURATION_NEXT_IMAGE)


def test_next_image_with_seg2() -> None:
    # Arrange
    model: CurationModel = CurationModel()
    viewer: FakeViewer = FakeViewer()
    test_service_with_model: CurationService = CurationService(
        curation_model=model, viewer=viewer
    )
    model.image_available = Mock(return_value=True)
    raw_path: Path = Path("raw_test")
    seg1_path: Path = Path("seg1_test")
    seg2_path: Path = Path("seg2_test")
    model.get_current_raw_image = Mock(return_value=raw_path)
    model.get_current_seg1_image = Mock(return_value=seg1_path)
    model.get_current_seg2_image = Mock(return_value=seg2_path)
    model.get_seg2_images = Mock(return_value=[seg2_path])
    test_service_with_model.add_image_to_viewer_from_path = Mock()
    test_service_with_model.update_curation_record = Mock()
    test_service_with_model.add_image_to_viewer_from_path = Mock()
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    model.subscribe(
        Event.PROCESS_CURATION_NEXT_IMAGE,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )

    # Act
    test_service_with_model.next_image(use_image=True)

    # Assert
    test_service_with_model.update_curation_record.assert_called_once_with(
        True
    )
    assert viewer.layers_cleared_count == 1
    assert model.get_current_merging_mask_path() == None
    assert model.get_current_excluding_mask_path() == None
    assert model.get_current_loaded_images() == (raw_path, seg1_path, seg2_path)
    assert fake_subscriber.was_handled(Event.PROCESS_CURATION_NEXT_IMAGE)


def test_next_image_finished(
    curation_service: CurationService, curation_model: CurationModel
) -> None:
    # Arrange
    curation_service.update_curation_record = Mock()
    curation_model.image_available.return_value = False
    curation_service.write_curation_record = Mock()
    curation_model.experiments_model = Mock(spec=ExperimentsModel)
    curation_model.experiments_model.get_user_experiments_path.return_value = (
        Path("path")
    )
    curation_model.experiments_model.get_experiment_name.return_value = Path(
        "test_exp"
    )

    # Act
    curation_service.next_image(use_image=True)

    # Assert
    curation_service.write_curation_record.assert_called_once()
