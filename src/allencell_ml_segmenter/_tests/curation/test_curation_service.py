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
from allencell_ml_segmenter.curation.curation_image_loader import (
    FakeCurationImageLoader,
    FakeCurationImageLoaderFactory,
)
from allencell_ml_segmenter.core.image_data_extractor import ImageData
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.viewer import Viewer
import allencell_ml_segmenter


def test_build_raw_images_list() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    cur_tests_path = Path(__file__).parent / "curation_tests"
    curation_model.set_raw_directory(cur_tests_path)
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    # Act
    raw_images: List[Path] = curation_service.build_raw_images_list()
    # Assert
    assert any(
        [
            raw.samefile(cur_tests_path / "file_test.ome.tiff")
            for raw in raw_images
        ]
    )
    assert any(
        [raw.samefile(cur_tests_path / "file1.ome.tiff") for raw in raw_images]
    )


def test_build_raw_images_list_invalid_path() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    # There is no raw direcotry set in the model- getter returns None
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.build_raw_images_list()


def test_build_seg1_images_list() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    cur_tests_path = Path(__file__).parent / "curation_tests"
    curation_model.set_seg1_directory(cur_tests_path)
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )

    # Act
    seg1_images: List[Path] = curation_service.build_seg1_images_list()
    # Assert
    assert any(
        [
            seg1.samefile(cur_tests_path / "file_test.ome.tiff")
            for seg1 in seg1_images
        ]
    )
    assert any(
        [
            seg1.samefile(cur_tests_path / "file1.ome.tiff")
            for seg1 in seg1_images
        ]
    )


def test_build_seg1_images_list_invalid_path() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    # There is no raw direcotry set in the model- getter returns None
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.build_seg1_images_list()


def test_build_seg2_images_list() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    cur_tests_path = Path(__file__).parent / "curation_tests"
    curation_model.set_seg2_directory(cur_tests_path)
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )

    # Act
    seg2_images: List[Path] = curation_service.build_seg2_images_list()
    # Assert
    assert any(
        [
            seg2.samefile(cur_tests_path / "file_test.ome.tiff")
            for seg2 in seg2_images
        ]
    )
    assert any(
        [
            seg2.samefile(cur_tests_path / "file1.ome.tiff")
            for seg2 in seg2_images
        ]
    )


def test_build_seg2_images_list_invalid_path() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    # There is no raw direcotry set in the model- getter returns None
    # Act/ assert
    with pytest.raises(ValueError):
        curation_service.build_seg2_images_list()


def test_remove_all_images_from_viewer_layers() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_service_fake_viewer: CurationService = CurationService(
        Mock(spec=CurationModel), fake_viewer, FakeCurationImageLoaderFactory()
    )

    # Act
    curation_service_fake_viewer.remove_all_images_from_viewer_layers()

    # Assert
    assert fake_viewer.layers_cleared_count == 1


def test_add_image_to_viewer() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_service_fake_viewer: CurationService = CurationService(
        Mock(spec=CurationModel), fake_viewer, FakeCurationImageLoaderFactory()
    )

    # Act
    mock_array: np.ndarray = np.ndarray([1, 1, 2])
    mock_img_data = ImageData(1, 2, 3, 4, mock_array, Path("fake"))
    curation_service_fake_viewer.add_image_to_viewer(
        mock_img_data, title="hello"
    )

    assert np.array_equal(fake_viewer.images_added["hello"], mock_array)


def test_enable_shape_selection_viewer_merging() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_model = CurationModel()
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    curation_model.subscribe(
        Event.ACTION_CURATION_DRAW_EXCLUDING,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )
    curation_service_fake_viewer: CurationService = CurationService(
        curation_model, fake_viewer
    )

    # Arrange
    curation_service_fake_viewer.enable_shape_selection_viewer(
        mode=SelectionMode.EXCLUDING
    )

    assert "Excluding_mask" in fake_viewer.shapes_layers_added
    assert (
        curation_model.get_excluding_mask_shape_layers()[0].name
        == "Excluding_mask"
    )
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_DRAW_EXCLUDING)


def test_enable_shape_selection_viewer_merging() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_model = CurationModel()
    fake_subscriber = FakeSubscriber()
    curation_model.subscribe(
        Event.ACTION_CURATION_DRAW_MERGING,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )
    curation_service_fake_viewer: CurationService = CurationService(
        curation_model, fake_viewer, FakeCurationImageLoaderFactory()
    )

    # Act
    curation_service_fake_viewer.enable_shape_selection_viewer(
        mode=SelectionMode.MERGING
    )

    assert "Merging Mask" in fake_viewer.shapes_layers_added
    assert (
        curation_model.get_merging_mask_shape_layers()[0].name
        == "Merging Mask"
    )
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_DRAW_MERGING)


@patch(
    "allencell_ml_segmenter.core.channel_extraction.ChannelExtractionThread.start"
)
def test_select_directory_raw(start_mock: Mock) -> None:
    # Arrange
    model: CurationModel = CurationModel()
    curation_service: CurationService = CurationService(
        model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    img_folder: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
    )

    fake_subscriber: FakeSubscriber = FakeSubscriber()
    model.subscribe(
        Event.ACTION_CURATION_RAW_CHANNELS_SET,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )

    # Act
    curation_service.select_directory_raw(img_folder)
    # manually run the thread task without creating a new thread so
    # that updates are testable
    curation_service._raw_thread.run()

    # Assert
    start_mock.assert_called()
    assert model.get_raw_directory() == img_folder
    assert model.get_raw_image_channel_count() == 3
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_RAW_CHANNELS_SET)


@patch(
    "allencell_ml_segmenter.core.channel_extraction.ChannelExtractionThread.start"
)
def test_select_directory_seg1(start_mock: Mock) -> None:
    # Arrange
    model: CurationModel = CurationModel()
    curation_service: CurationService = CurationService(
        model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    img_folder: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
    )

    fake_subscriber: FakeSubscriber = FakeSubscriber()
    model.subscribe(
        Event.ACTION_CURATION_SEG1_CHANNELS_SET,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )

    # Act
    curation_service.select_directory_seg1(img_folder)
    # manually run the thread task without creating a new thread so
    # that updates are testable
    curation_service._seg1_thread.run()

    # Assert
    start_mock.assert_called()
    assert model.get_seg1_directory() == img_folder
    assert model.get_seg1_image_channel_count() == 3
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_SEG1_CHANNELS_SET)


@patch(
    "allencell_ml_segmenter.core.channel_extraction.ChannelExtractionThread.start"
)
def test_select_directory_seg2(start_mock: Mock) -> None:
    # Arrange
    model = CurationModel()

    curation_service = CurationService(
        model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    img_folder: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "images"
    )

    fake_subscriber: FakeSubscriber = FakeSubscriber()
    model.subscribe(
        Event.ACTION_CURATION_SEG2_CHANNELS_SET,
        fake_subscriber,
        lambda e: fake_subscriber.handle(e),
    )

    # Act
    curation_service.select_directory_seg2(img_folder)
    # manually run the thread task without creating a new thread so
    # that updates are testable
    curation_service._seg2_thread.run()

    # Assert
    start_mock.assert_called()
    assert model.get_seg2_directory() == img_folder
    assert model.get_seg2_image_channel_count() == 3
    assert fake_subscriber.was_handled(Event.ACTION_CURATION_SEG2_CHANNELS_SET)


def test_write_curation_record() -> None:
    # Arrange
    curation_service: CurationService = CurationService(
        Mock(spec=CurationModel),
        Mock(spec=Viewer),
        FakeCurationImageLoaderFactory(),
    )
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
    with (
        patch(
            "allencell_ml_segmenter.curation.curation_service.open",
            mock_open(),
        ) as mock_file,
        patch(
            "allencell_ml_segmenter.curation.curation_service.Path"
        ) as mock_path,
        patch(
            "allencell_ml_segmenter.curation.curation_service.csv.writer"
        ) as mock_writer,
    ):
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
    use_this_image: str,
    expected_result: bool,
) -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    curation_model.set_image_loader(
        FakeCurationImageLoader(
            [Path("raw 1"), Path("raw 2"), Path("raw 3")],
            [Path("seg1 1"), Path("seg1 2"), Path("seg1 3")],
            [Path("seg2 1"), Path("seg2 2"), Path("seg2 3")],
        )
    )
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    curation_model.get_current_excluding_mask_path_and_reset_mask = Mock(
        return_value="excluding_mask_path"
    )
    curation_model.get_current_merging_mask_path = Mock(
        return_value="merging_mask_path"
    )
    curation_model.get_merging_mask_base_layer = Mock(return_value="seg2")

    # Act
    curation_service.update_curation_record(use_image=use_this_image)

    # Assert
    # Ensure last record in curation_record is the one we just added
    assert (
        CurationRecord(
            Path("raw 1"),
            Path("seg1 1"),
            Path("seg2 1"),
            "excluding_mask_path",
            "merging_mask_path",
            "seg2",
            use_this_image,
        )
        == curation_model.get_curation_record()[0]
    )


def test_finished_shape_selection_excluding() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    curation_model.get_excluding_mask_shape_layers = Mock()
    shapes: Shapes = Shapes()
    curation_model.get_excluding_mask_shape_layers.return_value = [shapes]

    # Act/Assert
    curation_service.finished_shape_selection(
        selection_mode=SelectionMode.EXCLUDING
    )
    assert shapes.mode == "pan_zoom"


def test_finished_shape_selection_merging() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    curation_model.get_merging_mask_shape_layers = Mock()
    shapes: Shapes = Shapes()
    curation_model.get_merging_mask_shape_layers.return_value = [shapes]

    # Act/Assert
    curation_service.finished_shape_selection(
        selection_mode=SelectionMode.MERGING
    )
    assert shapes.mode == "pan_zoom"


def test_clear_merging_mask_layers_all() -> None:
    # Arrange
    fake_viewer: FakeViewer = FakeViewer()
    curation_model: CurationModel = CurationModel()
    shapes_layers: List[Shapes] = [
        Shapes(name="merging_layer"),
        Shapes(name="merging_layer2"),
        Shapes(name="merging_layer3"),
    ]
    curation_model.set_merging_mask_shape_layers(shapes_layers)
    test_service_with_viewer: CurationService = CurationService(
        curation_model, fake_viewer, FakeCurationImageLoaderFactory()
    )
    assert len(curation_model.get_merging_mask_shape_layers()) > 0

    # act
    test_service_with_viewer.clear_merging_mask_layers_all()

    # Assert
    assert len(curation_model.get_merging_mask_shape_layers()) == 0
    fake_viewer.is_layer_removed(shapes_layers[0])
    fake_viewer.is_layer_removed(shapes_layers[1])
    fake_viewer.is_layer_removed(shapes_layers[2])


def test_clear_excluding_mask_layers_all() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    shapes_layers: List[Shapes] = [
        Shapes(name="merging_layer"),
        Shapes(name="merging_layer2"),
        Shapes(name="merging_layer3"),
    ]
    curation_model.set_excluding_mask_shape_layers(shapes_layers)

    fake_viewer: FakeViewer = FakeViewer()
    curation_service: CurationService = CurationService(
        curation_model, fake_viewer, FakeCurationImageLoaderFactory()
    )
    assert len(curation_model.get_excluding_mask_shape_layers()) > 0

    # act
    curation_service.clear_excluding_mask_layers_all()

    # Assert
    assert len(curation_model.get_excluding_mask_shape_layers()) == 0
    fake_viewer.is_layer_removed(shapes_layers[0])
    fake_viewer.is_layer_removed(shapes_layers[1])
    fake_viewer.is_layer_removed(shapes_layers[2])


def test_next_image_no_seg2() -> None:
    # Arrange
    model: CurationModel = CurationModel()
    model.set_image_loader(
        FakeCurationImageLoader(
            [Path("raw 1"), Path("raw 2"), Path("raw 3")],
            [Path("seg1 1"), Path("seg1 2"), Path("seg1 3")],
            [Path("seg2 1"), Path("seg2 2"), Path("seg2 3")],
        )
    )
    viewer: FakeViewer = FakeViewer()
    test_service_with_model: CurationService = CurationService(
        model, viewer, FakeCurationImageLoaderFactory()
    )
    raw_path: Mock = Mock(spec=Path)
    model.get_save_masks_path = Mock(return_value=Path("fake_path_save_mask"))
    model.is_user_experiment_selected = Mock(return_value=False)

    test_service_with_model.update_curation_record = Mock()

    # Act
    test_service_with_model.next_image(use_image=True)

    # Assert
    test_service_with_model.update_curation_record.assert_called_once_with(
        True
    )
    assert viewer.layers_cleared_count == 1
    assert model.get_current_merging_mask_path() == None
    assert model.get_current_excluding_mask_path_and_reset_mask() == None


def test_next_image_with_seg2() -> None:
    # Arrange
    model: CurationModel = CurationModel()
    model.set_image_loader(
        FakeCurationImageLoader(
            [Path("raw 1"), Path("raw 2"), Path("raw 3")],
            [Path("seg1 1"), Path("seg1 2"), Path("seg1 3")],
            [Path("seg2 1"), Path("seg2 2"), Path("seg2 3")],
        )
    )
    viewer: FakeViewer = FakeViewer()
    test_service_with_model: CurationService = CurationService(
        model, viewer, FakeCurationImageLoaderFactory()
    )
    test_service_with_model.update_curation_record = Mock()

    # Act
    test_service_with_model.next_image(use_image=True)

    # Assert
    test_service_with_model.update_curation_record.assert_called_once_with(
        True
    )
    assert viewer.layers_cleared_count == 1
    assert model.get_current_merging_mask_path() == None
    assert model.get_current_excluding_mask_path_and_reset_mask() == None


def test_next_image_finished() -> None:
    # Arrange
    curation_model: CurationModel = CurationModel()
    curation_model.set_image_loader(
        FakeCurationImageLoader(
            [Path("raw 1"), Path("raw 2"), Path("raw 3")],
            [Path("seg1 1"), Path("seg1 2"), Path("seg1 3")],
            [Path("seg2 1"), Path("seg2 2"), Path("seg2 3")],
        )
    )
    curation_model.get_image_loader().next()
    curation_model.get_image_loader().next()
    curation_service: CurationService = CurationService(
        curation_model, Mock(spec=Viewer), FakeCurationImageLoaderFactory()
    )
    curation_service.update_curation_record = Mock()
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
