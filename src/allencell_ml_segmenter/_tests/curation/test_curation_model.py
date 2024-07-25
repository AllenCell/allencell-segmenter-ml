import pytest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    CurationView,
    ImageType,
)
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.core.image_data_extractor import ImageData
import numpy as np


FAKE_IMAGE_DATA: ImageData = ImageData(
    28, 28, 28, 1, np.zeros((28, 28, 28)), Path("fake")
)


@pytest.fixture
def curation_model() -> CurationModel:
    # returns curation model with view set to input view
    return CurationModel(FakeExperimentsModel(), MainModel())


@pytest.fixture
def curation_model_main_view(curation_model: CurationModel) -> CurationModel:
    # returns curation model configured to main view
    curation_model.set_image_directory_paths(
        ImageType.RAW, [Path("r1"), Path("r2"), Path("r3")]
    )
    curation_model.set_image_directory_paths(
        ImageType.SEG1, [Path("s11"), Path("s12"), Path("s13")]
    )
    curation_model.set_image_directory_paths(
        ImageType.SEG2, [Path("s21"), Path("s22"), Path("s23")]
    )
    curation_model.set_current_view(CurationView.MAIN_VIEW)
    return curation_model


@pytest.fixture
def curation_model_loading_started(
    curation_model_main_view: CurationModel,
) -> CurationModel:
    # returns curation model with first images done loading
    curation_model_main_view.start_loading_images()

    curation_model_main_view.set_curr_image_data(
        ImageType.RAW, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_curr_image_data(
        ImageType.SEG1, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_curr_image_data(
        ImageType.SEG2, FAKE_IMAGE_DATA
    )

    curation_model_main_view.set_next_image_data(
        ImageType.RAW, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_next_image_data(
        ImageType.SEG1, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_next_image_data(
        ImageType.SEG2, FAKE_IMAGE_DATA
    )
    return curation_model_main_view


def test_set_raw_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    dir_set_slot: Mock = Mock()
    curation_model.image_directory_set.connect(dir_set_slot)

    # Act
    curation_model.set_image_directory(ImageType.RAW, directory)

    # Assert
    assert curation_model.get_image_directory(ImageType.RAW) == directory
    dir_set_slot.assert_called_once_with(ImageType.RAW)


def test_set_seg1_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    dir_set_slot: Mock = Mock()
    curation_model.image_directory_set.connect(dir_set_slot)

    # Act
    curation_model.set_image_directory(ImageType.SEG1, directory)

    # Assert
    assert curation_model.get_image_directory(ImageType.SEG1) == directory
    dir_set_slot.assert_called_once_with(ImageType.SEG1)


def test_set_seg2_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    dir_set_slot: Mock = Mock()
    curation_model.image_directory_set.connect(dir_set_slot)

    # Act
    curation_model.set_image_directory(ImageType.SEG2, directory)

    # Assert
    assert curation_model.get_image_directory(ImageType.SEG2) == directory
    dir_set_slot.assert_called_once_with(ImageType.SEG2)


def test_set_raw_image_channel_count(
    curation_model_loading_started: CurationModel,
) -> None:
    # Arrange
    channel_count_set_slot: Mock = Mock()
    curation_model_loading_started.channel_count_set.connect(
        channel_count_set_slot
    )

    # Act
    curation_model_loading_started.set_channel_count(ImageType.RAW, 4)

    # Assert
    assert curation_model_loading_started.get_channel_count(ImageType.RAW) == 4
    channel_count_set_slot.assert_called_once_with(ImageType.RAW)


def test_set_seg1_image_channel_count(
    curation_model_loading_started: CurationModel,
) -> None:
    # Arrange
    channel_count_set_slot: Mock = Mock()
    curation_model_loading_started.channel_count_set.connect(
        channel_count_set_slot
    )

    # Act
    curation_model_loading_started.set_channel_count(ImageType.SEG1, 5)

    # Assert
    assert (
        curation_model_loading_started.get_channel_count(ImageType.SEG1) == 5
    )
    channel_count_set_slot.assert_called_once_with(ImageType.SEG1)


def test_set_seg2_image_channel_count(
    curation_model_loading_started: CurationModel,
) -> None:
    # Arrange
    channel_count_set_slot: Mock = Mock()
    curation_model_loading_started.channel_count_set.connect(
        channel_count_set_slot
    )

    # Act
    curation_model_loading_started.set_channel_count(ImageType.SEG2, 6)

    # Assert
    assert (
        curation_model_loading_started.get_channel_count(ImageType.SEG2) == 6
    )
    channel_count_set_slot.assert_called_once_with(ImageType.SEG2)


def test_set_raw_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 0

    # Act
    curation_model.set_selected_channel(ImageType.RAW, channel)

    # Assert
    assert curation_model.get_selected_channel(ImageType.RAW) == channel


def test_set_seg1_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 1

    # Act
    curation_model.set_selected_channel(ImageType.SEG1, channel)

    # Assert
    assert curation_model.get_selected_channel(ImageType.SEG1) == channel


def test_set_seg2_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 2

    # Act
    curation_model.set_selected_channel(ImageType.SEG2, channel)

    # Assert
    assert curation_model.get_selected_channel(ImageType.SEG2) == channel


def test_set_current_view_to_main_view(curation_model: CurationModel) -> None:
    # Arrange

    # expect paths to be set before changing view
    curation_model.set_image_directory_paths(
        ImageType.RAW, [Path("r1"), Path("r2"), Path("r3")]
    )
    curation_model.set_image_directory_paths(
        ImageType.SEG1, [Path("s11"), Path("s12"), Path("s13")]
    )
    curation_model.set_image_directory_paths(
        ImageType.SEG2, [Path("s21"), Path("s22"), Path("s23")]
    )

    view_changed_slot: Mock = Mock()
    curation_model.current_view_changed.connect(view_changed_slot)

    # Act
    curation_model.set_current_view(CurationView.MAIN_VIEW)

    # Assert
    assert curation_model.get_current_view() == CurationView.MAIN_VIEW

    view_changed_slot.assert_called_once()


def test_start_loading_images(curation_model_main_view: CurationModel) -> None:
    # Arrange
    cursor_moved_slot: Mock = Mock()
    curation_model_main_view.cursor_moved.connect(cursor_moved_slot)
    img_loading_finished_slot: Mock = Mock()
    curation_model_main_view.image_loading_finished.connect(
        img_loading_finished_slot
    )

    # Act
    curation_model_main_view.start_loading_images()

    # Assert
    cursor_moved_slot.assert_called_once()
    img_loading_finished_slot.assert_not_called()

    # Act (pretending to be curation service)
    curation_model_main_view.set_curr_image_data(
        ImageType.RAW, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_curr_image_data(
        ImageType.SEG1, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_curr_image_data(
        ImageType.SEG2, FAKE_IMAGE_DATA
    )

    curation_model_main_view.set_next_image_data(
        ImageType.RAW, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_next_image_data(
        ImageType.SEG1, FAKE_IMAGE_DATA
    )

    # Assert
    # still one unwritten image data, so expect img_loading_finished not to be emitted yet
    img_loading_finished_slot.assert_not_called()

    # Act
    curation_model_main_view.set_next_image_data(
        ImageType.SEG2, FAKE_IMAGE_DATA
    )

    # Assert
    img_loading_finished_slot.assert_called_once()
    assert (
        curation_model_main_view.get_curr_image_data(ImageType.RAW) is not None
    )
    assert (
        curation_model_main_view.get_curr_image_data(ImageType.SEG1)
        is not None
    )
    assert (
        curation_model_main_view.get_curr_image_data(ImageType.SEG2)
        is not None
    )


def test_next_image(curation_model_main_view: CurationModel) -> None:
    # Arrange
    cursor_moved_slot: Mock = Mock()
    curation_model_main_view.cursor_moved.connect(cursor_moved_slot)
    img_loading_finished_slot: Mock = Mock()
    curation_model_main_view.image_loading_finished.connect(
        img_loading_finished_slot
    )
    curation_model_main_view.start_loading_images()
    curation_model_main_view.set_curr_image_data(
        ImageType.RAW, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_curr_image_data(
        ImageType.SEG1, FAKE_IMAGE_DATA
    )

    curation_model_main_view.set_next_image_data(
        ImageType.RAW, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_next_image_data(
        ImageType.SEG1, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_next_image_data(
        ImageType.SEG2, FAKE_IMAGE_DATA
    )

    # Assert
    # since not all images have finished loading, expect call to next to raise error
    with pytest.raises(RuntimeError):
        curation_model_main_view.next_image()

    # Act
    curation_model_main_view.set_curr_image_data(
        ImageType.SEG2, FAKE_IMAGE_DATA
    )
    curation_model_main_view.next_image()

    # Assert
    assert (
        cursor_moved_slot.call_count == 2
    )  # once after start, once after next
    assert img_loading_finished_slot.call_count == 1  # once after start

    # Act
    curation_model_main_view.set_next_image_data(
        ImageType.RAW, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_next_image_data(
        ImageType.SEG1, FAKE_IMAGE_DATA
    )
    curation_model_main_view.set_next_image_data(
        ImageType.SEG2, FAKE_IMAGE_DATA
    )

    # Assert
    assert img_loading_finished_slot.call_count == 2

    # Act
    curation_model_main_view.next_image()
    # we are at the last image, so there is no next image data to load

    # Assert
    assert cursor_moved_slot.call_count == 3
    assert img_loading_finished_slot.call_count == 3
    # there is no next image
    with pytest.raises(RuntimeError):
        curation_model_main_view.next_image()


# TODO: if we end up needing to change back from main to input view, add tests for that
# here


def test_set_merging_mask(
    curation_model_loading_started: CurationModel,
) -> None:
    # Arrange
    mask: np.ndarray = np.asarray([[1, 2], [3, 4]])
    # Act
    curation_model_loading_started.set_merging_mask(mask)
    # Assert
    assert np.array_equal(
        curation_model_loading_started.get_merging_mask(), mask
    )


def test_set_excluding_mask(
    curation_model_loading_started: CurationModel,
) -> None:
    # Arrange
    mask: np.ndarray = np.asarray([[8, 2], [3, 5]])
    # Act
    curation_model_loading_started.set_excluding_mask(mask)
    # Assert
    assert np.array_equal(
        curation_model_loading_started.get_excluding_mask(), mask
    )


def test_set_base_image(curation_model_loading_started: CurationModel) -> None:
    # Act
    curation_model_loading_started.set_base_image("seg2")

    # Assert
    assert curation_model_loading_started.get_base_image() == "seg2"


def test_set_use_image(curation_model_loading_started: CurationModel) -> None:
    # Act
    curation_model_loading_started.set_use_image(False)

    # Assert
    assert not curation_model_loading_started.get_use_image()


def test_set_curation_saved_to_disk(
    curation_model_loading_started: CurationModel,
) -> None:
    # Arrange
    saved_slot: Mock = Mock()
    curation_model_loading_started.saved_to_disk.connect(saved_slot)

    # Act
    curation_model_loading_started.set_curation_record_saved_to_disk(False)
    # Assert
    saved_slot.assert_called_with(False)

    # Act
    curation_model_loading_started.set_curation_record_saved_to_disk(True)
    # Assert
    saved_slot.assert_called_with(True)


def test_state_resets_on_next(
    curation_model_loading_started: CurationModel,
) -> None:
    # Act
    curation_model_loading_started.set_base_image("seg2")
    curation_model_loading_started.set_merging_mask(
        np.asarray([[1, 2], [3, 4]])
    )
    curation_model_loading_started.set_excluding_mask(
        np.asarray([[2, 3], [4, 5]])
    )
    curation_model_loading_started.set_use_image(False)
    curation_model_loading_started.next_image()

    # Assert
    assert curation_model_loading_started.get_base_image() == "seg1"
    assert curation_model_loading_started.get_merging_mask() == None
    assert curation_model_loading_started.get_excluding_mask() == None
    assert curation_model_loading_started.get_use_image() == True


def test_save_curation_record_to_disk(
    curation_model_loading_started: CurationModel,
) -> None:
    # Arrange
    save_requested_slot: Mock = Mock()
    curation_model_loading_started.save_to_disk_requested.connect(
        save_requested_slot
    )

    # Act
    curation_model_loading_started.save_curr_curation_record_to_disk()

    # Assert
    save_requested_slot.assert_called_once()
