from dataclasses import dataclass

import numpy as np

from allencell_ml_segmenter.curation.main_view import (
    CurationMainView,
    MERGING_MASK_LAYER_NAME,
    EXCLUDING_MASK_LAYER_NAME,
)
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.core.info_dialog_box import InfoDialogBox
from allencell_ml_segmenter.core.dialog_box import DialogBox
from qtpy.QtWidgets import QDialog
from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    CurationView,
    CurationRecord,
    ImageType,
)
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.core.image_data_extractor import ImageData
import allencell_ml_segmenter
from allencell_ml_segmenter.main.main_model import MainModel

from pytestqt.qtbot import QtBot
from unittest.mock import Mock
from pathlib import Path
import pytest
from pytest import MonkeyPatch

IMG_DIR_PATH = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "img_folder"
)

IMG_DIR_FILES = [path for path in IMG_DIR_PATH.iterdir()]
FAKE_IMG_DATA = [
    ImageData(28, 28, 28, 4, np.zeros([28, 28, 28, 4]), path)
    for path in IMG_DIR_FILES
]


@dataclass
class TestEnvironment:
    viewer: FakeViewer
    model: CurationModel
    view: CurationMainView


@pytest.fixture
def test_environment_with_seg2() -> TestEnvironment:
    curation_model: CurationModel = CurationModel(
        FakeExperimentsModel(), MainModel()
    )

    curation_model.set_image_directory_paths(ImageType.RAW, IMG_DIR_FILES)
    curation_model.set_image_directory_paths(ImageType.SEG1, IMG_DIR_FILES)
    curation_model.set_image_directory_paths(ImageType.SEG2, IMG_DIR_FILES)
    curation_model.set_current_view(CurationView.MAIN_VIEW)
    curation_model.start_loading_images()

    viewer: IViewer = FakeViewer()
    main_view: CurationMainView = CurationMainView(curation_model, viewer)

    return TestEnvironment(
        viewer,
        curation_model,
        main_view,
    )


@pytest.fixture
def test_environment_first_images_ready(
    test_environment_with_seg2: TestEnvironment,
) -> TestEnvironment:
    env: TestEnvironment = test_environment_with_seg2
    env.model.set_curr_image_data(ImageType.RAW, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG1, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG2, FAKE_IMG_DATA[0])

    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[1])
    return env


@pytest.fixture
def test_environment_without_seg2() -> TestEnvironment:
    curation_model: CurationModel = CurationModel(
        FakeExperimentsModel(), MainModel()
    )

    curation_model.set_image_directory_paths(ImageType.RAW, IMG_DIR_FILES)
    curation_model.set_image_directory_paths(ImageType.SEG1, IMG_DIR_FILES)
    curation_model.set_current_view(CurationView.MAIN_VIEW)
    curation_model.start_loading_images()

    viewer: IViewer = FakeViewer()
    main_view: CurationMainView = CurationMainView(curation_model, viewer)

    return TestEnvironment(
        viewer,
        curation_model,
        main_view,
    )


# NOTE: not using qtbot methods based on https://pytest-qt.readthedocs.io/en/latest/tutorial.html#note-about-qtbot-methods
### UI State Tests ----------------------------------------------------------------------------------


def test_initial_state_with_seg2(
    qtbot: QtBot, test_environment_with_seg2: TestEnvironment
) -> None:
    env: TestEnvironment = test_environment_with_seg2
    # Arrange
    loading_finished_mock: Mock = Mock()
    env.model.image_loading_finished.connect(loading_finished_mock)

    # everything should be disabled until signals from model fire
    # Assert
    assert not env.view.save_csv_button.isEnabled()
    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert not env.view.merging_base_combo.isEnabled()
    assert env.view.merging_base_combo.currentText() == "seg1"
    assert not env.view.merging_delete_button.isEnabled()

    assert not env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.use_img_stacked_spinner.is_spinning()
    assert not env.view.yes_radio.isEnabled()
    assert not env.view.no_radio.isEnabled()

    assert not env.view.next_button.isEnabled()

    # Act (pretend to be service)
    env.model.set_curr_image_data(ImageType.RAW, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG1, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG2, FAKE_IMG_DATA[0])

    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[1])

    # Assert
    # this behavior is already tested in the model tests, so if this fails, then you haven't
    # set all the required image data
    loading_finished_mock.assert_called_once()

    # save csv will not be enabled until >= 4 images have been curated
    assert not env.view.save_csv_button.isEnabled()
    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert not env.view.use_img_stacked_spinner.is_spinning()
    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert env.view.next_button.isEnabled()

    # TODO: these will fail if we change naming convention of added images...
    # is there a better way to check?
    assert env.viewer.contains_layer(f"[raw] {IMG_DIR_FILES[0].name}")
    assert env.viewer.contains_layer(f"[seg1] {IMG_DIR_FILES[0].name}")
    assert env.viewer.contains_layer(f"[seg2] {IMG_DIR_FILES[0].name}")

    assert env.view.progress_bar.value() == 1
    assert env.view.progress_bar.maximum() == len(IMG_DIR_FILES)


def test_initial_state_no_seg2(
    qtbot: QtBot, test_environment_without_seg2: TestEnvironment
) -> None:
    env: TestEnvironment = test_environment_without_seg2
    # Arrange
    loading_finished_mock: Mock = Mock()
    env.model.image_loading_finished.connect(loading_finished_mock)

    # everything should be disabled until signals from model fire
    # Assert
    assert not env.view.save_csv_button.isEnabled()
    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert not env.view.merging_base_combo.isEnabled()
    assert env.view.merging_base_combo.currentText() == "seg1"
    assert not env.view.merging_delete_button.isEnabled()

    assert not env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.use_img_stacked_spinner.is_spinning()
    assert not env.view.yes_radio.isEnabled()
    assert not env.view.no_radio.isEnabled()

    assert not env.view.next_button.isEnabled()

    # Act (pretend to be service)
    env.model.set_curr_image_data(ImageType.RAW, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG1, FAKE_IMG_DATA[0])

    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[1])

    # Assert
    loading_finished_mock.assert_called_once()

    # save csv will not be enabled until >= 4 images have been curated
    assert not env.view.save_csv_button.isEnabled()
    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert not env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert not env.view.use_img_stacked_spinner.is_spinning()
    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert env.view.next_button.isEnabled()

    assert env.viewer.contains_layer(f"[raw] {IMG_DIR_FILES[0].name}")
    assert env.viewer.contains_layer(f"[seg1] {IMG_DIR_FILES[0].name}")
    assert not env.viewer.contains_layer(f"[seg2] {IMG_DIR_FILES[0].name}")

    assert env.view.progress_bar.value() == 1
    assert env.view.progress_bar.maximum() == len(IMG_DIR_FILES)


def test_next_image(
    qtbot: QtBot, test_environment_with_seg2: TestEnvironment
) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    loading_finished_mock: Mock = Mock()
    env.model.image_loading_finished.connect(loading_finished_mock)
    # finish 'loading' the first set of images
    env.model.set_curr_image_data(ImageType.RAW, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG1, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG2, FAKE_IMG_DATA[0])

    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[1])

    # Act
    env.view.next_button.click()

    # Assert
    assert loading_finished_mock.call_count == 1
    assert env.view.progress_bar.value() == 2

    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert not env.view.next_button.isEnabled()

    assert env.viewer.contains_layer(f"[raw] {IMG_DIR_FILES[1].name}")
    assert env.viewer.contains_layer(f"[seg1] {IMG_DIR_FILES[1].name}")
    assert env.viewer.contains_layer(f"[seg2] {IMG_DIR_FILES[1].name}")

    # Act
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[2])

    # Assert
    assert loading_finished_mock.call_count == 2
    assert env.view.next_button.isEnabled()


def test_last_image(
    qtbot: QtBot,
    test_environment_with_seg2: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    # Arrange
    # standard way to deal with modal dialogs: https://pytest-qt.readthedocs.io/en/latest/note_dialogs.html
    monkeypatch.setattr(InfoDialogBox, "exec", lambda *args: 0)
    env: TestEnvironment = test_environment_with_seg2
    loading_finished_mock: Mock = Mock()
    env.model.image_loading_finished.connect(loading_finished_mock)
    # finish 'loading' the first set of images
    env.model.set_curr_image_data(ImageType.RAW, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG1, FAKE_IMG_DATA[0])
    env.model.set_curr_image_data(ImageType.SEG2, FAKE_IMG_DATA[0])

    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[1])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[1])

    # Act
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[2])
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[3])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[3])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[3])
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[4])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[4])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[4])
    env.view.next_button.click()

    # Assert
    # one finished signal should be emitted after 'start_loading_images', and one
    # should be emitted after clicking the next button (which we do 4 times)
    assert loading_finished_mock.call_count == 5
    assert env.view.progress_bar.value() == 5

    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert env.view.next_button.isEnabled()

    assert env.viewer.contains_layer(f"[raw] {IMG_DIR_FILES[-1].name}")
    assert env.viewer.contains_layer(f"[seg1] {IMG_DIR_FILES[-1].name}")
    assert env.viewer.contains_layer(f"[seg2] {IMG_DIR_FILES[-1].name}")

    # Act
    env.view.next_button.click()

    # reached end, so buttons should be disabled
    # Assert
    assert loading_finished_mock.call_count == len(IMG_DIR_FILES)
    assert env.view.progress_bar.value() == len(IMG_DIR_FILES)

    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert not env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert not env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert not env.view.yes_radio.isEnabled()
    assert not env.view.no_radio.isEnabled()

    assert not env.view.next_button.isEnabled()


def test_create_new_merging_mask(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
) -> None:
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    assert len(env.viewer.get_all_shapes()) == 0

    # Act
    env.view.merging_create_button.click()

    # Assert
    assert len(env.viewer.get_all_shapes()) == 1
    assert env.viewer.get_shapes(MERGING_MASK_LAYER_NAME) is not None
    assert env.view.merging_save_button.isEnabled()
    assert env.view.merging_delete_button.isEnabled()


def test_create_new_excluding_mask(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
) -> None:
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    assert len(env.viewer.get_all_shapes()) == 0

    # Act
    env.view.excluding_create_button.click()

    # Assert
    assert len(env.viewer.get_all_shapes()) == 1
    assert env.viewer.get_shapes(EXCLUDING_MASK_LAYER_NAME) is not None
    assert env.view.excluding_save_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()


def test_save_csv(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
) -> None:
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    save_requested_slot: Mock = Mock()
    env.model.save_to_disk_requested.connect(save_requested_slot)
    # need to get to a point where enough images are marked to use that saving is allowed
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[2])
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[3])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[3])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[3])
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[4])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[4])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[4])

    # Act / Assert
    env.view.save_csv_button.click()
    assert not env.view.save_csv_button.isEnabled()
    save_requested_slot.assert_called_once()

    env.model.set_curation_record_saved_to_disk(
        True
    )  # should re-enable the button
    assert env.view.save_csv_button.isEnabled()


def test_save_csv_enabled_state(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
) -> None:
    """
    Test that save csv button is enabled only when at least 4 images have been selected for use.
    """
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Assert
    assert not env.view.save_csv_button.isEnabled()

    # Act
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[2])
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[3])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[3])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[3])

    # Assert
    # we are at the 3rd image, so button should still be disabled
    assert not env.view.save_csv_button.isEnabled()

    # Act
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[4])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[4])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[4])

    # Assert
    # by default, use_image should be selected, so save csv button should be enabled
    assert env.view.save_csv_button.isEnabled()

    # Act
    env.view.no_radio.click()

    # Assert
    assert not env.view.save_csv_button.isEnabled()

    # Act
    env.view.yes_radio.click()
    env.view.next_button.click()

    # Assert
    assert env.view.save_csv_button.isEnabled()

    # Act
    env.view.no_radio.click()

    # Assert
    # since we are toggling between 4 and 5 images selected for use, should still be enabled
    assert env.view.save_csv_button.isEnabled()


def test_radio_button_enabled_state(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
) -> None:
    """
    Test that radio buttons get disabled when it's only possible to select 4 images for use.
    """
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Assert
    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    # Act
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[2])

    # Assert
    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    # Act
    env.view.no_radio.click()
    env.view.next_button.click()

    # Assert
    # since there are 5 images total, and we've said we don't want to use one, we shouldn't be able to
    # select not to use any more of them
    assert not env.view.yes_radio.isEnabled()
    assert not env.view.no_radio.isEnabled()


def test_delete_merging_mask(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
):
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.merging_create_button.click()
    env.view.merging_delete_button.click()

    # Assert
    assert len(env.viewer.get_all_shapes()) == 0
    assert env.viewer.get_shapes(MERGING_MASK_LAYER_NAME) is None
    assert not env.view.merging_delete_button.isEnabled()


def test_delete_excluding_mask(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
):
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.excluding_create_button.click()
    env.view.excluding_delete_button.click()

    # Assert
    assert len(env.viewer.get_all_shapes()) == 0
    assert env.viewer.get_shapes(EXCLUDING_MASK_LAYER_NAME) is None
    assert not env.view.excluding_delete_button.isEnabled()


def test_use_image_radios(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
) -> None:
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act / Assert
    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    env.view.no_radio.click()

    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert not env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert not env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    env.view.yes_radio.click()

    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()


### Model State Tests ------------------------------------------------------------------------


def test_set_use_image(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
):
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act / Assert
    assert env.view.yes_radio.isChecked()
    assert env.model.get_use_image()

    env.view.no_radio.click()
    assert not env.model.get_use_image()

    env.view.yes_radio.click()
    assert env.model.get_use_image()


def test_set_base_image(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
):
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act / Assert
    assert (
        env.model.get_base_image() == env.view.merging_base_combo.currentText()
    )
    env.view.merging_base_combo.setCurrentIndex(1)
    assert (
        env.model.get_base_image() == env.view.merging_base_combo.currentText()
    )
    env.view.merging_base_combo.setCurrentIndex(0)
    assert (
        env.model.get_base_image() == env.view.merging_base_combo.currentText()
    )


def test_set_merging_mask(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
):
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act / Assert
    env.view.merging_create_button.click()
    assert env.model.get_merging_mask() is None
    env.view.merging_save_button.click()
    assert env.model.get_merging_mask() is not None
    env.view.merging_delete_button.click()
    assert env.model.get_merging_mask() is None


def test_set_excluding_mask(
    qtbot: QtBot, test_environment_first_images_ready: TestEnvironment
):
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act / Assert
    env.view.excluding_create_button.click()
    assert env.model.get_excluding_mask() is None
    env.view.excluding_save_button.click()
    assert env.model.get_excluding_mask() is not None
    env.view.excluding_delete_button.click()
    assert env.model.get_excluding_mask() is None


def test_curation_record_on_next(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    # Arrange
    # standard way to deal with modal dialogs: https://pytest-qt.readthedocs.io/en/latest/note_dialogs.html
    monkeypatch.setattr(InfoDialogBox, "exec", lambda *args: 0)
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.yes_radio.click()
    env.view.merging_create_button.click()
    env.view.merging_save_button.click()
    env.view.merging_base_combo.setCurrentIndex(0)
    merging_base: str = env.view.merging_base_combo.currentText()
    env.view.next_button.click()
    env.model.set_next_image_data(ImageType.RAW, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG1, FAKE_IMG_DATA[2])
    env.model.set_next_image_data(ImageType.SEG2, FAKE_IMG_DATA[2])

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert record.to_use
    assert record.excluding_mask is None
    assert record.merging_mask is not None
    assert record.seg1 == IMG_DIR_FILES[0]
    assert record.seg2 == IMG_DIR_FILES[0]
    assert record.raw_file == IMG_DIR_FILES[0]
    assert record.base_image == merging_base

    # Act
    env.view.no_radio.click()
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[1]
    assert not record.to_use
    assert record.excluding_mask is None
    assert record.merging_mask is None
    assert record.seg1 == IMG_DIR_FILES[1]
    assert record.seg2 == IMG_DIR_FILES[1]
    assert record.raw_file == IMG_DIR_FILES[1]
    assert record.base_image == "seg1"  # should default to seg1

    # Act
    env.view.yes_radio.click()
    env.view.excluding_create_button.click()
    env.view.excluding_save_button.click()
    env.view.merging_base_combo.setCurrentIndex(1)
    merging_base: str = env.view.merging_base_combo.currentText()
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[2]
    assert record.to_use
    assert record.excluding_mask is not None
    assert record.merging_mask is None
    assert record.seg1 == IMG_DIR_FILES[2]
    assert record.seg2 == IMG_DIR_FILES[2]
    assert record.raw_file == IMG_DIR_FILES[2]
    assert record.base_image == merging_base


def test_merg_viewer_unsaved_model_saved_yes_to_prompt(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: merging mask in viewer is unsaved, and mask in viewer is different from previously saved mask. User
    clicks 'next' then responds 'yes' to the 'do you want to save the unsaved changes' prompt.

    Expectation: we expect that the mask in the viewer will be saved and persist in model state.
    """
    # Arrange
    monkeypatch.setattr(
        DialogBox, "exec", lambda *args: QDialog.DialogCode.Accepted
    )
    env: TestEnvironment = test_environment_first_images_ready
    unsaved_mask: np.ndarray = np.asarray([[4, 5], [6, 7]])

    # Act
    env.view.merging_create_button.click()
    env.view.merging_save_button.click()
    env.viewer.modify_shapes(MERGING_MASK_LAYER_NAME, unsaved_mask)
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert np.array_equal(record.merging_mask, unsaved_mask)


def test_merg_viewer_unsaved_model_saved_no_to_prompt(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: merging mask in viewer is unsaved, and mask in viewer is different from previously saved mask. User
    clicks 'next' then responds 'no' to the 'do you want to save the unsaved changes' prompt.

    Expectation: we expect that the mask in the viewer will not be saved and the previously saved mask will
    persist in model state.
    """
    # Arrange
    monkeypatch.setattr(
        DialogBox, "exec", lambda *args: QDialog.DialogCode.Rejected
    )
    env: TestEnvironment = test_environment_first_images_ready
    unsaved_mask: np.ndarray = np.asarray([[4, 5], [6, 7]])

    # Act
    env.view.merging_create_button.click()
    env.view.merging_save_button.click()
    env.viewer.modify_shapes(MERGING_MASK_LAYER_NAME, unsaved_mask)
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert not np.array_equal(record.merging_mask, unsaved_mask)


def test_merg_viewer_unsaved_model_unsaved_yes_to_prompt(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: mask in viewer is unsaved, and there is *no previously saved mask*. User
    clicks 'next' then responds 'yes' to the 'do you want to save the unsaved changes' prompt.

    Expectation: we expect that the mask in the viewer will be saved and persist in model state.
    """
    # Arrange
    monkeypatch.setattr(
        DialogBox, "exec", lambda *args: QDialog.DialogCode.Accepted
    )
    env: TestEnvironment = test_environment_first_images_ready
    unsaved_mask: np.ndarray = np.asarray([[4, 5], [6, 7]])

    # Act
    env.view.merging_create_button.click()
    env.viewer.modify_shapes(MERGING_MASK_LAYER_NAME, unsaved_mask)
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert np.array_equal(record.merging_mask, unsaved_mask)


def test_merg_viewer_empty_model_unsaved(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: there is an empty merging mask in the viewer and no mask saved yet

    Expectation: user should click next and have no dialog pop up / no mask get saved.
    """
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.merging_create_button.click()
    # make it so that merging mask has no shapes
    env.viewer.modify_shapes(MERGING_MASK_LAYER_NAME, np.asarray([]))
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert record.merging_mask is None


def test_merg_viewer_empty_model_saved_yes_to_prompt(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: there is an empty merging mask in the viewer and some non-empty mask saved

    Expectation: clicking next causes dialog to pop up; empty mask will get saved when we select 'yes'.
    """
    # Arrange
    monkeypatch.setattr(
        DialogBox, "exec", lambda *args: QDialog.DialogCode.Accepted
    )
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.merging_create_button.click()
    env.view.merging_save_button.click()
    # make it so that merging mask has no shapes
    env.viewer.modify_shapes(MERGING_MASK_LAYER_NAME, np.array([]))
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert len(record.merging_mask) == 0


def test_merg_viewer_unsaved_not_to_use(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: there is an unsaved merging mask in the viewer, but the user has opted not to use this image

    Expectation: no dialog will pop up since we are not using the image anyways
    """
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.merging_create_button.click()
    env.view.no_radio.click()
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert record.merging_mask is None


def test_exclud_viewer_unsaved_model_saved_yes_to_prompt(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: excluding mask in viewer is unsaved, and mask in viewer is different from previously saved mask. User
    clicks 'next' then responds 'yes' to the 'do you want to save the unsaved changes' prompt.

    Expectation: we expect that the mask in the viewer will be saved and persist in model state.
    """
    # Arrange
    monkeypatch.setattr(
        DialogBox, "exec", lambda *args: QDialog.DialogCode.Accepted
    )
    env: TestEnvironment = test_environment_first_images_ready
    unsaved_mask: np.ndarray = np.asarray([[4, 5], [6, 7]])

    # Act
    env.view.excluding_create_button.click()
    env.view.excluding_save_button.click()
    env.viewer.modify_shapes(EXCLUDING_MASK_LAYER_NAME, unsaved_mask)
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert np.array_equal(record.excluding_mask, unsaved_mask)


def test_exclud_viewer_unsaved_model_saved_no_to_prompt(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: excluding mask in viewer is unsaved, and mask in viewer is different from previously saved mask. User
    clicks 'next' then responds 'no' to the 'do you want to save the unsaved changes' prompt.

    Expectation: we expect that the mask in the viewer will not be saved and the previously saved mask will
    persist in model state.
    """
    # Arrange
    monkeypatch.setattr(
        DialogBox, "exec", lambda *args: QDialog.DialogCode.Rejected
    )
    env: TestEnvironment = test_environment_first_images_ready
    unsaved_mask: np.ndarray = np.asarray([[4, 5], [6, 7]])

    # Act
    env.view.excluding_create_button.click()
    env.view.excluding_save_button.click()
    env.viewer.modify_shapes(EXCLUDING_MASK_LAYER_NAME, unsaved_mask)
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert not np.array_equal(record.excluding_mask, unsaved_mask)


def test_exclud_viewer_unsaved_model_unsaved_yes_to_prompt(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: excluding mask in viewer is unsaved, and there is *no previously saved mask*. User
    clicks 'next' then responds 'yes' to the 'do you want to save the unsaved changes' prompt.

    Expectation: we expect that the mask in the viewer will be saved and persist in model state.
    """
    # Arrange
    monkeypatch.setattr(
        DialogBox, "exec", lambda *args: QDialog.DialogCode.Accepted
    )
    env: TestEnvironment = test_environment_first_images_ready
    unsaved_mask: np.ndarray = np.asarray([[4, 5], [6, 7]])

    # Act
    env.view.excluding_create_button.click()
    env.viewer.modify_shapes(EXCLUDING_MASK_LAYER_NAME, unsaved_mask)
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert np.array_equal(record.excluding_mask, unsaved_mask)


def test_exclud_viewer_empty_model_unsaved(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: there is an empty excluding mask in the viewer and no mask saved yet

    Expectation: user should click next and have no dialog pop up / no mask get saved.
    """
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.excluding_create_button.click()
    # make it so that excluding mask has no shapes
    env.viewer.modify_shapes(EXCLUDING_MASK_LAYER_NAME, np.asarray([]))
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert record.excluding_mask is None


def test_exclud_viewer_empty_model_saved_yes_to_prompt(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: there is an empty excluding mask in the viewer and some non-empty mask saved

    Expectation: clicking next causes dialog to pop up; empty mask will get saved when we select 'yes'.
    """
    # Arrange
    monkeypatch.setattr(
        DialogBox, "exec", lambda *args: QDialog.DialogCode.Accepted
    )
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.excluding_create_button.click()
    env.view.excluding_save_button.click()
    # make it so that excluding mask has no shapes
    env.viewer.modify_shapes(EXCLUDING_MASK_LAYER_NAME, np.array([]))
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert len(record.excluding_mask) == 0


def test_exclud_viewer_unsaved_not_to_use(
    qtbot: QtBot,
    test_environment_first_images_ready: TestEnvironment,
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Scenario: there is an unsaved excluding mask in the viewer, but the user has opted not to use this image

    Expectation: no dialog will pop up since we are not using the image anyways
    """
    # Arrange
    env: TestEnvironment = test_environment_first_images_ready

    # Act
    env.view.excluding_create_button.click()
    env.view.no_radio.click()
    env.view.next_button.click()

    # Assert
    record: CurationRecord = env.model.get_curation_record()[0]
    assert record.excluding_mask is None
