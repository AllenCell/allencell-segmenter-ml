from dataclasses import dataclass
from allencell_ml_segmenter.curation.main_view import CurationMainView, MERGING_MASK_LAYER_NAME, EXCLUDING_MASK_LAYER_NAME
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.curation.curation_model import CurationModel, CurationView, CurationRecord
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import FakeExperimentsModel
from allencell_ml_segmenter.curation.curation_image_loader import FakeCurationImageLoaderFactory
import allencell_ml_segmenter

from pytestqt.qtbot import QtBot
from unittest.mock import Mock
from pathlib import Path
import pytest

IMG_DIR_PATH = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "img_folder"
)

IMG_DIR_FILES = [path for path in IMG_DIR_PATH.iterdir()]


@dataclass
class TestEnvironment:
    viewer: IViewer
    model: CurationModel
    view: CurationMainView

@pytest.fixture
def test_environment_with_seg2() -> TestEnvironment:
    curation_model: CurationModel = CurationModel(FakeExperimentsModel(), FakeCurationImageLoaderFactory())

    curation_model.set_raw_directory_paths(IMG_DIR_FILES)
    curation_model.set_seg1_directory_paths(IMG_DIR_FILES)
    curation_model.set_seg2_directory_paths(IMG_DIR_FILES)
    curation_model.set_current_view(CurationView.MAIN_VIEW)

    viewer: IViewer = FakeViewer()
    main_view: CurationMainView = CurationMainView(
        curation_model, viewer
    )

    return TestEnvironment(
        viewer,
        curation_model,
        main_view,
    )

@pytest.fixture
def test_environment_without_seg2() -> TestEnvironment:
    curation_model: CurationModel = CurationModel(FakeExperimentsModel(), FakeCurationImageLoaderFactory())

    curation_model.set_raw_directory_paths(IMG_DIR_FILES)
    curation_model.set_seg1_directory_paths(IMG_DIR_FILES)
    curation_model.set_current_view(CurationView.MAIN_VIEW)

    viewer: IViewer = FakeViewer()
    main_view: CurationMainView = CurationMainView(
        curation_model, viewer
    )
    
    return TestEnvironment(
        viewer,
        curation_model,
        main_view,
    )

# NOTE: not using qtbot methods based on https://pytest-qt.readthedocs.io/en/latest/tutorial.html#note-about-qtbot-methods
### UI State Tests ----------------------------------------------------------------------------------

def test_initial_state_with_seg2(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    env: TestEnvironment = test_environment_with_seg2

    # everything should be disabled until signals from model fire
    # Assert
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

    # Act
    env.model.first_image_data_ready.emit()
    # Assert
    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert not env.view.next_button.isEnabled()

    # Act
    env.model.next_image_data_ready.emit()
    # Assert
    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert env.view.next_button.isEnabled()

    assert env.viewer.contains_layer(f"[raw] {IMG_DIR_FILES[0].name}")
    assert env.viewer.contains_layer(f"[seg1] {IMG_DIR_FILES[0].name}")
    assert env.viewer.contains_layer(f"[seg2] {IMG_DIR_FILES[0].name}")

    assert env.view.progress_bar.value() == 1
    assert env.view.progress_bar.maximum() == 3


def test_initial_state_no_seg2(qtbot: QtBot, test_environment_without_seg2: TestEnvironment) -> None:
    env: TestEnvironment = test_environment_without_seg2

    # everything should be disabled until signals from model fire
    # Assert
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

    # Act
    env.model.first_image_data_ready.emit()
    # Assert
    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    # assert env.view.merging_base_combo.isEnabled() this change comes in later PR
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert not env.view.next_button.isEnabled()

    # Act
    env.model.next_image_data_ready.emit()
    # Assert
    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    # assert env.view.merging_base_combo.isEnabled() this change comes in later PR
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert env.view.next_button.isEnabled()

    assert env.viewer.contains_layer(f"[raw] {IMG_DIR_FILES[0].name}")
    assert env.viewer.contains_layer(f"[seg1] {IMG_DIR_FILES[0].name}")
    assert not env.viewer.contains_layer(f"[seg2] {IMG_DIR_FILES[0].name}")

    assert env.view.progress_bar.value() == 1
    assert env.view.progress_bar.maximum() == 3


def test_next_image(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act
    env.view.next_button.click()

    # Assert
    assert env.view.progress_bar.value() == 2

    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert not env.view.next_button.isEnabled()
    env.model.next_image_data_ready.emit()
    assert env.view.next_button.isEnabled()

    assert env.viewer.contains_layer(f"[raw] {IMG_DIR_FILES[1].name}")
    assert env.viewer.contains_layer(f"[seg1] {IMG_DIR_FILES[1].name}")
    assert env.viewer.contains_layer(f"[seg2] {IMG_DIR_FILES[1].name}")


def test_last_image(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act
    env.view.next_button.click()
    env.model.next_image_data_ready.emit()

    env.view.next_button.click()
    # since there is no next image after this click, we do not expect next_image_data_ready signal

    # Assert
    assert env.view.progress_bar.value() == 3

    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.yes_radio.isEnabled()
    assert env.view.no_radio.isEnabled()

    assert env.view.next_button.isEnabled()

    assert env.viewer.contains_layer(f"[raw] {IMG_DIR_FILES[2].name}")
    assert env.viewer.contains_layer(f"[seg1] {IMG_DIR_FILES[2].name}")
    assert env.viewer.contains_layer(f"[seg2] {IMG_DIR_FILES[2].name}")

    # Act
    env.view.next_button.click() # reached end, so buttons should be disabled

    # Assert
    assert env.view.progress_bar.value() == 3

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

def test_create_new_merging_mask(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    assert len(env.viewer.get_all_shapes()) == 0

    # Act
    env.view.merging_create_button.click()

    # Assert
    assert len(env.viewer.get_all_shapes()) == 1
    assert env.viewer.get_shapes(MERGING_MASK_LAYER_NAME) is not None
    assert env.view.merging_save_button.isEnabled()

def test_create_new_excluding_mask(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    assert len(env.viewer.get_all_shapes()) == 0

    # Act
    env.view.excluding_create_button.click()

    # Assert
    assert len(env.viewer.get_all_shapes()) == 1
    assert env.viewer.get_shapes(EXCLUDING_MASK_LAYER_NAME) is not None
    assert env.view.excluding_save_button.isEnabled()

def test_save_csv(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()
    save_requested_slot: Mock = Mock()
    env.model.save_to_disk_requested.connect(save_requested_slot)

    # Act / Assert
    env.view.save_csv_button.click()
    assert not env.view.save_csv_button.isEnabled()
    save_requested_slot.assert_called_once()

    env.model.saved_to_disk.emit() # should re-enable the button
    assert env.view.save_csv_button.isEnabled()

@pytest.mark.skip(reason="support in later PR")
def test_delete_merging_mask(qtbot: QtBot, test_environment_with_seg2: TestEnvironment):
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act
    env.view.merging_create_button.click()
    env.view.merging_delete_button.click()

    # Assert
    assert len(env.viewer.get_all_shapes()) == 0
    assert env.viewer.get_shapes(MERGING_MASK_LAYER_NAME) is None

@pytest.mark.skip(reason="support in later PR")
def test_delete_excluding_mask(qtbot: QtBot, test_environment_with_seg2: TestEnvironment):
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act
    env.view.excluding_create_button.click()
    env.view.excluding_delete_button.click()

    # Assert
    assert len(env.viewer.get_all_shapes()) == 0
    assert env.viewer.get_shapes(EXCLUDING_MASK_LAYER_NAME) is None

### View + Model Integration Tests ------------------------------------------------------------------------

def test_set_use_image(qtbot: QtBot, test_environment_with_seg2: TestEnvironment):
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act / Assert
    assert env.view.yes_radio.isChecked()
    assert env.model.get_use_image()

    env.view.no_radio.click()
    assert not env.model.get_use_image()

    env.view.yes_radio.click()
    assert env.model.get_use_image()

def test_set_base_image(qtbot: QtBot, test_environment_with_seg2: TestEnvironment):
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act / Assert
    env.view.merging_base_combo.setCurrentIndex(0)
    assert env.model.get_base_image() == env.view.merging_base_combo.currentText()
    env.view.merging_base_combo.setCurrentIndex(1)
    assert env.model.get_base_image() == env.view.merging_base_combo.currentText()

def test_set_merging_mask(qtbot: QtBot, test_environment_with_seg2: TestEnvironment):
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act / Assert
    env.view.merging_create_button.click()
    assert env.model.get_merging_mask() is None
    env.view.merging_save_button.click()
    assert env.model.get_merging_mask() is not None
    # env.view.merging_delete_button.click() coming later PR
    # assert env.model.get_merging_mask() is None

def test_set_excluding_mask(qtbot: QtBot, test_environment_with_seg2: TestEnvironment):
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act / Assert
    env.view.excluding_create_button.click()
    assert env.model.get_excluding_mask() is None
    env.view.excluding_save_button.click()
    assert env.model.get_excluding_mask() is not None
    # env.view.excluding_delete_button.click() coming later PR
    # assert env.model.get_excluding_mask() is None

def test_curation_record_on_next(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act
    env.view.yes_radio.click()
    env.view.merging_create_button.click()
    env.view.merging_save_button.click()
    env.view.merging_base_combo.setCurrentIndex(0)
    merging_base: str = env.view.merging_base_combo.currentText()
    env.view.next_button.click()

    # Assert
    assert len(env.model.get_curation_record()) == 1
    record: CurationRecord = env.model.get_curation_record()[-1]
    assert record.to_use
    assert record.excluding_mask is None
    assert record.merging_mask is not None
    assert record.seg1 == IMG_DIR_FILES[0]
    assert record.seg2 == IMG_DIR_FILES[0]
    assert record.raw_file == IMG_DIR_FILES[0]
    assert record.base_image_index == merging_base

    # Act
    env.model.next_image_data_ready.emit()
    env.view.no_radio.click()
    env.view.next_button.click()

    # Assert
    assert len(env.model.get_curation_record()) == 2
    record: CurationRecord = env.model.get_curation_record()[-1]
    assert not record.to_use
    assert record.excluding_mask is None
    assert record.merging_mask is None
    assert record.seg1 == IMG_DIR_FILES[1]
    assert record.seg2 == IMG_DIR_FILES[1]
    assert record.raw_file == IMG_DIR_FILES[1]
    assert record.base_image_index == "seg1" # should default to seg1

    # Act
    env.view.yes_radio.click()
    env.view.excluding_create_button.click()
    env.view.excluding_save_button.click()
    env.view.merging_base_combo.setCurrentIndex(1)
    merging_base: str = env.view.merging_base_combo.currentText()
    env.view.next_button.click()

    # Assert
    assert len(env.model.get_curation_record()) == 3
    record: CurationRecord = env.model.get_curation_record()[-1]
    assert record.to_use
    assert record.excluding_mask is not None
    assert record.merging_mask is None
    assert record.seg1 == IMG_DIR_FILES[2]
    assert record.seg2 == IMG_DIR_FILES[2]
    assert record.raw_file == IMG_DIR_FILES[2]
    assert record.base_image_index == merging_base

def test_curation_record_on_save(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act
    env.view.yes_radio.click()
    env.view.merging_create_button.click()
    env.view.merging_save_button.click()
    env.view.merging_base_combo.setCurrentIndex(0)
    env.view.excluding_create_button.click()
    env.view.excluding_save_button.click()
    env.view.save_csv_button.click()

    # Assert
    assert len(env.model.get_curation_record()) == 1
    record: CurationRecord = env.model.get_curation_record()[-1]
    assert record.to_use
    assert record.excluding_mask is not None
    assert record.merging_mask is not None
    assert record.seg1 == IMG_DIR_FILES[0]
    assert record.seg2 == IMG_DIR_FILES[0]
    assert record.raw_file == IMG_DIR_FILES[0]
    assert record.base_image_index == env.view.merging_base_combo.currentText()

    # Act
    env.model.saved_to_disk.emit() # necessary in order to re-enable save csv button
    env.view.merging_base_combo.setCurrentIndex(1)
    env.view.save_csv_button.click()

    # Assert
    assert len(env.model.get_curation_record()) == 1
    record: CurationRecord = env.model.get_curation_record()[-1]
    assert record.to_use
    assert record.excluding_mask is not None
    assert record.merging_mask is not None
    assert record.seg1 == IMG_DIR_FILES[0]
    assert record.seg2 == IMG_DIR_FILES[0]
    assert record.raw_file == IMG_DIR_FILES[0]
    assert record.base_image_index == env.view.merging_base_combo.currentText()


