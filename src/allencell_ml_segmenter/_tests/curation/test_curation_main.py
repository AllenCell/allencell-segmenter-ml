from dataclasses import dataclass
from allencell_ml_segmenter.curation.main_view import CurationMainView
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.curation.curation_model import CurationModel, CurationView
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
    # 1 = left click: https://het.as.utexas.edu/HET/Software/html/qt.html#MouseButton-enum
    qtbot.mouseClick(env.view.next_button, 1)

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
    # 1 = left click: https://het.as.utexas.edu/HET/Software/html/qt.html#MouseButton-enum
    qtbot.mouseClick(env.view.next_button, 1)
    env.model.next_image_data_ready.emit()

    qtbot.mouseClick(env.view.next_button, 1)
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
    qtbot.mouseClick(env.view.next_button, 1) # reached end, so buttons should be disabled

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

    # Act
    qtbot.mouseClick(env.view.merging_create_button, 1)

### View + Model Integration Tests ------------------------------------------------------------------------

def test_curation_record_on_next_with_seg2(qtbot: QtBot, test_environment_with_seg2: TestEnvironment) -> None:
    # Arrange
    env: TestEnvironment = test_environment_with_seg2
    env.model.first_image_data_ready.emit()
    env.model.next_image_data_ready.emit()

    # Act
    #qtbot.mouseClick(env.view.no_radio)