from dataclasses import dataclass
from allencell_ml_segmenter.curation.main_view import CurationMainView
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.curation.curation_image_loader import (
    FakeCurationImageLoaderFactory,
)
import allencell_ml_segmenter

from pytestqt.qtbot import QtBot
from pathlib import Path

IMG_DIR_PATH = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "img_folder"
)


@dataclass
class TestEnvironment:
    viewer: IViewer
    model: CurationModel
    service: CurationService
    view: CurationMainView


def get_test_environment(
    include_seg_2: bool,
) -> TestEnvironment:
    curation_model: CurationModel = CurationModel()

    curation_model.set_raw_directory(IMG_DIR_PATH)
    curation_model.set_seg1_directory(IMG_DIR_PATH)
    if include_seg_2:
        curation_model.set_seg2_directory(IMG_DIR_PATH)

    viewer: IViewer = FakeViewer()
    curation_service: CurationService = CurationService(
        curation_model, viewer, FakeCurationImageLoaderFactory()
    )
    main_view: CurationMainView = CurationMainView(
        curation_model, curation_service
    )
    # Note: we assume that when CurationMainView is shown, curation_setup will be called
    main_view.curation_setup(first_setup=True)
    return TestEnvironment(
        viewer,
        curation_model,
        curation_service,
        main_view,
    )


def test_initial_state_with_seg2(qtbot: QtBot) -> None:
    # Arrange
    env: TestEnvironment = get_test_environment(True)

    # Act

    # Assert
    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()

    assert env.view.progress_bar.value() == 1
    assert env.view.progress_bar.maximum() == 2

    assert "[raw] t1.tiff" in env.viewer.images_added
    assert "[seg1] t1.tiff" in env.viewer.images_added
    assert "[seg2] t1.tiff" in env.viewer.images_added


def test_initial_state_no_seg2(qtbot: QtBot) -> None:
    # Arrange
    env: TestEnvironment = get_test_environment(False)

    # Act

    # Assert
    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert not env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()

    assert env.view.progress_bar.value() == 1
    assert env.view.progress_bar.maximum() == 2

    assert "[raw] t1.tiff" in env.viewer.images_added
    assert "[seg1] t1.tiff" in env.viewer.images_added
    assert "[seg2] t1.tiff" not in env.viewer.images_added


def test_next_image_with_seg2(qtbot: QtBot) -> None:
    # Arrange
    env: TestEnvironment = get_test_environment(True)

    # Act
    # 1 = left click: https://het.as.utexas.edu/HET/Software/html/qt.html#MouseButton-enum
    qtbot.mouseClick(env.view.next_button, 1)

    # Assert
    assert env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert env.view.merging_base_combo.isEnabled()
    assert env.view.merging_delete_button.isEnabled()

    assert not env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()
    assert not env.view.excluding_delete_button.isEnabled()

    assert env.view.progress_bar.value() == 2

    assert "[raw] t2.tiff" in env.viewer.images_added
    assert "[seg1] t2.tiff" in env.viewer.images_added
    assert "[seg2] t2.tiff" in env.viewer.images_added


def test_next_image_no_seg2(qtbot: QtBot) -> None:
    # Arrange
    env: TestEnvironment = get_test_environment(False)

    # Act
    # 1 = left click: https://het.as.utexas.edu/HET/Software/html/qt.html#MouseButton-enum
    qtbot.mouseClick(env.view.next_button, 1)

    # Assert
    assert not env.view.merging_create_button.isEnabled()
    assert not env.view.merging_save_button.isEnabled()
    assert not env.view.merging_base_combo.isEnabled()
    assert not env.view.merging_delete_button.isEnabled()

    assert env.view.excluding_create_button.isEnabled()
    assert not env.view.excluding_save_button.isEnabled()
    assert env.view.excluding_delete_button.isEnabled()

    assert env.view.progress_bar.value() == 2

    assert "[raw] t2.tiff" in env.viewer.images_added
    assert "[seg1] t2.tiff" in env.viewer.images_added
    assert "[seg2] t2.tiff" not in env.viewer.images_added
