import pytest
from typing import Tuple
from allencell_ml_segmenter.curation.main_view import CurationMainView
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.curation.curation_image_loader import (
    FakeCurationImageLoader,
)
import allencell_ml_segmenter

from unittest.mock import Mock
from pytestqt.qtbot import QtBot
from pathlib import Path

IMG_DIR_PATH = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "img_folder"
)


def get_test_instances(
    include_seg_2: bool,
) -> Tuple[FakeViewer, CurationModel, CurationService, CurationMainView]:
    curation_model = CurationModel()

    curation_model.set_raw_directory(IMG_DIR_PATH)
    curation_model.set_seg1_directory(IMG_DIR_PATH)
    if include_seg_2:
        curation_model.set_seg2_directory(IMG_DIR_PATH)

    curation_model.set_image_loader(
        FakeCurationImageLoader(
            [Path("raw 1"), Path("raw 2"), Path("raw 3")],
            [Path("seg1 1"), Path("seg1 2"), Path("seg1 3")],
            [Path("seg2 1"), Path("seg2 2"), Path("seg2 3")],
        )
    )
    viewer = FakeViewer()
    curation_service = CurationService(curation_model, viewer)
    main_view = CurationMainView(curation_model, curation_service)
    # Note: we assume that when CurationMainView is shown, curation_setup will be called

    # with our current setup, need to mock this since this will set up a real
    # image loader and start trying to load images into memory
    main_view._curation_service.curation_setup = Mock()
    main_view.curation_setup(first_setup=True)
    return (
        viewer,
        curation_model,
        curation_service,
        main_view,
    )


def test_initial_state_with_seg2(qtbot: QtBot) -> None:
    # Arrange
    viewer, model, service, view = get_test_instances(True)

    # Act

    # Assert
    assert view.merging_create_button.isEnabled()
    assert not view.merging_save_button.isEnabled()
    assert view.merging_base_combo.isEnabled()
    assert view.merging_delete_button.isEnabled()

    # why are these buttons disabled until we create a merging mask?
    assert not view.excluding_create_button.isEnabled()
    assert not view.excluding_save_button.isEnabled()
    assert not view.excluding_delete_button.isEnabled()

    assert view.progress_bar.value() == 1
    assert view.progress_bar.maximum() == 3


def test_initial_state_no_seg2(qtbot: QtBot) -> None:
    # Arrange
    viewer, model, service, view = get_test_instances(False)

    # Act

    # Assert
    assert not view.merging_create_button.isEnabled()
    assert not view.merging_save_button.isEnabled()
    assert not view.merging_base_combo.isEnabled()
    assert not view.merging_delete_button.isEnabled()

    assert view.excluding_create_button.isEnabled()
    assert not view.excluding_save_button.isEnabled()
    assert view.excluding_delete_button.isEnabled()

    assert view.progress_bar.value() == 1
    assert view.progress_bar.maximum() == 3


def test_next_image_with_seg2(qtbot: QtBot) -> None:
    # Arrange
    viewer, model, service, view = get_test_instances(True)

    # Act
    # 1 = left click: https://het.as.utexas.edu/HET/Software/html/qt.html#MouseButton-enum
    qtbot.mouseClick(view.next_button, 1)

    # Assert
    assert view.merging_create_button.isEnabled()
    assert not view.merging_save_button.isEnabled()
    assert view.merging_base_combo.isEnabled()
    assert view.merging_delete_button.isEnabled()

    assert not view.excluding_create_button.isEnabled()
    assert not view.excluding_save_button.isEnabled()
    assert not view.excluding_delete_button.isEnabled()

    assert view.progress_bar.value() == 2

    assert "[raw] raw 2" in viewer.images_added
    assert "[seg1] seg1 2" in viewer.images_added
    assert "[seg2] seg2 2" in viewer.images_added


def test_next_image_no_seg2(qtbot: QtBot) -> None:
    # Arrange
    viewer, model, service, view = get_test_instances(False)

    # Act
    # 1 = left click: https://het.as.utexas.edu/HET/Software/html/qt.html#MouseButton-enum
    qtbot.mouseClick(view.next_button, 1)

    # Assert
    assert not view.merging_create_button.isEnabled()
    assert not view.merging_save_button.isEnabled()
    assert not view.merging_base_combo.isEnabled()
    assert not view.merging_delete_button.isEnabled()

    assert view.excluding_create_button.isEnabled()
    assert not view.excluding_save_button.isEnabled()
    assert view.excluding_delete_button.isEnabled()

    assert view.progress_bar.value() == 2

    assert "[raw] raw 2" in viewer.images_added
    assert "[seg1] seg1 2" in viewer.images_added
    assert "[seg2] seg2 2" in viewer.images_added
