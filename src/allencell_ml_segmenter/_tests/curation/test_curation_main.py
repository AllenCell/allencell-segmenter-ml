import pytest
from allencell_ml_segmenter.curation.main_view import CurationMainView
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import CurationService

from unittest.mock import Mock
from pytestqt.qtbot import QtBot
from pathlib import Path


@pytest.fixture
def curation_main_view(qtbot: QtBot) -> CurationMainView:
    curation_model: CurationModel = CurationModel()
    experiments_model: FakeExperimentsModel = FakeExperimentsModel()
    curation_service: Mock = Mock(spec=CurationService)
    return CurationMainView(
        curation_model, curation_service
    )


def test_curation_setup(curation_main_view: CurationMainView) -> None:
    # Arrange
    curation_main_view._curation_service.build_raw_images_list = Mock(
        return_value=[Path("path_raw")]
    )
    curation_main_view._curation_service.build_seg1_images_list = Mock(
        return_value=[Path("path_seg1")]
    )
    curation_main_view.init_progress_bar = Mock()

    # Act
    curation_main_view.curation_setup(first_setup=True)

    # Assert
    curation_main_view.init_progress_bar.assert_called_once()
    curation_main_view._curation_service.add_image_to_viewer.called_once_with(
        ["path_raw"], f"[raw] path_raw"
    )
    curation_main_view._curation_service.add_image_to_viewer.called_once_with(
        ["path_seg1"], f"[raw] path_seg1"
    )


def test_init_progress_bar(curation_main_view: CurationMainView) -> None:
    # Act
    curation_main_view.init_progress_bar()

    # Assert
    assert curation_main_view.progress_bar.value() == 0

def test_next_image(curation_main_view: CurationMainView) -> None:
    # Arrange
    curation_main_view._update_curation_record = Mock()
    curation_main_view.raw_images = [None, Path("path_raw")]
    curation_main_view.seg1_images = [None, Path("path_seg1")]
    assert curation_main_view._curation_model.curation_index == 0

    # Act
    curation_main_view._next_image()

    # Assert
    curation_main_view._curation_service.next_image.assert_called_once()

def test_increment_progress_bar(curation_main_view: CurationMainView) -> None:
    # Arrange
    curation_main_view.init_progress_bar()
    curation_main_view._curation_model.raw_images = [Path(), Path(), Path()]
    initial_value: int = curation_main_view.progress_bar.value()
    # Act
    curation_main_view._increment_progress_bar()
    # Assert
    assert curation_main_view.progress_bar.value() == initial_value + 1

def test_stop_increment_progress_bar_when_curation_finished(curation_main_view: CurationMainView) -> None:
    # Arrange
    curation_main_view.init_progress_bar()
    curation_main_view._curation_model.raw_images = [Path(), Path(), Path()]
    curation_main_view._curation_model.curation_index = 3
    initial_value: int = curation_main_view.progress_bar.value()

    # Act
    curation_main_view._increment_progress_bar()
    # Assert
    assert curation_main_view.progress_bar.value() == initial_value
