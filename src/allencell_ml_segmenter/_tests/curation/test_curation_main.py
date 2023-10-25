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
        curation_model, experiments_model, curation_service
    )


def test_curation_setup(curation_main_view: CurationMainView) -> None:
    # Arrange
    curation_main_view._curation_service.get_raw_images_list = Mock(
        return_value=[Path("path_raw")]
    )
    curation_main_view._curation_service.get_seg1_images_list = Mock(
        return_value=[Path("path_seg1")]
    )
    curation_main_view.init_progress_bar = Mock()

    # Act
    curation_main_view.curation_setup()

    # Assert
    curation_main_view.init_progress_bar.assert_called_once()
    curation_main_view._curation_service.remove_all_images_from_viewer_layers.assert_called_once()
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
    assert curation_main_view.progress_bar.value() == 1


def test_next_image(curation_main_view: CurationMainView) -> None:
    # Arrange
    curation_main_view._update_curation_record = Mock()
    curation_main_view.raw_images = [None, Path("path_raw")]
    curation_main_view.seg1_images = [None, Path("path_seg1")]
    assert curation_main_view.curation_index == 0

    # Act
    curation_main_view.next_image()

    # Assert
    curation_main_view._update_curation_record.assert_called_once()
    assert curation_main_view.curation_index == 1
    curation_main_view._curation_service.remove_all_images_from_viewer_layers.assert_called_once()
    assert (
        curation_main_view._curation_service.add_image_to_viewer.call_count
        == 2
    )


def test_increment_progress_bar(curation_main_view: CurationMainView) -> None:
    # Arrange
    initial_value: int = curation_main_view.progress_bar.value()
    # Act
    curation_main_view._increment_progress_bar()
    # Assert
    assert curation_main_view.progress_bar.value() == initial_value + 1


# parametized test to test when we want to use the image in curation vs when we dont
# either way we save the raw image path and the seg1 image path as a CurationRecord
@pytest.mark.parametrize(
    "use_this_image, expected_result", [(True, True), (False, False)]
)
def test_update_curation_record(
    curation_main_view: CurationMainView,
    use_this_image: str,
    expected_result: bool,
) -> None:
    # Arrange
    curation_main_view.no_radio.setChecked(not use_this_image)
    raw_test_path: Path = Path("test_path_raw")
    curation_main_view.raw_images = [raw_test_path]
    seg1_test_path: Path = Path("test_path_seg1")
    curation_main_view.seg1_images = [seg1_test_path]

    # Act
    curation_main_view._update_curation_record()

    # Assert
    # Ensure last record in curation_record is the one we just added
    assert curation_main_view.curation_record[-1].to_use == expected_result
    assert curation_main_view.curation_record[-1].raw_file == raw_test_path
    assert curation_main_view.curation_record[-1].seg1 == seg1_test_path
