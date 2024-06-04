from dataclasses import dataclass
from unittest.mock import Mock, patch
from typing import List
from pathlib import Path

import pytest
from qtpy.QtWidgets import QComboBox, QFileDialog
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.curation.input_view import CurationInputView
from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    CurationImageType,
    CurationView,
)
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from napari.utils.notifications import show_info


@dataclass
class TestEnvironment:
    model: CurationModel
    view: CurationInputView


@pytest.fixture
def test_env() -> TestEnvironment:
    model: CurationModel = CurationModel(FakeExperimentsModel())
    return TestEnvironment(model, CurationInputView(model))


MOCK_STR_PATH: str = "mock_path"
MOCK_DIR_PATHS: List[Path] = [Path("p1"), Path("p2")]

### UI State Tests ----------------------------------------------------------------------------------


def assert_combo_box_matches_channel_count(
    combo_box: QComboBox, count: int
) -> None:
    """
    Assert that the given combo box contains options 0 through count - 1 inclusive
    in ascending order.
    """
    expected_items = [str(x) for x in range(count)]
    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]


@patch.multiple(
    QFileDialog,
    exec_=Mock(return_value=QFileDialog.Accepted),
    getExistingDirectory=Mock(return_value=MOCK_STR_PATH),
)
class TestsWithStubbedFileDialog:
    def test_raw_dir_selected(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Assert sanity check
        assert not test_env.view.raw_dir_stacked_spinner.spinner.is_spinning
        # Act
        test_env.view.raw_directory_select.button.click()
        # Assert
        assert test_env.view.raw_dir_stacked_spinner.spinner.is_spinning

    def test_seg1_dir_selected(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Assert sanity check
        assert not test_env.view.seg1_dir_stacked_spinner.spinner.is_spinning
        # Act
        test_env.view.seg1_directory_select.button.click()
        # Assert
        assert test_env.view.seg1_dir_stacked_spinner.spinner.is_spinning

    def test_seg2_dir_selected(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Assert sanity check
        assert not test_env.view.seg2_dir_stacked_spinner.spinner.is_spinning
        # Act
        test_env.view.seg2_directory_select.button.click()
        # Assert
        assert test_env.view.seg2_dir_stacked_spinner.spinner.is_spinning

    def test_raw_channel_count_set(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Arrange
        test_env.view.raw_directory_select.button.click()
        # Simulate the model's channel count being set
        count: int = 2
        combo_box: QComboBox = test_env.view.raw_image_channel_combo

        # Act
        test_env.model.set_channel_count(CurationImageType.RAW, count)

        # Assert
        assert not test_env.view.raw_dir_stacked_spinner.spinner.is_spinning
        assert_combo_box_matches_channel_count(combo_box, count)

    def test_seg1_channel_count_set(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Arrange
        test_env.view.seg1_directory_select.button.click()
        # Simulate the model's channel count being set
        count: int = 3
        combo_box: QComboBox = test_env.view.seg1_image_channel_combo

        # Act
        test_env.model.set_channel_count(CurationImageType.SEG1, count)

        # Assert
        assert not test_env.view.seg1_dir_stacked_spinner.spinner.is_spinning
        assert_combo_box_matches_channel_count(combo_box, count)

    def test_seg2_channel_count_set(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Arrange
        test_env.view.seg2_directory_select.button.click()
        # Simulate the model's channel count being set
        count: int = 4
        combo_box: QComboBox = test_env.view.seg2_image_channel_combo

        # Act
        test_env.model.set_channel_count(CurationImageType.SEG2, count)

        # Assert
        assert not test_env.view.seg2_dir_stacked_spinner.spinner.is_spinning
        assert_combo_box_matches_channel_count(combo_box, count)

    def test_start_raw_selected(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Arrange
        test_env.view.raw_directory_select.button.click()
        test_env.model.set_channel_count(CurationImageType.RAW, 5)
        test_env.view.raw_image_channel_combo.setCurrentIndex(1)

        # Assert (sanity check)
        assert test_env.model.get_current_view() == CurationView.INPUT_VIEW

        # Act
        test_env.view.start_btn.click()

        # Assert
        # we expect the start button to fail since we haven't selected seg1 -- should still be input view
        assert test_env.model.get_current_view() == CurationView.INPUT_VIEW

    def test_start_raw_seg1_selected(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Arrange
        test_env.view.raw_directory_select.button.click()
        test_env.model.set_channel_count(CurationImageType.RAW, 5)
        test_env.model.set_image_directory_paths(
            CurationImageType.RAW, MOCK_DIR_PATHS
        )
        test_env.view.raw_image_channel_combo.setCurrentIndex(1)

        test_env.view.seg1_directory_select.button.click()
        test_env.model.set_channel_count(CurationImageType.SEG1, 5)
        test_env.model.set_image_directory_paths(
            CurationImageType.SEG1, MOCK_DIR_PATHS
        )
        test_env.view.seg1_image_channel_combo.setCurrentIndex(2)

        # Assert (sanity check)
        assert test_env.model.get_current_view() == CurationView.INPUT_VIEW

        # Act
        test_env.view.start_btn.click()

        # Assert
        # we expect the start button to work since we have at least raw and seg1
        assert test_env.model.get_current_view() == CurationView.MAIN_VIEW

    def test_start_all_selected(
        self, qtbot: QtBot, test_env: TestEnvironment
    ) -> None:
        # Arrange
        test_env.view.raw_directory_select.button.click()
        test_env.model.set_channel_count(CurationImageType.RAW, 5)
        test_env.model.set_image_directory_paths(
            CurationImageType.RAW, MOCK_DIR_PATHS
        )
        test_env.view.raw_image_channel_combo.setCurrentIndex(1)

        test_env.view.seg1_directory_select.button.click()
        test_env.model.set_channel_count(CurationImageType.SEG1, 5)
        test_env.model.set_image_directory_paths(
            CurationImageType.SEG1, MOCK_DIR_PATHS
        )
        test_env.view.seg1_image_channel_combo.setCurrentIndex(2)

        test_env.view.seg2_directory_select.button.click()
        test_env.model.set_channel_count(CurationImageType.SEG2, 5)
        test_env.model.set_image_directory_paths(
            CurationImageType.SEG2, MOCK_DIR_PATHS
        )
        test_env.view.seg2_image_channel_combo.setCurrentIndex(3)

        # Assert (sanity check)
        assert test_env.model.get_current_view() == CurationView.INPUT_VIEW

        # Act
        test_env.view.start_btn.click()

        # Assert
        # we expect the start button to work since we have all selected
        assert test_env.model.get_current_view() == CurationView.MAIN_VIEW


### View + Model Integration Tests ------------------------------------------------------------------------


def test_raw_channel_selected(qtbot: QtBot, test_env: TestEnvironment) -> None:
    # Arrange
    test_env.model.set_channel_count(CurationImageType.RAW, 3)
    combo_box: QComboBox = test_env.view.raw_image_channel_combo
    # Act / Assert
    assert test_env.model.get_selected_channel(CurationImageType.RAW) == 0
    combo_box.setCurrentIndex(0)
    assert test_env.model.get_selected_channel(CurationImageType.RAW) == 0
    combo_box.setCurrentIndex(1)
    assert test_env.model.get_selected_channel(CurationImageType.RAW) == 1
    combo_box.setCurrentIndex(2)
    assert test_env.model.get_selected_channel(CurationImageType.RAW) == 2


def test_seg1_channel_selected(
    qtbot: QtBot, test_env: TestEnvironment
) -> None:
    # Arrange
    test_env.model.set_channel_count(CurationImageType.SEG1, 3)
    combo_box: QComboBox = test_env.view.seg1_image_channel_combo
    # Act / Assert
    assert test_env.model.get_selected_channel(CurationImageType.SEG1) == 0
    combo_box.setCurrentIndex(0)
    assert test_env.model.get_selected_channel(CurationImageType.SEG1) == 0
    combo_box.setCurrentIndex(1)
    assert test_env.model.get_selected_channel(CurationImageType.SEG1) == 1
    combo_box.setCurrentIndex(2)
    assert test_env.model.get_selected_channel(CurationImageType.SEG1) == 2


def test_seg2_channel_selected(
    qtbot: QtBot, test_env: TestEnvironment
) -> None:
    # Arrange
    test_env.model.set_channel_count(CurationImageType.SEG2, 3)
    combo_box: QComboBox = test_env.view.seg2_image_channel_combo
    # Act / Assert
    assert test_env.model.get_selected_channel(CurationImageType.SEG2) == 0
    combo_box.setCurrentIndex(0)
    assert test_env.model.get_selected_channel(CurationImageType.SEG2) == 0
    combo_box.setCurrentIndex(1)
    assert test_env.model.get_selected_channel(CurationImageType.SEG2) == 1
    combo_box.setCurrentIndex(2)
    assert test_env.model.get_selected_channel(CurationImageType.SEG2) == 2
