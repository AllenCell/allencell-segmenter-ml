from typing import List
from unittest.mock import Mock
from dataclasses import dataclass

import pytest
from qtpy.QtWidgets import QComboBox
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.input_view import CurationInputView
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import FakeExperimentsModel
from allencell_ml_segmenter.curation.curation_image_loader import FakeCurationImageLoaderFactory


@dataclass
class TestEnvironment:
    model: CurationModel
    view: CurationInputView

@pytest.fixture
def test_env() -> TestEnvironment:
    model: CurationModel = CurationModel(FakeExperimentsModel(), FakeCurationImageLoaderFactory())
    return TestEnvironment(model, CurationInputView(model))

# TODO: figure out how to test the qfiledialog portion of this view

### UI State Tests ----------------------------------------------------------------------------------

def test_raw_channel_set(qtbot: QtBot, test_env: TestEnvironment) -> None:
    # Arrange
    # Simulate the model's channel count being set
    count: int = 2
    test_env.model.set_raw_image_channel_count(count)
    combo_box: QComboBox = test_env.view.raw_image_channel_combo
    # Act / Assert
    expected_items = [
        str(x) for x in range(count)
    ]
    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]


def test_seg1_channel_set(qtbot: QtBot, test_env: TestEnvironment) -> None:
    # Arrange
    # Simulate the model's channel count being set
    count: int = 3
    test_env.model.set_seg1_image_channel_count(count)
    combo_box: QComboBox = test_env.view.seg1_image_channel_combo
    # Act / Assert
    expected_items = [
        str(x) for x in range(count)
    ]
    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]


def test_seg2_channel_set(qtbot: QtBot, test_env: TestEnvironment) -> None:
    # Arrange
    # Simulate the model's channel count being set
    count: int = 4
    test_env.model.set_seg2_image_channel_count(count)
    combo_box: QComboBox = test_env.view.seg2_image_channel_combo
    # Act / Assert
    expected_items = [
        str(x) for x in range(count)
    ]
    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]

### View + Model Integration Tests ------------------------------------------------------------------------

def test_raw_channel_selected(qtbot: QtBot, test_env: TestEnvironment) -> None:
    # Arrange
    test_env.model.set_raw_image_channel_count(3)
    combo_box: QComboBox = test_env.view.raw_image_channel_combo
    # Act / Assert
    assert test_env.model.get_raw_channel() == 0
    combo_box.setCurrentIndex(0)
    assert test_env.model.get_raw_channel() == 0
    combo_box.setCurrentIndex(1)
    assert test_env.model.get_raw_channel() == 1
    combo_box.setCurrentIndex(2)
    assert test_env.model.get_raw_channel() == 2


def test_seg1_channel_selected(qtbot: QtBot, test_env: TestEnvironment) -> None:
    # Arrange
    test_env.model.set_seg1_image_channel_count(3)
    combo_box: QComboBox = test_env.view.seg1_image_channel_combo
    # Act / Assert
    assert test_env.model.get_seg1_channel() == 0
    combo_box.setCurrentIndex(0)
    assert test_env.model.get_seg1_channel() == 0
    combo_box.setCurrentIndex(1)
    assert test_env.model.get_seg1_channel() == 1
    combo_box.setCurrentIndex(2)
    assert test_env.model.get_seg1_channel() == 2


def test_seg2_channel_selected(qtbot: QtBot, test_env: TestEnvironment) -> None:
    # Arrange
    test_env.model.set_seg2_image_channel_count(3)
    combo_box: QComboBox = test_env.view.seg2_image_channel_combo
    # Act / Assert
    assert test_env.model.get_seg2_channel() == 0
    combo_box.setCurrentIndex(0)
    assert test_env.model.get_seg2_channel() == 0
    combo_box.setCurrentIndex(1)
    assert test_env.model.get_seg2_channel() == 1
    combo_box.setCurrentIndex(2)
    assert test_env.model.get_seg2_channel() == 2
