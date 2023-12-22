from typing import List
from unittest.mock import Mock

import pytest
from qtpy.QtWidgets import QLabel, QComboBox, QPushButton
from qtpy.QtCore import Qt
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.input_view import CurationInputView
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import CurationService


@pytest.fixture
def curation_model() -> CurationModel:
    return CurationModel()


@pytest.fixture
def curation_input_view(curation_model: CurationModel, qtbot: QtBot) -> None:
    return CurationInputView(curation_model, Mock(spec=CurationService))


def test_raw_channel_selected(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Arrange
    # Simulate selecting a raw channel in the combo box
    combo_box: QComboBox = curation_input_view._raw_image_channel_combo
    combo_box.setCurrentIndex(1)

    # Act
    curation_input_view.raw_channel_selected(1)

    # Assert
    assert curation_model.get_raw_channel() == 1


def test_seg1_channel_selected(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Arrange
    # Simulate selecting a seg1 channel in the combo box
    combo_box: QComboBox = curation_input_view._seg1_image_channel_combo
    combo_box.setCurrentIndex(2)

    # Act
    curation_input_view.seg1_channel_selected(2)

    # Assert
    assert curation_model.get_seg1_channel() == 2


def test_seg2_channel_selected(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Arrange
    # Simulate selecting a seg2 channel in the combo box
    combo_box: QComboBox = curation_input_view._seg2_image_channel_combo
    combo_box.setCurrentIndex(0)

    # Act
    curation_input_view.seg2_channel_selected(0)

    # Assert
    assert curation_model.get_seg2_channel() == 0


def test_update_raw_channels(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Arrange
    # Simulate the raw image directory being selected
    event: Event = Event.ACTION_CURATION_RAW_SELECTED
    curation_model.set_raw_image_channel_count(3)

    # Act
    curation_input_view.update_raw_channels(event)
    combo_box: QComboBox = curation_input_view._raw_image_channel_combo
    expected_items: List[int] = [
        str(x) for x in range(curation_model.get_total_num_channels_raw())
    ]

    # Assert
    # See if combo box contains the correct selections for channels
    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]


def test_update_seg1_channels(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Arrange
    # Simulate the seg1 image directory being selected
    event: Event = Event.ACTION_CURATION_SEG1_SELECTED
    curation_model.set_seg1_image_channel_count(5)

    # Act
    curation_input_view.update_seg1_channels(event)

    # Assert
    # See if combo box contains the correct selections for channels
    combo_box: QComboBox = curation_input_view._seg1_image_channel_combo
    expected_items: List[int] = [
        str(x) for x in range(curation_model.get_total_num_channels_seg1())
    ]
    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]


def test_update_seg2_channels(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Arrange
    # Simulate the seg2 image directory being selected
    event: Event = Event.ACTION_CURATION_SEG2_SELECTED
    curation_model.set_seg2_image_channel_count(10)

    # Act
    curation_input_view.update_seg2_channels(event)

    # Assert
    # See if combo box contains the correct selections for channels
    combo_box: QComboBox = curation_input_view._seg2_image_channel_combo
    expected_items = [
        str(x) for x in range(curation_model.get_total_num_channels_seg2())
    ]
    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]
