from typing import List

import pytest
from qtpy.QtWidgets import QLabel, QComboBox, QPushButton
from qtpy.QtCore import Qt
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.input_view import CurationInputView
from allencell_ml_segmenter.curation.curation_model import CurationModel


@pytest.fixture
def curation_model() -> CurationModel:
    return CurationModel()


@pytest.fixture
def curation_input_view(curation_model: CurationModel, qtbot: QtBot) -> None:
    return CurationInputView(curation_model)


def test_raw_channel_selected(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Simulate selecting a raw channel in the combo box
    combo_box: QComboBox = curation_input_view._raw_image_channel_combo
    combo_box.setCurrentIndex(1)
    curation_input_view.raw_channel_selected(1)

    assert curation_model.get_raw_channel() == 1


def test_seg1_channel_selected(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Simulate selecting a seg1 channel in the combo box
    combo_box: QComboBox = curation_input_view._seg1_image_channel_combo
    combo_box.setCurrentIndex(2)
    curation_input_view.seg1_channel_selected(2)

    assert curation_model.get_seg1_channel() == 2


def test_seg2_channel_selected(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Simulate selecting a seg2 channel in the combo box
    combo_box: QComboBox = curation_input_view._seg2_image_channel_combo
    combo_box.setCurrentIndex(0)
    curation_input_view.seg2_channel_selected(0)

    assert curation_model.get_seg2_channel() == 0


def test_update_raw_channels(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Simulate the raw image directory being selected
    event: Event = Event.ACTION_CURATION_RAW_SELECTED
    curation_model._raw_image_channel_count = 3

    curation_input_view.update_raw_channels(event)

    combo_box: QComboBox = curation_input_view._raw_image_channel_combo
    expected_items: List[int] = [
        str(x) for x in range(curation_model.get_total_num_channels_raw())
    ]

    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]


def test_update_seg1_channels(
    curation_input_view: CurationInputView, curation_model: CurationModel
) -> None:
    # Simulate the seg1 image directory being selected
    event: Event = Event.ACTION_CURATION_SEG1_SELECTED
    curation_model._seg1_image_channel_count = 5

    curation_input_view.update_seg1_channels(event)

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
    # Simulate the seg2 image directory being selected
    event: Event = Event.ACTION_CURATION_SEG2_SELECTED
    curation_model._seg2_image_channel_count = 10
    curation_input_view.update_seg2_channels(event)

    combo_box: QComboBox = curation_input_view._seg2_image_channel_combo
    expected_items = [
        str(x) for x in range(curation_model.get_total_num_channels_seg2())
    ]
    assert combo_box.count() == len(expected_items)
    for i in range(combo_box.count()):
        assert combo_box.itemText(i) == expected_items[i]
