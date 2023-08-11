from pathlib import Path

import pytest
from unittest.mock import patch

from qtpy.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.prediction.model_input_widget import (
    ModelInputWidget,
)
from allencell_ml_segmenter.prediction.model import PredictionModel


@pytest.fixture
def model_input_widget(qtbot: QtBot):
    """
    Fixture that creates an instance of ModelInputWidget for testing.
    """
    return ModelInputWidget(PredictionModel())


def test_top_radio_button_slot(
    qtbot: QtBot, model_input_widget: ModelInputWidget
) -> None:
    """
    Test the _top_radio_button_slot method of ModelInputWidget.
    """
    # ARRANGE - explicitly enable all input UI elements (slider and combo box)
    model_input_widget._simple_thresh_slider.native.setEnabled(True)
    model_input_widget._auto_thresh_selection.setEnabled(True)

    # ACT
    with qtbot.waitSignals([model_input_widget._top_postproc_button.toggled]):
        model_input_widget._top_postproc_button.click()

    # ASSERT - both input UI elements should be disabled, postprocessing method should be set to TOP_TEXT
    assert not model_input_widget._simple_thresh_slider.native.isEnabled()
    assert not model_input_widget._auto_thresh_selection.isEnabled()
    assert (
        model_input_widget._model.get_postprocessing_method()
        == ModelInputWidget.TOP_TEXT
    )


def test_mid_radio_button_slot(
    qtbot: QtBot, model_input_widget: ModelInputWidget
) -> None:
    """
    Test the _mid_radio_button_slot method of ModelInputWidget.
    """
    # ARRANGE - explicitly disable the slider and enable the combo box
    model_input_widget._simple_thresh_slider.native.setEnabled(False)
    model_input_widget._auto_thresh_selection.setEnabled(True)

    # ACT
    with qtbot.waitSignals([model_input_widget._mid_postproc_button.toggled]):
        model_input_widget._mid_postproc_button.click()

    # ASSERT - slider should be enabled, combo box should be disabled, postprocessing method should be set to MID_TEXT
    assert model_input_widget._simple_thresh_slider.native.isEnabled()
    assert not model_input_widget._auto_thresh_selection.isEnabled()
    assert (
        model_input_widget._model.get_postprocessing_method()
        == ModelInputWidget.MID_TEXT
    )


def test_bottom_radio_button_slot(
    qtbot: QtBot, model_input_widget: ModelInputWidget
) -> None:
    """
    Test the _bottom_radio_button_slot method of ModelInputWidget.
    """
    # ARRANGE - explicitly enable the slider and disable the combo box
    model_input_widget._simple_thresh_slider.native.setEnabled(True)
    model_input_widget._auto_thresh_selection.setEnabled(False)

    # ACT
    with qtbot.waitSignals(
        [model_input_widget._bottom_postproc_button.toggled]
    ):
        model_input_widget._bottom_postproc_button.click()

    # ASSERT - slider should be disabled, combo box should be enabled, postprocessing method should be set to BOTTOM_TEXT
    assert not model_input_widget._simple_thresh_slider.native.isEnabled()
    assert model_input_widget._auto_thresh_selection.isEnabled()
    assert (
        model_input_widget._model.get_postprocessing_method()
        == ModelInputWidget.BOTTOM_TEXT
    )


def test_call_setters(model_input_widget: ModelInputWidget) -> None:
    """
    Test the _call_setters method of ModelInputWidget.
    """
    # TODO: fix after magicgui fix
    model_input_widget._call_setters()

    # Test default values for input fields
    # assert model_input_widget._bottom_input_box.count() == 12
    # assert model_input_widget._bottom_input_box.currentIndex() == -1
    # assert model_input_widget._bottom_input_box.isEditable() is False
    # assert model_input_widget._bottom_input_box.placeholderText() == "select a method"
    #
    # # Test disabling input fields
    # assert not model_input_widget._top_input_box.isEnabled()
    # assert not model_input_widget._bottom_input_box.isEnabled()


def test_configure_slots(
    qtbot: QtBot, model_input_widget: ModelInputWidget
) -> None:
    """
    Test the _configure_slots method of ModelInputWidget.
    """
    # Simulate selecting an option in the bottom input box
    with patch.object(
        model_input_widget._model, "set_postprocessing_auto_threshold"
    ) as mock_set_threshold:
        model_input_widget._auto_thresh_selection.setCurrentIndex(0)
        qtbot.wait(100)

    # Verify that the corresponding method was called on the model
    mock_set_threshold.assert_called_once_with("isodata")


def test_model_path(
    qtbot: QtBot, model_input_widget: ModelInputWidget
) -> None:
    """
    Tests that selecting a model file updates the model path.
    """
    # ARRANGE
    dummy_path: str = "/path/to/file"
    with patch.object(
        QFileDialog, "getOpenFileName", return_value=(dummy_path, "")
    ):
        with qtbot.waitSignals(
            [model_input_widget._input_button._button.clicked]
        ):
            model_input_widget._input_button._button.click()

    # ASSERT
    assert model_input_widget._model.get_model_path() == Path(dummy_path)


def test_postprocessing_method(
    qtbot: QtBot, model_input_widget: ModelInputWidget
) -> None:
    # ACT
    with qtbot.waitSignals([model_input_widget._top_postproc_button.toggled]):
        model_input_widget._top_postproc_button.click()

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_method()
        == ModelInputWidget.TOP_TEXT
    )

    # ACT
    with qtbot.waitSignals([model_input_widget._mid_postproc_button.toggled]):
        model_input_widget._mid_postproc_button.click()

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_method()
        == ModelInputWidget.MID_TEXT
    )

    # ACT
    with qtbot.waitSignals(
        [model_input_widget._bottom_postproc_button.toggled]
    ):
        model_input_widget._bottom_postproc_button.click()

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_method()
        == ModelInputWidget.BOTTOM_TEXT
    )


def test_postprocessing_auto_threshold(
    model_input_widget: ModelInputWidget,
) -> None:
    # ACT
    model_input_widget._auto_thresh_selection.setCurrentIndex(4)

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_auto_threshold()
        == "minimum"
    )

    # ACT
    model_input_widget._auto_thresh_selection.setCurrentIndex(6)

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_auto_threshold()
        == "niblack"
    )
