import pytest
from qtpy.QtCore import Qt
from unittest.mock import patch, Mock
from allencell_ml_segmenter.prediction.model_input_widget import (
    ModelInputWidget,
)
from allencell_ml_segmenter.prediction.model import PredictionModel


@pytest.fixture
def model_input_widget(qtbot):
    """
    Fixture that creates an instance of ModelInputWidget for testing.
    """
    return ModelInputWidget(PredictionModel())


def test_top_radio_button_slot(qtbot, model_input_widget):
    """
    Test the _top_radio_button_slot method of ModelInputWidget.
    TODO: replace test once magicgui is being used for top widget
    """
    pass


def test_bottom_radio_button_slot(qtbot, model_input_widget):
    """
    Test the _bottom_radio_button_slot method of ModelInputWidget.
    """
    # Enable the bottom input box and disable the top input box
    model_input_widget._bottom_postproc_button_slot()

    assert not model_input_widget._simple_thresh_slider.native.isEnabled()
    assert model_input_widget._auto_thresh_selection.isEnabled()
    assert (
        model_input_widget._model.get_postprocessing_method()
        == ModelInputWidget.BOTTOM_TEXT
    )


def test_call_setters(model_input_widget):
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


def test_configure_slots(qtbot, model_input_widget):
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


def test_postprocessing_method(model_input_widget, qtbot):
    # ACT
    qtbot.mouseClick(model_input_widget._top_postproc_button, Qt.LeftButton)

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_method()
        == "simple threshold"
    )

    # ACT
    qtbot.mouseClick(model_input_widget._bottom_postproc_button, Qt.LeftButton)

    # ASSERT
    assert (
        model_input_widget._model.get_postprocessing_method()
        == "auto threshold"
    )


def test_postprocessing_auto_threshold(model_input_widget):
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
