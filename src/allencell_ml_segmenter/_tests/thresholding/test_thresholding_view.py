import pytest
from pathlib import Path

from PyQt5.QtCore import Qt

from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.thresholding.thresholding_model import ThresholdingModel
from allencell_ml_segmenter.core.file_input_model import FileInputModel, InputMode
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.main.i_viewer import FakeViewer  # Assuming this exists
from allencell_ml_segmenter.prediction.prediction_folder_progress_tracker import PredictionFolderProgressTracker
from allencell_ml_segmenter.views.thresholding_view import ThresholdingView

@pytest.fixture
def main_model() -> MainModel:
    return MainModel()

@pytest.fixture
def thresholding_model() -> ThresholdingModel:
    model = ThresholdingModel()
    model.set_thresholding_value(128)
    return model

@pytest.fixture
def file_input_model(tmp_path: Path) -> FileInputModel:
    model = FileInputModel()
    model.set_output_directory(tmp_path / "output")
    model.set_input_image_path(tmp_path / "input")
    model.set_input_mode(InputMode.FROM_PATH)
    return model

@pytest.fixture
def experiments_model() -> IExperimentsModel:
    return IExperimentsModel()

@pytest.fixture
def viewer() -> FakeViewer:
    return FakeViewer()

@pytest.fixture
def thresholding_view(main_model, thresholding_model, file_input_model, experiments_model, viewer, qtbot):
    view = ThresholdingView(main_model, thresholding_model, file_input_model, experiments_model, viewer)
    qtbot.addWidget(view)
    return view


def test_update_spinbox_from_slider_drag_and_click(thresholding_view, qtbot):
    """
    Test the behavior of _update_spinbox_from_slider when dragging and clicking the slider.
    """

    # Arrange: Set initial values for the slider and spinbox
    initial_value = 50
    thresholding_view._threshold_value_slider.setValue(initial_value)
    thresholding_view._threshold_value_spinbox.setValue(initial_value)
    assert thresholding_view._threshold_value_spinbox.value() == initial_value

    # Act: Simulate dragging the slider
    for new_value in range(60, 101, 10):  # Simulate dragging from 60 to 100 in steps
        thresholding_view._threshold_value_slider.setValue(new_value)
        qtbot.wait(10)  # Simulate a small delay during dragging

        # Assert: Spinbox value updates during dragging
        assert thresholding_view._threshold_value_spinbox.value() == new_value

    # Act: Simulate clicking on the slider
    qtbot.mouseClick(thresholding_view._threshold_value_slider, Qt.LeftButton)

    # Set a specific value programmatically for the slider to simulate user click
    clicked_value = 120
    thresholding_view._threshold_value_slider.setValue(clicked_value)

    # Assert: Spinbox updates after clicking
    assert thresholding_view._threshold_value_spinbox.value() == clicked_value

def test_update_slider_from_spinbox(thresholding_view, qtbot):
    """
    Test the behavior of _update_slider_from_spinbox using qtbot to simulate spinbox value change.
    """

    # Arrange: Set an initial value for the slider
    initial_value = 50
    thresholding_view._threshold_value_slider.setValue(initial_value)
    assert thresholding_view._threshold_value_slider.value() == initial_value

    # Act: Simulate changing the spinbox value
    new_value = 100
    thresholding_view._threshold_value_spinbox.setValue(new_value)
    qtbot.keyPress(thresholding_view._threshold_value_spinbox, Qt.Key_Enter)  # Simulate user pressing "Enter"

    # Assert: Slider value should update to match the spinbox
    assert thresholding_view._threshold_value_slider.value() == new_value

def test_enable_specific_threshold_widgets(thresholding_view, qtbot):
    """
    Test the behavior of _enable_specific_threshold_widgets by simulating enabling and disabling.
    """

    # Assert: Widgets are initially disabled
    assert not thresholding_view._threshold_value_slider.isEnabled()
    assert not thresholding_view._threshold_value_spinbox.isEnabled()

    # Act: Enable the specific threshold widgets
    qtbot.waitUntil(lambda: thresholding_view._enable_specific_threshold_widgets(True))

    # Assert: Widgets are enabled
    assert thresholding_view._threshold_value_slider.isEnabled()
    assert thresholding_view._threshold_value_spinbox.isEnabled()

    # Act: Disable the specific threshold widgets
    qtbot.waitUntil(lambda: thresholding_view._enable_specific_threshold_widgets(False))

    # Assert: Widgets are disabled
    assert not thresholding_view._threshold_value_slider.isEnabled()
    assert not thresholding_view._threshold_value_spinbox.isEnabled()

def test_autothreshold_radio_enables_combobox(thresholding_view, qtbot):
    """
    Test that clicking the "Autothreshold" radio button enables the combobox.
    """

    # Arrange: Ensure the combobox is initially disabled
    assert not thresholding_view._autothreshold_method_combo.isEnabled()

    # Act: Simulate clicking the "Autothreshold" radio button
    qtbot.mouseClick(thresholding_view._autothreshold_radio_button, Qt.LeftButton)

    # Assert: The combobox should now be enabled
    assert thresholding_view._autothreshold_method_combo.isEnabled()

    # Act: Switch to "None" radio button and verify combobox is disabled again
    qtbot.mouseClick(thresholding_view._none_radio_button, Qt.LeftButton)
    assert not thresholding_view._autothreshold_method_combo.isEnabled()

def test_update_state_from_radios(thresholding_view, thresholding_model, qtbot):
    # Initially all radio buttons are unchecked
    assert not thresholding_view._none_radio_button.isChecked()
    assert not thresholding_view._specific_value_radio_button.isChecked()
    assert not thresholding_view._autothreshold_radio_button.isChecked()

    # Act: Simulate toggling "Specific Value" radio button
    qtbot.mouseClick(thresholding_view._specific_value_radio_button, Qt.LeftButton)
    thresholding_view._update_state_from_radios()

    # Assert: Model state reflects specific value thresholding is enabled
    assert thresholding_model.is_threshold_enabled()
    assert not thresholding_model.is_autothresholding_enabled()
    assert thresholding_view._apply_save_button.isEnabled()

    # Act: Simulate toggling "Autothreshold" radio button
    qtbot.mouseClick(thresholding_view._autothreshold_radio_button, Qt.LeftButton)
    thresholding_view._update_state_from_radios()

    # Assert: Model state reflects autothresholding is enabled
    assert not thresholding_model.is_threshold_enabled()
    assert thresholding_model.is_autothresholding_enabled()
    assert thresholding_view._apply_save_button.isEnabled()

    # Act: Simulate toggling "None" radio button
    qtbot.mouseClick(thresholding_view._none_radio_button, Qt.LeftButton)
    thresholding_view._update_state_from_radios()

    # Assert: Model state reflects no thresholding is enabled
    assert not thresholding_model.is_threshold_enabled()
    assert not thresholding_model.is_autothresholding_enabled()
    assert not thresholding_view._apply_save_button.isEnabled()

def test_check_able_to_threshold(thresholding_view, file_input_model):
    file_input_model.set_output_directory(None)

    # Act
    able_to_threshold = thresholding_view._check_able_to_threshold()

    # Assert
    assert not able_to_threshold

    # Arrange: Add output directory, but remove input path
    file_input_model.set_output_directory(Path("/valid/output"))
    file_input_model.set_input_image_path(None)

    # Act
    able_to_threshold = thresholding_view._check_able_to_threshold()

    # Assert
    assert not able_to_threshold

    # Arrange: Restore valid preconditions
    file_input_model.set_input_image_path(Path("/valid/input"))
    thresholding_view._specific_value_radio_button.setChecked(True)

    # Act
    able_to_threshold = thresholding_view._check_able_to_threshold()

    # Assert
    assert able_to_threshold

def test_model_updates_on_slider_release(thresholding_view, thresholding_model, qtbot):
    """
    Test that the model updates when the slider is released.
    """

    # Arrange: Set an initial threshold value in the model
    initial_value = 50
    thresholding_model.set_thresholding_value(initial_value)
    assert thresholding_model.get_thresholding_value() == initial_value

    # Act: Simulate moving the slider and releasing it
    new_value = 100
    thresholding_view._threshold_value_slider.setValue(new_value)
    qtbot.mouseRelease(thresholding_view._threshold_value_slider, Qt.LeftButton)

    # Assert: Model value should update to match the slider's new value
    assert thresholding_model.get_thresholding_value() == new_value


def test_model_updates_on_spinbox_editing_finished(thresholding_view, thresholding_model, qtbot):
    """
    Test that the model updates when the spinbox editing is finished.
    """

    # Arrange: Set an initial threshold value in the model
    initial_value = 50
    thresholding_model.set_thresholding_value(initial_value)
    assert thresholding_model.get_thresholding_value() == initial_value

    # Act: Simulate changing the spinbox value and finishing editing
    new_value = 120
    thresholding_view._threshold_value_spinbox.setValue(new_value)
    qtbot.keyPress(thresholding_view._threshold_value_spinbox, Qt.Key_Enter)  # Simulate "Enter" to finish editing

    # Assert: Model value should update to match the spinbox's new value
    assert thresholding_model.get_thresholding_value() == new_value

def test_update_spinbox_from_slider(thresholding_view, qtbot):
    """
    Test the behavior of _update_spinbox_from_slider using qtbot to simulate slider value change.
    """

    # Arrange: Set an initial value for the spinbox
    initial_value = 50
    thresholding_view._threshold_value_spinbox.setValue(initial_value)
    assert thresholding_view._threshold_value_spinbox.value() == initial_value

    # Act: Simulate changing the slider value
    new_value = 100
    qtbot.mouseClick(thresholding_view._threshold_value_slider, Qt.LeftButton)
    thresholding_view._threshold_value_slider.setValue(new_value)  # Simulate user moving the slider

    # Assert: Spinbox value should update to match the slider
    assert thresholding_view._threshold_value_spinbox.value() == new_value

def test_update_spinbox_from_slider_and_model(thresholding_view, thresholding_model, qtbot):
    """
    Test the behavior of _update_spinbox_from_slider and ensure the model updates when slider value changes.
    """

    # Arrange: Set initial values for the slider, spinbox, and model
    initial_value = 50
    thresholding_view._threshold_value_slider.setValue(initial_value)
    thresholding_view._threshold_value_spinbox.setValue(initial_value)
    thresholding_model.set_thresholding_value(initial_value)
    assert thresholding_view._threshold_value_spinbox.value() == initial_value
    assert thresholding_model.get_thresholding_value() == initial_value

    # Act: Simulate changing the slider value
    new_value = 100
    qtbot.mouseClick(thresholding_view._threshold_value_slider, Qt.LeftButton)
    thresholding_view._threshold_value_slider.setValue(new_value)  # Simulate user moving the slider
    qtbot.mouseRelease(thresholding_view._threshold_value_slider, Qt.LeftButton)  # Simulate releasing the slider

    # Assert: Spinbox value should update to match the slider
    assert thresholding_view._threshold_value_spinbox.value() == new_value

    # Assert: Model value should update to match the slider's value after release
    assert thresholding_model.get_thresholding_value() == new_value




