import pytest
from pathlib import Path

from PyQt5.QtCore import Qt

from allencell_ml_segmenter._tests.fakes.fake_experiments_model import FakeExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.thresholding.thresholding_model import ThresholdingModel
from allencell_ml_segmenter.core.file_input_model import FileInputModel, InputMode
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.thresholding.thresholding_view import ThresholdingView

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
def experiments_model() -> FakeExperimentsModel:
    return FakeExperimentsModel()

@pytest.fixture
def viewer() -> FakeViewer:
    return FakeViewer()

@pytest.fixture
def thresholding_view(main_model, thresholding_model, file_input_model, experiments_model, viewer, qtbot):
    view = ThresholdingView(main_model, thresholding_model, file_input_model, experiments_model, viewer)
    qtbot.addWidget(view)
    return view

def test_model_updates_on_slider_release(thresholding_view, thresholding_model):
    # Arrange
    initial_value: int = thresholding_model.get_thresholding_value()

    # Act: simulate slider was moved and released
    thresholding_view._threshold_value_slider.setValue(111)
    thresholding_view._threshold_value_slider.sliderReleased.emit()

    # Assert: Model value should update to match the slider's new value
    assert thresholding_model.get_thresholding_value() == 111
    assert thresholding_model.get_thresholding_value() != initial_value


def test_model_updates_on_spinbox_editing_finished(thresholding_view, thresholding_model, qtbot):
    # Arrange
    initial_value = thresholding_model.get_thresholding_value()

    # Act
    new_value = 122
    thresholding_view._threshold_value_spinbox.setValue(new_value)
    thresholding_view._threshold_value_spinbox.editingFinished.emit()

    # Assert: Model value should update to match the spinbox's new value
    assert thresholding_model.get_thresholding_value() == new_value
    assert thresholding_model.get_thresholding_value() != initial_value

def test_update_spinbox_from_slider(thresholding_view, qtbot):
    # Act: dragging
    for new_value in range(60, 101, 10):  # Simulate dragging from 60 to 100 in steps
        thresholding_view._threshold_value_slider.setValue(new_value)
        qtbot.wait(10)  # small delay

        # Assert that spinbox always stays updated
        assert thresholding_view._threshold_value_spinbox.value() == new_value

    # Act: clicking a value
    qtbot.mouseClick(thresholding_view._threshold_value_slider, Qt.LeftButton)
    clicked_value = 120
    thresholding_view._threshold_value_slider.setValue(clicked_value)

    # Assert that spinbox updated correctly
    assert thresholding_view._threshold_value_spinbox.value() == clicked_value

def test_update_slider_from_spinbox(thresholding_view, qtbot):
    # Act
    new_value = 100
    thresholding_view._threshold_value_spinbox.setValue(new_value)
    qtbot.keyPress(thresholding_view._threshold_value_spinbox, Qt.Key_Enter)  # Simulate user pressing "Enter"

    # Assert
    assert thresholding_view._threshold_value_slider.value() == new_value

def test_update_state_from_radios(thresholding_view, thresholding_model):
    # Arrange
    assert not thresholding_view._none_radio_button.isChecked()
    assert not thresholding_view._specific_value_radio_button.isChecked()
    assert not thresholding_view._autothreshold_radio_button.isChecked()

    # Act
    thresholding_view._specific_value_radio_button.setChecked(True)
    thresholding_view._update_state_from_radios()

    # Assert
    assert thresholding_model.is_threshold_enabled()
    assert not thresholding_model.is_autothresholding_enabled()
    assert thresholding_view._apply_save_button.isEnabled()

    thresholding_view._specific_value_radio_button.setChecked(False)
    thresholding_view._autothreshold_radio_button.setChecked(True)
    thresholding_view._update_state_from_radios()

    # Assert
    assert not thresholding_model.is_threshold_enabled()
    assert thresholding_model.is_autothresholding_enabled()
    assert thresholding_view._apply_save_button.isEnabled()


def test_check_able_to_threshold_valid(main_model, file_input_model, experiments_model, viewer):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    thresholding_model.set_threshold_enabled(True)
    thresholding_model.set_thresholding_value(100)
    thresholding_view: ThresholdingView = ThresholdingView(main_model, thresholding_model, file_input_model, experiments_model, viewer)

    assert thresholding_view._check_able_to_threshold

def test_check_able_to_threshold_no_output_dir(main_model, experiments_model, viewer):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    thresholding_model.set_threshold_enabled(True)
    thresholding_model.set_thresholding_value(100)
    file_input_model: FileInputModel = FileInputModel()
    file_input_model.set_input_mode(InputMode.FROM_PATH)
    file_input_model.set_input_image_path(Path("fake_path"))
    thresholding_view: ThresholdingView = ThresholdingView(main_model, thresholding_model, file_input_model, experiments_model, viewer)

    assert not thresholding_view._check_able_to_threshold

def test_check_able_to_threshold_no_input_dir(main_model, experiments_model, viewer):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    thresholding_model.set_threshold_enabled(True)
    thresholding_model.set_thresholding_value(100)
    file_input_model: FileInputModel = FileInputModel()
    file_input_model.set_input_mode(InputMode.FROM_PATH)
    file_input_model.set_output_directory(Path("fake_path"))
    thresholding_view: ThresholdingView = ThresholdingView(main_model, thresholding_model, file_input_model, experiments_model, viewer)

    assert not thresholding_view._check_able_to_threshold

def test_check_able_to_threshold_no_input_method(main_model, experiments_model, viewer):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    thresholding_model.set_threshold_enabled(True)
    thresholding_model.set_thresholding_value(100)
    file_input_model: FileInputModel = FileInputModel()
    file_input_model.set_input_image_path(Path("fake_path"))
    file_input_model.set_output_directory(Path("fake_path"))
    thresholding_view: ThresholdingView = ThresholdingView(main_model, thresholding_model, file_input_model, experiments_model, viewer)

    assert not thresholding_view._check_able_to_threshold



