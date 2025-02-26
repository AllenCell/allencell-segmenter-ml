import pytest
from pathlib import Path

from qtpy.QtCore import Qt

from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.thresholding.thresholding_model import (
    ThresholdingModel,
)
from allencell_ml_segmenter.core.file_input_model import (
    InputMode,
)
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.thresholding.thresholding_view import (
    ThresholdingView,
)


@pytest.fixture
def main_model() -> MainModel:
    return MainModel()


@pytest.fixture
def thresholding_model(tmp_path: Path) -> ThresholdingModel:
    model = ThresholdingModel()
    model.set_thresholding_value(128)
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
def thresholding_view(
    main_model,
    thresholding_model,
    experiments_model,
    viewer,
    qtbot,
):
    view = ThresholdingView(
        main_model,
        thresholding_model,
        experiments_model,
        viewer,
    )
    qtbot.addWidget(view)
    return view


def test_model_updates_on_slider_release(
    thresholding_view, thresholding_model
):
    # this tests to see if the model updates when the slider releases
    # Arrange
    initial_value: int = thresholding_model.get_thresholding_value()

    # Act
    thresholding_view._threshold_value_slider.setValue(111)
    thresholding_view._threshold_value_slider.sliderReleased.emit()

    # Assert
    assert thresholding_model.get_thresholding_value() == 111
    assert thresholding_model.get_thresholding_value() != initial_value


def test_model_updates_on_spinbox_editing_finished(
    thresholding_view, thresholding_model
):
    # this tests to see if the model updates when the spinbox is edited
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
    # This tests to see if the spinbox updates live along with the slider
    # Act: dragging
    for new_value in range(
        60, 101, 10
    ):  # Simulate dragging from 60 to 100 in steps
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
    # This tests to see if the slider updates live when the spinbox is changed
    # Act
    new_value = 100
    thresholding_view._threshold_value_spinbox.setValue(new_value)
    qtbot.keyPress(
        thresholding_view._threshold_value_spinbox, Qt.Key_Enter
    )  # Simulate user pressing "Enter"

    # Assert
    assert thresholding_view._threshold_value_slider.value() == new_value


def test_update_state_from_radios(thresholding_view, thresholding_model):
    # This tests if ui/model updates correctly when the radio buttons are toggled
    # Arrange
    assert not thresholding_view._none_radio_button.isChecked()
    assert not thresholding_view._specific_value_radio_button.isChecked()
    assert not thresholding_view._autothreshold_radio_button.isChecked()
    assert not thresholding_view._apply_save_button.isEnabled()
    assert not thresholding_view._threshold_value_slider.isEnabled()
    assert not thresholding_view._threshold_value_spinbox.isEnabled()

    # Act
    thresholding_view._specific_value_radio_button.setChecked(True)
    thresholding_view._update_state_from_radios()

    # Assert
    assert thresholding_model.is_threshold_enabled()
    assert not thresholding_model.is_autothresholding_enabled()
    assert thresholding_view._apply_save_button.isEnabled()
    assert thresholding_view._threshold_value_slider.isEnabled()
    assert thresholding_view._threshold_value_spinbox.isEnabled()

    thresholding_view._specific_value_radio_button.setChecked(False)
    thresholding_view._autothreshold_radio_button.setChecked(True)
    thresholding_view._update_state_from_radios()

    # Assert
    assert not thresholding_model.is_threshold_enabled()
    assert thresholding_model.is_autothresholding_enabled()
    assert thresholding_view._apply_save_button.isEnabled()
    assert thresholding_view._autothreshold_method_combo.isEnabled()
    assert not thresholding_view._threshold_value_slider.isEnabled()
    assert not thresholding_view._threshold_value_spinbox.isEnabled()


def test_check_able_to_threshold_valid(
    main_model, experiments_model, viewer, tmp_path
):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    thresholding_model.set_threshold_enabled(True)
    thresholding_model.set_thresholding_value(100)
    thresholding_model.set_output_directory(tmp_path / "output")
    thresholding_model.set_input_image_path(tmp_path / "input")
    thresholding_model.set_input_mode(InputMode.FROM_PATH)

    thresholding_view: ThresholdingView = ThresholdingView(
        main_model,
        thresholding_model,
        experiments_model,
        viewer,
    )

    assert thresholding_view._check_able_to_threshold()


def test_check_able_to_threshold_no_output_dir(
    main_model, experiments_model, viewer
):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    thresholding_model.set_threshold_enabled(True)
    thresholding_model.set_thresholding_value(100)
    thresholding_model.set_input_mode(InputMode.FROM_PATH)
    thresholding_model.set_input_image_path(Path("fake_path"))
    thresholding_view: ThresholdingView = ThresholdingView(
        main_model,
        thresholding_model,
        experiments_model,
        viewer,
    )

    assert not thresholding_view._check_able_to_threshold()


def test_check_able_to_threshold_no_input_dir(
    main_model, experiments_model, viewer
):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    thresholding_model.set_threshold_enabled(True)
    thresholding_model.set_thresholding_value(100)
    thresholding_model.set_input_mode(InputMode.FROM_PATH)
    thresholding_model.set_output_directory(Path("fake_path"))
    thresholding_view: ThresholdingView = ThresholdingView(
        main_model,
        thresholding_model,
        experiments_model,
        viewer,
    )

    assert not thresholding_view._check_able_to_threshold()


def test_check_able_to_threshold_no_input_method(
    main_model, experiments_model, viewer
):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    thresholding_model.set_threshold_enabled(True)
    thresholding_model.set_thresholding_value(100)
    thresholding_model.set_input_image_path(Path("fake_path"))
    thresholding_model.set_output_directory(Path("fake_path"))
    thresholding_view: ThresholdingView = ThresholdingView(
        main_model,
        thresholding_model,
        experiments_model,
        viewer,
    )

    assert not thresholding_view._check_able_to_threshold()


def check_button_press_dispatches_event(
    thresholding_view, thresholding_model, qtbot
):
    # arrange
    fake_subscriber: FakeSubscriber = FakeSubscriber()
    thresholding_model.subscribe(
        Event.ACTION_SAVE_THRESHOLDING_IMAGES,
        fake_subscriber,
        fake_subscriber.handle,
    )

    # act
    qtbot.mouseClick(thresholding_view._apply_save_button, Qt.LeftButton)

    # assert that event was dispatched
    assert fake_subscriber.handled[Event.ACTION_SAVE_THRESHOLDING_IMAGES]
