import pytest
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.sample.sample_model import SampleModel
from allencell_ml_segmenter.sample.sample_state_widget import (
    TRAINING_NOT_RUNNING,
    TRAINING_RUNNING,
    SampleStateWidget,
)


@pytest.fixture
def sample_model():
    return SampleModel()


@pytest.fixture
def sample_state_widget(sample_model):
    return SampleStateWidget(sample_model)


def test_handles_process_training_evt(
    qtbot, sample_state_widget, sample_model
):
    # ARRANGE
    qtbot.addWidget(sample_state_widget)

    # ACT
    sample_model.set_process_running(True)

    # ASSERT
    assert sample_state_widget._label.text() == TRAINING_RUNNING


def test_handles_process_training_show_error_evt(
    qtbot, sample_state_widget, sample_model
):
    # ARRANGE
    qtbot.addWidget(sample_state_widget)

    # ACT
    sample_model.set_error_message("Error")

    # ASSERT
    assert sample_state_widget._label.text() == "Error"

    # ACT
    sample_model.dispatch(Event.PROCESS_TRAINING_CLEAR_ERROR)

    # ASSERT
    assert sample_state_widget._label.text() == TRAINING_NOT_RUNNING
