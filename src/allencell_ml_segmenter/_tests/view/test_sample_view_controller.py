import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.widgets.sample_widget import SampleWidget
from allencell_ml_segmenter.model.training_model import TrainingModel
from allencell_ml_segmenter.view.sample_view_controller import (
    SampleViewController,
)

from qtpy.QtWidgets import QVBoxLayout


@pytest.fixture
def sample_model():
    return TrainingModel()


@pytest.fixture
def sample_view_controller(sample_model, qtbot):
    return SampleViewController(sample_model)


def test_model_property(sample_view_controller, sample_model):
    assert sample_view_controller.model == sample_model


def test_handle_event_training(sample_view_controller, sample_model):
    mock_set_label_text = Mock()
    # mock set label text method
    sample_view_controller.widget.setLabelText = mock_set_label_text

    sample_model.set_model_training(True)

    expected_label_text = (
        f"training is running {sample_model.get_model_training()}"
    )
    mock_set_label_text.assert_called_once_with(expected_label_text)


def test_change_label(sample_view_controller, sample_model):
    assert sample_model.get_model_training() == False

    sample_view_controller.change_label()
    assert sample_model.get_model_training() == True


def test_load_and_setup_ui(sample_view_controller):
    sample_view_controller.load()
    assert isinstance(sample_view_controller.layout(), QVBoxLayout)
    assert isinstance(sample_view_controller.widget, SampleWidget)
