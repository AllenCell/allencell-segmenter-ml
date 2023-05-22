import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.view.sample_widget import SampleWidget
from allencell_ml_segmenter.model.sample_model import SampleModel, Event
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.view.sample_view import SampleViewController

@pytest.fixture
def sample_model():
    return SampleModel()

@pytest.fixture
def sample_view_controller(sample_model):
    return SampleViewController(sample_model)

def test_model_property():
    assert sample_view_controller.model == sample_model

def test_handle_event_training(sample_view_controller, sample_model):
    mock_set_label_text = Mock()
    # mock set label text method
    sample_view_controller.widget.setLabelText = mock_set_label_text

    sample_model.set_model_training(True)

    expected_label_text = f"training is running {sample_model.get_model_training()}"
    mock_set_label_text.assert_called_once_with(expected_label_text)

def test_change_label(sample_view_controller, sample_model):
    sample_view_controller.change_label()

    assert sample_model.set_model_training.called_once_with(not sample_model.get_model_training())

def test_load(sample_view_controller):
    mock_layout = Mock()
    mock_widget = Mock()
    sample_view_controller.setLayout = mock_layout
    sample_view_controller.widget = mock_widget

    sample_view_controller.load()

    mock_layout.setContentsMargins.assert_called_once_with(0, 0, 0, 0)
    mock_layout.addWidget.assert_called_once_with(mock_widget)
    mock_widget.connectSlots.assert_called_once_with(sample_view_controller.change_label)

