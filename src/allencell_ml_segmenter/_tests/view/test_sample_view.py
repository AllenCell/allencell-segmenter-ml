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
def sample_view_controller(sample_model, qtbot):
    return SampleViewController(sample_model)

def test_model_property(sample_view_controller, sample_model):
    assert sample_view_controller.model == sample_model

def test_handle_event_training(sample_view_controller, sample_model):
    mock_set_label_text = Mock()
    # mock set label text method
    sample_view_controller.widget.setLabelText = mock_set_label_text

    sample_model.set_model_training(True)

    expected_label_text = f"training is running {sample_model.get_model_training()}"
    mock_set_label_text.assert_called_once_with(expected_label_text)

