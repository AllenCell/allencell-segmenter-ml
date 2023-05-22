import pytest
from unittest.mock import Mock, patch
from allencell_ml_segmenter.model.sample_model import SampleModel
from allencell_ml_segmenter.view.sample_view import SampleViewController
from allencell_ml_segmenter.model.pub_sub import Event


@pytest.fixture
def sample_model() -> SampleModel:
    return SampleModel()

@pytest.fixture
def sample_view_controller(sample_model: SampleModel) -> SampleViewController:
    with patch("allencell_ml_segmenter.view.sample_view.SampleWidget"):
        return SampleViewController(sample_model)

def test_handle_event(sample_view_controller, sample_model):
    controller = sample_view_controller
    sample_model.set_model_training(True)

    sample_view_controller.widget.setLabelText.assert_called_once_with(
        f"training is running {sample_model.get_model_training()}")
    # mock samplewidget but keep a real instance of the model
