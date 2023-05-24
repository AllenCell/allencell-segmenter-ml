from allencell_ml_segmenter.controller.training_controller import (
    TrainingController,
)
import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.model.training_model import TrainingModel
from allencell_ml_segmenter.model.publisher import Event


@pytest.fixture
def mock_application() -> Mock:
    return Mock()


@pytest.fixture
def mock_model() -> Mock:
    return Mock(spec=TrainingModel)


@pytest.fixture
def mock_training_service() -> Mock:
    return Mock()


@pytest.fixture
def training_controller(mock_application, mock_model):
    return TrainingController(
        mock_application, mock_model
    )


def test_handle_event_starts_training_when_model_training_is_true(
    training_controller, mock_model
):
    mock_model.set_model_training(True)

    event = Event.TRAINING
    training_controller.handle_event(event)

    #TODO once method is implemented, test call
