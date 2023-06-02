import pytest
from unittest.mock import Mock, patch
from allencell_ml_segmenter.model.training_model import TrainingModel
from allencell_ml_segmenter.view.training_view import (
    TrainingView,
)
from allencell_ml_segmenter.model.publisher import Event
from allencell_ml_segmenter.controller.training_controller import (
    TrainingController,
)


@pytest.fixture
def mock_application() -> Mock:
    return Mock()


@pytest.fixture
def mock_model() -> Mock:
    return Mock(spec=TrainingModel)


@pytest.fixture
def ui_controller(
    mock_application: Mock, mock_model: Mock
) -> TrainingController:
    with patch(
        "allencell_ml_segmenter.controller.ui_controller.SampleViewController"
    ):
        return TrainingController(mock_application, mock_model)


def test_handle_event(ui_controller: TrainingController) -> None:
    pass


def test_index(
    ui_controller: TrainingController, mock_application: Mock
) -> None:
    ui_controller.index()

    # ensure view is loaded into app and label is changed
    mock_application.view_manager.load_view.assert_called_once_with(
        ui_controller.view
    )
    ui_controller.view.connect_slots.assert_called_once()


def test_load_view(ui_controller, mock_application):
    ui_controller.load_view()

    # ensure view is loaded into app and label is changed
    mock_application.view_manager.load_view.assert_called_once_with(
        ui_controller.view
    )
