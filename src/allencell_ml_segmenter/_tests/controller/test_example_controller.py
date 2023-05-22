import pytest
from unittest.mock import Mock, patch
from allencell_ml_segmenter.model.sample_model import SampleModel
from allencell_ml_segmenter.view.sample_view import SampleViewController
from allencell_ml_segmenter.model.pub_sub import Event
from allencell_ml_segmenter.controller.example_controller import UiController

@pytest.fixture
def mock_application() -> Mock:
    return Mock()


@pytest.fixture
def mock_model() -> Mock:
    return Mock(spec=SampleModel)


@pytest.fixture
def ui_controller(mock_application: Mock, mock_model: Mock) -> UiController:
    with patch("allencell_ml_segmenter.controller.example_controller.SampleView"):
        return UiController(mock_application, mock_model)


def test_handle_event(ui_controller: UiController) -> None:
    # training event
    event: Event = Event.TRAINING
    ui_controller.handle_event(event)

    # ensure view method is called
    ui_controller.view.widget.label.setText.assert_called_once_with(
        f"training is running {ui_controller._model.get_model_training()}"
    )


def test_index(ui_controller: UiController, mock_application: Mock) -> None:
    ui_controller.index()

    # ensure view is loaded into app and label is changed
    mock_application.view_manager.load_view.assert_called_once_with(ui_controller.view)
    ui_controller.view.widget.btn.clicked.connect.assert_called_once_with(ui_controller.change_label)


def test_change_label(ui_controller: UiController, mock_model: Mock):
    # Call the change_label method
    ui_controller.change_label()

    # Assert
    mock_model.set_model_training.assert_called_once_with(
        not ui_controller._model.get_model_training()
    )