from pathlib import Path
import pytest
from pytestqt.qtbot import QtBot
from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.view import PredictionView


@pytest.fixture
def main_model() -> MainModel:
    """
    Returns a MainModel instance for testing.
    """
    return MainModel(ExperimentsModel(CytoDlConfig(Path(), Path())))


@pytest.fixture
def prediction_view(main_model: MainModel, qtbot: QtBot) -> PredictionView:
    """
    Returns a PredictionView instance for testing.
    """
    return PredictionView(main_model)


def test_prediction_view(
    prediction_view: PredictionView, main_model: MainModel
) -> None:
    """
    Tests that the PredictionView correctly sets the current view to itself
    """
    # ACT
    main_model.dispatch(Event.PROCESS_TRAINING_COMPLETE)

    # ASSERT
    assert main_model.get_current_view() == prediction_view
