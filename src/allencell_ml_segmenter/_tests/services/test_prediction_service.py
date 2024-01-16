from pathlib import Path
from typing import List

from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter.config.i_user_settings import IUserSettings
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.prediction.model import PredictionModel
import pytest
from unittest.mock import patch

from allencell_ml_segmenter.services.prediction_service import (
    PredictionService,
)


@pytest.fixture
def fake_user_settings() -> IUserSettings:
    """
    Fixture for ExperimentsModel Testing
    """
    return FakeUserSettings()


@pytest.mark.skip(reason="cyto_dl mock currently breaks this test on CI")
def test_predict_model() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    experiments_model: ExperimentsModel = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
            user_experiments_path=Path(__file__).parent.parent
            / "main"
            / "experiments_home",
        )
    )
    experiments_model.set_experiment_name("0_exp")
    experiments_model.set_checkpoint("1.ckpt")
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )

    # Act
    with patch("cyto_dl.api.model.CytoDLModel.predict") as patched_api:
        prediction_model.dispatch(Event.PROCESS_PREDICTION)

    # Assert
    patched_api.assert_called_once()


@pytest.mark.skip(reason="cyto_dl mock currently breaks this test on CI")
def test_predict_model_no_experiment_selected() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    experiments_model: ExperimentsModel = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
            user_experiments_path=Path(__file__).parent.parent
            / "main"
            / "experiments_home",
        )
    )
    experiments_model.set_checkpoint("1.ckpt")
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )

    # Act
    with patch("cyto_dl.api.model.CytoDLModel.predict") as patched_api:
        prediction_model.dispatch(Event.PROCESS_PREDICTION)

    # Assert
    patched_api.assert_not_called()


@pytest.mark.skip(reason="cyto_dl mock currently breaks this test on CI")
def test_predict_model_no_checkpoint_selected() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    experiments_model: ExperimentsModel = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
            user_experiments_path=Path(__file__).parent.parent
            / "main"
            / "experiments_home",
        )
    )
    experiments_model.set_experiment_name("0_exp")
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )

    # Act
    with patch("cyto_dl.api.model.CytoDLModel.predict") as patched_api:
        prediction_model.dispatch(Event.PROCESS_PREDICTION)

    # Assert
    patched_api.assert_not_called()


def test_build_default_prediction_overrides() -> None:
    # Arrange
    prediction_model: PredictionModel = PredictionModel()
    experiments_model: ExperimentsModel = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
            user_experiments_path=Path(__file__).parent.parent
            / "main"
            / "experiments_home",
        )
    )
    experiments_model.set_experiment_name("0_exp")
    experiments_model.set_checkpoint("1.ckpt")
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )

    # act
    overrides: List[
        str
    ] = prediction_service._build_default_prediction_overrides(
        experiments_model.get_experiment_name(),
        experiments_model.get_experiment_name(),
    )

    # assert
    # need these for prediction runs
    assert "test=False" in overrides
    assert "train=False" in overrides
    assert "mode=predict" in overrides
    assert (
        f"ckpt_path={str(experiments_model.get_model_checkpoints_path(experiments_model.get_experiment_name(), experiments_model.get_experiment_name()))}"
        in overrides
    )
