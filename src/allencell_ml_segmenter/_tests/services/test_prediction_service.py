import csv
from pathlib import Path
from typing import List, Dict, Union

from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter.config.i_user_settings import IUserSettings
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.prediction.model import PredictionModel
import pytest
from unittest.mock import patch, MagicMock, mock_open, call

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
    experiments_model.apply_experiment_name("0_exp")
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
    experiments_model.apply_experiment_name("0_exp")
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )

    # Act
    with patch("cyto_dl.api.model.CytoDLModel.predict") as patched_api:
        prediction_model.dispatch(Event.PROCESS_PREDICTION)

    # Assert
    patched_api.assert_not_called()


def test_build_overrides() -> None:
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
    experiments_model.apply_experiment_name("one_ckpt_exp")
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )
    prediction_model.set_output_directory(
        Path(__file__).parent.parent
        / "main"
        / "0_exp"
        / "prediction_output_test"
    )
    prediction_model.set_image_input_channel_index(3)

    # act
    overrides: Dict[str, Union[str, int, float, bool]] = (
        prediction_service.build_overrides(
            experiments_model.get_experiment_name(),
            experiments_model.get_checkpoint(),
        )
    )

    # assert
    # Requried overrides- need these for prediction runs
    assert overrides["test"] == False
    assert overrides["train"] == False
    assert overrides["mode"] == "predict"
    assert overrides["task_name"] == "predict_task_from_app"
    assert overrides["ckpt_path"] == str(
        Path(__file__).parent.parent
        / "main"
        / "experiments_home"
        / "one_ckpt_exp"
        / "checkpoints"
        / "epoch_000.ckpt"
    )

    # optional overrides
    assert overrides["paths.output_dir"] == str(
        Path(__file__).parent.parent
        / "main"
        / "0_exp"
        / "prediction_output_test"
    )
    assert overrides["data.transforms.predict.transforms[1].reader[0].C"] == 3
    assert overrides["data.columns"] == ["raw", "split"]
    assert overrides["data.split_column"] == "split"


def test_build_overrides_experiment_none() -> None:
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
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )
    prediction_model.set_output_directory(
        Path(__file__).parent.parent
        / "main"
        / "0_exp"
        / "prediction_output_test"
    )
    prediction_model.set_image_input_channel_index(3)

    # act/assert
    # Experiment name is None, so build_overrides should throw a ValueError
    with pytest.raises(ValueError):
        overrides: Dict[str, Union[str, int, float, bool]] = (
            prediction_service.build_overrides(
                experiments_model.get_experiment_name(),
                experiments_model.get_checkpoint(),
            )
        )


def test_build_overrides_checkpoint_none() -> None:
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
    experiments_model.apply_experiment_name("0_exp")
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )
    prediction_model.set_output_directory(
        Path(__file__).parent.parent
        / "main"
        / "0_exp"
        / "prediction_output_test"
    )
    prediction_model.set_image_input_channel_index(3)

    # act/assert
    # Checkpoint is None, so build_overrides should throw a ValueError
    with pytest.raises(ValueError):
        overrides: Dict[str, Union[str, int, float, bool]] = (
            prediction_service.build_overrides(
                experiments_model.get_experiment_name(),
                experiments_model.get_checkpoint(),
            )
        )


def test_write_csv_for_inputs() -> None:
    # Arrange
    experiments_model: ExperimentsModel = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
            user_experiments_path=Path(__file__).parent.parent
            / "main"
            / "experiments_home",
        )
    )
    experiments_model.apply_experiment_name("0_exp")
    prediction_model: PredictionModel = PredictionModel()
    prediction_service: PredictionService = PredictionService(
        prediction_model, experiments_model
    )
    mock_csv_write = MagicMock(spec=csv.writer)

    # Act
    with patch("builtins.open", mock_open()) as mock_file_open:
        with patch("csv.writer", mock_csv_write):
            prediction_service.write_csv_for_inputs(["image1", "image2"])

    # Assert that CSV.write is called with correct rows
    assert call().writerow(["", "raw", "split"]) in mock_csv_write.mock_calls
    assert (
        call().writerow(["0", "image1", "test"]) in mock_csv_write.mock_calls
    )
    assert (
        call().writerow(["1", "image2", "test"]) in mock_csv_write.mock_calls
    )
