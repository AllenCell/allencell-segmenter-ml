# from pathlib import Path
from pathlib import Path
import pytest
from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter.config.i_user_settings import IUserSettings

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
import allencell_ml_segmenter


@pytest.fixture
def experiments_model() -> ExperimentsModel:
    exp_path: Path = Path(allencell_ml_segmenter.__file__).parent / "_tests" / "main" / "experiments_home"
    experiments_model = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(), user_experiments_path=exp_path
        )
    )
    experiments_model.set_experiment_name("2_exp")
    return experiments_model


def test_refresh_experiments(experiments_model: ExperimentsModel) -> None:
    expected = ["0_exp", "1_exp", "2_exp"]
    assert experiments_model.get_experiments() == expected


def test_get_cyto_dl_config() -> None:
    expected_config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=Path(__file__).parent / "experiments_home",
    )
    model = ExperimentsModel(expected_config)
    assert model.get_user_settings() == expected_config


def test_get_user_experiments_path() -> None:
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)
    assert model.get_user_experiments_path() == user_experiments_path


def test_get_model_checkpoints() -> None:
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    expected = user_experiments_path / "foo" / "checkpoints" / "bar"
    model = ExperimentsModel(config)
    assert model.get_model_checkpoints_path("foo", "bar") == expected


def test_get_model_checkpoints_no_experiment_name() -> None:
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)

    with pytest.raises(ValueError):
        model.get_model_checkpoints_path(None, "bar")


def test_get_model_checkpoints_no_checkpoint() -> None:
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)

    with pytest.raises(ValueError):
        model.get_model_checkpoints_path("foo", None)


def test_get_train_config_path() -> None:
    # Arrange
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    expected = user_experiments_path / "test_experiment" / "train_config.yaml"
    model = ExperimentsModel(config)

    # Act / Assert
    assert model.get_train_config_path("test_experiment") == expected


def test_get_csv_path() -> None:
    # Arrange
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)
    model.set_experiment_name("0_exp")
    expected = user_experiments_path / "0_exp" / "data"

    # Act / Assert
    assert model.get_csv_path() == expected


def test_get_metrics_csv_path() -> None:
    # Arrange
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)
    model.set_experiment_name("0_exp")
    expected = user_experiments_path / "0_exp" / "csv"

    # Act / Assert
    assert model.get_metrics_csv_path() == expected


def test_get_latest_metrics_csv_version_no_versions() -> None:
    # Arrange
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)
    model.set_experiment_name("0_exp")

    # Act / Assert
    assert model.get_latest_metrics_csv_version() == -1


def test_get_latest_metrics_csv_version_no_directory() -> None:
    # Arrange
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)
    model.set_experiment_name("2_exp")

    # Act / Assert
    assert model.get_latest_metrics_csv_version() == -1


def test_get_latest_metrics_csv_version_version_1() -> None:
    # Arrange
    user_experiments_path = Path(__file__).parent / "experiments_home"
    config = FakeUserSettings(
        cyto_dl_home_path=Path(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)
    model.set_experiment_name("1_exp")

    # Act / Assert
    assert model.get_latest_metrics_csv_version() == 1
