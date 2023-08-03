import pytest
from pathlib import Path

from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    TrainingType,
    Hardware,
    PatchSize,
)


@pytest.fixture
def training_model() -> TrainingModel:
    return TrainingModel()


def test_get_experiment_type(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_experiment_type() is None
    experiment: TrainingType = TrainingType("segmentation")
    training_model._experiment_type = experiment

    # Act/Assert
    assert training_model.get_experiment_type() == experiment


def test_set_experiment_type(training_model: TrainingModel) -> None:
    # Arrange/Act
    training_model.set_experiment_type("segmentation")

    # Assert
    assert training_model._experiment_type == TrainingType("segmentation")


def test_set_experiment_type_invalid_experiment(
    training_model: TrainingModel,
) -> None:
    # Act
    with pytest.raises(ValueError):
        training_model.set_experiment_type("invalid_experiment")


def test_get_hardware_type(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_hardware_type() is None
    hardware: Hardware = Hardware("cpu")
    training_model._hardware_type = hardware

    # Act/Assert
    assert training_model.get_hardware_type() == hardware


def test_set_hardware_type(training_model: TrainingModel) -> None:
    # Arrange/Act
    training_model.set_hardware_type("gpu")

    # Assert
    assert training_model._hardware_type == Hardware("gpu")


def test_get_image_dims(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_image_dims() is None
    training_model._image_dims = 2

    # Act/Assert
    assert training_model.get_image_dims() == 2

    # Arrange
    training_model._image_dims = 3

    # Act/Assert
    assert training_model.get_image_dims() == 3


def test_set_invalid_image_dims(training_model: TrainingModel) -> None:
    # Arrange/Act/Assert
    with pytest.raises(ValueError):
        training_model.set_image_dims(1)

    # Arrange/Act/Assert
    with pytest.raises(ValueError):
        training_model.set_image_dims(4)


def test_get_max_epoch(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_max_epoch() is None
    training_model._max_epoch = 100

    # Act/Assert
    assert training_model.get_max_epoch() == 100


def test_set_max_epoch(training_model: TrainingModel) -> None:
    # Arrange/Act
    training_model.set_max_epoch(100)

    # Assert
    assert training_model._max_epoch == 100


def test_get_images_directory(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_images_directory() is None
    path: Path = Path("/path/to/images")
    training_model._images_directory = path

    # Act/Assert
    assert training_model.get_images_directory() == path


def test_set_images_directory(training_model: TrainingModel) -> None:
    # Arrange/Act
    path: Path = Path("/path/to/images")
    training_model.set_images_directory(path)

    # Assert
    assert training_model._images_directory == path


def test_get_channel_index(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_channel_index() is None
    training_model._channel_index = 1

    # Act/Assert
    assert training_model.get_channel_index() == 1


def test_set_channel_index(training_model: TrainingModel) -> None:
    # Arrange/Act
    training_model.set_channel_index(1)

    # Assert
    assert training_model._channel_index == 1


def test_get_patch_size(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_patch_size() is None
    training_model._patch_size = PatchSize.SMALL

    # Act/Assert
    assert training_model.get_patch_size() == PatchSize.SMALL


def test_set_patch_size(training_model: TrainingModel) -> None:
    # Arrange/Act
    training_model.set_patch_size("small")

    # Assert
    assert training_model._patch_size == PatchSize.SMALL

    # Arrange/Act
    training_model.set_patch_size("SMaLL")

    # Assert
    assert training_model._patch_size == PatchSize.SMALL

    # Arrange/Act
    training_model.set_patch_size("MEDIUM")

    # Assert
    assert training_model._patch_size == PatchSize.MEDIUM

    # Arrange/Act
    training_model.set_patch_size("large")

    # Assert
    assert training_model._patch_size == PatchSize.LARGE


def test_get_max_time(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_max_time() is None
    training_model._max_time = 100

    # Act/Assert
    assert training_model.get_max_time() == 100


def test_set_max_time(training_model: TrainingModel) -> None:
    # Arrange/Act
    training_model.set_max_time(100)

    # Assert
    assert training_model._max_time == 100


def test_get_config_dir(training_model: TrainingModel) -> None:
    # Arrange
    assert training_model.get_config_dir() is None
    path: Path = Path("/path/to/config")
    training_model._config_dir = path

    # Act/Assert
    assert training_model.get_config_dir() == path


def test_set_config_dir(training_model: TrainingModel) -> None:
    # Arrange/Act
    path: Path = Path("/path/to/config")
    training_model.set_config_dir(path)

    # Assert
    assert training_model._config_dir == path
