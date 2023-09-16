import pytest
from pathlib import Path
from allencell_ml_segmenter.main.main_model import MainModel

from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    TrainingType,
    Hardware,
    PatchSize,
)


@pytest.fixture
def training_model() -> TrainingModel:
    """
    Returns a TrainingModel instance for testing.
    """
    return TrainingModel(
        MainModel()
    )


def test_get_experiment_type(training_model: TrainingModel) -> None:
    """
    Tests that get_experiment_type returns the correct experiment type.
    """
    # ASSERT
    assert training_model.get_experiment_type() is None

    # ARRANGE
    experiment: TrainingType = TrainingType("segmentation")
    training_model._experiment_type = experiment

    # ACT/ASSERT
    assert training_model.get_experiment_type() == experiment


def test_set_experiment_type(training_model: TrainingModel) -> None:
    """
    Tests that set_experiment_type sets the correct experiment type.
    """
    # ACT
    training_model.set_experiment_type("segmentation")

    # ASSERT
    assert training_model._experiment_type == TrainingType("segmentation")


def test_set_experiment_type_invalid_experiment(
    training_model: TrainingModel,
) -> None:
    """
    Tests that set_experiment_type raises a ValueError when given an invalid experiment type.
    """
    # ACT
    with pytest.raises(ValueError):
        training_model.set_experiment_type("invalid_experiment")


def test_get_hardware_type(training_model: TrainingModel) -> None:
    """
    Tests that get_hardware_type returns the correct hardware type.
    """
    # ASSERT
    assert training_model.get_hardware_type() is None

    # ARRANGE
    hardware: Hardware = Hardware("cpu")
    training_model._hardware_type = hardware

    # ACT/ASSERT
    assert training_model.get_hardware_type() == hardware


def test_set_hardware_type(training_model: TrainingModel) -> None:
    """
    Tests that set_hardware_type sets the correct hardware type.
    """
    # ACT
    training_model.set_hardware_type("gpu")

    # ASSERT
    assert training_model._hardware_type == Hardware("gpu")


def test_get_image_dims(training_model: TrainingModel) -> None:
    """
    Tests that get_image_dims returns the correct image dimensions.
    """
    # ASSERT
    assert training_model.get_image_dims() is None

    # ARRANGE
    training_model._image_dims = 2

    # ACT/ASSERT
    assert training_model.get_image_dims() == 2

    # ARRANGE
    training_model._image_dims = 3

    # ACT/ASSERT
    assert training_model.get_image_dims() == 3


def test_set_invalid_image_dims(training_model: TrainingModel) -> None:
    """
    Tests that set_image_dims raises a ValueError when given an invalid image dimension (not 2 or 3).
    """
    # ACT
    with pytest.raises(ValueError):
        training_model.set_image_dims(1)

    # ACT
    with pytest.raises(ValueError):
        training_model.set_image_dims(4)


def test_get_max_epoch(training_model: TrainingModel) -> None:
    """
    Tests that get_max_epoch returns the correct max epoch.
    """
    # ASSERT
    assert training_model.get_max_epoch() is None

    # ARRANGE
    training_model._max_epoch = 100

    # ACT/ASSERT
    assert training_model.get_max_epoch() == 100


def test_set_max_epoch(training_model: TrainingModel) -> None:
    """
    Tests that set_max_epoch sets the correct max epoch.
    """
    # ACT
    training_model.set_max_epoch(100)

    # ASSERT
    assert training_model._max_epoch == 100


def test_get_images_directory(training_model: TrainingModel) -> None:
    """
    Tests that get_images_directory returns the correct images directory.
    """
    # ASSERT
    assert training_model.get_images_directory() is None

    # ARRANGE
    path: Path = Path("/path/to/images")
    training_model._images_directory = path

    # ACT/ASSERT
    assert training_model.get_images_directory() == path


def test_set_images_directory(training_model: TrainingModel) -> None:
    """
    Tests that set_images_directory sets the correct images directory.
    """
    # ARRANGE
    path: Path = Path("/path/to/images")

    # ACT
    training_model.set_images_directory(path)

    # ASSERT
    assert training_model._images_directory == path


def test_get_channel_index(training_model: TrainingModel) -> None:
    """
    Tests that get_channel_index returns the correct channel index.
    """
    # ASSERT
    assert training_model.get_channel_index() is None

    # ARRANGE
    training_model._channel_index = 1

    # ACT/ASSERT
    assert training_model.get_channel_index() == 1


def test_set_channel_index(training_model: TrainingModel) -> None:
    """
    Tests that set_channel_index sets the correct channel index.
    """
    # ACT
    training_model.set_channel_index(1)

    # ASSERT
    assert training_model._channel_index == 1


def test_get_patch_size(training_model: TrainingModel) -> None:
    """
    Tests that get_patch_size returns the correct patch size.
    """
    # ASSERT
    assert training_model.get_patch_size() is None

    # ARRANGE
    training_model._patch_size = PatchSize.SMALL

    # ACT/ASSERT
    assert training_model.get_patch_size() == PatchSize.SMALL


def test_set_patch_size(training_model: TrainingModel) -> None:
    """
    Tests that set_patch_size sets the correct patch size.
    """
    # ACT
    training_model.set_patch_size("small")

    # ASSERT
    assert training_model._patch_size == PatchSize.SMALL

    # ACT
    training_model.set_patch_size("SMaLL")

    # ASSERT
    assert training_model._patch_size == PatchSize.SMALL

    # ACT
    training_model.set_patch_size("MEDIUM")

    # ASSERT
    assert training_model._patch_size == PatchSize.MEDIUM

    # ACT
    training_model.set_patch_size("large")

    # ASSERT
    assert training_model._patch_size == PatchSize.LARGE


def test_get_max_time(training_model: TrainingModel) -> None:
    """
    Tests that get_max_time returns the correct max time.
    """
    # ASSERT
    assert training_model.get_max_time() is None

    # ARRANGE
    training_model._max_time = 100

    # ACT/ASSERT
    assert training_model.get_max_time() == 100


def test_set_max_time(training_model: TrainingModel) -> None:
    """
    Tests that set_max_time sets the correct max time.
    """
    # ACT
    training_model.set_max_time(100)

    # ASSERT
    assert training_model._max_time == 100


def test_get_config_dir(training_model: TrainingModel) -> None:
    """
    Tests that get_config_dir returns the correct config directory.
    """
    # ASSERT
    assert training_model.get_config_dir() is None

    # ARRANGE
    path: Path = Path("/path/to/config")
    training_model._config_dir = path

    # ACT/ASSERT
    assert training_model.get_config_dir() == path


def test_set_config_dir(training_model: TrainingModel) -> None:
    """
    Tests that set_config_dir sets the correct config directory.
    """
    # ARRANGE
    path: Path = Path("/path/to/config")

    # ACT
    training_model.set_config_dir(path)

    # ASSERT
    assert training_model._config_dir == path
