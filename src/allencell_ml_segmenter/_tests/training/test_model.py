import pytest
from pathlib import Path
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.main.main_model import MainModel

from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    TrainingType,
    ImageType,
)
from unittest.mock import Mock


@pytest.fixture
def training_model() -> TrainingModel:
    """
    Returns a TrainingModel instance for testing.
    """
    return TrainingModel(MainModel(), FakeExperimentsModel())


def test_get_experiment_type(training_model: TrainingModel) -> None:
    """
    Tests that get_experiment_type returns the correct experiment type.
    """
    # ASSERT
    assert training_model.get_experiment_type() is None

    # ARRANGE
    experiment: TrainingType = TrainingType("segmentation")
    training_model.set_experiment_type(experiment.value)

    # ACT/ASSERT
    assert training_model.get_experiment_type() == experiment.value


def test_set_experiment_type(training_model: TrainingModel) -> None:
    """
    Tests that set_experiment_type sets the correct experiment type.
    """
    training_type_str = "segmentation"

    # ACT
    training_model.set_experiment_type(training_type_str)

    # ASSERT
    assert training_model.get_experiment_type() == training_type_str


def test_set_experiment_type_invalid_experiment(
    training_model: TrainingModel,
) -> None:
    """
    Tests that set_experiment_type raises a ValueError when given an invalid experiment type.
    """
    # ACT
    with pytest.raises(ValueError):
        training_model.set_experiment_type("invalid_experiment")


def test_get_image_dims(training_model: TrainingModel) -> None:
    """
    Tests that get_image_dims returns the correct image dimensions.
    """
    # ASSERT
    assert training_model.get_spatial_dims() is None

    # ARRANGE
    training_model._spatial_dims = 2

    # ACT/ASSERT
    assert training_model.get_spatial_dims() == 2

    # ARRANGE
    training_model._spatial_dims = 3

    # ACT/ASSERT
    assert training_model.get_spatial_dims() == 3


def test_get_num_epochs(training_model: TrainingModel) -> None:
    """
    Tests that get_max_epoch returns the correct max epoch.
    """
    # ASSERT
    assert training_model.get_num_epochs() is None

    # ARRANGE
    training_model._num_epochs = 100

    # ACT/ASSERT
    assert training_model.get_num_epochs() == 100


def test_set_num_epochs(training_model: TrainingModel) -> None:
    """
    Tests that set_max_epoch sets the correct max epoch.
    """
    # ACT
    training_model.set_num_epochs(100)

    # ASSERT
    assert training_model._num_epochs == 100


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
    img_dir_listener: Mock = Mock()
    path: Path = Path("/path/to/images")
    training_model.signals.images_directory_set.connect(img_dir_listener)

    # ACT
    training_model.set_images_directory(path)

    # ASSERT
    assert training_model._images_directory == path
    img_dir_listener.assert_called_once()


def test_selected_channel(training_model: TrainingModel) -> None:
    """
    Tests that get_channel_index returns the correct channel index.
    """
    # ASSERT
    assert training_model.get_selected_channel(ImageType.RAW) is None
    assert training_model.get_selected_channel(ImageType.SEG1) is None
    assert training_model.get_selected_channel(ImageType.SEG2) is None

    # ACT
    training_model.set_selected_channel(ImageType.RAW, 1)
    training_model.set_selected_channel(ImageType.SEG1, 2)
    training_model.set_selected_channel(ImageType.SEG2, 3)

    # ACT/ASSERT
    assert training_model.get_selected_channel(ImageType.RAW) == 1
    assert training_model.get_selected_channel(ImageType.SEG1) == 2
    assert training_model.get_selected_channel(ImageType.SEG2) == 3


def test_num_channels(training_model: TrainingModel) -> None:
    # ASSSERT
    assert training_model.get_num_channels(ImageType.RAW) is None
    assert training_model.get_num_channels(ImageType.SEG1) is None
    assert training_model.get_num_channels(ImageType.SEG2) is None

    # ARRANGE
    num_channel_listener: Mock = Mock()
    training_model.signals.num_channels_set.connect(num_channel_listener)

    # ACT
    training_model.set_all_num_channels(
        {
            ImageType.RAW: 1,
            ImageType.SEG1: 2,
            ImageType.SEG2: 3,
        }
    )

    # ASSERT
    assert training_model.get_num_channels(ImageType.RAW) == 1
    assert training_model.get_num_channels(ImageType.SEG1) == 2
    assert training_model.get_num_channels(ImageType.SEG2) == 3
    num_channel_listener.assert_called_once()


def test_get_patch_size(training_model: TrainingModel) -> None:
    """
    Tests that get_patch_size returns the correct patch size.
    """
    # ASSERT
    assert training_model.get_patch_size() is None
    expected_patch = [4, 8]

    # ARRANGE
    training_model._patch_size = expected_patch

    # ACT/ASSERT
    assert training_model.get_patch_size() == expected_patch


def test_set_patch_size(training_model: TrainingModel) -> None:
    """
    Tests that set_patch_size sets the correct patch size.
    """
    # expected
    patch_2d = [4, 8]
    patch_3d = [1, 2, 3]

    # ACT
    training_model.set_patch_size(patch_2d)

    # ASSERT
    assert training_model._patch_size == patch_2d

    # ACT
    training_model.set_patch_size(patch_3d)

    # ASSERT
    assert training_model._patch_size == patch_3d


def test_set_patch_size_invalid(training_model: TrainingModel) -> None:
    """
    Tests that set_patch_size throws an error if an invalid patch size is provided
    """

    # ACT/ASSERT
    with pytest.raises(ValueError):
        training_model.set_patch_size([1])

    with pytest.raises(ValueError):
        training_model.set_patch_size([1, 2, 3, 4])

    with pytest.raises(TypeError):
        training_model.set_patch_size(2)

    with pytest.raises(TypeError):
        training_model.set_patch_size(None)


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
