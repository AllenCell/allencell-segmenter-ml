import pytest
import sys

from allencell_ml_segmenter.services.training_service import (
    _list_to_string,
    TrainingService,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
)


@pytest.fixture
def training_service() -> TrainingService:
    model = TrainingModel()
    model.set_experiment_type("segmentation")
    model.set_hardware_type("cpu")
    model.set_image_dims(2)
    model.set_images_directory("/path/to/images")
    model.set_channel_index(9)
    model.set_max_time(9992)
    model.set_config_dir("/path/to/configs")
    model.set_patch_size("small")
    model.set_max_epoch(100)
    return TrainingService(model)


def test_list_to_string() -> None:
    assert _list_to_string([1, 2, 3]) == "[1, 2, 3]"
    assert _list_to_string([1]) == "[1]"
    assert _list_to_string([]) == "[]"
    assert _list_to_string("abc") == "[a, b, c]"
    with pytest.raises(TypeError):
        _list_to_string(24)
    with pytest.raises(TypeError):
        _list_to_string(None)


def test_init(training_service: TrainingService) -> None:
    # Assert
    # Check to see if training model set properly
    assert training_service._training_model._events_to_subscriber_handlers[
        "training"
    ] == {training_service: training_service.train_model}


# TODO include when on artifactory
# def test_train_model(training_service: TrainingService) -> None:
#     # Act
#     with patch(
#         "allencell_ml_segmenter.services.training_service.cyto_train"
#     ) as mock_train:
#         training_service.train_model()
#
#     # Assert
#     mock_train.assert_called_once()


def test_set_experiment(training_service: TrainingService) -> None:
    # Act
    training_service._set_experiment()

    # Assert
    assert f"experiment=im2im/{training_service._training_model.get_experiment_type().value}.yaml" in sys.argv


def test_set_hardware(training_service: TrainingService) -> None:
    # Act
    training_service._set_hardware()

    # Assert
    assert f"trainer={training_service._training_model.get_hardware_type().value}" in sys.argv


def test_set_image_dims(training_service: TrainingService) -> None:
    # Act
    training_service._set_image_dims()

    # Assert
    assert f"++spatial_dims=[{training_service._training_model.get_image_dims()}]" in sys.argv


def test_set_max_epoch(training_service: TrainingService) -> None:
    # Act
    training_service._set_max_epoch()

    # Assert
    assert f"++trainer.max_epochs={training_service._training_model.get_max_epoch()}" in sys.argv


def test_set_images_directory(training_service: TrainingService) -> None:
    # Act
    training_service._set_images_directory()

    # Assert
    assert f"++data.path={training_service._training_model.get_images_directory()}" in sys.argv


def test_set_patch_shape_from_size(training_service: TrainingService) -> None:
    # Act
    training_service._set_patch_shape_from_size()

    # Assert
    assert f"++data._aux.patch_shape={training_service._training_model.get_patch_size().value}" in sys.argv


def test_set_config_dir(training_service: TrainingService) -> None:
    # Act
    training_service._set_config_dir()

    # Assert
    assert ("--config-dir" in sys.argv) and (training_service._training_model.get_config_dir() in sys.argv)
