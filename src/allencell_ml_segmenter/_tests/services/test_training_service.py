import pytest
import sys

from allencell_ml_segmenter.services.cyto_service import (
    _list_to_string,
    CytoService,
    CytodlMode,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
)


@pytest.fixture
def training_service() -> CytoService:
    model: TrainingModel = TrainingModel()
    model.set_channel_index(9)
    model.set_max_time(9992)
    return CytoService(model, mode=CytodlMode.TRAIN)


@pytest.fixture
def prediction_service() -> CytoService:
    model: TrainingModel = TrainingModel()
    return CytoService(model, mode=CytodlMode.PREDICT)


def test_list_to_string() -> None:
    """
    Tests the _list_to_string helper function.
    """
    # ACT/ASSERT
    assert _list_to_string([1, 2, 3]) == "[1, 2, 3]"
    assert _list_to_string([1]) == "[1]"
    assert _list_to_string([]) == "[]"
    assert _list_to_string("abc") == "[a, b, c]"

    with pytest.raises(TypeError):
        _list_to_string(24)
    with pytest.raises(TypeError):
        _list_to_string(None)


def test_init_training(training_service: CytoService) -> None:
    # Assert
    # Check to see if training model set properly
    assert training_service._model._events_to_subscriber_handlers[
        "training"
    ] == {training_service: training_service.train_model}


def test_init_preidction(prediction_service: CytoService) -> None:
    assert prediction_service._model._events_to_subscriber_handlers[
        "prediction"
    ] == {prediction_service: prediction_service.predict_model}


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


def test_set_experiment(training_service: CytoService) -> None:
    """
    Tests the _set_experiment method.
    """
    # Act
    training_service._model.set_experiment_type("segmentation")
    training_service._set_experiment()

    # ASSERT
    assert (
        f"experiment=im2im/{training_service._model.get_experiment_type().value}.yaml"
        in sys.argv
    )


def test_set_experiment_not_set(training_service: CytoService) -> None:
    # Assert
    with pytest.raises(ValueError):
        training_service._set_experiment()


def test_set_hardware(training_service: CytoService) -> None:
    """
    Tests the _set_hardware method.
    """
    # Act
    training_service._model.set_hardware_type("cpu")
    training_service._set_hardware()

    # ASSERT
    assert (
        f"trainer={training_service._model.get_hardware_type().value}"
        in sys.argv
    )


def test_set_hardware_not_set(training_service: CytoService) -> None:
    # Assert
    with pytest.raises(ValueError):
        training_service._set_hardware()


def test_set_image_dims(training_service: CytoService) -> None:
    """
    Tests the _set_image_dims method.
    """
    # Act
    training_service._model.set_image_dims(2)
    training_service._set_image_dims()

    # ASSERT
    assert (
        f"++spatial_dims=[{training_service._model.get_image_dims()}]"
        in sys.argv
    )


def test_set_image_dims_not_set(training_service: CytoService) -> None:
    # Act
    length_argv = len(sys.argv)
    training_service._set_image_dims()

    # Assert
    assert length_argv == len(
        sys.argv
    )  # no argument variables added since image_dims not set


def test_set_max_epoch(training_service: CytoService) -> None:
    """
    Tests the _set_max_epoch method.
    """
    # Act
    training_service._model.set_max_epoch(100)
    training_service._set_max_epoch()

    # ASSERT
    assert (
        f"++trainer.max_epochs={training_service._model.get_max_epoch()}"
        in sys.argv
    )


def test_set_max_epoch_not_set(training_service: CytoService) -> None:
    # Act
    length_argv = len(sys.argv)
    training_service._set_max_epoch()

    # Assert
    assert length_argv == len(
        sys.argv
    )  # no argument variables added since max_epoch not set


def test_set_images_directory(training_service: CytoService) -> None:
    """
    Tests the _set_images_directory method.
    """
    # Act
    training_service._model.set_images_directory("/path/to/images")
    training_service._set_images_directory()

    # ASSERT
    assert (
        f"++data.path={training_service._model.get_images_directory()}"
        in sys.argv
    )


def test_set_images_directory_not_set(training_service: CytoService) -> None:
    # Act
    length_argv = len(sys.argv)
    training_service._set_images_directory()

    # Assert
    assert length_argv == len(
        sys.argv
    )  # no argument variables added since images_directory not set


def test_set_patch_shape_from_size(training_service: CytoService) -> None:
    """
    Tests the _set_patch_shape_from_size method.
    """
    # Act
    training_service._model.set_patch_size("small")
    training_service._set_patch_shape_from_size()

    # ASSERT
    assert (
        f"++data._aux.patch_shape={training_service._model.get_patch_size().value}"
        in sys.argv
    )


def test_set_patch_shape_not_set(training_service: CytoService) -> None:
    # Act
    length_argv = len(sys.argv)
    training_service._set_patch_shape_from_size()

    # Assert
    assert length_argv == len(
        sys.argv
    )  # no argument variables added since patch_size not set


def test_set_config_dir(training_service: CytoService) -> None:
    """
    Tests the _set_config_dir method.
    """
    # Act
    training_service._model.set_config_dir("/path/to/configs")
    training_service._set_config_dir()

    # ASSERT
    assert ("--config-dir" in sys.argv) and (
        training_service._model.get_config_dir() in sys.argv
    )


def test_set_config_dir_not_set(training_service: CytoService) -> None:
    # Assert
    with pytest.raises(ValueError):
        training_service._set_config_dir()


def test_training_set_config_name(training_service: CytoService) -> None:
    # Act
    training_service._model.set_config_name("segmentation")
    training_service._set_config_name()

    # Assert
    assert ("--config-name" in sys.argv) and (
        training_service._model.get_config_name() in sys.argv
    )


def test_training_set_config_name_not_set(
    training_service: CytoService,
) -> None:
    # Act
    length_argv: int = len(sys.argv)
    training_service._set_config_name()

    # Assert
    assert (
        length_argv == length_argv
    )  # no argument variables added since config_name not set


def test_prediction_set_config_name(prediction_service: CytoService) -> None:
    # Act
    prediction_service._model.set_config_name("train.yaml")
    prediction_service._set_config_name()

    # Assert
    assert ("--config-name" in sys.argv) and (
        str(prediction_service._model.get_config_name()) in sys.argv
    )


def test_prediction_set_config_name_not_set(
    prediction_service: CytoService,
) -> None:
    # Act
    with pytest.raises(ValueError):
        prediction_service._set_config_name()
