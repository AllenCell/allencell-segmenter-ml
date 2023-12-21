from pathlib import Path
import pytest
import sys
from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel

from allencell_ml_segmenter.services.training_service import (
    _list_to_string,
    TrainingService,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
)
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)


@pytest.fixture
def training_service() -> TrainingService:
    """
    Returns a TrainingService object with arbitrary-set fields in the model for testing.
    """
    experiments_model = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(), user_experiments_path=Path()
        )
    )
    training_model: TrainingModel = TrainingModel(
        MainModel(), FakeExperimentsModel()
    )
    training_model.set_experiment_type("segmentation")
    training_model.set_hardware_type("cpu")
    training_model.set_image_dims(2)
    training_model.set_images_directory("/path/to/images")
    training_model.set_channel_index(9)
    training_model.set_max_time(9992)
    training_model.set_config_dir("/path/to/configs")
    training_model.set_patch_size("small")
    training_model.set_max_epoch(100)
    return TrainingService(
        training_model=training_model, experiments_model=experiments_model
    )


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


def test_init(training_service: TrainingService) -> None:
    """
    Tests the initialization of the TrainingService object.
    """
    # ASSERT - check if training model is set properly
    assert training_service._training_model._events_to_subscriber_handlers[
        "training"
    ] == {training_service: training_service.train_model_handler}


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
    """
    Tests the _set_experiment method.
    """
    # ACT
    training_service._set_experiment()

    # ASSERT
    assert (
        f"experiment=im2im/{training_service._training_model.get_experiment_type().value}.yaml"
        in sys.argv
    )


def test_set_hardware(training_service: TrainingService) -> None:
    """
    Tests the _set_hardware method.
    """
    # ACT
    training_service._set_hardware()

    # ASSERT
    assert (
        f"trainer={training_service._training_model.get_hardware_type().value}"
        in sys.argv
    )


def test_set_image_dims(training_service: TrainingService) -> None:
    """
    Tests the _set_image_dims method.
    """
    # ACT
    training_service._set_image_dims()

    # ASSERT
    assert (
        f"++spatial_dims=[{training_service._training_model.get_image_dims()}]"
        in sys.argv
    )


def test_set_max_epoch(training_service: TrainingService) -> None:
    """
    Tests the _set_max_epoch method.
    """
    # ACT
    training_service._set_max_epoch()

    # ASSERT
    assert (
        f"++trainer.max_epochs={training_service._training_model.get_max_epoch()}"
        in sys.argv
    )


def test_set_images_directory(training_service: TrainingService) -> None:
    """
    Tests the _set_images_directory method.
    """
    # ACT
    training_service._set_images_directory()

    # ASSERT
    assert (
        f"++data.path={training_service._training_model.get_images_directory()}"
        in sys.argv
    )


def test_set_patch_shape_from_size(training_service: TrainingService) -> None:
    """
    Tests the _set_patch_shape_from_size method.
    """
    # ACT
    training_service._set_patch_shape_from_size()

    # ASSERT
    assert (
        f"++data._aux.patch_shape={training_service._training_model.get_patch_size().value}"
        in sys.argv
    )


def test_set_config_dir(training_service: TrainingService) -> None:
    """
    Tests the _set_config_dir method.
    """
    # ACT
    training_service._set_config_dir()

    # ASSERT
    assert ("--config-dir" in sys.argv) and (
        training_service._training_model.get_config_dir() in sys.argv
    )
