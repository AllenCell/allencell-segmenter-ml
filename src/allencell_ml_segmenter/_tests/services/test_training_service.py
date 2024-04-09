from pathlib import Path
from typing import List

import pytest

import allencell_ml_segmenter
from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter._tests.fakes.fake_channel_extraction import (
    FakeChannelExtractionThread,
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


@pytest.fixture
def experiments_model() -> ExperimentsModel:
    experiments_model = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(), user_experiments_path=Path()
        )
    )
    experiments_model.set_experiment_name("testing_experiment")
    experiments_model.set_checkpoint("test_path_checkpoint")
    return experiments_model


@pytest.fixture
def training_model(experiments_model: ExperimentsModel) -> TrainingModel:
    model: TrainingModel = TrainingModel(MainModel(), experiments_model)
    model.set_experiment_type("segmentation")
    model.set_hardware_type("cpu")
    model.set_spatial_dims(2)
    model.set_images_directory("/path/to/images")
    model.set_channel_index(9)
    model.set_max_time(9992)
    model.set_config_dir("/path/to/configs")
    model.set_patch_size("small")
    model.set_max_epoch(100)
    return model


@pytest.fixture
def training_service(
    training_model: TrainingModel, experiments_model: ExperimentsModel
) -> TrainingService:
    """
    Returns a TrainingService object with arbitrary-set fields in the model for testing.
    """
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


def test_get_hardware_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_hardware_override method.
    """
    # ACT
    hardware_override: str = training_service._get_hardware_override()

    # ASSERT
    assert (
        hardware_override
        == f"trainer={training_model.get_hardware_type().value}"
    )


def test_get_spatial_dims_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_spatial_dims_override method.
    """
    # ACT
    spatial_dims_override: str = training_service._get_spatial_dims_override()

    # ASSERT
    assert (
        spatial_dims_override
        == f"spatial_dims={training_model.get_spatial_dims()}"
    )


def test_get_experiment_name_override(
    training_service: TrainingService, experiments_model
) -> None:
    """
    Tests the _get_experiment_name_override method.
    """
    # ACT
    override_str: str = training_service._get_experiment_name_override()

    # ASSERT
    assert (
        override_str
        == f"experiment_name={experiments_model.get_experiment_name()}"
    )


def test_get_max_epoch_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_max_epoch_override method.
    """
    # ACT
    max_epoch_override: str = training_service._get_max_epoch_override()

    # ASSERT
    assert (
        max_epoch_override
        == f"trainer.max_epochs={training_model.get_max_epoch()}"
    )


def test_get_images_directory_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_images_directory_override method.
    """
    # ACT
    im_dir_override: str = training_service._get_images_directory_override()

    # ASSERT
    assert (
        im_dir_override
        == f"data.path={str(training_model.get_images_directory())}"
    )


def test_get_patch_shape_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_patch_shape_override method.
    """
    # ACT
    patch_shape_override: str = training_service._get_patch_shape_override()

    # ASSERT
    assert (
        patch_shape_override
        == f"data._aux.patch_shape={_list_to_string(training_model.get_patch_size().value)}"
    )


def test_get_checkpoint_override(
    training_service: TrainingService, experiments_model: ExperimentsModel
) -> None:
    """
    Tests the _get_checkpoint_override method.
    """
    # Act
    ckpt_path_override = training_service._get_checkpoint_override()

    # Assert
    assert (
        ckpt_path_override
        == f"ckpt_path={experiments_model.get_model_checkpoints_path(experiments_model.get_experiment_name(), experiments_model.get_checkpoint())}"
    )


def test_build_overrrides(
    training_service,
    training_model: TrainingModel,
    experiments_model: ExperimentsModel,
) -> None:
    """
    Tests the _build_overrides method.
    """
    # Act
    all_overrides: List[str] = training_service._build_overrides()

    # Assert
    assert training_service._get_hardware_override() in all_overrides
    assert training_service._get_spatial_dims_override() in all_overrides
    assert training_service._get_experiment_name_override() in all_overrides
    assert training_service._get_images_directory_override() in all_overrides
    assert training_service._get_max_epoch_override() in all_overrides
    assert training_service._get_patch_shape_override() in all_overrides
    assert training_service._get_checkpoint_override() in all_overrides


def test_training_image_directory_selected_subscription(
    training_model: TrainingModel,
    experiments_model: ExperimentsModel,
) -> None:
    # Arrange
    training_service: TrainingService = TrainingService(
        training_model=training_model, experiments_model=experiments_model
    )
    fake_extraction_thread: FakeChannelExtractionThread = (
        FakeChannelExtractionThread()
    )
    training_service.set_channel_extraction_thread_for_test(
        fake_extraction_thread
    )

    # Act
    training_model.set_images_directory(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
    )

    # Assert
    assert fake_extraction_thread.started
    assert (
        fake_extraction_thread.channels_ready.connected
        == training_model.set_max_channel
    )
