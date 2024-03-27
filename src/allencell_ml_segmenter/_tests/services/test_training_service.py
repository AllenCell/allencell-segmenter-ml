from pathlib import Path
from typing import List

import pytest
from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel

from allencell_ml_segmenter.services.training_service import (
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
    model.set_use_max_time(True)
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


def test_init(training_service: TrainingService) -> None:
    """
    Tests the initialization of the TrainingService object.
    """
    # ASSERT - check if training model is set properly
    assert training_service._training_model._events_to_subscriber_handlers[
        "training"
    ] == {training_service: training_service._train_model_handler}


def test_hardware_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_hardware_override method.
    """
    # ACT
    training_service._hardware_override()

    # ASSERT
    assert (
        training_service.get_overrides()["trainer.accelerator"]
        == training_model.get_hardware_type().value
    )


def test_spatial_dims_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_spatial_dims_override method.
    """
    # ACT
    training_service._spatial_dims_override()

    # ASSERT
    assert (
        training_service.get_overrides()["spatial_dims"]
        == training_model.get_spatial_dims()
    )


def test_experiment_name_override(
    training_service: TrainingService, experiments_model
) -> None:
    """
    Tests the _get_experiment_name_override method.
    """
    # ACT
    training_service._experiment_name_override()

    # ASSERT
    assert (
        training_service.get_overrides()["experiment_name"]
        == experiments_model.get_experiment_name()
    )


def test_max_run_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_max_epoch_override method.
    """
    # ACT
    training_service._max_run_override()

    # ASSERT
    assert (
        training_service.get_overrides()["trainer.max_epochs"]
        == training_model.get_max_epoch()
    )
    assert (
        training_service.get_overrides()["trainer.max_time"]["minutes"]
        == training_model.get_max_time()
    )


def test_images_directory_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_images_directory_override method.
    """
    # ACT
    training_service._images_directory_override()

    # ASSERT
    assert training_service.get_overrides()["data.path"] == str(
        training_model.get_images_directory()
    )


def test_patch_shape_override(
    training_service: TrainingService, training_model: TrainingModel
) -> None:
    """
    Tests the _get_patch_shape_override method.
    """
    # ACT
    training_service._patch_shape_override()

    # ASSERT
    assert (
        training_service.get_overrides()["data._aux.patch_shape"]
        == training_model.get_patch_size().value
    )


def test_checkpoint_override(
    training_service: TrainingService, experiments_model: ExperimentsModel
) -> None:
    """
    Tests the _get_checkpoint_override method.
    """
    # Act
    training_service._checkpoint_override()

    # Assert
    assert training_service.get_overrides()["ckpt_path"] == str(
        experiments_model.get_model_checkpoints_path(
            experiments_model.get_experiment_name(),
            experiments_model.get_checkpoint(),
        )
    )
