from pathlib import Path
from typing import Dict, Union

import pytest
from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
)
from allencell_ml_segmenter.utils.cyto_overrides_manager import (
    CytoDLOverridesManager,
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


def test_get_training_overrides(
    experiments_model: ExperimentsModel, training_model: TrainingModel
):
    cyto_overrides_manager: CytoDLOverridesManager = CytoDLOverridesManager(
        experiments_model, training_model
    )

    training_overrides: Dict[str, Union[str, int, float, bool, Dict]] = (
        cyto_overrides_manager.get_training_overrides()
    )

    # ASSERT
    assert (
        training_overrides["trainer.accelerator"]
        == training_model.get_hardware_type().value
    )

    assert (
        training_overrides["spatial_dims"] == training_model.get_spatial_dims()
    )

    assert (
        training_overrides["trainer.max_epochs"]
        == training_model.get_max_epoch()
    )
    assert (
        training_overrides["trainer.max_time"]["minutes"]
        == training_model.get_max_time()
    )

    assert training_overrides["data.path"] == str(
        training_model.get_images_directory()
    )

    assert (
        training_overrides["data._aux.patch_shape"]
        == training_model.get_patch_size().value
    )

    assert training_overrides["ckpt_path"] == str(
        experiments_model.get_model_checkpoints_path(
            experiments_model.get_experiment_name(),
            experiments_model.get_checkpoint(),
        )
    )