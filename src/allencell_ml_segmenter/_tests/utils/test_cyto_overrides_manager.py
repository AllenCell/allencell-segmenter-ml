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
    ImageType,
)
from allencell_ml_segmenter.utils.cyto_overrides_manager import (
    CytoDLOverridesManager,
)
import allencell_ml_segmenter


@pytest.fixture
def experiments_model() -> ExperimentsModel:
    exp_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "main"
        / "experiments_home"
    )
    experiments_model = ExperimentsModel(
        FakeUserSettings(
            cyto_dl_home_path=Path(), user_experiments_path=exp_path
        )
    )
    return experiments_model


@pytest.fixture
def training_model(experiments_model: ExperimentsModel) -> TrainingModel:
    model: TrainingModel = TrainingModel(MainModel(), experiments_model)
    model.set_experiment_type("segmentation")
    model.set_spatial_dims(3)
    model.set_images_directory("/path/to/images")
    model.set_selected_channel(ImageType.RAW, 1)
    model.set_selected_channel(ImageType.SEG1, 2)
    model.set_selected_channel(ImageType.SEG2, 3)
    model.set_use_max_time(True)
    model.set_max_time(9992)
    model.set_config_dir("/path/to/configs")
    model.set_patch_size([1, 2, 4])
    model.set_num_epochs(100)
    model.set_model_size("medium")
    return model


def test_get_training_overrides(
    experiments_model: ExperimentsModel, training_model: TrainingModel
):
    experiments_model.apply_experiment_name("one_ckpt_exp")
    cyto_overrides_manager: CytoDLOverridesManager = CytoDLOverridesManager(
        experiments_model, training_model
    )

    training_overrides: Dict[str, Union[str, int, float, bool, Dict]] = (
        cyto_overrides_manager.get_training_overrides()
    )

    assert (
        training_overrides["spatial_dims"] == training_model.get_spatial_dims()
    )

    # the experiments model fixture is set to the experiment @ _tests/experiments_home/one_ckpt_exp,
    # for which the last 'best' checkpoint is epoch_000.ckpt. Since we are currently at the first epoch checkpoint,
    # in order to run training_model.get_num_epochs() more epochs, we need to increment by 1
    assert (
        training_overrides["trainer.max_epochs"]
        == training_model.get_num_epochs() + 1
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
        == training_model.get_patch_size()
    )

    assert training_overrides[
        "input_channel"
    ] == training_model.get_selected_channel(ImageType.RAW)

    assert training_overrides[
        "target_col1_channel"
    ] == training_model.get_selected_channel(ImageType.SEG1)

    assert training_overrides[
        "target_col2_channel"
    ] == training_model.get_selected_channel(ImageType.SEG2)


def test_get_training_overrides_2d_spatial_dims(experiments_model) -> None:
    # Arrange
    model: TrainingModel = TrainingModel(MainModel(), experiments_model)
    model.set_experiment_type("segmentation")
    model.set_images_directory("/path/to/images")
    model.set_use_max_time(True)
    model.set_selected_channel(ImageType.RAW, 0)
    model.set_selected_channel(ImageType.SEG1, 0)
    model.set_selected_channel(
        ImageType.SEG2, 0
    )  # set to 0 if 2d image loaded
    model.set_max_time(9992)
    model.set_config_dir("/path/to/configs")
    model.set_num_epochs(100)
    model.set_model_size("medium")

    model.set_spatial_dims(2)
    model.set_patch_size([4, 8])
    cyto_overrides_manager: CytoDLOverridesManager = CytoDLOverridesManager(
        experiments_model, model
    )

    # Act
    training_overrides: Dict[str, Union[str, int, float, bool, Dict]] = (
        cyto_overrides_manager.get_training_overrides()
    )

    # Assert
    assert len(training_overrides["data._aux.patch_shape"]) == 2
    assert training_overrides["data._aux.patch_shape"] == [4, 8]
    assert training_overrides["input_channel"] == 0


def test_max_epochs_no_existing_ckpt(
    experiments_model: ExperimentsModel, training_model: TrainingModel
):
    experiments_model.apply_experiment_name("0_exp")
    cyto_overrides_manager: CytoDLOverridesManager = CytoDLOverridesManager(
        experiments_model, training_model
    )

    training_overrides: Dict[str, Union[str, int, float, bool, Dict]] = (
        cyto_overrides_manager.get_training_overrides()
    )

    # the experiments model fixture is set to the experiment @ _tests/experiments_home/0_exp,
    # for which there are no existing checkpoints. So, we expect the max epoch to be equal to what the
    # user has entered in the field
    assert (
        training_overrides["trainer.max_epochs"]
        == training_model.get_num_epochs()
    )
