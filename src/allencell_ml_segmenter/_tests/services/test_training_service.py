from pathlib import Path
from typing import List
from unittest.mock import patch
import pytest

import allencell_ml_segmenter
from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter._tests.fakes.fake_channel_extraction import (
    FakeChannelExtractionThread,
)
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.extractor_factory import FakeExtractorFactory
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel

from allencell_ml_segmenter.services.training_service import (
    TrainingService,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
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
    experiments_model.apply_experiment_name("2_exp")
    return experiments_model


@pytest.fixture
def training_model(experiments_model: ExperimentsModel) -> TrainingModel:
    model: TrainingModel = TrainingModel(MainModel(), experiments_model)
    model.set_experiment_type("segmentation")
    model.set_hardware_type("cpu")
    model.set_spatial_dims(2)
    model.set_images_directory(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "csv"
    )
    model.set_channel_index(9)
    model.set_use_max_time(True)
    model.set_max_time(9992)
    model.set_config_dir("/path/to/configs")
    model.set_patch_size([1, 2])
    model.set_num_epochs(100)
    return model


@pytest.fixture
def training_service(
    training_model: TrainingModel, experiments_model: ExperimentsModel
) -> TrainingService:
    """
    Returns a TrainingService object with arbitrary-set fields in the model for testing.
    """
    return TrainingService(
        training_model=training_model,
        experiments_model=experiments_model,
        extractor_factory=FakeExtractorFactory(0),
    )


def test_init(training_service: TrainingService) -> None:
    """
    Tests the initialization of the TrainingService object.
    """
    # ASSERT - check if training model is set properly
    assert training_service._training_model._events_to_subscriber_handlers[
        "training"
    ] == {training_service: training_service._train_model_handler}


def test_training_image_directory_selected(
    training_model: TrainingModel, experiments_model: ExperimentsModel
) -> None:
    """
    Tests to see if service starts image data extraction and handles
    image data accordingly after a dataset is selected.
    """
    # arrange
    service: TrainingService = TrainingService(
        training_model=training_model,
        experiments_model=experiments_model,
        extractor_factory=FakeExtractorFactory(),
    )

    # act
    training_model.dispatch(Event.ACTION_TRAINING_DATASET_SELECTED)

    # assert
    assert training_model.get_max_channel() == 4
    assert training_model.get_image_dimensions() == [3, 2, 1]
