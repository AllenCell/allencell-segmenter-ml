from pathlib import Path
import pytest
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    FakeImageDataExtractor,
)
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel

from allencell_ml_segmenter.services.training_service import (
    TrainingService,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    ImageType,
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
    return model


def test_service_reacts_to_image_dir_set(
    qtbot: QtBot,
    training_model: TrainingModel,
    experiments_model: ExperimentsModel,
) -> None:
    # Arrange
    service: TrainingService = TrainingService(
        training_model,
        experiments_model,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
    )
    img_dir: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "multiple_csv"
    )

    # Act / Assert
    # we expect the service to listen for the signal emitted when the image directory is set
    # do some async work, then set num channels, which will emit the signal we are waiting on
    with qtbot.waitSignal(training_model.signals.num_channels_set):
        training_model.set_images_directory(img_dir)
