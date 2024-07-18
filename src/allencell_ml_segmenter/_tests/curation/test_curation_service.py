from pathlib import Path
from dataclasses import dataclass

import pytest
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    ImageType,
    CurationView,
)
from allencell_ml_segmenter.curation.curation_service import (
    CurationService,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    FakeImageDataExtractor,
)
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.utils.file_writer import FakeFileWriter
from allencell_ml_segmenter.core.task_executor import SynchroTaskExecutor
import allencell_ml_segmenter


FAKE_CHANNEL_SELECTION_PATH: Path = Path("channel_sel")


@dataclass
class TestEnvironment:
    model: CurationModel
    service: CurationService
    file_writer: FakeFileWriter


@pytest.fixture
def test_env_input_view() -> TestEnvironment:
    exp_mod: FakeExperimentsModel = FakeExperimentsModel()
    exp_mod.apply_experiment_name("0_exp")
    model: CurationModel = CurationModel(exp_mod)
    writer: FakeFileWriter = FakeFileWriter()
    service: CurationService = CurationService(
        model,
        exp_mod,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
        file_writer=writer,
    )
    return TestEnvironment(model, service, writer)


@pytest.fixture
def test_env_input_view_synchro_executor() -> TestEnvironment:
    exp_mod: FakeExperimentsModel = FakeExperimentsModel(
        channel_selection_path=FAKE_CHANNEL_SELECTION_PATH
    )
    exp_mod.apply_experiment_name("0_exp")
    model: CurationModel = CurationModel(exp_mod)
    writer: FakeFileWriter = FakeFileWriter()
    service: CurationService = CurationService(
        model,
        exp_mod,
        img_data_extractor=FakeImageDataExtractor.global_instance(),
        task_executor=SynchroTaskExecutor.global_instance(),
        file_writer=writer,
    )
    return TestEnvironment(model, service, writer)


@pytest.fixture
def test_env_main_view(
    test_env_input_view: TestEnvironment,
) -> TestEnvironment:
    model: CurationModel = test_env_input_view.model
    model.set_image_directory_paths(ImageType.RAW, IMG_DIR_FILES)
    model.set_image_directory_paths(ImageType.SEG1, IMG_DIR_FILES)
    model.set_image_directory_paths(ImageType.SEG2, IMG_DIR_FILES)
    model.set_current_view(CurationView.MAIN_VIEW)

    return test_env_input_view


IMG_DIR_PATH = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "img_folder"
)

IMG_DIR_FILES = [path for path in IMG_DIR_PATH.iterdir()]

# NOTE: the service tests follow this general format:
# 1. act on the model to trigger a signal which service should be connected to
# 2. wait for signal indicating that service work triggered by 1 is complete
# 3. verify that expected side effects (most likely to model state) of completed work in 2
#    are present


def test_service_reacts_to_set_raw_dir(
    qtbot: QtBot, test_env_input_view: TestEnvironment
) -> None:
    test_env: TestEnvironment = test_env_input_view
    # Assert (sanity check)
    assert test_env.model.get_channel_count(ImageType.RAW) is None
    assert (
        test_env.model.get_image_directory_paths(ImageType.RAW) is None
    )

    # Act
    with qtbot.waitSignal(test_env.model.channel_count_set):
        test_env.model.set_image_directory(ImageType.RAW, IMG_DIR_PATH)

    # Assert
    # we expect that when the raw directory is set, CurationService will extract the paths
    # and the number of channels from that directory
    assert test_env.model.get_channel_count(ImageType.RAW) is not None
    for file in test_env.model.get_image_directory_paths(
        ImageType.RAW
    ):
        assert file in IMG_DIR_FILES


def test_service_reacts_to_set_seg1_dir(
    qtbot: QtBot, test_env_input_view: TestEnvironment
) -> None:
    test_env: TestEnvironment = test_env_input_view
    # Assert (sanity check)
    assert test_env.model.get_channel_count(ImageType.SEG1) is None
    assert (
        test_env.model.get_image_directory_paths(ImageType.SEG1)
        is None
    )

    # Act
    with qtbot.waitSignal(test_env.model.channel_count_set):
        test_env.model.set_image_directory(
            ImageType.SEG1, IMG_DIR_PATH
        )

    # Assert
    assert test_env.model.get_channel_count(ImageType.SEG1) is not None
    for file in test_env.model.get_image_directory_paths(
        ImageType.SEG1
    ):
        assert file in IMG_DIR_FILES


def test_service_reacts_to_set_seg2_dir(
    qtbot: QtBot, test_env_input_view: TestEnvironment
) -> None:
    test_env: TestEnvironment = test_env_input_view
    # Assert (sanity check)
    assert test_env.model.get_channel_count(ImageType.SEG2) is None
    assert (
        test_env.model.get_image_directory_paths(ImageType.SEG2)
        is None
    )

    # Act
    with qtbot.waitSignal(test_env.model.channel_count_set):
        test_env.model.set_image_directory(
            ImageType.SEG2, IMG_DIR_PATH
        )

    # Assert
    assert test_env.model.get_channel_count(ImageType.SEG2) is not None
    for file in test_env.model.get_image_directory_paths(
        ImageType.SEG2
    ):
        assert file in IMG_DIR_FILES


def test_service_reacts_to_cursor_moved(
    qtbot: QtBot, test_env_main_view: TestEnvironment
) -> None:
    test_env: TestEnvironment = test_env_main_view
    # Assert (sanity check)
    assert test_env.model.get_curr_image_data(ImageType.RAW) is None
    assert test_env.model.get_curr_image_data(ImageType.SEG1) is None
    assert test_env.model.get_curr_image_data(ImageType.SEG2) is None

    # Act
    with qtbot.waitSignal(test_env.model.image_loading_finished):
        test_env.model.start_loading_images()

    # Assert
    assert not test_env.model.is_waiting_for_images()
    assert (
        test_env.model.get_curr_image_data(ImageType.RAW) is not None
    )
    assert (
        test_env.model.get_curr_image_data(ImageType.SEG1) is not None
    )
    assert (
        test_env.model.get_curr_image_data(ImageType.SEG2) is not None
    )


def test_service_reacts_to_save_csv(
    qtbot: QtBot, test_env_main_view: TestEnvironment
) -> None:
    test_env: TestEnvironment = test_env_main_view

    # Arrange
    with qtbot.waitSignal(test_env.model.image_loading_finished):
        test_env.model.start_loading_images()

    # Assert (sanity check)
    assert len(test_env.file_writer.csv_state) == 0

    # Act
    with qtbot.waitSignal(test_env.model.saved_to_disk):
        test_env.model.save_curr_curation_record_to_disk()

    # Assert
    assert len(test_env.file_writer.csv_state) > 0


def test_service_reacts_to_view_change(
    qtbot: QtBot, test_env_input_view_synchro_executor: TestEnvironment
) -> None:
    """
    When the view changes to main, we expect a copy of the selected channels to be
    saved to disk.
    """
    test_env: TestEnvironment = test_env_input_view_synchro_executor

    # Arrange
    test_env.model.set_selected_channel(ImageType.RAW, 1)
    test_env.model.set_selected_channel(ImageType.SEG1, 2)
    test_env.model.set_selected_channel(ImageType.SEG2, 3)
    test_env.model.set_image_directory_paths(
        ImageType.RAW, IMG_DIR_FILES
    )
    test_env.model.set_image_directory_paths(
        ImageType.SEG1, IMG_DIR_FILES
    )
    test_env.model.set_image_directory_paths(
        ImageType.SEG2, IMG_DIR_FILES
    )

    # Assert (sanity check)
    assert len(test_env.file_writer.json_state) == 0

    # Act
    test_env.model.set_current_view(CurationView.MAIN_VIEW)

    # Assert
    assert len(test_env.file_writer.json_state) == 1
    assert test_env.file_writer.json_state[FAKE_CHANNEL_SELECTION_PATH] == {
        "raw": 1,
        "seg1": 2,
        "seg2": 3,
    }
