from allencell_ml_segmenter.main.main_service import MainService
from allencell_ml_segmenter.main.main_model import MainModel, ImageType
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.utils.file_writer import FakeFileWriter
from allencell_ml_segmenter.core.task_executor import SynchroTaskExecutor
from typing import Optional
from pathlib import Path
import allencell_ml_segmenter


# raw: 5, seg1: 2, seg2: 1
mixed_channel_path: Path = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "channel_selection_json"
    / "valid_mixed.json"
)

nonexistent_channel_path: Path = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "channel_selection_json"
    / "DNE.json"
)


def test_service_sets_channels_on_exp_applied_mixed():
    # Arrange
    main_model: MainModel = MainModel()
    exp_model: FakeExperimentsModel = FakeExperimentsModel(
        channel_selection_path=mixed_channel_path
    )
    service: MainService = MainService(
        main_model,
        exp_model,
        task_executor=SynchroTaskExecutor.global_instance(),
        file_writer=FakeFileWriter(),
    )

    # Assert (sanity check)
    channels: dict[ImageType, Optional[int]] = (
        main_model.get_selected_channels()
    )
    assert channels[ImageType.RAW] is None
    assert channels[ImageType.SEG1] is None
    assert channels[ImageType.SEG2] is None

    # Act
    exp_model.apply_experiment_name("test")

    # Assert
    channels = main_model.get_selected_channels()
    assert channels[ImageType.RAW] == 5
    assert channels[ImageType.SEG1] == 2
    assert channels[ImageType.SEG2] == 1


def test_service_does_not_set_channels_on_exp_applied_nonexistent():
    # Arrange
    main_model: MainModel = MainModel()
    exp_model: FakeExperimentsModel = FakeExperimentsModel(
        channel_selection_path=nonexistent_channel_path
    )
    service: MainService = MainService(
        main_model,
        exp_model,
        task_executor=SynchroTaskExecutor.global_instance(),
        file_writer=FakeFileWriter(),
    )

    # Assert (sanity check)
    channels: dict[ImageType, Optional[int]] = (
        main_model.get_selected_channels()
    )
    assert channels[ImageType.RAW] is None
    assert channels[ImageType.SEG1] is None
    assert channels[ImageType.SEG2] is None

    # Act
    exp_model.apply_experiment_name("test")

    # Assert
    channels = main_model.get_selected_channels()
    # since the JSON is not available, channels should all be None still
    assert channels[ImageType.RAW] is None
    assert channels[ImageType.SEG1] is None
    assert channels[ImageType.SEG2] is None


def test_service_writes_to_disk_when_channels_change():
    # Arrange
    main_model: MainModel = MainModel()
    exp_model: FakeExperimentsModel = FakeExperimentsModel(
        channel_selection_path=nonexistent_channel_path
    )
    file_writer: FakeFileWriter = FakeFileWriter()
    service: MainService = MainService(
        main_model,
        exp_model,
        task_executor=SynchroTaskExecutor.global_instance(),
        file_writer=file_writer,
    )

    # Assert (sanity check)
    channels: dict[ImageType, Optional[int]] = (
        main_model.get_selected_channels()
    )
    assert channels[ImageType.RAW] is None
    assert channels[ImageType.SEG1] is None
    assert channels[ImageType.SEG2] is None
    assert len(file_writer.json_state) == 0

    # Act
    main_model.set_selected_channels(
        {
            ImageType.RAW: 8,
            ImageType.SEG1: 2,
            ImageType.SEG2: 7,
        }
    )

    # Assert
    assert len(file_writer.json_state) == 1
    assert file_writer.json_state[nonexistent_channel_path] == {
        "raw": 8,
        "seg1": 2,
        "seg2": 7,
    }
