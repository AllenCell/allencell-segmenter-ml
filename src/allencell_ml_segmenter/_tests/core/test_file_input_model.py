from pathlib import Path

import allencell_ml_segmenter
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.file_input_model import (
    FileInputModel,
    InputMode,
)


def test_set_selected_paths_no_extract_channels() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_FILEINPUT_EXTRACT_CHANNELS,
        dummy_subscriber,
        dummy_subscriber.handle,
    )

    # Act
    file_input_model.set_selected_paths([Path()], False)

    # Assert nothing happened

    assert not dummy_subscriber.was_handled(
        Event.ACTION_FILEINPUT_EXTRACT_CHANNELS
    )


def test_set_selected_paths_no_paths() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_FILEINPUT_EXTRACT_CHANNELS,
        dummy_subscriber,
        dummy_subscriber.handle,
    )

    # Act
    file_input_model.set_selected_paths(None, True)

    # Assert nothing happened

    assert not dummy_subscriber.was_handled(
        Event.ACTION_FILEINPUT_EXTRACT_CHANNELS
    )


def test_set_selected_paths_dispatched() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_FILEINPUT_EXTRACT_CHANNELS,
        dummy_subscriber,
        dummy_subscriber.handle,
    )

    # Act
    file_input_model.set_selected_paths([Path()], True)

    # Assert dispatched
    assert dummy_subscriber.was_handled(
        Event.ACTION_FILEINPUT_EXTRACT_CHANNELS
    )


def test_set_max_channels_no_channel() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_FILEINPUT_MAX_CHANNELS_SET,
        dummy_subscriber,
        dummy_subscriber.handle,
    )

    # Act
    file_input_model.set_max_channels(None)

    # Assert nothing happened
    assert not dummy_subscriber.was_handled(
        Event.ACTION_FILEINPUT_MAX_CHANNELS_SET
    )


def test_set_max_channels_dispatch() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_FILEINPUT_MAX_CHANNELS_SET,
        dummy_subscriber,
        dummy_subscriber.handle,
    )

    # Act
    file_input_model.set_max_channels(2)

    # Assert nothing happened
    dummy_subscriber.was_handled(Event.ACTION_FILEINPUT_MAX_CHANNELS_SET)


def test_get_input_files_as_list_from_path() -> None:
    """
    Test to see if all paths from a directory are returned as a list
    """
    # ARRANGE
    file_input_model: FileInputModel = FileInputModel()
    file_input_model.set_input_mode(InputMode.FROM_PATH)
    file_input_model.set_input_image_path(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "img_folder"
    )

    # Act
    files: list[Path] = file_input_model.get_input_files_as_list()

    # Assert
    assert len(files) == 5


def test_get_input_files_as_list_from_viewer() -> None:
    """
    Test to see if all paths from viewer displayed images are returned as a list
    """
    # ARRANGE
    file_input_model: FileInputModel = FileInputModel()
    file_input_model.set_input_mode(InputMode.FROM_NAPARI_LAYERS)
    fake_selected_paths: list[Path] = [Path("fake_path1"), Path("fake_path2")]
    file_input_model.set_selected_paths(fake_selected_paths)

    # Act
    files: list[Path] = file_input_model.get_input_files_as_list()

    # Assert
    assert len(files) == 2
    assert files == fake_selected_paths


def test_get_input_files_as_list_from_no_directory_selected() -> None:
    """
    Test to see if an empty list is returned when no directory is selected
    """
    # ARRANGE
    file_input_model: FileInputModel = FileInputModel()

    # Act
    files: list[Path] = file_input_model.get_input_files_as_list()

    # Assert
    assert len(files) == 0


def test_get_input_files_as_list_from_no_selected_paths() -> None:
    """
    Test to see if an empty list is returned when no layers were selected
    """
    # ARRANGE
    file_input_model: FileInputModel = FileInputModel()

    # Act
    files: list[Path] = file_input_model.get_input_files_as_list()

    # Assert
    assert len(files) == 0


def test_get_input_files_as_list_from_no_selected_paths() -> None:
    """
    Test to see if an empty list is returned when no input mode is selected
    """
    # ARRANGE
    file_input_model: FileInputModel = FileInputModel()

    # Act
    files: list[Path] = file_input_model.get_input_files_as_list()

    # Assert
    assert len(files) == 0
