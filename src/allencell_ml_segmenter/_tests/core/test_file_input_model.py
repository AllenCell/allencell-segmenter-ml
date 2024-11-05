from pathlib import Path

from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.file_input_model import FileInputModel


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
