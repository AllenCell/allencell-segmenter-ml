from pathlib import Path

from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.file_input_model import FileInputModel


def test_set_selected_paths_no_extract_channels() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_PREDICTION_EXTRACT_CHANNELS,
        dummy_subscriber,
        dummy_subscriber.handle
    )

    # Act
    file_input_model.set_selected_paths([Path()], False)

    # Assert nothing happened

    assert len(dummy_subscriber.handled) == 0


def test_set_selected_paths_no_paths() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_PREDICTION_EXTRACT_CHANNELS,
        dummy_subscriber,
        dummy_subscriber.handle
    )

    # Act
    file_input_model.set_selected_paths([], True)

    # Assert nothing happened

    assert len(dummy_subscriber.handled) == 0

def test_set_selected_paths_dispatched() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_PREDICTION_EXTRACT_CHANNELS,
        dummy_subscriber,
        dummy_subscriber.handle
    )

    # Act
    file_input_model.set_selected_paths([Path()], True)

    # Assert dispatched
    assert len(dummy_subscriber.handled) == 1
    assert dummy_subscriber.handled[Event.ACTION_PREDICTION_EXTRACT_CHANNELS] = True

def test_set_max_channels_no_channel() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_PREDICTION_EXTRACT_CHANNELS,
        dummy_subscriber,
        dummy_subscriber.handle
    )

    # Act
    file_input_model.set_max_channels(None)

    # Assert nothing happened
    assert len(dummy_subscriber.handled) == 0

def test_set_max_channels_dispatch() -> None:
    # Arrange
    file_input_model: FileInputModel = FileInputModel()
    dummy_subscriber: FakeSubscriber = FakeSubscriber()
    file_input_model.subscribe(
        Event.ACTION_PREDICTION_EXTRACT_CHANNELS,
        dummy_subscriber,
        dummy_subscriber.handle
    )

    # Act
    file_input_model.set_max_channels(2)

    # Assert nothing happened
    assert len(dummy_subscriber.handled) == 1
    assert dummy_subscriber.handled[Event.ACTION_PREDICTION_MAX_CHANNELS_SET] = True





