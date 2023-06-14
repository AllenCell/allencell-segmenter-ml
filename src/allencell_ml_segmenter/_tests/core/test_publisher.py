import pytest
from allencell_ml_segmenter.core.publisher import Publisher, Event
from allencell_ml_segmenter._tests.fakes.fake_event_handler import (
    FakeEventHandler,
)
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def publisher():
    return Publisher()


def test_pub_dispatch(publisher):
    subscriber = FakeSubscriber()
    publisher.subscribe(Event.TRAINING, subscriber)

    publisher.dispatch(Event.TRAINING)

    assert subscriber.handled_event == Event.TRAINING


def test_pub_dispatch_explicit_handler(publisher: Publisher):
    subscriber = FakeSubscriber()
    fake_event_handler = FakeEventHandler()
    publisher.subscribe(Event.TRAINING, subscriber, fake_event_handler.handle)

    # ACT
    publisher.dispatch(Event.TRAINING)

    assert fake_event_handler.is_handled()


def test_pub_dispatch_multiple(publisher: Publisher):
    subscriber1 = FakeSubscriber()
    subscriber2 = FakeSubscriber()
    publisher.subscribe(Event.TRAINING, subscriber1)
    publisher.subscribe(Event.TRAINING, subscriber2)

    publisher.dispatch(Event.TRAINING)

    assert subscriber1.handled_event == Event.TRAINING
    assert subscriber2.handled_event == Event.TRAINING


def test_pub_unsubscribe(publisher: Publisher):
    subscriber = FakeSubscriber()

    # ARRANGE
    publisher.subscribe(Event.TRAINING, subscriber)
    publisher.unsubscribe(Event.TRAINING, subscriber)

    # ACT
    publisher.dispatch(Event.TRAINING)

    assert subscriber.handled_event is None


def test_pub_unsubscribe_unknown(publisher: Publisher):
    subscriber = FakeSubscriber()

    # ACT
    publisher.unsubscribe(Event.TRAINING, subscriber)

    assert 1 == 1  # no error raised
