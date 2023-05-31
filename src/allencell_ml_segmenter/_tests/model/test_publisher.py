import pytest
from allencell_ml_segmenter.model.publisher import Publisher, Event
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def publisher():
    return Publisher()


def test_pub_dispatch(publisher):
    subscriber = FakeSubscriber()
    publisher.subscribe(subscriber)

    publisher.dispatch(Event.TRAINING)

    assert subscriber.handled_event == Event.TRAINING


def test_pub_dispatch_multiple(publisher):
    subscriber1 = FakeSubscriber()
    subscriber2 = FakeSubscriber()
    publisher.subscribe(subscriber1)
    publisher.subscribe(subscriber2)

    publisher.dispatch(Event.TRAINING)

    assert subscriber1.handled_event == Event.TRAINING
    assert subscriber2.handled_event == Event.TRAINING
