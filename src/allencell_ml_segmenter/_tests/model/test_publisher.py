import pytest
from allencell_ml_segmenter.model.publisher import Publisher, Subscriber, Event


@pytest.fixture
def publisher():
    return Publisher()


class MockSubscriber(Subscriber):
    def __init__(self):
        self.handled_event = None

    def handle_event(self, event: Event):
        self.handled_event = event


def test_pub_dispatch(publisher):
    subscriber = MockSubscriber()
    publisher.subscribe(subscriber)
    event = Event.TRAINING

    publisher.dispatch(event)

    assert subscriber.handled_event == event


def test_pub_dispatch_multiple(publisher):
    subscriber1 = MockSubscriber()
    subscriber2 = MockSubscriber()
    publisher.subscribe(subscriber1)
    publisher.subscribe(subscriber2)

    publisher.dispatch(Event.TRAINING)

    assert subscriber1.handled_event == Event.TRAINING
    assert subscriber2.handled_event == Event.TRAINING
