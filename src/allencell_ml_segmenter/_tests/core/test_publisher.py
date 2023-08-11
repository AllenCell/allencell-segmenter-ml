import pytest
from allencell_ml_segmenter.core.publisher import Publisher, Event
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def publisher() -> Publisher:
    """
    Fixture for Publisher.
    """
    return Publisher()


def test_pub_dispatch(publisher: Publisher) -> None:
    """
    Tests that a publisher can properly dispatch an event to a subscriber.
    """
    # ARRANGE
    subscriber: FakeSubscriber = FakeSubscriber()
    event_under_test: Event = Event.PROCESS_TRAINING
    publisher.subscribe(event_under_test, subscriber, subscriber.handle)

    # ACT
    publisher.dispatch(event_under_test)

    # ASSERT
    assert subscriber.was_handled(event_under_test)


# TODO from brian: add better methods for testing publisher event to subscriber handlers


def test_pub_dispatch_multiple(publisher: Publisher) -> None:
    """
    Tests that a publisher can properly dispatch an event to multiple subscribers.
    """

    # ARRANGE
    subscriber1: FakeSubscriber = FakeSubscriber()
    subscriber2: FakeSubscriber = FakeSubscriber()

    publisher.subscribe(
        Event.PROCESS_TRAINING, subscriber1, subscriber1.handle
    )
    publisher.subscribe(
        Event.PROCESS_TRAINING, subscriber2, subscriber2.handle
    )

    # ACT
    publisher.dispatch(Event.PROCESS_TRAINING)

    # ASSERT
    assert subscriber1.was_handled(Event.PROCESS_TRAINING)
    assert subscriber2.was_handled(Event.PROCESS_TRAINING)


def test_pub_unsubscribe(publisher: Publisher) -> None:
    """
    Tests that a publisher can properly unsubscribe a subscriber.
    """

    # ARRANGE
    subscriber: FakeSubscriber = FakeSubscriber()
    event_under_test: Event = Event.PROCESS_TRAINING

    publisher.subscribe(event_under_test, subscriber, subscriber.handle)
    publisher.unsubscribe(event_under_test, subscriber)

    # ACT
    publisher.dispatch(event_under_test)

    # ASSERT
    assert subscriber.was_handled(event_under_test) is False


def test_pub_unsubscribe_unknown(publisher: Publisher) -> None:
    """
    Tests that unsubscribing a nonexistent subscriber does not cause errors.
    """
    subscriber: FakeSubscriber = FakeSubscriber()

    # ACT
    publisher.unsubscribe(Event.PROCESS_TRAINING, subscriber)

    # If no error is raised, the test will pass (no arbitrary assertion needed)
