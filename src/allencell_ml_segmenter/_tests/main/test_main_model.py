import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.core.publisher import Event
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def main_model() -> MainModel:
    """
    Fixture for MainModel testing.
    """
    return MainModel()


@pytest.fixture
def fake_subscriber() -> FakeSubscriber:
    """
    Fixture for FakeSubscriber testing.
    """
    return FakeSubscriber()


def test_get_current_view(main_model: MainModel) -> None:
    """
    Tests that the current view is correctly retrievable.
    """
    # ASSERT
    assert main_model.get_current_view() is None

    # ARRANGE
    mock_view: AicsWidget = Mock(spec=AicsWidget)
    main_model._current_view = mock_view

    # ACT/ASSERT
    assert main_model.get_current_view() == mock_view


def test_set_new_model(
    main_model: MainModel, fake_subscriber: FakeSubscriber
) -> None:
    """
    Tests that a new model can be set.
    """
    # ARRANGE
    main_model.subscribe(
        Event.ACTION_NEW_MODEL, fake_subscriber, fake_subscriber.handle
    )

    # ACT
    main_model.set_new_model(True)

    # ASSERT
    assert fake_subscriber.was_handled(Event.ACTION_NEW_MODEL)


def test_set_current_view(
    main_model: MainModel, fake_subscriber: FakeSubscriber
) -> None:
    """
    Tests that the current view is correctly settable.
    """
    # ARRANGE
    mock_view: AicsWidget = Mock(spec=AicsWidget)

    main_model.subscribe(
        Event.ACTION_CHANGE_VIEW, fake_subscriber, fake_subscriber.handle
    )

    # ACT
    main_model.set_current_view(mock_view)

    # ASSERT
    assert fake_subscriber.was_handled(Event.ACTION_CHANGE_VIEW)
