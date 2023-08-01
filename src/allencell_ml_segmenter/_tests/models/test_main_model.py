import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.core.publisher import Event
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def main_model() -> MainModel:
    return MainModel()


@pytest.fixture
def fake_subscriber() -> FakeSubscriber:
    return FakeSubscriber()


def test_get_current_view(main_model: MainModel) -> None:
    assert main_model.get_current_view() is None
    mock_view: View = Mock(spec=View)
    main_model._current_view = mock_view

    assert main_model.get_current_view() == mock_view


def test_set_current_view(
    main_model: MainModel, fake_subscriber: FakeSubscriber
) -> None:
    # set a mock views
    mock_view: View = Mock(spec=View)

    # ARRANGE
    main_model.subscribe(
        Event.ACTION_CHANGE_VIEW, fake_subscriber, fake_subscriber.handle
    )

    # ACT
    main_model.set_current_view(mock_view)

    # ASSERT
    assert fake_subscriber.was_handled(Event.ACTION_CHANGE_VIEW)
