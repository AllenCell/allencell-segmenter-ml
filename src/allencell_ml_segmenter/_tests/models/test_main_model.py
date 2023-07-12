import pytest
from unittest.mock import Mock
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.core.publisher import Event
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def main_model():
    return MainModel()


@pytest.fixture
def fake_subscriber():
    return FakeSubscriber()


def test_get_current_view(main_model):
    assert main_model.get_current_view() is None
    mock_view = Mock(spec=View)
    main_model._current_view = mock_view

    assert main_model.get_current_view() == mock_view


def test_set_current_view(main_model, fake_subscriber):
    # set a mock views
    mock_view = Mock(spec=View)

    # ARRANGE
    main_model.subscribe(
        Event.ACTION_CHANGE_VIEW, fake_subscriber, fake_subscriber.handle
    )

    # ACT
    main_model.set_current_view(mock_view)

    # ASSERT
    assert fake_subscriber.was_handled(Event.ACTION_CHANGE_VIEW)
