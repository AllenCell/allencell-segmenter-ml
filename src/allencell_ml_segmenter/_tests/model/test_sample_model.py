import pytest
from allencell_ml_segmenter.model.sample_model import SampleModel
from allencell_ml_segmenter.model.pub_sub import Event


class MockSubscriber:
    def __init__(self):
        self.event_received = None

    def handle_event(self, event):
        self.event_received = event

@pytest.fixture
def sample_model():
    return SampleModel()


def test_set_model_training(sample_model):
    subscriber = MockSubscriber()
    sample_model.subscribe(subscriber)

    sample_model.set_model_training(True)

    assert sample_model._model_training
    assert subscriber.event_received == Event.TRAINING


def test_get_model_training(sample_model):
    assert not sample_model.get_model_training()

    sample_model.set_model_training(True)
    assert sample_model.get_model_training()