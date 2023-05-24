import pytest
from allencell_ml_segmenter.model.training_model import TrainingModel
from allencell_ml_segmenter.model.publisher import Event
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber

@pytest.fixture
def sample_model():
    return TrainingModel()


def test_set_model_training(sample_model):
    subscriber = FakeSubscriber()
    sample_model.subscribe(subscriber)

    sample_model.set_model_training(True)

    assert sample_model._model_training
    assert subscriber.handled_event == Event.TRAINING


def test_get_model_training(sample_model):
    assert not sample_model.get_model_training()

    sample_model.set_model_training(True)
    assert sample_model.get_model_training()
