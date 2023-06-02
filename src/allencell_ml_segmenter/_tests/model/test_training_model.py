import pytest
from allencell_ml_segmenter.model.training_model import TrainingModel
from allencell_ml_segmenter.model.publisher import Event
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def training_model():
    return TrainingModel()


def test_set_model_training(training_model):
    subscriber = FakeSubscriber()
    training_model.subscribe(subscriber)

    training_model.set_model_training(True)

    assert training_model._model_training
    assert subscriber.handled_event == Event.TRAINING


def test_get_model_training(training_model):
    assert not training_model.get_model_training()

    training_model.set_model_training(True)
    assert training_model.get_model_training()
