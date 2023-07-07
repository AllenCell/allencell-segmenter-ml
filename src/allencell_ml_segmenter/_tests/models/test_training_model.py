import pytest
from allencell_ml_segmenter.models.training_model import TrainingModel
from allencell_ml_segmenter.core.publisher import Event
from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber


@pytest.fixture
def training_model():
    return TrainingModel()


def test_set_model_training(training_model):

    # ARRANGE
    subscriber = FakeSubscriber()
    training_model.subscribe(Event.PROCESS_TRAINING, subscriber, subscriber.handle)

    # ACT
    training_model.set_model_training(True)

    # ASSERT
    assert training_model.get_model_training()
    assert subscriber.was_handled(Event.PROCESS_TRAINING)