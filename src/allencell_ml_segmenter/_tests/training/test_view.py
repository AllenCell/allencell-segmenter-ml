from typing import List

from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.training.training_model import (
    PatchSize,
    TrainingModel,
    ModelSize,
)
from allencell_ml_segmenter.training.view import TrainingView
import pytest
from pytestqt.qtbot import QtBot


@pytest.fixture
def main_model():
    return MainModel()


@pytest.fixture
def experiments_model():
    return FakeExperimentsModel()


@pytest.fixture
def training_model(main_model, experiments_model):
    return TrainingModel(
        main_model=main_model, experiments_model=experiments_model
    )


@pytest.fixture
def viewer():
    return FakeViewer()


@pytest.fixture
def training_view(
    qtbot: QtBot, main_model: MainModel, training_model: TrainingModel
) -> TrainingView:
    """
    Returns a PredictionView instance for testing.
    """
    experimentsModel = FakeExperimentsModel()
    return TrainingView(
        main_model=main_model,
        experiments_model=experimentsModel,
        training_model=training_model,
        viewer=FakeViewer(),
    )


def test_handle_dimensions_available_3d(training_view: TrainingView, training_model: TrainingModel) -> None:
    # arrange
    test_dims: List[int] = [1, 3, 6] # Z, Y, X

    # Act
    training_model.set_image_dimensions(test_dims)

    # assert
    # check max patch sizes set for QSpinBoxes based on image dims given
    assert training_view._z_patch_size.maximum() == test_dims[0]
    assert training_view._y_patch_size.maximum() == test_dims[1]
    assert training_view._x_patch_size.maximum() == test_dims[2]

    # check label displays correct number of dims
    assert training_view._dimension_label.text() == "3D"

    # check model updated with correct number of spatial dims
    assert training_model.get_spatial_dims() == len(test_dims)


def test_handle_dimensions_available_2d(training_view: TrainingView, training_model: TrainingModel) -> None:
    # arrange
    test_dims: List[int] = [2, 4]  #Y, X

    # Act
    training_model.set_image_dimensions(test_dims)

    # assert
    # 2d- so z still disabled
    assert training_view._z_patch_size.maximum() == 0
    assert not training_view._z_patch_size.isEnabled()
    # check max patch sizes set for QSpinBoxes and were enabled
    assert training_view._y_patch_size.maximum() == test_dims[0]
    assert training_view._y_patch_size.isEnabled()
    assert training_view._x_patch_size.maximum() == test_dims[1]
    assert training_view._x_patch_size.isEnabled()

    # check label displays correct number of dims
    assert training_view._dimension_label.text() == "2D"

    # check model updated with correct number of spatial dims
    assert training_model.get_spatial_dims() == len(test_dims)

def test_set_max_epoch(
    qtbot: QtBot, training_view: TrainingView, training_model: TrainingModel
) -> None:
    """
    Tests that the max epoch field is properly set by the associated QLineEdit.
    """
    # ACT
    qtbot.keyClicks(training_view._num_epochs_input, "100")

    # ASSERT
    assert training_model.get_num_epochs() == 100


def test_set_max_time(
    qtbot: QtBot, training_view: TrainingView, training_model: TrainingModel
) -> None:
    """
    Tests that the max time field is properly set by the associated QLineEdit.
    """
    # ACT
    with qtbot.waitSignal(training_view._max_time_checkbox.toggled):
        training_view._max_time_checkbox.click()  # enables the QLineEdit

    qtbot.keyClicks(training_view._max_time_in_minutes_input, "30")

    # ASSERT
    assert training_model.get_max_time() == 30


def test_checkbox_slot(
    qtbot: QtBot, training_view: TrainingView, training_model: TrainingModel
) -> None:
    """
    Test the slot connected to the timeout checkbox.
    """
    # ASSERT (QLineEdit related to timeout limit is disabled by default)
    assert not training_view._max_time_in_minutes_input.isEnabled()

    # ACT (enable QLineEdit related to timeout limit)
    with qtbot.waitSignal(training_view._max_time_checkbox.stateChanged):
        training_view._max_time_checkbox.click()

    # ASSERT
    assert training_view._max_time_in_minutes_input.isEnabled()

    # ACT (disabled QLineEdit related to timeout limit)
    with qtbot.waitSignal(training_view._max_time_checkbox.stateChanged):
        training_view._max_time_checkbox.click()

    # ASSERT
    assert not training_view._max_time_in_minutes_input.isEnabled()


def test_set_model_size(
    training_view: TrainingView, training_model: TrainingModel
) -> None:
    """
    Tests that using the associated combo box properly sets the model size field.
    """
    for idx, model_size in enumerate(ModelSize):
        # ACT
        training_view._model_size_combo_box.setCurrentIndex(idx)

        # ASSERT
        assert training_model.get_model_size() == model_size
