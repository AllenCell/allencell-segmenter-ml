from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.main.main_service import MainService
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    ModelSize,
    ImageType,
)
from allencell_ml_segmenter.training.view import TrainingView
from allencell_ml_segmenter.core.task_executor import SynchroTaskExecutor
import pytest
from pytestqt.qtbot import QtBot
import allencell_ml_segmenter
from pathlib import Path


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


def test_set_image_dimensions(
    qtbot: QtBot,
    training_view: TrainingView,
    training_model: TrainingModel,
) -> None:
    """
    Tests that checking the associated radio buttons properly sets the image dimensions.
    """
    # ACT
    with qtbot.waitSignal(training_view._radio_2d.toggled):
        training_view._radio_2d.click()

    # ASSERT
    assert training_model.get_spatial_dims() == 2

    # ACT
    with qtbot.waitSignal(training_view._radio_3d.toggled):
        training_view._radio_3d.click()

    # ASSERT
    assert training_model.get_spatial_dims() == 3


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


def test_set_patch_size(
    qtbot: QtBot,
    main_model: MainModel,
    experiments_model: FakeExperimentsModel,
    training_model: TrainingModel,
) -> None:
    # ARRANGE
    view: TrainingView = TrainingView(
        main_model=main_model,
        experiments_model=FakeExperimentsModel(),
        training_model=training_model,
        viewer=FakeViewer(),
    )
    view.z_patch_size.setText("1")
    view.y_patch_size.setText("4")
    view.x_patch_size.setText("12")

    # ACT
    view.set_patch_size()

    # ASSERT
    assert training_model.get_patch_size() == [1, 4, 12]


def test_navigate_to_training_populates_channel_selection(
    qtbot: QtBot,
) -> None:
    """
    Validates that when there is a valid channel selection JSON, navigating to training
    will auto-populate the channel selection dropdowns with the previously chosen channels
    from curation.
    """
    # Arrange
    test_channel_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "channel_selection_json"
        / "valid_mixed.json"
    )
    main_model: MainModel = MainModel()
    experiments_model: FakeExperimentsModel = FakeExperimentsModel(
        channel_selection_path=test_channel_path
    )
    # must init main service and set experiment name to pull channel data into main model
    main_service: MainService = MainService(
        main_model,
        experiments_model,
        task_executor=SynchroTaskExecutor.global_instance(),
    )
    experiments_model.apply_experiment_name("test")

    training_model: TrainingModel = TrainingModel(
        main_model, experiments_model
    )
    view: TrainingView = TrainingView(
        main_model=main_model,
        experiments_model=experiments_model,
        training_model=training_model,
        viewer=FakeViewer(),
    )

    # Act
    # simulate service completing its channel extraction work
    training_model.set_all_num_channels(
        {
            ImageType.RAW: 8,
            ImageType.SEG1: 6,
            ImageType.SEG2: 4,
        }
    )

    # Assert (these values come from valid_mixed.json)
    assert training_model.get_selected_channel(ImageType.RAW) == 5
    assert training_model.get_selected_channel(ImageType.SEG1) == 2
    assert training_model.get_selected_channel(ImageType.SEG2) == 1
    assert (
        view.image_selection_widget._raw_channel_combo_box.currentIndex() == 5
    )
    assert (
        view.image_selection_widget._seg1_channel_combo_box.currentIndex() == 2
    )
    assert (
        view.image_selection_widget._seg2_channel_combo_box.currentIndex() == 1
    )


def test_navigate_to_training_populates_channel_selection_no_json(
    qtbot: QtBot,
) -> None:
    """
    Validates that when there is no channel selection JSON, navigating to training
    will auto-populate the channel selection dropdowns with 0.
    """
    # Arrange
    test_channel_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "channel_selection_json"
        / "nonexistent.json"
    )
    main_model: MainModel = MainModel()
    experiments_model: FakeExperimentsModel = FakeExperimentsModel(
        channel_selection_path=test_channel_path
    )
    # must init main service and set experiment to pull channel data into main model
    main_service: MainService = MainService(
        main_model,
        experiments_model,
        task_executor=SynchroTaskExecutor.global_instance(),
    )
    experiments_model.apply_experiment_name("test")

    training_model: TrainingModel = TrainingModel(
        main_model, experiments_model
    )
    view: TrainingView = TrainingView(
        main_model=main_model,
        experiments_model=experiments_model,
        training_model=training_model,
        viewer=FakeViewer(),
    )

    # Act
    # simulate service completing its channel extraction work
    training_model.set_all_num_channels(
        {
            ImageType.RAW: 8,
            ImageType.SEG1: 6,
            ImageType.SEG2: 4,
        }
    )

    # Assert (these values come from valid_mixed.json)
    assert training_model.get_selected_channel(ImageType.RAW) == 0
    assert training_model.get_selected_channel(ImageType.SEG1) == 0
    assert training_model.get_selected_channel(ImageType.SEG2) == 0
    assert (
        view.image_selection_widget._raw_channel_combo_box.currentIndex() == 0
    )
    assert (
        view.image_selection_widget._seg1_channel_combo_box.currentIndex() == 0
    )
    assert (
        view.image_selection_widget._seg2_channel_combo_box.currentIndex() == 0
    )
