from pathlib import Path
from typing import List

import pytest
from pytestqt.qtbot import QtBot
from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel

from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    PatchSize,
)


@pytest.fixture
def training_model() -> TrainingModel:
    """
    Fixture that creates an instance of TrainingModel for testing.
    """
    return TrainingModel(
        MainModel(ExperimentsModel(CytoDlConfig(Path(), Path())))
    )


@pytest.fixture
def model_selection_widget(
    qtbot: QtBot,
    training_model: TrainingModel,
) -> ModelSelectionWidget:
    """
    Fixture that creates an instance of ModelSelectionWidget for testing.
    """
    return ModelSelectionWidget(training_model)


# def test_radio_new_slot(
#     qtbot: QtBot, model_selection_widget: ModelSelectionWidget
# ) -> None:
#     """
#     Test the slot connected to the "new model" radio button.
#     """
#     # ARRANGE
#     model_selection_widget._combo_box_existing_models.setEnabled(True)

#     # ACT (disable combo box)
#     with qtbot.waitSignal(model_selection_widget._radio_new_model.toggled):
#         model_selection_widget._radio_new_model.click()

#     # ASSERT
#     assert not model_selection_widget._combo_box_existing_models.isEnabled()


def test_radio_existing_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the "existing model" radio button.
    """
    # ARRANGE
    model_selection_widget._combo_box_existing_models.setEnabled(False)

    # ACT (enable combo box)
    with qtbot.waitSignal(
        model_selection_widget._radio_existing_model.toggled
    ):
        model_selection_widget._radio_existing_model.click()

    # ASSERT
    assert model_selection_widget._combo_box_existing_models.isEnabled()


def test_checkbox_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the timeout checkbox.
    """
    # ASSERT (QLineEdit related to timeout limit is disabled by default)
    assert not model_selection_widget._max_time_in_hours_input.isEnabled()

    # ACT (enable QLineEdit related to timeout limit)
    with qtbot.waitSignal(
        model_selection_widget._max_time_checkbox.stateChanged
    ):
        model_selection_widget._max_time_checkbox.click()

    # ASSERT
    assert model_selection_widget._max_time_in_hours_input.isEnabled()

    # ACT (disabled QLineEdit related to timeout limit)
    with qtbot.waitSignal(
        model_selection_widget._max_time_checkbox.stateChanged
    ):
        model_selection_widget._max_time_checkbox.click()

    # ASSERT
    assert not model_selection_widget._max_time_in_hours_input.isEnabled()


def test_select_existing_model_option(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the slots connected to the "start a new model" radio button and the existing model QCombBox properly set the model path field.
    """
    # ARRANGE - add arbitrary model path options to the QComboBox, since it does not come with default choices
    mock_choices: List[str] = [f"dummy path {i}" for i in range(3)]
    model_selection_widget._combo_box_existing_models.addItems(mock_choices)
    with qtbot.waitSignal(
        model_selection_widget._radio_existing_model.toggled
    ):
        model_selection_widget._radio_existing_model.click()  # enables the combo box

    for i, choice in enumerate(mock_choices):
        # ACT
        model_selection_widget._combo_box_existing_models.setCurrentIndex(i)
        training_model.set_checkpoint("dummy_checkpoint")

        # ASSERT
        assert training_model.get_model_checkpoints_path() == Path(
            f"{choice}/checkpoints/dummy_checkpoint"
        )

# def test_select_new_model_radio(
#     qtbot: QtBot,
#     model_selection_widget: ModelSelectionWidget,
#     training_model: TrainingModel,
# ) -> None:
    
#     # ARRANGE
#     training_model.set_checkpoint("dummy_checkpoint")
#     training_model.set_experiment_name("dummy_experiment")

#     # ACT - press "start a new model" radio button, which should set model_path to None
#     with qtbot.waitSignal(
#         model_selection_widget._radio_new_model.toggled
#     ):
#         model_selection_widget._radio_new_model.click()  # enables the combo box

#     # ASSERT
#     assert training_model.get_model_checkpoints_path() is None


def test_set_patch_size(
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that using the associated combo box properly sets the patch size field.
    """

    for index, patch in enumerate(PatchSize):
        # ACT
        model_selection_widget._patch_size_combo_box.setCurrentIndex(index)

        # ASSERT
        assert training_model.get_patch_size() == patch


def test_set_image_dimensions(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that checking the associated radio buttons properly sets the image dimensions.
    """
    # ACT
    with qtbot.waitSignal(model_selection_widget._radio_2d.toggled):
        model_selection_widget._radio_2d.click()

    # ASSERT
    assert training_model.get_image_dims() == 2

    # ACT
    with qtbot.waitSignal(model_selection_widget._radio_3d.toggled):
        model_selection_widget._radio_3d.click()

    # ASSERT
    assert training_model.get_image_dims() == 3


def test_set_max_epoch(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the max epoch field is properly set by the associated QLineEdit.
    """
    # ACT
    qtbot.keyClicks(model_selection_widget._max_epoch_input, "100")

    # ASSERT
    assert training_model.get_max_epoch() == 100


def test_set_max_time(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    training_model: TrainingModel,
) -> None:
    """
    Tests that the max time field is properly set by the associated QLineEdit.
    """
    # ACT
    with qtbot.waitSignal(model_selection_widget._max_time_checkbox.toggled):
        model_selection_widget._max_time_checkbox.click()  # enables the QLineEdit

    qtbot.keyClicks(model_selection_widget._max_time_in_hours_input, "1")

    # ASSERT
    assert training_model.get_max_time() == 3600
