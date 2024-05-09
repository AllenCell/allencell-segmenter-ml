import pytest
from pytestqt.qtbot import QtBot
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel

from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
)


@pytest.fixture
def experiment_model() -> IExperimentsModel:
    return FakeExperimentsModel()


@pytest.fixture
def main_model() -> MainModel:
    """
    Fixture for MainModel testing.
    """
    return MainModel()


@pytest.fixture
def training_model(main_model: MainModel) -> TrainingModel:
    """
    Fixture that creates an instance of TrainingModel for testing.
    """
    return TrainingModel(  # instead of this, how about i mock os.listdir ?
        main_model, FakeExperimentsModel()
    )


@pytest.fixture
def model_selection_widget(
    main_model: MainModel,
    experiment_model: IExperimentsModel,
) -> ModelSelectionWidget:
    """
    Fixture that creates an instance of ModelSelectionWidget for testing.
    """
    return ModelSelectionWidget(
        main_model=main_model, experiments_model=experiment_model
    )


def test_radio_new_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the "new model" radio button.
    """
    # ARRANGE
    model_selection_widget._combo_box_existing_models.setEnabled(True)

    # ACT (disable combo box)
    with qtbot.waitSignal(model_selection_widget._radio_new_model.toggled):
        model_selection_widget._radio_new_model.click()

    # ASSERT
    assert not model_selection_widget._combo_box_existing_models.isEnabled()


def test_radio_existing_slot(
    qtbot: QtBot, model_selection_widget: ModelSelectionWidget
) -> None:
    """
    Test the slot connected to the "existing model" radio button.
    """
    # ARRANGE radios selected in the inverted condition that the action will set them to
    model_selection_widget._radio_new_model.setChecked(True)
    model_selection_widget._radio_existing_model.setChecked(False)
    model_selection_widget._combo_box_existing_models.setEnabled(False)

    # ACT (enable combo box)
    with qtbot.waitSignal(
        model_selection_widget._radio_existing_model.toggled
    ):
        model_selection_widget._radio_existing_model.click()

    # ASSERT
    assert model_selection_widget._combo_box_existing_models.isEnabled()


def test_select_new_model_radio(
    qtbot: QtBot,
    model_selection_widget: ModelSelectionWidget,
    experiment_model: IExperimentsModel,
) -> None:
    # ARRANGE radios selected in the inverted condition that the action will set them to
    experiment_model.apply_experiment_name("dummy_experiment")
    model_selection_widget._radio_new_model.setChecked(False)
    model_selection_widget._radio_existing_model.setChecked(True)

    # ACT - press "start a new model" radio button, which should set model_path to None
    with qtbot.waitSignal(model_selection_widget._radio_new_model.toggled):
        model_selection_widget._radio_new_model.click()  # enables the combo box

    # ASSERT
    assert experiment_model.get_checkpoint() is None


def test_select_existing_model_option(
    qtbot: QtBot,
    experiment_model: IExperimentsModel,
    model_selection_widget: ModelSelectionWidget,
) -> None:
    """
    Tests that the slots connected to the "start a new model" radio button and the existing model QCombBox properly set the model path field.
    """
    # ARRANGE - add arbitrary model path options to the QComboBox, since it does not come with default choices
    model_selection_widget._radio_new_model.setChecked(True)
    model_selection_widget._radio_existing_model.setChecked(False)
    with qtbot.waitSignal(
        model_selection_widget._radio_existing_model.toggled
    ):
        model_selection_widget._radio_existing_model.click()  # enables the combo box

    for i, experiment in enumerate(experiment_model.get_experiments()):
        # ACT
        # Invariant: options in existing_models combo were added in the order the appear in the model.
        model_selection_widget._combo_box_existing_models.setCurrentIndex(i)

        # ASSERT
        assert experiment == experiment_model.get_experiment_name_selection()


def test_apply_button_enabled(
    model_selection_widget: ModelSelectionWidget,
    experiment_model: IExperimentsModel,
) -> None:
    """
    Test that the apply button is enabled when a model is selected.
    """
    # ARRANGE
    assert not model_selection_widget._apply_btn.isEnabled()

    # ACT
    experiment_model.select_experiment_name("dummy_experiment")

    # ASSERT
    assert model_selection_widget._apply_btn.isEnabled()


def test_text_input_enables_apply_button(
    model_selection_widget: ModelSelectionWidget,
) -> None:
    """
    Test that the apply button is disabled when a model is not selected.
    """
    # ARRANGE
    assert not model_selection_widget._apply_btn.isEnabled()

    # ACT
    model_selection_widget._experiment_name_input.setText("dummy_experiment")

    # ASSERT
    assert model_selection_widget._apply_btn.isEnabled()


def test_combo_input_enables_apply_button_new_radio_disables(
    model_selection_widget: ModelSelectionWidget,
    experiment_model: IExperimentsModel,
    qtbot: QtBot,
) -> None:
    """
    Test that the apply button Reacts to a model being selevted then deselected.
    """
    # ARRANGE
    assert not model_selection_widget._apply_btn.isEnabled()
    model_selection_widget._radio_new_model.setChecked(True)
    model_selection_widget._radio_existing_model.setChecked(False)

    # Initially no model is selected, so the apply button should NOT be enabled
    assert not model_selection_widget._apply_btn.isEnabled()

    # ACT - select a model
    experiment_model.select_experiment_name("dummy_experiment")

    # ASSERT - apply button SHOULD be enabled
    assert model_selection_widget._apply_btn.isEnabled()

    # ACT - select the "start a new model" radio button, clearing the model selection
    with qtbot.waitSignal(
        model_selection_widget._radio_existing_model.toggled
    ):
        model_selection_widget._radio_existing_model.click()  # enables the combo box

    # ASSERT - apply button should NOT be enabled
    assert not model_selection_widget._apply_btn.isEnabled()

    # ACT - select an existing model
    model_selection_widget._combo_box_existing_models.setCurrentIndex(1)

    # ASSERT - apply button SHOULD be enabled
    assert model_selection_widget._apply_btn.isEnabled()


def test_click_apply_btn(
    model_selection_widget: ModelSelectionWidget,
    experiment_model: IExperimentsModel,
) -> None:
    """
    Test that the apply button updates model.
    """
    # ARRANGE
    experiment_model.select_experiment_name("dummy_experiment")

    # Sanity check
    assert experiment_model.get_experiment_name() is None

    # ACT
    model_selection_widget._apply_btn.click()

    # ASSERT
    assert experiment_model.get_experiment_name() == "dummy_experiment"


def test_type_name_apply_change(
    model_selection_widget: ModelSelectionWidget,
    experiment_model: IExperimentsModel,
) -> None:
    """
    Test that the apply button updates model.
    """
    # ACT
    model_selection_widget._experiment_name_input.setText("dummy_experiment")

    # ASSERT note that the model name is selected but not applied until the apply button is clicked
    assert (
        experiment_model.get_experiment_name_selection() == "dummy_experiment"
    )
    assert experiment_model.get_experiment_name() is None

    # ACT
    model_selection_widget._apply_btn.click()

    # ASSERT
    assert experiment_model.get_experiment_name() == "dummy_experiment"


def test_select_existing_combo_apply_click_change(
    model_selection_widget: ModelSelectionWidget,
    experiment_model: IExperimentsModel,
) -> None:
    """
    Test that the apply button updates model.
    """
    # ACT
    model_selection_widget._combo_box_existing_models.setIndex(1)

    # ASSERT note that the model name is selected but not applied until the apply button is clicked
    assert experiment_model.get_experiment_name_selection() == "dummy_experiment"
    assert experiment_model.get_experiment_name() is None

    # ACT
    model_selection_widget._apply_btn.click()

    #ASSERT
    assert experiment_model.get_experiment_name() == "dummy_experiment"