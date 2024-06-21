from pathlib import Path
from typing import Set

import pytest
from pytestqt.qtbot import QtBot
from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.config.i_user_settings import IUserSettings

from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.main.main_widget import MainWidget
from unittest.mock import Mock
import napari

# IMPORTANT NOTE: MainWidget is different from the other widgets since we do not directly
# instantiate it in our code. So, it will always receive a napari.Viewer object in
# production. We cannot initialize with our FakeViewer because our 'Viewer' is created during
# initialization of MainWidget. We could supply a "viewer factory" to the MainWidget,
# but for now I'm just mocking it here.


@pytest.fixture
def main_widget(qtbot: QtBot) -> MainWidget:
    """
    Returns a MainWidget instance for testing.
    """
    settings: IUserSettings = FakeUserSettings()
    settings.set_cyto_dl_home_path(Path())
    settings.set_user_experiments_path(Path())
    return MainWidget(viewer=Mock(), settings=settings)


def test_tabs_react_to_new_model_event(
    main_widget: MainWidget,
) -> None:
    """
    Tests that the main widget handles the action new model event correctly.
    """

    # ACT: have the model dispatch the action new model event
    main_widget._model.set_new_model(True)

    # ASSERT: check that the main widget's current view (after setting) is curation
    assert (
        main_widget._window_container.currentIndex()
        == main_widget._window_to_index[main_widget._curation_view]
    )
    # ASSERT: check that the correct tabs are enabled
    assert (
        main_widget._window_container.isTabEnabled(
            main_widget._window_to_index[main_widget._prediction_view]
        )
        == True
    )
    assert (
        main_widget._window_container.isTabEnabled(
            main_widget._window_to_index[main_widget._curation_view]
        )
        == True
    )
    assert (
        main_widget._window_container.isTabEnabled(
            main_widget._window_to_index[main_widget._training_view]
        )
        == True
    )


def test_tabs_react_to_existing_model_event(
    main_widget: MainWidget,
) -> None:
    """
    Tests that the main widget handles the action new model event correctly.
    """

    # ACT: have the model dispatch the action new model event
    main_widget._model.set_new_model(False)

    # ASSERT: check that the main widget's current view (after setting) is prediction
    assert (
        main_widget._window_container.currentIndex()
        == main_widget._window_to_index[main_widget._prediction_view]
    )
    # ASSERT: check that the correct tabs are enabled
    assert (
        main_widget._window_container.isTabEnabled(
            main_widget._window_to_index[main_widget._prediction_view]
        )
        == True
    )
    assert (
        main_widget._window_container.isTabEnabled(
            main_widget._window_to_index[main_widget._curation_view]
        )
        == False
    )
    assert (
        main_widget._window_container.isTabEnabled(
            main_widget._window_to_index[main_widget._training_view]
        )
        == False
    )


def test_handle_action_change_view_event(
    main_widget: MainWidget,
) -> None:
    """
    Tests that the main widget handles the action change view event correctly.
    """
    # ARRANGE
    views: Set[AicsWidget] = main_widget._window_to_index.keys()

    for view in views:
        # ACT: have the model dispatch the action change view event
        main_widget._model.set_current_view(view)

        # ASSERT: check that the main widget's current view (after setting) is same as the model's current view
        assert (
            main_widget._window_container.currentIndex()
            == main_widget._window_to_index[view]
        )


def test_experiments_home_initialized(qtbot: QtBot) -> None:
    """
    Tests that the main widget promtps the user to choose a. 'experiments home' dir if one is not found in user settings.
    """
    # ARRANGE
    EXPECTED_EXPERIMENTS_HOME = Path(
        __file__
    ).parent  # simulates (in the fake settings) the location chosed by user
    settings = FakeUserSettings(
        init_prompt_response=EXPECTED_EXPERIMENTS_HOME,
        cyto_dl_home_path=Path("foo/cyto/path"),
    )
    settings.set_user_experiments_path(
        None
    )  # Simulates state where users has not yet chosen an experiments home.

    # ACT
    MainWidget(
        Mock(spec=napari.Viewer), settings
    )  # If the users settings does not find an experiments home path, it will prompt the user for one and persist it.

    # ASSERT
    assert (
        settings.get_user_experiments_path() == EXPECTED_EXPERIMENTS_HOME
    )  # The path chosen by the user should have been persisted in settings.


def test_tab_enabled(main_widget) -> None:
    """
    Tests that the main widget enables the correct tabs when the experiment is applied.
    """

    # ARRANGE
    main_widget._experiments_model.apply_experiment_name("foo")

    # Sanity check
    assert main_widget._window_container.isEnabled() == True

    # ACT
    main_widget._experiments_model.apply_experiment_name(None)

    # ASSERT
    assert main_widget._window_container.isEnabled() == False

    # ACT
    main_widget._experiments_model.apply_experiment_name("foo")

    # Sanity check
    assert main_widget._window_container.isEnabled() == True
