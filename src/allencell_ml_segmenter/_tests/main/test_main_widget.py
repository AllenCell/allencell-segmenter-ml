from pathlib import Path, PurePath
from typing import Set

import pytest
import napari
from pytestqt.qtbot import QtBot
from allencell_ml_segmenter.config.fake_user_settings import FakeUserSettings
from allencell_ml_segmenter.config.i_user_settings import IUserSettings

from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.main.main_widget import MainWidget
from unittest.mock import Mock


@pytest.fixture
def viewer() -> napari.Viewer:
    """
    Returns a mock napari viewer.
    """
    return Mock(spec=napari.Viewer)


@pytest.fixture
def main_widget(qtbot: QtBot) -> MainWidget:
    """
    Returns a MainWidget instance for testing.
    """
    settings: IUserSettings = FakeUserSettings()
    settings.set_cyto_dl_home_path(Path())
    settings.set_user_experiments_path(Path())
    return MainWidget(viewer=viewer, settings=settings)


def test_handle_action_change_view_event(
    main_widget: MainWidget,
) -> None:
    """
    Tests that the main widget handles the action change view event correctly.
    """
    # ARRANGE
    views: Set[AicsWidget] = main_widget._view_to_index.keys()

    for view in views:
        # ACT: have the model dispatch the action change view event
        main_widget._model.set_current_view(view)

        # ASSERT: check that the main widget's current view (after setting) is same as the model's current view
        assert (
            main_widget._view_container.currentIndex()
            == main_widget._view_to_index[view]
        )


def test_experiments_home_initialized(qtbot: QtBot) -> None:
    """
    Tests that the main widget promtps the user to choose a. 'experiments home' dir if one is not found in user settings.
    """
    # ARRANGE
    EXPECTED_EXPERIMENTS_HOME = PurePath(
        __file__
    ).parent  # simulates (in the fake settings) the location chosed by user
    settings = FakeUserSettings(EXPECTED_EXPERIMENTS_HOME)
    settings.set_cyto_dl_home_path(Path("foo/cyto/path"))
    settings.set_user_experiments_path(
        None
    )  # Simulates state where users has not yet chosen an experiments home.

    # ACT
    MainWidget(
        viewer, settings
    )  # If the users settings does not find an experiments home path, it will prompt the user for one and persist it.

    # ASSERT
    assert (
        settings.get_user_experiments_path() == EXPECTED_EXPERIMENTS_HOME
    )  # The path chosen by the user should have been persisted in settings.
