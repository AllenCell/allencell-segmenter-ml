from pathlib import Path
from typing import Set

import pytest
import napari
from pytestqt.qtbot import QtBot
from allencell_ml_segmenter.config.user_settings import CytoDlConfig

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
    return MainWidget(viewer, CytoDlConfig(Path(), Path()))


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
