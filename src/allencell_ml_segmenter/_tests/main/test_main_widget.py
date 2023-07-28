import pytest
import napari

from allencell_ml_segmenter.main.main_widget import MainTabWidget
from unittest.mock import Mock


@pytest.fixture
def viewer():
    return Mock(spec=napari.Viewer)


@pytest.fixture
def main_tab_widget(qtbot):
    return MainTabWidget(viewer)


def test_handle_action_change_view_event(main_tab_widget):
    # ACT: have the model dispatch the action change view event
    main_tab_widget._model.set_current_view(main_tab_widget._training_view)

    # ASSERT: check that the main widget's current view (after setting) is same as the model's current view
    assert (
        main_tab_widget.currentIndex()
        == main_tab_widget._view_to_index[main_tab_widget._training_view]
    )

    # ACT: have the model dispatch the action change view event, in case that the first act statement didn't do anything
    main_tab_widget._model.set_current_view(main_tab_widget._prediction_view)

    # ASSERT: check that the main widget's current view (after setting) is same as the model's current view
    assert (
        main_tab_widget.currentIndex()
        == main_tab_widget._view_to_index[main_tab_widget._prediction_view]
    )
