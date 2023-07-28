import pytest
import napari
from PyQt5.QtWidgets import QTabWidget
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.view import PredictionView
from allencell_ml_segmenter.sample.sample_view import SampleView
from allencell_ml_segmenter.main.main_widget import MainTabWidget
from unittest.mock import Mock


@pytest.fixture
def viewer():
    return Mock(spec=napari.Viewer)


def test_manual_tab_switching(viewer: napari.Viewer, qtbot: QtBot) -> None:
    # ARRANGE
    main_widget: MainTabWidget = MainTabWidget(viewer)

    # ASSERT (default view is prediction)
    assert isinstance(main_widget.currentWidget(), PredictionView)

    # ACT
    main_widget.setCurrentIndex(1)

    # ASSERT
    assert main_widget.currentIndex() == 1
    assert isinstance(main_widget.currentWidget(), SampleView)

    # ACT
    main_widget.setCurrentIndex(0)

    # ASSERT
    assert main_widget.currentIndex() == 0
    assert isinstance(main_widget.currentWidget(), PredictionView)


def test_init(viewer: napari.Viewer, qtbot: QtBot) -> None:
    main_widget: MainTabWidget = MainTabWidget(viewer)
    assert isinstance(main_widget, QTabWidget)
    assert isinstance(main_widget.model, MainModel)
    assert len(main_widget.view_to_index) > 0  # need at least one view loaded


def test_main_widget_set_view(viewer: napari.Viewer, qtbot: QtBot):
    main_widget: MainTabWidget = MainTabWidget(viewer)
    view: View = View()
    main_widget.initialize_view(view, "Example")

    main_widget.set_view(view)

    assert main_widget.currentIndex() == main_widget.view_to_index[view]


def test_main_widget_initialize_view(viewer: napari.Viewer, qtbot: QtBot):
    main_widget: MainTabWidget = MainTabWidget(viewer)
    view: View = View()

    main_widget.initialize_view(view, "Example")

    # index value assigned before widget is added
    widget_count = main_widget.count()
    assert main_widget.view_to_index[view] == widget_count - 1
    assert main_widget.widget(widget_count - 1) == view
