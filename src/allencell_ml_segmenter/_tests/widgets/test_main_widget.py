import pytest
import napari
from qtpy.QtWidgets import QStackedWidget
from allencell_ml_segmenter.view.view import View
from allencell_ml_segmenter.model.main_model import MainModel
from allencell_ml_segmenter.view.training_view import TrainingView
from allencell_ml_segmenter.widgets.main_widget import MainWidget
from unittest.mock import Mock


@pytest.fixture
def viewer():
    return Mock(spec=napari.Viewer)


def test_init(viewer, qtbot):
    main_widget = MainWidget(viewer)
    assert isinstance(main_widget, QStackedWidget)
    assert isinstance(main_widget.model, MainModel)
    assert len(main_widget.view_to_index) > 0 # need at least one view loaded

def test_handle_event(viewer, qtbot):
    main_widget = MainWidget(viewer)
    training_view = TrainingView(main_widget.model)
    main_widget.initialize_view(training_view)
    assert main_widget.currentIndex() != main_widget.view_to_index[training_view]

    main_widget.model.set_current_view(training_view)

    assert main_widget.currentIndex() == main_widget.view_to_index[training_view]


def test_main_widget_set_view(viewer, qtbot):
    main_widget = MainWidget(viewer)
    view = View()
    main_widget.initialize_view(view)

    main_widget.set_view(view)

    assert main_widget.currentIndex() == main_widget.view_to_index[view]


def test_main_widget_initialize_view(viewer, qtbot):
    main_widget = MainWidget(viewer)
    view = View()

    main_widget.initialize_view(view)

    # index value assigned before widget is added
    widget_count = main_widget.count()
    assert main_widget.view_to_index[view] == widget_count - 1
    assert main_widget.widget(widget_count - 1) == view