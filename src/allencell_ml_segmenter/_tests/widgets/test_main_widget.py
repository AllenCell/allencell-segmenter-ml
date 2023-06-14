import pytest
import napari
from qtpy.QtWidgets import QStackedWidget
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.sample.sample_view import SampleView
from allencell_ml_segmenter.main.main_widget import MainWidget
from unittest.mock import Mock


@pytest.fixture
def viewer():
    return Mock(spec=napari.Viewer)


def test_init(viewer, qtbot):
    main_widget = MainWidget(viewer)
    assert isinstance(main_widget, QStackedWidget)
    assert isinstance(main_widget.model, MainModel)
    assert len(main_widget.view_to_index) > 0  # need at least one views loaded


def test_handle_event(viewer, qtbot):
    main_widget = MainWidget(viewer)
    sample_view = SampleView(main_widget.model)
    main_widget.initialize_view(sample_view)
    assert (
        main_widget.currentIndex() != main_widget.view_to_index[sample_view]
    )

    main_widget.model.set_current_view(sample_view)

    assert (
        main_widget.currentIndex() == main_widget.view_to_index[sample_view]
    )


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
