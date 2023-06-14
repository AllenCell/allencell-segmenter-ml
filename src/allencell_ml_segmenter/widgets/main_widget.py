import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QStackedWidget,
)

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.models.main_model import MainModel
from allencell_ml_segmenter.models.example_model import ExampleModel
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.views.training_view import TrainingView
from allencell_ml_segmenter.widgets.selection_widget import SelectionWidget
from allencell_ml_segmenter.views.example_view import ExampleView
from allencell_ml_segmenter.services.example_service import ExampleService


class MainMeta(type(QStackedWidget), type(Subscriber)):
    """
    Metaclass for MainWidget

    """

    pass


class MainWidget(QStackedWidget, Subscriber, metaclass=MainMeta):
    """
    Main widget that is displayed in the plugin window. This widget is an empty Qwidget that supports adding and removing different
    views from the main layout by responding to MainEvents.

    """

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer: napari.Viewer = viewer

        # basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Model
        self.model: MainModel = MainModel()
        self.model.subscribe(Event.CHANGE_VIEW, self)

        # Dictionaries of views to index values
        self.view_to_index = dict()

        # add training page
        training_view = TrainingView(self.model)
        self.initialize_view(training_view)

        # add example page
        example_view = ExampleView(self.model)
        self.initialize_view(example_view)

        # add main page
        selection_view = SelectionWidget(self.model)
        self.initialize_view(selection_view)

        # start on selection views
        self.model.set_current_view(selection_view)

    def handle_event(self, event: Event) -> None:
        """
        Handle event function for the main widget, which handles MainEvents.

        inputs:
            event - MainEvent
        """
        self.set_view(self.model.get_current_view())

    def set_view(self, view: View) -> None:
        """
        Set the current views, must be initialized first
        """
        self.setCurrentIndex(self.view_to_index[view])

    def initialize_view(self, view: View) -> None:
        # QStackedWidget count method keeps track of how many child widgets have been added
        self.view_to_index[view] = self.count()
        self.addWidget(view)
