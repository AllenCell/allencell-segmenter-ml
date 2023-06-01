import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QStackedWidget,
)

from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.model.main_model import MainModel
from allencell_ml_segmenter.model.subscriber import Subscriber
from allencell_ml_segmenter.model.event import Event
from allencell_ml_segmenter.view.sample_view_controller import (
    SampleViewController,
)
from allencell_ml_segmenter.widgets.selection_widget import SelectionWidget


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
        # basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Model
        self.model: MainModel = MainModel()
        self.model.subscribe(self)

        self.view_to_index = dict()

        # add training page
        self.training_view = SampleViewController(self.model)
        self.initalize_view(self.training_view)

        # add main page
        self.selection_view = SelectionWidget(self.model, self.training_view)
        self.initalize_view(self.selection_view)

        self.viewer: napari.Viewer = viewer

        self.model.set_current_view(self.selection_view)

    def handle_event(self, event: Event) -> None:
        """
        Handle event function for the main widget, which handles MainEvents.

        inputs:
            event - MainEvent
        """
        self.set_view(self.model.get_current_view())

    def set_view(self, view: View) -> None:
        """
        Set the current view
        """
        self.setCurrentIndex(self.view_to_index[view])

    def initalize_view(self, view: View) -> None:
        # QStackedWidget count method keeps track of how many child widgets have been added
        self.view_to_index[view] = self.count()
        self.addWidget(view)



