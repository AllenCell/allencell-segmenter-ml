import napari
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QSizePolicy, QStackedWidget
from allencell_ml_segmenter.model.main_model import MainModel, Page
from allencell_ml_segmenter.model.subscriber import Subscriber
from allencell_ml_segmenter.model.event import MainEvent
from allencell_ml_segmenter.view.sample_view_controller import (
    SampleViewController,
)
from allencell_ml_segmenter.widgets.selection_widget import SelectionWidget

class MainMeta(type(QWidget), type(Subscriber)):
    """
    Metaclass for MainWidget

    """

    pass


class MainWidget(QWidget, Subscriber, metaclass=MainMeta):
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
        self.mainmodel: MainModel = MainModel()
        self.mainmodel.subscribe(self)

        # Widget stack
        self.stacked_widget = QStackedWidget()
        # add main page
        self.selection_widget = SelectionWidget(self.mainmodel)
        self.stacked_widget.addWidget(self.selection_widget)
        # add training page
        self.training_view_controller = SampleViewController(self.mainmodel)
        self.stacked_widget.addWidget(self.training_view_controller)

        self.layout().addWidget(self.stacked_widget)
        self.open_main_view()

        self.viewer: napari.Viewer = viewer

        self.stacked_widget.setCurrentIndex(1)

    def handle_event(self, event: MainEvent) -> None:
        """
        Handle event function for the main widget, which handles MainEvents.

        inputs:
            event - MainEvent
        """
        print("main handle event called")
        # remove a view if already being displayed
        if event == MainEvent.MAIN:
            self.stacked_widget.setCurrentIndex(0)
        elif event == MainEvent.TRAINING:
            self.stacked_widget.setCurrentIndex(1)

    def open_training_view(self) -> None:
        """
        Open the training view
        """
        self.mainmodel.set_current_page(Page.TRAINING)

    def open_main_view(self) -> None:
        """
        Open the main view
        """
        self.mainmodel.set_current_page(Page.MAIN)
