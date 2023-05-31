import napari
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QSizePolicy
from allencell_ml_segmenter.model.main_model import MainModel, Page
from allencell_ml_segmenter.model.subscriber import Subscriber
from allencell_ml_segmenter.model.event import MainEvent
from allencell_ml_segmenter.view.sample_view_controller import (
    SampleViewController,
)
from allencell_ml_segmenter.core.view_manager import ViewManager


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

        # Buttons
        self.training_button = QPushButton("Training View")
        self.training_button.clicked.connect(self.open_training_view)
        self.prediction_button = QPushButton("Prediction View")
        self.active_view = None

        # Model
        self.mainmodel: MainModel = MainModel()
        self.mainmodel.subscribe(self)

        # Controller and view manager
        self.training_view_controller = SampleViewController(self.mainmodel)
        self.view_manager = ViewManager(self.layout())
        self.open_main_view()
        self.viewer: napari.Viewer = viewer

    def handle_event(self, event: MainEvent) -> None:
        """
        Handle event function for the main widget, which handles MainEvents.

        inputs:
            event - MaintEvent
        """
        print("main handle event called")
        # remove a view if already being displayed
        if self.active_view is not None:
            self.layout().removeWidget(self.active_view)

        if event == MainEvent.MAIN:
            # add buttons
            self.layout().addWidget(self.training_button)
            self.layout().addWidget(self.prediction_button)
        elif event == MainEvent.TRAINING:
            self.layout().removeWidget(self.training_button)
            self.layout().removeWidget(self.prediction_button)

            self.active_view = self.training_view_controller
            self.layout().addWidget(self.active_view)

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
