from allencell_ml_segmenter.view.view import View
from allencell_ml_segmenter.widgets.training_widget import TrainingWidget
from qtpy.QtWidgets import QVBoxLayout
from allencell_ml_segmenter.models.training_model import Event
from allencell_ml_segmenter.core.publisher import Subscriber
from allencell_ml_segmenter.models.main_model import MainModel


class TrainingView(View, Subscriber):
    """
    View that is a subscriber for TrainingWidget, responsible for handling events and updating the models + UI.
    """

    def __init__(self, main_model: MainModel):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # models
        self._main_model = main_model
        self._main_model.subscribe(self)

        # init widget and connect slots
        widget = TrainingWidget()
        widget.connectSlots([self.back_to_main, self.back_to_main])
        layout.addWidget(widget)

    def handle_event(self, event: Event) -> None:
        """
        Handles Events from the Training Model
        """
        if event == Event.TRAINING_SELECTED:
            self._main_model.set_current_view(self)

    def back_to_main(self) -> None:
        """
        Updates models in order to change page back to main.
        """
        self._main_model.dispatch(Event.MAIN_SELECTED)
