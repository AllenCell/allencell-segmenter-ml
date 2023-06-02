from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.widgets.training_widget import TrainingWidget
from qtpy.QtWidgets import QVBoxLayout
from allencell_ml_segmenter.model.training_model import Event, TrainingModel
from allencell_ml_segmenter.model.publisher import Subscriber
from allencell_ml_segmenter.model.main_model import MainModel


class TrainingViewController(View, Subscriber):
    """
    ViewController for SampleWidget, responsible for handling events and updating the model + UI.
    """

    def __init__(self, main_model: MainModel):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # models
        self._main_model = main_model
        # self._model = TrainingModel()
        self._main_model.subscribe(self)

        # init widget and connect slots
        widget = TrainingWidget()
        widget.connectSlots([self.change_label, self.back_to_main])
        layout.addWidget(widget)

    def handle_event(self, event: Event) -> None:
        """
        Handles Events from the Training Model
        """
        print("recieved event in training view controller")
        if event == Event.TRAINING_SELECTED:
            self._main_model.set_current_view(self)

    def change_label(self) -> None:
        """
        Updates model in order to change label in UI
        """
        print("change label called")
        # self._model.set_model_training(not self._model.get_model_training())

    def back_to_main(self) -> None:
        """
        Updates model in order to change page back to main.
        """
        self._main_model.dispatch(Event.MAIN_SELECTED)
