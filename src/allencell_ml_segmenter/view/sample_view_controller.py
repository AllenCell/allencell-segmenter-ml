from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.widgets.sample_widget import SampleWidget
from qtpy.QtWidgets import QVBoxLayout
from allencell_ml_segmenter.model.training_model import Event, TrainingModel
from allencell_ml_segmenter.model.publisher import Subscriber
from allencell_ml_segmenter.model.main_model import MainModel
from allencell_ml_segmenter.model.main_model import Page


class SampleViewController(View, Subscriber):
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
        self._model = TrainingModel()
        self.model.subscribe(self)

        # init widget and connect slots
        widget = SampleWidget()
        widget.connectSlots([self.change_label, self.back_to_main])
        layout.addWidget(widget)

    @property
    def model(self) -> TrainingModel:
        """
        Model property
        """
        return self._model

    def handle_event(self, event: Event) -> None:
        """
        Handles Events from the Training Model
        """
        print("sampleviewcontroller handle event called")

    def change_label(self) -> None:
        """
        Updates model in order to change label in UI
        """
        print("change label called")
        self._model.set_model_training(not self._model.get_model_training())

    def back_to_main(self) -> None:
        """
        Updates model in order to change page back to main.
        """
        print("back to main called")
        self._main_model.set_current_page(Page.MAIN)
