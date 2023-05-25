from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.widgets.sample_widget import SampleWidget
from allencell_ml_segmenter.view._main_template import MainTemplate
from qtpy.QtWidgets import QVBoxLayout
from allencell_ml_segmenter.model.training_model import Event, TrainingModel
from allencell_ml_segmenter.model.publisher import Subscriber
from allencell_ml_segmenter.model.main_model import MainModel
from allencell_ml_segmenter.model.main_model import Page


class SampleViewController(View, Subscriber):
    def __init__(self, main_model: MainModel):
        # hook up controller here when ready
        super().__init__(template_class=MainTemplate)
        self._main_model = main_model
        self.widget = SampleWidget()
        self._model = TrainingModel()
        self.model.subscribe(self)
        self.load()

    @property
    def model(self):
        return self._model

    def handle_event(self, event: Event):
        print("sampleviewcontroller handle event called")

        if event == event.TRAINING:
            self.widget.setLabelText("training")

    def change_label(self):
        print("change label called")
        self._model.set_model_training(not self._model.get_model_training())

    def back_to_main(self):
        print("back to main called")
        self._main_model.set_current_page(Page.MAIN)

    ###################################################
    #                 Setup Section                   #
    ###################################################
    def load(self):
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self.widget)
        self.widget.connectSlots([self.change_label, self.back_to_main])
