from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.view.sample_widget import SampleWidget
from allencell_ml_segmenter.view._main_template import MainTemplate
from qtpy.QtWidgets import QVBoxLayout, QPushButton
from allencell_ml_segmenter.model.sample_model import Event, SampleModel
from allencell_ml_segmenter.model.pub_sub import Subscriber

class SampleViewController(View, Subscriber):

    def __init__(self, model: SampleModel):
        # hook up controller here when ready
        super().__init__(template_class=MainTemplate)
        self.widget = SampleWidget()
        self._model = model
        model.subscribe(self)

    @property
    def model(self):
        return self._model

    def handle_event(self, event: Event):
        if event == Event.TRAINING:
            self.widget.setLabelText(f"training is running {self._model.get_model_training()}")

    def change_label(self):
        self._model.set_model_training(not self._model.get_model_training())

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
        self.widget.connectSlots(self.change_label)