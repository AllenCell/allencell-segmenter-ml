from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QLabel,
)

from allencell_ml_segmenter.models.main_model import MainModel
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event

TRAINING_NOT_RUNNING = "Training not running"
TRAINING_RUNNING = "Training running"


class SampleStateWidget(View, Subscriber):
    """
    A sample widget for training a models.

    """

    def __init__(self, model: MainModel):
        super().__init__()
        self.main_model = model
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(TRAINING_NOT_RUNNING)
        self.layout().addWidget(self.label)

        model.subscribe(Event.TRAINING, self, self.update_label)

    def update_label(self, event):
        if self.main_model.get_training_running():
            self.label.setText(TRAINING_RUNNING)
        else:
            self.label.setText(TRAINING_NOT_RUNNING)

    def handle_event(self, event):
        pass
