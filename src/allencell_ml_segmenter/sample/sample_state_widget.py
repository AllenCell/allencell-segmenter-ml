from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QLabel,
)

from allencell_ml_segmenter.sample.sample_model import SampleModel
from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.core.event import Event

TRAINING_NOT_RUNNING = "Training not running"
TRAINING_RUNNING = "Training running"


class SampleStateWidget(AicsWidget):
    """
    A sample widget for training a models.

    """

    def __init__(self, model: SampleModel):
        super().__init__()
        self._sample_model = model

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(TRAINING_NOT_RUNNING)
        self.layout().addWidget(self._label)

        self._sample_model.subscribe(
            Event.PROCESS_TRAINING, self, self.update_label_with_state
        )
        self._sample_model.subscribe(
            Event.PROCESS_TRAINING_SHOW_ERROR,
            self,
            self.update_label_with_error,
        )
        self._sample_model.subscribe(
            Event.PROCESS_TRAINING_CLEAR_ERROR,
            self,
            self.update_label_with_state,
        )

    def update_label_with_state(self, event):
        if self._sample_model.get_process_running():
            self._label.setText(TRAINING_RUNNING)
        else:
            self._label.setText(TRAINING_NOT_RUNNING)

    def update_label_with_error(self, event):
        self._label.setText(self._sample_model.get_error_message())

    def handle_event(self, event):
        pass
