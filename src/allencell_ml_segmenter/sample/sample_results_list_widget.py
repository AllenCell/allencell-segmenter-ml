from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.sample.sample_model import SampleModel
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.core.event import Event
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QLabel,
)

class SampleResultsListWidget(View, Subscriber):
    """
    """

    def __init__(self, model: SampleModel) -> None:
        super().__init__()
        self._model = model

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        model.subscribe(event=Event.PROCESS_TRAINING_PROGRESS, subscriber=self, handler=self.update)

    def update(self, event: Event):
        for i in reversed(range(self.layout().count())): 
            self.layout().itemAt(i).widget().setParent(None)
        for file in self._model.get_training_output_files():
            self.layout().addWidget(QLabel(file))

    def handle_event(self, event: Event) -> None:
        pass

    