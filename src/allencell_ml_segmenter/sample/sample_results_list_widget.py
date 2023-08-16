from allencell_ml_segmenter.sample.sample_model import SampleModel
from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.core.event import Event
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QLabel,
    QPushButton,
)


class SampleResultsListWidget(AicsWidget):
    """ """

    def __init__(self, model: SampleModel) -> None:
        super().__init__()
        self._model = model

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear_training_output_files)

        model.subscribe(
            event=Event.PROCESS_TRAINING_PROGRESS,
            subscriber=self,
            handler=self.update,
        )

    def update(self, event: Event):
        self.clear_training_output_files()
        for file in self._model.get_training_output_files():
            self.layout().addWidget(QLabel(file))
        self.layout().addWidget(QLabel("Training complete!"))
        self.layout().addWidget(self._clear_btn)

    def clear_training_output_files(self):
        for i in reversed(range(self.layout().count())):
            self.layout().itemAt(i).widget().setParent(None)

    def handle_event(self, event: Event) -> None:
        pass
