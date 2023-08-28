from allencell_ml_segmenter.training.training_model import TrainingModel
from qtpy.QtWidgets import (
    QWidget,
    QFrame,
    QGridLayout,
    QLineEdit,
    QVBoxLayout,
    QSizePolicy,
)

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint

class ExperimentInfoWidget(QWidget):
    """
    A widget for experiment information.
    """

    TITLE_TEXT: str = "Experiment information"

    def __init__(self, model: TrainingModel):
        super().__init__()

        self._model: TrainingModel = model

        # widget skeleton
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        frame: QFrame = QFrame()
        frame.setLayout(QGridLayout())
        frame.layout().setSpacing(0)
        frame.setObjectName("frame-no-border")
        self.layout().addWidget(frame)

        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        experiment_name_label: LabelWithHint = LabelWithHint("Experiment name")
        self._experiment_name_input: QLineEdit = QLineEdit()

        frame.layout().addWidget(experiment_name_label, 0, 0)
        frame.layout().addWidget(self._experiment_name_input, 0, 1)

        self._experiment_name_input.textChanged.connect(
            lambda text: self._model.set_experiment_name(text)
        )

    def set_enabled(self, enabled: bool) -> None:
        """
        Enables/disables the widget.

        enabled (bool): whether or not the widget should be enabled
        """
        self._experiment_name_input.setEnabled(enabled)

    def clear(self) -> None:
        """
        Clears the widget.
        """
        self._experiment_name_input.clear()
