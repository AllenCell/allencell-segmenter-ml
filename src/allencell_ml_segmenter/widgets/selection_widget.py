from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.models.main_model import MainModel


class SelectionWidget(QWidget):
    """
    A sample widget with two buttons for selecting between training and example views.
    """

    def __init__(self, model: MainModel):
        super().__init__()
        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Controller
        self.training_button = QPushButton("Training View")
        self.training_button.clicked.connect(
            lambda: self.model.dispatch(Event.TRAINING_SELECTED)
        )
        self.example_button = QPushButton("Example View")
        self.example_button.clicked.connect(
            lambda: self.model.dispatch(Event.EXAMPLE_SELECTED)
        )

        # add buttons
        self.layout().addWidget(self.training_button)
        self.layout().addWidget(self.example_button)

        # models
        self.model = model
        self.model.subscribe(self)

    def handle_event(self, event: Event) -> None:
        if event == Event.MAIN_SELECTED:
            self.model.set_current_view(self)
