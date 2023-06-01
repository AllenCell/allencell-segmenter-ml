from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from allencell_ml_segmenter.model.main_model import MainModel, Page


class SelectionWidget(QWidget):
    """
    A sample widget with two buttons for selecting between training and prediction views.
    """

    def __init__(self, model: MainModel):
        super().__init__()
        # self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Controller
        self.training_button = QPushButton("Training View")
        self.training_button.clicked.connect(
            lambda: self.model.set_current_page(Page.TRAINING)
        )
        self.prediction_button = QPushButton("Prediction View")
        self.prediction_button.clicked.connect(
            lambda: self.model.set_current_page(Page.PREDICTION)
        )

        # add buttons
        self.layout().addWidget(self.training_button)
        self.layout().addWidget(self.prediction_button)

        # model
        self.model = model
