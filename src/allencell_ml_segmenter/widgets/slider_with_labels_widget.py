from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QWidget,
    QSizePolicy,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QLineEdit,
)

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.model import PredictionModel


class SliderWithLabels(QWidget):
    """
    Compound widget: slider with lower and upper bounds clearly indicated
    and an adjacent textbox displaying the current value.
    """

    def __init__(
        self, lower_bound: int, upper_bound: int, model: PredictionModel
    ):
        super().__init__()

        self._model: PredictionModel = model

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setLayout(QHBoxLayout())

        self._left_layout: QVBoxLayout = QVBoxLayout()
        self._left_layout.setSpacing(0)
        self._upper_layout: QHBoxLayout = QHBoxLayout()

        self._low: QLabel = QLabel(str(lower_bound))
        self._high: QLabel = QLabel(str(upper_bound))

        self._slider: QSlider = QSlider(Qt.Horizontal)
        self._slider.setRange(
            lower_bound, round(100 * upper_bound)
        )  # divide by 100 to get the actual value

        self._label: QLineEdit = QLineEdit()
        self._label.setPlaceholderText(
            f"{lower_bound}.xx - {upper_bound}"
        )  # does this placeholder text set off any weird event that breaks stuff

        # TODO: when they hit "run", check to see that this input is valid and halt running if not
        validator: QDoubleValidator = QDoubleValidator()
        validator.setRange(lower_bound, upper_bound, 2)
        self._label.setValidator(validator)
        self._label.setMaxLength(4)

        self._add_to_layouts()
        self._connect_to_handlers()

        self._model.subscribe(
            Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_MOVED,
            self,
            lambda e: self._update_simple_threshold_label(),
        )

        self._model.subscribe(
            Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_TYPED,
            self,
            lambda e: self._update_simple_threshold_slider(),
        )

    def _add_to_layouts(self) -> None:
        """
        Adds pertinent widgets and layouts to the overall layout.
        """
        self._upper_layout.addWidget(self._low, alignment=Qt.AlignLeft)
        self._upper_layout.addWidget(self._high, alignment=Qt.AlignRight)

        self._left_layout.addLayout(self._upper_layout)
        self._left_layout.addWidget(self._slider)

        self.layout().addLayout(self._left_layout)
        self.layout().addWidget(self._label)

    def _label_slot(self, s: str) -> None:
        """
        Updates slider in response to label.
        """
        if not s:
            num: float = 0
        else:
            num: float = float(s)

        self._model.set_postprocessing_simple_threshold(num)
        self._model.dispatch(
            Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_TYPED
        )

    def _slider_slot(self, v: int) -> None:
        """
        Updates label in response to slider.
        """
        self._model.set_postprocessing_simple_threshold(v / 100)
        self._model.dispatch(
            Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_MOVED
        )

    def _connect_to_handlers(self) -> None:
        """
        Connects slider and label to their respective slots.
        """
        self._label.textChanged.connect(self._label_slot)
        self._slider.valueChanged.connect(self._slider_slot)

    def set_slider_value(self, v: float) -> None:
        """
        Adjusts on-screen slider positioning in relation to stored state.
        """
        self._slider.setValue(round(v * 100))

    def set_label_value(self, v: float) -> None:
        """
        Adjusts on-screen label in relation to stored state.
        """
        self._label.setText(str(v))

    def _update_simple_threshold_label(self) -> None:
        threshold: float = self._model.get_postprocessing_simple_threshold()
        self.set_label_value(threshold)

    def _update_simple_threshold_slider(self) -> None:
        threshold: float = self._model.get_postprocessing_simple_threshold()
        self.set_slider_value(threshold)
