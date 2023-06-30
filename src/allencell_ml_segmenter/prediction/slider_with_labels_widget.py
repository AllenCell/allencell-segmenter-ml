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

from allencell_ml_segmenter.core.publisher import Publisher


class SliderWithLabels(QWidget):
    """
    Compound widget: slider with lower and upper bounds clearly indicated
    and an adjacent textbox displaying the current value.
    """

    def __init__(self, lower_bound: int, upper_bound: int, model: Publisher):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self._model: Publisher = model

        layout: QHBoxLayout = QHBoxLayout()
        self.setLayout(layout)

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

    def _connect_to_handlers(self):
        # TODO: instead of having the slider respond to the label and vice versa,
        #  have both set the field in the model, which dispatches a "slider change" event
        #  that both listen and respond to when appropriate
        pass
