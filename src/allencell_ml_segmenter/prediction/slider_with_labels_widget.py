from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QSizePolicy,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QLineEdit,
)


class SliderWithLabels(QWidget):
    """
    Compound widget: slider with lower and upper bounds clearly indicated
    and an adjacent textbox displaying the current value.
    """

    def __init__(self):
        super().__init__()

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        layout: QHBoxLayout = QHBoxLayout()
        self.setLayout(layout)

        self.left_layout: QVBoxLayout = QVBoxLayout()
        self.left_layout.setSpacing(0)
        self.upper_layout: QHBoxLayout = QHBoxLayout()

        self.lower_bound: QLabel = QLabel("0")
        self.upper_bound: QLabel = QLabel("1")

        self.slider: QSlider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)  # divide by 100 to get the actual value

        self.label: QLineEdit = QLineEdit()
        self.label.setPlaceholderText("0")
        self.label.setMaxLength(4)

        self.add_to_layouts()
        self.connect_to_handlers()

    def add_to_layouts(self) -> None:
        """
        Adds pertinent widgets and layouts to the overall layout.
        """
        self.upper_layout.addWidget(self.lower_bound, alignment=Qt.AlignLeft)
        self.upper_layout.addWidget(self.upper_bound, alignment=Qt.AlignRight)

        self.left_layout.addLayout(self.upper_layout)
        self.left_layout.addWidget(self.slider)

        self.layout().addLayout(self.left_layout)
        self.layout().addWidget(self.label)

    def connect_to_handlers(self):
        # TODO: instead of having the slider respond to the label and vice versa,
        #  have both set the field in the model, which dispatches a "slider change" event
        #  that both listen and respond to when appropriate
        pass
