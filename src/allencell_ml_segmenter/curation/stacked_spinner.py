from allencell_ml_segmenter.widgets.input_button_widget import (
    InputButton,
)
from qtpy.QtWidgets import QStackedWidget
from pyqtspinner import WaitingSpinner
from PyQt5.QtGui import QColor


class StackedSpinner(QStackedWidget):
    def __init__(self, input_button: InputButton = None):
        super().__init__()
        self.setMaximumSize(260, 50)
        self.spinner = WaitingSpinner(
            parent=self,
            center_on_parent=True,
            disable_parent_when_spinning=True,
            color=QColor(244, 244, 244),
        )
        self.input_button = input_button
        self.addWidget(self.spinner)
        self.addWidget(self.input_button)
        self.setCurrentWidget(self.input_button)

    def start(self):
        print()
        self.setCurrentWidget(self.spinner)
        self.spinner.start()

    def stop(self):
        print()
        self.setCurrentWidget(self.input_button)
        self.spinner.stop()
