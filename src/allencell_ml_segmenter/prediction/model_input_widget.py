from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QApplication,
    QMainWindow,
    QSizePolicy,
    QLineEdit,
    QComboBox,
)
from qtpy.QtCore import Qt

from allencell_ml_segmenter.prediction.radio_button_list_widget import (
    RadioButtonList,
)

from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.input_button_widget import InputButton
from allencell_ml_segmenter.prediction.label_with_hint_widget import (
    LabelWithHint,
)


class ModelInputWidget(View, Subscriber):
    """
    Handles model input, preprocessing selection, and
    postprocessing selection for prediction.
    """

    def __init__(self):
        super().__init__()
        # self._model = model -> pass model in as a parameter later

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # title + hint at the top
        self.model_label_with_hint: LabelWithHint = LabelWithHint("Model")

        # horizontal layout containing widgets related to file selection
        selection_layout: QHBoxLayout = QHBoxLayout()
        selection_layout.setSpacing(0)
        self.selection_label_with_hint: LabelWithHint = LabelWithHint(
            "Select an existing model"
        )

        self.input_button: InputButton = InputButton()
        self.input_button.button.clicked.connect(self.get_file_name)

        selection_layout.addWidget(
            self.selection_label_with_hint, alignment=Qt.AlignLeft
        )
        selection_layout.addWidget(self.input_button, alignment=Qt.AlignLeft)

        # horizontal layout containing widgets related to preprocessing
        preprocessing_layout: QHBoxLayout = QHBoxLayout()
        preprocessing_layout.setSpacing(0)
        self.preprocessing_label_with_hint: LabelWithHint = LabelWithHint(
            "Preprocessing method"
        )

        # TODO: make this dynamic
        self.method: QLabel = QLabel("simple cutoff")
        self.method.setStyleSheet("margin-left: 25px")

        preprocessing_layout.addWidget(
            self.preprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        preprocessing_layout.addWidget(self.method, alignment=Qt.AlignLeft)

        # label and hint for postprocessing
        self.postprocessing_label_with_hint: LabelWithHint = LabelWithHint(
            "Postprocessing methods"
        )

        # horizontal layout with radio button list and various input boxes to its right
        bottom_layout: QHBoxLayout = QHBoxLayout()
        bottom_layout.setSpacing(0)

        self.radio_button_list: RadioButtonList = RadioButtonList(
            [
                "simple threshold cutoff",
                "auto threshold",
                "customized operations",
            ]
        )
        bottom_layout.addWidget(self.radio_button_list)

        bottom_layout.addStretch(4)

        bottom_right_layout: QVBoxLayout = QVBoxLayout()
        bottom_right_layout.setSpacing(0)

        self.top_input_box: QLineEdit = QLineEdit()
        self.top_input_box.setPlaceholderText("0.5")
        bottom_right_layout.addWidget(self.top_input_box)

        self.middle_input_box: QComboBox = QComboBox()
        self.middle_input_box.addItems(
            ["Select value", "Example 1", "Example 2"]
        )
        bottom_right_layout.addWidget(self.middle_input_box)

        self.bottom_input_box: QLineEdit = QLineEdit()
        self.bottom_input_box.setPlaceholderText("input value")
        bottom_right_layout.addWidget(self.bottom_input_box)

        bottom_layout.addLayout(bottom_right_layout)

        # add inner widgets and layouts to overarching layout
        self.layout().addWidget(
            self.model_label_with_hint, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(selection_layout)
        self.layout().addLayout(preprocessing_layout)
        self.layout().addWidget(
            self.postprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(bottom_layout)

    def handle_event(self, event: Event):
        pass

    def get_file_name(self):
        file_name = QFileDialog.getOpenFileName(self, "Open file")
        self.input_button.text_display.setReadOnly(False)
        self.input_button.text_display.setText(file_name[0])
        self.input_button.text_display.setReadOnly(True)


class MainWindow(QMainWindow):
    # remove once widget is completely figured out
    """For display/debugging purposes."""

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("WIP - Model Input Widget")

        widget: ModelInputWidget = ModelInputWidget()
        self.setCentralWidget(widget)


app: QApplication = QApplication([])

window: MainWindow = MainWindow()
window.show()

app.exec_()
