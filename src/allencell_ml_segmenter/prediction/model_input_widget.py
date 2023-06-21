from qtpy.QtWidgets import (
    QWidget,
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
from qtpy.QtGui import QPalette, QColor, QPixmap

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
    # TODO: figure out metaclass so a widget does not have to be a view
    """
    Handles model input, preprocessing selection, and
    postprocessing selection for prediction.
    """

    def __init__(self):
        super().__init__()
        # self._model = model -> pass model in as a parameter later

        # TODO: make sure this is the desired size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # self.setStyleSheet("margin: 5px")

        # horizontal layout with title + question button
        title_layout: QHBoxLayout = QHBoxLayout()
        title_layout.setSpacing(0)
        self.model_label_with_hint: LabelWithHint = LabelWithHint("Model")
        title_layout.addWidget(
            self.model_label_with_hint, alignment=Qt.AlignLeft
        )

        # horizontal layout with select label, question button, & file selector
        selection_layout: QHBoxLayout = QHBoxLayout()
        selection_layout.setSpacing(0)
        self.selection_label_with_hint: LabelWithHint = LabelWithHint(
            "Select an existing model"
        )
        selection_layout.addWidget(
            self.selection_label_with_hint, alignment=Qt.AlignLeft
        )
        self.input_button: InputButton = InputButton()
        selection_layout.addWidget(self.input_button, alignment=Qt.AlignLeft)

        # horizontal layout with preprocessing label, question button, & label
        preprocessing_layout: QHBoxLayout = QHBoxLayout()
        preprocessing_layout.setSpacing(0)
        self.preprocessing_label_with_hint: LabelWithHint = LabelWithHint(
            "Preprocessing method"
        )
        preprocessing_layout.addWidget(
            self.preprocessing_label_with_hint, alignment=Qt.AlignLeft
        )

        # TODO: make this dynamic
        self.method: QLabel = QLabel("simple cutoff")
        self.method.setStyleSheet("margin-left: 25px")
        preprocessing_layout.addWidget(self.method, alignment=Qt.AlignLeft)

        # horizontal layout with postprocessing label & question button
        postprocessing_layout: QHBoxLayout = QHBoxLayout()
        postprocessing_layout.setSpacing(0)
        self.postprocessing_label_with_hint: LabelWithHint = LabelWithHint(
            "Postprocessing methods"
        )
        postprocessing_layout.addWidget(
            self.postprocessing_label_with_hint, alignment=Qt.AlignLeft
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

        # add layouts to overarching layout
        self.layout().addLayout(title_layout)
        self.layout().addLayout(selection_layout)
        self.layout().addLayout(preprocessing_layout)
        self.layout().addLayout(postprocessing_layout)
        self.layout().addLayout(bottom_layout)

    def handle_event(self, event: Event):
        pass


class MainWindow(QMainWindow):
    # remove once widget is completely figured out
    """For display/debugging purposes."""

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("WIP - Model Input Widget")

        widget: ModelInputWidget = ModelInputWidget()
        self.setCentralWidget(widget)


app: QApplication = QApplication([])

window: MainWindow = MainWindow()
window.show()

app.exec_()
