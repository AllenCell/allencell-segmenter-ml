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
    QGridLayout,
    QRadioButton,
)
from qtpy.QtCore import Qt

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

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # title + hint at the top
        self.model_label_with_hint: LabelWithHint = LabelWithHint()
        self.model_label_with_hint.set_label_text("Model")
        self.model_label_with_hint.set_hint("this is a test")

        # horizontal layout containing widgets related to file selection
        selection_layout: QHBoxLayout = QHBoxLayout()
        selection_layout.setSpacing(0)

        self.selection_label_with_hint: LabelWithHint = LabelWithHint()
        self.selection_label_with_hint.set_label_text(
            "Select an existing model"
        )
        self.selection_label_with_hint.set_hint("this is another test")

        self.input_button: InputButton = InputButton()
        self.input_button.button.clicked.connect(self.get_file_name)

        selection_layout.addWidget(
            self.selection_label_with_hint, alignment=Qt.AlignLeft
        )
        selection_layout.addWidget(self.input_button, alignment=Qt.AlignLeft)

        # horizontal layout containing widgets related to preprocessing
        preprocessing_layout: QHBoxLayout = QHBoxLayout()
        preprocessing_layout.setSpacing(0)

        self.preprocessing_label_with_hint: LabelWithHint = LabelWithHint()
        self.preprocessing_label_with_hint.set_label_text(
            "Preprocessing method"
        )
        self.preprocessing_label_with_hint.set_hint(
            "this is the penultimate test"
        )

        # TODO: make this dynamic
        self.method: QLabel = QLabel("simple cutoff")
        self.method.setStyleSheet("margin-left: 25px")

        preprocessing_layout.addWidget(
            self.preprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        preprocessing_layout.addWidget(self.method, alignment=Qt.AlignLeft)

        # label and hint for postprocessing
        self.postprocessing_label_with_hint: LabelWithHint = LabelWithHint()
        self.postprocessing_label_with_hint.set_label_text(
            "Postprocessing methods"
        )
        self.postprocessing_label_with_hint.set_hint("this is the final test")

        # grid layout containing widgets related to postprocessing
        grid_layout: QGridLayout = QGridLayout()
        grid_layout.setSpacing(0)

        # initialize and add radio buttons to grid
        self.top_button: QRadioButton = QRadioButton()
        self.mid_button: QRadioButton = QRadioButton()
        self.bottom_button: QRadioButton = QRadioButton()

        for idx, button in enumerate(
            [self.top_button, self.mid_button, self.bottom_button]
        ):
            button.setStyleSheet("margin-left: 25px; margin-right: 6 px")
            grid_layout.addWidget(button, idx, 0)

        # initialize and add radio button labels to grid
        top_label: QLabel = QLabel("simple threshold cutoff")
        mid_label: QLabel = QLabel("auto threshold")
        bottom_label: QLabel = QLabel("customized operations")

        for idx, label in enumerate([top_label, mid_label, bottom_label]):
            label.setStyleSheet("margin-right: 25px")
            grid_layout.addWidget(label, idx, 1)

        # initialize and add input fields to grid
        self.top_input_box: QLineEdit = QLineEdit()
        self.top_input_box.setPlaceholderText("0.5")

        self.mid_input_box: QComboBox = QComboBox()
        self.mid_input_box.addItems(["Select value", "Example 1", "Example 2"])

        self.bottom_input_box: QLineEdit = QLineEdit()
        self.bottom_input_box.setPlaceholderText("input value")

        for idx, box in enumerate(
            [self.top_input_box, self.mid_input_box, self.bottom_input_box]
        ):
            # prohibit edits until the appropriate radio button is checked
            if isinstance(box, QLineEdit):
                box.setReadOnly(True)
            else:  # only other possibility is a combo box
                box.setEditable(False)
            grid_layout.addWidget(box, idx, 2)

        # connect radio buttons to slots
        self.top_button.toggled.connect(self.top_radio_button_slot)
        self.mid_button.toggled.connect(self.mid_radio_button_slot)
        self.bottom_button.toggled.connect(self.bottom_radio_button_slot)

        # add inner widgets and layouts to overarching layout
        self.layout().addWidget(
            self.model_label_with_hint, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(selection_layout)
        self.layout().addLayout(preprocessing_layout)
        self.layout().addWidget(
            self.postprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(grid_layout)

    def handle_event(self, event: Event) -> None:
        pass

    def get_file_name(self) -> None:
        file_name = QFileDialog.getOpenFileName(self, "Open file")
        self.input_button.text_display.setReadOnly(False)
        self.input_button.text_display.setText(file_name[0])
        self.input_button.text_display.setReadOnly(True)

    def top_radio_button_slot(self) -> None:
        # TODO: gray out styling
        # make only the top input field editable
        if self.top_button.isChecked():
            self.top_input_box.setReadOnly(False)
            self.mid_input_box.setEditable(False)
            self.bottom_input_box.setReadOnly(True)
        else:
            self.top_input_box.setReadOnly(True)

    def mid_radio_button_slot(self) -> None:
        # make only middle input field editable
        if self.mid_button.isChecked():
            self.top_input_box.setReadOnly(True)
            self.mid_input_box.setEditable(True)
            self.bottom_input_box.setReadOnly(True)
        else:
            self.mid_input_box.setEditable(False)

    def bottom_radio_button_slot(self) -> None:
        # make only bottom input field editable
        if self.bottom_button.isChecked():
            self.top_input_box.setReadOnly(True)
            self.mid_input_box.setEditable(False)
            self.bottom_input_box.setReadOnly(False)
        else:
            self.bottom_input_box.setReadOnly(True)


class MainWindow(QMainWindow):
    # TODO: remove once widget is completely figured out
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
