from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QLineEdit,
    QComboBox,
    QGridLayout,
    QRadioButton,
)
from qtpy.QtCore import Qt

from allencell_ml_segmenter.prediction.model import PredictionModel
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

    def __init__(self, model: PredictionModel):
        super().__init__()

        self.model = model

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # instantiate widgets
        self.model_label_with_hint: LabelWithHint = LabelWithHint()
        self.selection_label_with_hint: LabelWithHint = LabelWithHint()
        self.input_button: InputButton = InputButton()
        self.preprocessing_label_with_hint: LabelWithHint = LabelWithHint()
        self.method: QLabel = QLabel("simple cutoff")
        self.postprocessing_label_with_hint: LabelWithHint = LabelWithHint()

        # radio buttons
        self.top_button: QRadioButton = QRadioButton()
        self.mid_button: QRadioButton = QRadioButton()
        self.bottom_button: QRadioButton = QRadioButton()

        self.buttons = [self.top_button, self.mid_button, self.bottom_button]

        # labels for the radio buttons
        top_label: QLabel = QLabel("simple threshold cutoff")
        mid_label: QLabel = QLabel("auto threshold")
        bottom_label: QLabel = QLabel("customized operations")

        self.labels = [top_label, mid_label, bottom_label]

        # input fields corresponding to radio buttons & their labels
        self.top_input_box: QLineEdit = QLineEdit()
        self.mid_input_box: QComboBox = QComboBox()
        self.bottom_input_box: QLineEdit = QLineEdit()

        self.boxes = [
            self.top_input_box,
            self.mid_input_box,
            self.bottom_input_box,
        ]

        self.model.subscribe(
            Event.ACTION_PREDICTION_PREPROCESSING_METHOD_SELECTED,
            self,
            lambda e: self.method.setText(self.model.get_preprocessing_method()),
        )

        # finish default set-up
        self.call_setters()
        self.build_layouts()
        self.configure_slots()

    def handle_event(self, event: Event) -> None:
        pass

    def get_file_name(self) -> None:
        """
        Displays file path on label portion of input button.
        """
        file_path = QFileDialog.getOpenFileName(self, "Open file")[0]
        self.input_button.text_display.setReadOnly(False)
        self.input_button.text_display.setText(file_path)
        self.input_button.text_display.setReadOnly(True)

        self.model.set_file_path(file_path)

    def top_radio_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if top button is checked.
        """
        if self.top_button.isChecked():
            self.top_input_box.setEnabled(True)
            self.mid_input_box.setEnabled(False)
            self.bottom_input_box.setEnabled(False)
        else:
            self.top_input_box.setEnabled(False)

    def mid_radio_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if middle button is checked.
        """
        if self.mid_button.isChecked():
            self.top_input_box.setEnabled(False)
            self.mid_input_box.setEnabled(True)
            self.bottom_input_box.setEnabled(False)
        else:
            self.mid_input_box.setEnabled(False)

    def bottom_radio_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if bottom button is checked.
        """
        if self.bottom_button.isChecked():
            self.top_input_box.setEnabled(False)
            self.mid_input_box.setEnabled(False)
            self.bottom_input_box.setEnabled(True)
        else:
            self.bottom_input_box.setEnabled(False)

    def call_setters(self) -> None:
        """
        Sets pertinent default values for all widget fields.
        """
        # title + hint
        self.model_label_with_hint.set_label_text("Model")
        self.model_label_with_hint.set_hint("this is a test")

        # selection label + hint
        self.selection_label_with_hint.set_label_text("Select an existing model")
        self.selection_label_with_hint.set_hint("this is another test")

        # preprocessing label + hint
        self.preprocessing_label_with_hint.set_label_text("Preprocessing method")
        self.preprocessing_label_with_hint.set_hint("this is the penultimate test")

        # styling for label for preprocessing method
        self.method.setStyleSheet("margin-left: 25px")

        # postprocessing label + hint
        self.postprocessing_label_with_hint.set_label_text("Postprocessing methods")
        self.postprocessing_label_with_hint.set_hint("this is the final test")

        # add styling to buttons and labels
        for button in self.buttons:
            button.setStyleSheet("margin-left: 25px; margin-right: 6 px")
        for label in self.labels:
            label.setStyleSheet("margin-right: 25px")

        # set default values for input fields
        self.top_input_box.setPlaceholderText("0.5")
        self.mid_input_box.addItems(["Select value", "Example 1", "Example 2"])
        self.bottom_input_box.setPlaceholderText("input value")

        # prohibit input until a radio button is selected
        for box in self.boxes:
            box.setEnabled(False)

    def build_layouts(self) -> None:
        """
        Places previously instantiated widgets into respective layouts.
        """
        # initial set-up
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # horizontal layout containing widgets related to file selection
        selection_layout: QHBoxLayout = QHBoxLayout()
        selection_layout.setSpacing(0)

        selection_layout.addWidget(self.selection_label_with_hint, alignment=Qt.AlignLeft)
        selection_layout.addWidget(self.input_button, alignment=Qt.AlignLeft)

        # horizontal layout containing widgets related to preprocessing
        preprocessing_layout: QHBoxLayout = QHBoxLayout()
        preprocessing_layout.setSpacing(0)

        preprocessing_layout.addWidget(self.preprocessing_label_with_hint, alignment=Qt.AlignLeft)
        preprocessing_layout.addWidget(self.method, alignment=Qt.AlignLeft)

        # grid layout containing widgets related to postprocessing
        grid_layout: QGridLayout = QGridLayout()
        grid_layout.setSpacing(0)

        # add all pertinent widgets to the grid
        for idx, button in enumerate(self.buttons):
            grid_layout.addWidget(button, idx, 0)
        for idx, label in enumerate(self.labels):
            grid_layout.addWidget(label, idx, 1)
        for idx, box in enumerate(self.boxes):
            grid_layout.addWidget(box, idx, 2)

        # add inner widgets and layouts to overarching layout
        self.layout().addWidget(self.model_label_with_hint, alignment=Qt.AlignLeft)
        self.layout().addLayout(selection_layout)
        self.layout().addLayout(preprocessing_layout)
        self.layout().addWidget(self.postprocessing_label_with_hint, alignment=Qt.AlignLeft)
        self.layout().addLayout(grid_layout)

    def configure_slots(self) -> None:
        """
        Connects widgets to their respective event handlers.
        """
        # connect input button to file-retrieving slot
        self.input_button.button.clicked.connect(self.get_file_name)

        # connect radio buttons to slots
        self.top_button.toggled.connect(self.top_radio_button_slot)
        self.mid_button.toggled.connect(self.mid_radio_button_slot)
        self.bottom_button.toggled.connect(self.bottom_radio_button_slot)
