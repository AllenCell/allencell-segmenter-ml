from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QComboBox,
    QGridLayout,
    QRadioButton,
)
from qtpy.QtCore import Qt

from allencell_ml_segmenter.prediction.slider_with_labels_widget import (
    SliderWithLabels,
)
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

    TOP_TEXT: str = "simple threshold cutoff"
    BOTTOM_TEXT: str = "auto threshold"

    def __init__(self, model: PredictionModel):
        super().__init__()

        self.model = model

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # instantiate widgets
        self.model_label_with_hint: LabelWithHint = LabelWithHint()
        self.selection_label_with_hint: LabelWithHint = LabelWithHint()
        self.input_button: InputButton = InputButton()
        self.preprocessing_label_with_hint: LabelWithHint = LabelWithHint()
        self.method: QLabel = QLabel("n/a")
        self.postprocessing_label_with_hint: LabelWithHint = LabelWithHint()

        # radio buttons
        self.top_button: QRadioButton = QRadioButton()
        self.bottom_button: QRadioButton = QRadioButton()

        self.buttons = [self.top_button, self.bottom_button]

        # labels for the radio buttons
        self.top_label: QLabel = QLabel(self.TOP_TEXT)
        self.bottom_label: QLabel = QLabel(self.BOTTOM_TEXT)

        self.labels = [self.top_label, self.bottom_label]

        # input fields corresponding to radio buttons & their labels
        self.top_input_box: SliderWithLabels = SliderWithLabels()
        self.bottom_input_box: QComboBox = QComboBox()

        self.boxes = [
            self.top_input_box,
            self.bottom_input_box,
        ]

        self.model.subscribe(
            Event.ACTION_PREDICTION_PREPROCESSING_METHOD_SELECTED,
            self,
            lambda e: self.method.setText(
                self.model.get_preprocessing_method()
            ),
        )

        self.model.subscribe(
            Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_MOVED,
            self,
            lambda e: self.update_simple_threshold_label(),
        )

        self.model.subscribe(
            Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_TYPED,
            self,
            lambda e: self.update_simple_threshold_slider(),
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
        self.top_input_box.setEnabled(True)
        self.bottom_input_box.setEnabled(False)
        self.model.set_postprocessing_method(self.TOP_TEXT)

    def bottom_radio_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if bottom button is checked.
        """
        self.top_input_box.setEnabled(False)
        self.bottom_input_box.setEnabled(True)
        self.model.set_postprocessing_method(self.BOTTOM_TEXT)

    def update_simple_threshold_label(self) -> None:
        threshold: float = self.model.get_postprocessing_simple_threshold()
        self.top_input_box.label.setText(str(threshold))

    def update_simple_threshold_slider(self) -> None:
        threshold: float = self.model.get_postprocessing_simple_threshold()
        # TODO: figure out minimal combo
        self.top_input_box.slider.setTracking(True)
        self.top_input_box.slider.setValue(round(threshold * 100))
        self.top_input_box.slider.setSliderPosition(round(threshold * 100))
        self.top_input_box.slider.update()
        self.top_input_box.slider.repaint()

    def call_setters(self) -> None:
        """
        Sets pertinent default values for all widget fields.
        """
        # title + hint
        self.model_label_with_hint.set_label_text("Model")
        self.model_label_with_hint.set_hint("this is a test")

        # selection label + hint
        self.selection_label_with_hint.set_label_text(
            "Select an existing model"
        )
        self.selection_label_with_hint.set_hint("this is another test")

        # preprocessing label + hint
        self.preprocessing_label_with_hint.set_label_text(
            "Preprocessing method"
        )
        self.preprocessing_label_with_hint.set_hint(
            "this is the penultimate test"
        )

        # styling for label for preprocessing method
        self.method.setStyleSheet("margin-left: 25px")

        # postprocessing label + hint
        self.postprocessing_label_with_hint.set_label_text(
            "Postprocessing methods"
        )
        self.postprocessing_label_with_hint.set_hint("this is the final test")

        # add styling to buttons and labels
        for button in self.buttons:
            button.setStyleSheet("margin-left: 25px; margin-right: 6 px")
        for label in self.labels:
            label.setStyleSheet("margin-right: 25px")

        # set default values for input fields
        self.bottom_input_box.addItems(
            [
                "isodata",
                "li",
                "local",
                "mean",
                "minimum",
                "multiotsu",
                "niblack",
                "otsu",
                "savola",
                "triangle",
                "yen",
                "try all",
            ]
        )

        # set up disappearing placeholder text
        self.bottom_input_box.setEditable(True)
        self.bottom_input_box.setCurrentIndex(-1)

        # TODO: check that the user has selected an option other than the placeholder text when "run" is pressed
        # TODO: also check that one of the two radio buttons has been selected
        self.bottom_input_box.setPlaceholderText("select a method")
        self.bottom_input_box.setEditable(False)

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

        selection_layout.addWidget(
            self.selection_label_with_hint, alignment=Qt.AlignLeft
        )
        selection_layout.addWidget(self.input_button, alignment=Qt.AlignLeft)

        # horizontal layout containing widgets related to preprocessing
        preprocessing_layout: QHBoxLayout = QHBoxLayout()
        preprocessing_layout.setSpacing(0)

        preprocessing_layout.addWidget(
            self.preprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
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
        self.layout().addWidget(
            self.model_label_with_hint, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(selection_layout)
        self.layout().addLayout(preprocessing_layout)
        self.layout().addWidget(
            self.postprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(grid_layout)

    def configure_slots(self) -> None:
        """
        Connects widgets to their respective event handlers.
        """
        # connect input button to file-retrieving slot
        self.input_button.button.clicked.connect(self.get_file_name)

        # connect radio buttons to slots
        self.top_button.toggled.connect(self.top_radio_button_slot)
        self.bottom_button.toggled.connect(self.bottom_radio_button_slot)

        # connect input boxes to slots
        self.top_input_box.label.textChanged.connect(
            lambda s: self.model.set_postprocessing_simple_threshold_from_label(
                float(s)
            )
        )
        self.top_input_box.slider.valueChanged.connect(
            lambda v: self.model.set_postprocessing_simple_threshold_from_slider(
                v / 100
            )
        )

        self.bottom_input_box.currentTextChanged.connect(
            lambda s: self.model.set_postprocessing_auto_threshold(s)
        )
