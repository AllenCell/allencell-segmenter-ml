from qtpy.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QComboBox,
    QGridLayout,
    QRadioButton,
)
from qtpy.QtCore import Qt

from allencell_ml_segmenter.widgets.slider_with_labels_widget import (
    SliderWithLabels,
)
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.widgets.input_button_widget import InputButton
from allencell_ml_segmenter.widgets.label_with_hint_widget import (
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

        self._model = model

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # instantiate widgets
        self._model_label_with_hint: LabelWithHint = LabelWithHint()
        self._selection_label_with_hint: LabelWithHint = LabelWithHint()
        self._input_button: InputButton = InputButton(self._model)
        self._preprocessing_label_with_hint: LabelWithHint = LabelWithHint()
        self._method_label: QLabel = QLabel("n/a")
        self._postprocessing_label_with_hint: LabelWithHint = LabelWithHint()

        # radio buttons
        self._top_button: QRadioButton = QRadioButton()
        self._bottom_button: QRadioButton = QRadioButton()

        self._buttons = [self._top_button, self._bottom_button]

        # labels for the radio buttons
        self._top_label: QLabel = QLabel(self.TOP_TEXT)
        self._bottom_label: QLabel = QLabel(self.BOTTOM_TEXT)

        self._labels = [self._top_label, self._bottom_label]

        # input fields corresponding to radio buttons & their labels
        self._top_input_box: SliderWithLabels = SliderWithLabels(
            0, 1, self._model
        )
        self._bottom_input_box: QComboBox = QComboBox()

        self._boxes = [
            self._top_input_box,
            self._bottom_input_box,
        ]

        self._model.subscribe(
            Event.ACTION_PREDICTION_PREPROCESSING_METHOD,
            self,
            lambda e: self._method_label.setText(
                self._model.get_preprocessing_method()
            ),
        )

        # finish default set-up
        self._call_setters()
        self._build_layouts()
        self._configure_slots()

    def handle_event(self, event: Event) -> None:
        pass

    def _top_radio_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if top button is checked.
        """
        self._top_input_box.setEnabled(True)
        self._bottom_input_box.setEnabled(False)
        self._model.set_postprocessing_method(self.TOP_TEXT)

    def _bottom_radio_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if bottom button is checked.
        """
        self._top_input_box.setEnabled(False)
        self._bottom_input_box.setEnabled(True)
        self._model.set_postprocessing_method(self.BOTTOM_TEXT)

    def _call_setters(self) -> None:
        """
        Sets pertinent default values for all widget fields.
        """
        # title + hint
        self._model_label_with_hint.set_label_text("Model")
        self._model_label_with_hint.set_hint("this is a test")

        # selection label + hint
        self._selection_label_with_hint.set_label_text(
            "Select an existing model"
        )
        self._selection_label_with_hint.set_hint("this is another test")

        # preprocessing label + hint
        self._preprocessing_label_with_hint.set_label_text(
            "Preprocessing method"
        )
        self._preprocessing_label_with_hint.set_hint(
            "this is the penultimate test"
        )

        # styling for label for preprocessing method
        self._method_label.setStyleSheet("margin-left: 25px")

        # postprocessing label + hint
        self._postprocessing_label_with_hint.set_label_text(
            "Postprocessing methods"
        )
        self._postprocessing_label_with_hint.set_hint("this is the final test")

        # add styling to buttons and labels
        for button in self._buttons:
            button.setStyleSheet("margin-left: 25px; margin-right: 6 px")
        for label in self._labels:
            label.setStyleSheet("margin-right: 25px")

        # set default values for input fields
        self._bottom_input_box.addItems(
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
        self._bottom_input_box.setEditable(True)
        self._bottom_input_box.setCurrentIndex(-1)

        # TODO: check that the user has selected an option other than the placeholder text when "run" is pressed
        # TODO: also check that one of the two radio buttons has been selected
        self._bottom_input_box.setPlaceholderText("select a method")
        self._bottom_input_box.setEditable(False)

        # prohibit input until a radio button is selected
        for box in self._boxes:
            box.setEnabled(False)

    def _build_layouts(self) -> None:
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
            self._selection_label_with_hint, alignment=Qt.AlignLeft
        )
        selection_layout.addWidget(self._input_button, alignment=Qt.AlignLeft)

        # horizontal layout containing widgets related to preprocessing
        preprocessing_layout: QHBoxLayout = QHBoxLayout()
        preprocessing_layout.setSpacing(0)

        preprocessing_layout.addWidget(
            self._preprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        preprocessing_layout.addWidget(
            self._method_label, alignment=Qt.AlignLeft
        )

        # grid layout containing widgets related to postprocessing
        grid_layout: QGridLayout = QGridLayout()
        grid_layout.setSpacing(0)

        # add all pertinent widgets to the grid
        for idx, button in enumerate(self._buttons):
            grid_layout.addWidget(button, idx, 0)
        for idx, label in enumerate(self._labels):
            grid_layout.addWidget(label, idx, 1)
        for idx, box in enumerate(self._boxes):
            grid_layout.addWidget(box, idx, 2)

        # add inner widgets and layouts to overarching layout
        self.layout().addWidget(
            self._model_label_with_hint, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(selection_layout)
        self.layout().addLayout(preprocessing_layout)
        self.layout().addWidget(
            self._postprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(grid_layout)

    def _configure_slots(self) -> None:
        """
        Connects widgets to their respective event handlers.
        """
        # connect radio buttons to slots
        self._top_button.toggled.connect(self._top_radio_button_slot)
        self._bottom_button.toggled.connect(self._bottom_radio_button_slot)

        # connect bottom input box to slot
        self._bottom_input_box.currentTextChanged.connect(
            lambda s: self._model.set_postprocessing_auto_threshold(s)
        )
