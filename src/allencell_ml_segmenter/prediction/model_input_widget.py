from PyQt5.QtWidgets import QFrame
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
from magicgui.widgets import FloatSlider

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

    TOP_TEXT: str = "simple threshold"
    BOTTOM_TEXT: str = "auto threshold"

    def __init__(self, model: PredictionModel):
        super().__init__()

        self._model = model

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # instantiate widgets
        self.title_frame = QFrame()
        self.title = QLabel("Model", self)

        self._selection_label_with_hint: LabelWithHint = LabelWithHint()

        self._input_button: InputButton = InputButton(self._model)
        self._preprocessing_label_with_hint: LabelWithHint = LabelWithHint()
        self._method_label: QLabel = QLabel("n/a")
        self._postprocessing_label_with_hint: LabelWithHint = LabelWithHint()

        # radio buttons
        self._top_postproc_button: QRadioButton = QRadioButton()
        self._bottom_postproc_button: QRadioButton = QRadioButton()

        self._postproc_buttons = [
            self._top_postproc_button,
            self._bottom_postproc_button,
        ]

        # labels for the radio buttons
        self._top_postproc_label: QLabel = QLabel(self.TOP_TEXT)
        self._bottom_postproc_label: QLabel = QLabel(self.BOTTOM_TEXT)

        self._postproc_labels = [
            self._top_postproc_label,
            self._bottom_postproc_label,
        ]

        # input fields corresponding to radio buttons & their labels
        # TODO: show bounds of slider; make sure typed input field looks editable; add carat controls if possible
        self._simple_thresh_slider: FloatSlider = FloatSlider(
            min=0, max=100, step=1, readout=True
        )
        self._auto_thresh_selection: QComboBox = QComboBox()

        self._postprocessing_selections = [
            self._simple_thresh_slider,
            self._auto_thresh_selection,
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

    def _top_postproc_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if top button is checked.
        """
        self._simple_thresh_slider.native.setEnabled(True)
        self._auto_thresh_selection.setEnabled(False)
        self._model.set_postprocessing_method(self.TOP_TEXT)

    def _bottom_postproc_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if bottom button is checked.
        """
        self._simple_thresh_slider.native.setEnabled(False)
        self._auto_thresh_selection.setEnabled(True)
        self._model.set_postprocessing_method(self.BOTTOM_TEXT)

    def _call_setters(self) -> None:
        """
        Sets pertinent default values for all widget fields.
        """
        # title frame + title
        self.title_frame.setObjectName("tf")
        self.title_frame.setStyleSheet(
            "#tf {border: 1px solid #AAAAAA; border-radius: 5px}"
        )
        self.title.setStyleSheet("font-weight: bold")

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
        self._method_label.setStyleSheet(f"margin-right: 160px")

        # postprocessing label + hint
        self._postprocessing_label_with_hint.set_label_text(
            "Postprocessing methods"
        )
        self._postprocessing_label_with_hint.set_hint("this is the final test")

        # add styling to buttons and labels
        for button in self._postproc_buttons:
            button.setStyleSheet("margin-left: 25px; margin-right: 6 px")
        for label in self._postproc_labels:
            label.setStyleSheet("margin-right: 25px")

        # set default values for input fields
        self._auto_thresh_selection.addItems(
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
        self._auto_thresh_selection.setEditable(True)
        self._auto_thresh_selection.setCurrentIndex(-1)

        # TODO: check that the user has selected an option other than the placeholder text when "run" is pressed
        # TODO: also check that one of the two radio buttons has been selected
        self._auto_thresh_selection.setPlaceholderText("select a method")
        self._auto_thresh_selection.setEditable(False)

        # prohibit input until a radio button is selected
        for selection in self._postprocessing_selections:
            if isinstance(selection, FloatSlider):
                selection.native.setEnabled(False)
            else:
                selection.setEnabled(False)

    def _build_layouts(self) -> None:
        """
        Places previously instantiated widgets into respective layouts.
        """
        # initial set-up
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.title_frame.setLayout(QVBoxLayout())

        self.layout().addWidget(self.title)
        self.layout().addWidget(self.title_frame)

        # horizontal layout containing widgets related to file selection
        selection_layout: QHBoxLayout = QHBoxLayout()
        selection_layout.setSpacing(0)

        selection_layout.addWidget(
            self._selection_label_with_hint, alignment=Qt.AlignLeft
        )
        selection_layout.addWidget(self._input_button, alignment=Qt.AlignRight)

        # horizontal layout containing widgets related to preprocessing
        preprocessing_layout: QHBoxLayout = QHBoxLayout()
        preprocessing_layout.setSpacing(0)

        preprocessing_layout.addWidget(
            self._preprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        preprocessing_layout.addWidget(
            self._method_label, alignment=Qt.AlignRight
        )

        # grid layout containing widgets related to postprocessing
        grid_layout: QGridLayout = QGridLayout()
        grid_layout.setSpacing(0)

        # add all pertinent widgets to the grid
        for idx, button in enumerate(self._postproc_buttons):
            grid_layout.addWidget(button, idx, 0)
        for idx, label in enumerate(self._postproc_labels):
            grid_layout.addWidget(label, idx, 1)
        for idx, selection in enumerate(self._postprocessing_selections):
            if isinstance(selection, FloatSlider):
                grid_layout.addWidget(selection.native, idx, 2)
            else:
                grid_layout.addWidget(selection, idx, 2)

        grid_layout.setColumnStretch(1, 1)

        # add inner widgets and layouts to overarching layout
        self.title_frame.layout().addLayout(selection_layout)
        self.title_frame.layout().addLayout(preprocessing_layout)
        self.title_frame.layout().addWidget(
            self._postprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        self.title_frame.layout().addLayout(grid_layout)

    def _configure_slots(self) -> None:
        """
        Connects widgets to their respective event handlers.
        """
        # connect radio buttons to slots
        self._top_postproc_button.toggled.connect(
            self._top_postproc_button_slot
        )
        self._bottom_postproc_button.toggled.connect(
            self._bottom_postproc_button_slot
        )

        # connect postprocessing simple threshold slider to slot
        self._simple_thresh_slider.changed.connect(
            lambda v: self._model.set_postprocessing_simple_threshold(v)
        )

        # connect auto threshold selection to slot
        self._auto_thresh_selection.currentTextChanged.connect(
            lambda s: self._model.set_postprocessing_auto_threshold(s)
        )
