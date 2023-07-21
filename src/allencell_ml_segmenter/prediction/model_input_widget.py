from PyQt5.QtWidgets import QFrame, QDoubleSpinBox, QSlider
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
    MIN: int = 0
    MAX: int = 100
    STEP: int = 1

    def __init__(self, model: PredictionModel):
        super().__init__()

        self._model = model

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # instantiate widgets
        self._title_frame = QFrame()
        self._title = QLabel("Model", self)

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
        self._top_postproc_label: LabelWithHint = LabelWithHint(self.TOP_TEXT)
        self._bottom_postproc_label: LabelWithHint = LabelWithHint(
            self.BOTTOM_TEXT
        )

        self._postproc_labels = [
            self._top_postproc_label,
            self._bottom_postproc_label,
        ]

        # input fields corresponding to radio buttons & their labels
        # TODO: show bounds of slider; make sure typed input field looks editable; add carat controls if possible
        self._simple_thresh_slider: FloatSlider = FloatSlider(
            min=ModelInputWidget.MIN,
            max=ModelInputWidget.MAX,
            step=ModelInputWidget.STEP,
            readout=True,
        )
        self._lower_bound = QLabel(str(ModelInputWidget.MIN))
        self._upper_bound = QLabel(str(ModelInputWidget.MAX))

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
        # overall
        self.setStyleSheet(
            """
            QComboBox {
                color: #F0F1F2
            }
            QComboBox:disabled {
                color: #868E93
            }
        """
        )

        # title frame + title
        self._title_frame.setObjectName("tf")
        self._title_frame.setStyleSheet(
            "#tf {border: 1px solid #AAAAAA; border-radius: 5px}"
        )
        self._title.setStyleSheet("font-weight: bold")

        # selection label + hint
        self._selection_label_with_hint.set_label_text(
            "Select an existing model"
        )
        self._selection_label_with_hint.set_hint(
            "Path to packaged model output from training."
        )

        # preprocessing label + hint
        self._preprocessing_label_with_hint.set_label_text(
            "Preprocessing method"
        )
        self._preprocessing_label_with_hint.set_hint(
            "Image processing steps (e.g. normalization, smoothing) to run prior to segmentation. NOTE: this should be consistent with the image preprocessing used for training and only changed in rare circumstances."
        )

        # styling for label for preprocessing method
        self._method_label.setStyleSheet(f"margin-right: 160px")

        # postprocessing label + hint
        self._postprocessing_label_with_hint.set_label_text(
            "Postprocessing methods"
        )
        self._postprocessing_label_with_hint.set_hint(
            "Method for turning model-predicted probabilities into binary masks."
        )

        # Radio label hints
        self._top_postproc_label.set_hint(
            "Threshold predicted probabilities at a specific value."
        )
        self._bottom_postproc_label.set_hint(
            "Automatically choose a threshold at which to binarize predicted probabilities."
        )

        # add styling to buttons and labels
        for button in self._postproc_buttons:
            button.setStyleSheet("margin-left: 25px; margin-right: 6 px")
        for label in self._postproc_labels:
            label.add_right_space(25)

        # slider + spinbox
        slider: QSlider = (
            self._simple_thresh_slider.native.layout().itemAt(0).widget()
        )
        slider.setStyleSheet("margin-right: 20px")

        spinbox: QDoubleSpinBox = (
            self._simple_thresh_slider.native.layout().itemAt(1).widget()
        )
        spinbox.setObjectName("sb")
        spinbox.setStyleSheet(
            "#sb {padding: 0px; margin-bottom: 5px; background-color: #414851}"
        )

        # slider's bounds
        self._lower_bound.setStyleSheet("margin-right: 57px; font-size: 10px")
        self._upper_bound.setStyleSheet("font-size: 10px")

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

        self._title_frame.setLayout(QVBoxLayout())

        self.layout().addWidget(self._title)
        self.layout().addWidget(self._title_frame)

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

        # slider bounds
        horiz_layout: QHBoxLayout = QHBoxLayout()
        horiz_layout.addWidget(self._lower_bound)
        horiz_layout.addWidget(self._upper_bound, Qt.AlignLeft)
        horiz_layout.setSpacing(0)

        # slider + bounds
        vert_layout: QVBoxLayout = QVBoxLayout()
        vert_layout.addLayout(horiz_layout)
        vert_layout.addWidget(self._simple_thresh_slider.native)

        # add all pertinent widgets to the grid
        for idx, button in enumerate(self._postproc_buttons):
            grid_layout.addWidget(button, idx, 0)
        for idx, label in enumerate(self._postproc_labels):
            grid_layout.addWidget(label, idx, 1)
        for idx, selection in enumerate(
            [vert_layout, self._auto_thresh_selection]
        ):
            if isinstance(selection, QVBoxLayout):
                grid_layout.addLayout(selection, idx, 2)
            else:
                grid_layout.addWidget(selection, idx, 2)

        grid_layout.setColumnStretch(1, 1)

        # add inner widgets and layouts to overarching layout
        self._title_frame.layout().addLayout(selection_layout)
        self._title_frame.layout().addLayout(preprocessing_layout)
        self._title_frame.layout().addWidget(
            self._postprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        self._title_frame.layout().addLayout(grid_layout)

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
