from typing import List, Any

from qtpy.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QComboBox,
    QGridLayout,
    QRadioButton,
    QFrame,
    QDoubleSpinBox,
    QSlider,
)
from qtpy.QtCore import Qt
from magicgui.widgets import FloatSlider
from allencell_ml_segmenter.core.aics_widget import AicsWidget

from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.widgets.label_with_hint_widget import (
    LabelWithHint,
)


class ModelInputWidget(AicsWidget):
    """
    Handles model input, preprocessing selection, and
    postprocessing selection for prediction.
    """

    TOP_TEXT: str = "none"
    MID_TEXT: str = "simple threshold"
    BOTTOM_TEXT: str = "auto threshold"
    MIN: int = 0
    MAX: int = 100
    STEP: int = 1

    def __init__(self, model: PredictionModel):
        super().__init__()

        self._model: PredictionModel = model

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # instantiate widgets
        self._frame: QFrame = QFrame()
        self._title: LabelWithHint = LabelWithHint("Pre/Post Processing")
        # TODO: hints for widget titles?
        self._preprocessing_label_with_hint: LabelWithHint = LabelWithHint()
        self._method_label: QLabel = QLabel("n/a")
        self._postprocessing_label_with_hint: LabelWithHint = LabelWithHint()

        # radio buttons
        self._top_postproc_button: QRadioButton = QRadioButton()
        self._mid_postproc_button: QRadioButton = QRadioButton()
        self._bottom_postproc_button: QRadioButton = QRadioButton()

        self._postproc_buttons: List[QRadioButton] = [
            self._top_postproc_button,
            self._mid_postproc_button,
            self._bottom_postproc_button,
        ]

        # labels for the radio buttons
        self._top_postproc_label: QLabel = QLabel(ModelInputWidget.TOP_TEXT)
        self._mid_postproc_label: LabelWithHint = LabelWithHint(
            ModelInputWidget.MID_TEXT
        )
        self._bottom_postproc_label: LabelWithHint = LabelWithHint(
            ModelInputWidget.BOTTOM_TEXT
        )

        self._postproc_labels: List[Any] = [
            self._top_postproc_label,
            self._mid_postproc_label,
            self._bottom_postproc_label,
        ]

        # input fields corresponding to radio buttons & their labels
        self._simple_thresh_slider: FloatSlider = FloatSlider(
            min=ModelInputWidget.MIN,
            max=ModelInputWidget.MAX,
            step=ModelInputWidget.STEP,
            readout=True,
        )
        self._lower_bound: QLabel = QLabel(str(ModelInputWidget.MIN))
        self._upper_bound: QLabel = QLabel(str(ModelInputWidget.MAX))

        self._auto_thresh_selection: QComboBox = QComboBox()

        self._postprocessing_selections: List[Any] = [
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
        Prohibits usage of non-related input fields if "none" button is checked.
        """
        # disable middle
        self._simple_thresh_slider.native.setEnabled(False)

        # disable bottom
        self._auto_thresh_selection.setEnabled(False)

        # set postprocessing method
        self._model.set_postprocessing_method(ModelInputWidget.TOP_TEXT)

    def _mid_postproc_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if top button is checked.
        """
        self._simple_thresh_slider.native.setEnabled(True)
        self._auto_thresh_selection.setEnabled(False)
        self._model.set_postprocessing_method(ModelInputWidget.MID_TEXT)

    def _bottom_postproc_button_slot(self) -> None:
        """
        Prohibits usage of non-related input fields if bottom button is checked.
        """
        self._simple_thresh_slider.native.setEnabled(False)
        self._auto_thresh_selection.setEnabled(True)
        self._model.set_postprocessing_method(ModelInputWidget.BOTTOM_TEXT)

    def _call_setters(self) -> None:
        """
        Sets pertinent default values for all widget fields.
        """
        # title frame + title
        self._frame.setObjectName("frame")
        self._title.setObjectName("title")

        # preprocessing label + hint
        self._preprocessing_label_with_hint.set_label_text(
            "Preprocessing method"
        )
        self._preprocessing_label_with_hint.set_hint(
            "Image processing steps (e.g. normalization, smoothing) to run prior to segmentation. NOTE: this should be consistent with the image preprocessing used for training and only changed in rare circumstances."
        )

        # styling for label for preprocessing method
        self._method_label.setObjectName("methodLabel")

        # postprocessing label + hint
        self._postprocessing_label_with_hint.set_label_text(
            "Postprocessing methods"
        )
        self._postprocessing_label_with_hint.set_hint(
            "Method for turning model-predicted probabilities into binary masks."
        )

        # Radio label hints
        self._mid_postproc_label.set_hint(
            "Threshold predicted probabilities at a specific value."
        )
        self._bottom_postproc_label.set_hint(
            "Automatically choose a threshold at which to binarize predicted probabilities."
        )

        # slider + spinbox
        slider: QSlider = (
            self._simple_thresh_slider.native.layout().itemAt(0).widget()
        )
        slider.setObjectName("slider")

        spinbox: QDoubleSpinBox = (
            self._simple_thresh_slider.native.layout().itemAt(1).widget()
        )
        spinbox.setObjectName("spinbox")
        spinbox.setStyleSheet(
            """
            #spinbox {
                padding: 0px;
                margin-bottom: 5px;
                background-color: #414851;
            }
        """
        )

        # slider's bounds
        self._lower_bound.setObjectName("lowerBound")
        self._upper_bound.setObjectName("upperBound")

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
        # TODO: also check that one of the three radio buttons has been selected
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

        self._frame.setLayout(QVBoxLayout())

        self.layout().addWidget(self._title)
        self.layout().addWidget(self._frame)

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
                grid_layout.addLayout(selection, idx + 1, 2)
            else:
                grid_layout.addWidget(selection, idx + 1, 2)

        grid_layout.setColumnStretch(1, 1)

        # add inner widgets and layouts to overarching layout
        self._frame.layout().addLayout(preprocessing_layout)
        self._frame.layout().addWidget(
            self._postprocessing_label_with_hint, alignment=Qt.AlignLeft
        )
        self._frame.layout().addLayout(grid_layout)

    def _configure_slots(self) -> None:
        """
        Connects widgets to their respective event handlers.
        """
        # connect radio buttons to slots
        self._top_postproc_button.toggled.connect(
            self._top_postproc_button_slot
        )
        self._mid_postproc_button.toggled.connect(
            self._mid_postproc_button_slot
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
