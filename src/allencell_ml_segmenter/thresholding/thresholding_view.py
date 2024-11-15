from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.view import View, MainWindow
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.thresholding.thresholding_model import ThresholdingModel

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.core.file_input_widget import (
    FileInputWidget,
)
from allencell_ml_segmenter.core.file_input_model import FileInputModel

from qtpy.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QComboBox,
    QGroupBox,
    QRadioButton,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QFileDialog,
)
from qtpy.QtCore import Qt


class ThresholdingView(View, MainWindow):
    """
    View for thresholding
    """

    def __init__(
        self,
        main_model: MainModel,
        thresholding_model: ThresholdingModel,
        file_input_model: FileInputModel,
        experiments_model: IExperimentsModel,
        viewer: IViewer,
    ):
        super().__init__()

        self._main_model: MainModel = main_model
        self._experiments_model: IExperimentsModel = experiments_model
        self._viewer: IViewer = viewer
        self._thresholding_model: ThresholdingModel = thresholding_model


        # To manage input files:
        self._file_input_model: FileInputModel = file_input_model
        self._input_files_service: ModelFileService = ModelFileService(
            self._file_input_model
        )

        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        self.setLayout(layout)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )

        # title
        self._title: QLabel = QLabel("THRESHOLD", self)
        self._title.setObjectName("title")
        layout.addWidget(self._title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # selecting input image
        self._file_input_widget: FileInputWidget = FileInputWidget(
            self._file_input_model, self._viewer, self._input_files_service
        )
        self._file_input_widget.setObjectName("fileInput")
        layout.addWidget(self._file_input_widget)

        # thresholding values
        self._threshold_label: LabelWithHint = LabelWithHint("Threshold")
        self._threshold_label.set_hint("Values to threshold with.")
        self._threshold_label.setObjectName("title")
        layout.addWidget(self._threshold_label)

        threshold_group_box = QGroupBox()
        threshold_group_layout = QVBoxLayout()

        # none thresholding selection
        none_radio_layout: QHBoxLayout = QHBoxLayout()
        self._none_radio_button: QRadioButton = QRadioButton()
        none_radio_layout.addWidget(self._none_radio_button)

        none_radio_label: LabelWithHint = LabelWithHint("None")
        none_radio_label.set_hint("No thresholding applied.")
        none_radio_layout.addWidget(none_radio_label)
        threshold_group_layout.addLayout(none_radio_layout)

        # specific value thresholding selection
        specific_value_layout = QHBoxLayout()

        self._specific_value_radio_button: QRadioButton = QRadioButton()
        specific_value_layout.addWidget(self._specific_value_radio_button)
        specific_radio_label: LabelWithHint = LabelWithHint("Specific Value")
        specific_radio_label.set_hint(
            "Set thresholding value you'd like to apply."
        )
        specific_value_layout.addWidget(specific_radio_label)

        self._threshold_value_slider: QSlider = QSlider(
            Qt.Orientation.Horizontal
        )

        # TODO: see if i can set range and step similarly to spinbox, to avoid conversion each time we update
        self._threshold_value_slider.setRange(
            0, 100
        )  # slider values from 0 to 100 (representing 0.0 to 1.0)
        self._threshold_value_slider.setValue(self._threshold_value_to_slider(
            self._thresholding_model.get_thresholding_value()))

        self._threshold_value_spinbox: QDoubleSpinBox = QDoubleSpinBox()
        self._threshold_value_spinbox.setRange(0.0, 1.0)
        self._threshold_value_spinbox.setSingleStep(0.01)
        self._threshold_value_spinbox.setValue(0.5)

        # sync slider and spinbox
        self._threshold_value_slider.valueChanged.connect(
            self._update_spinbox_from_slider
        )
        self._threshold_value_spinbox.valueChanged.connect(
            self._update_slider_from_spinbox
        )

        # add slider and spinbox
        specific_value_layout.addWidget(self._specific_value_radio_button)
        specific_value_layout.addWidget(self._threshold_value_slider)
        specific_value_layout.addWidget(self._threshold_value_spinbox)
        threshold_group_layout.addLayout(specific_value_layout)

        # autothresholding
        autothreshold_layout = QHBoxLayout()
        self._autothreshold_radio_button: QRadioButton = QRadioButton()
        auto_thresh_label: LabelWithHint = LabelWithHint("Autothreshold")
        auto_thresh_label.set_hint("Apply an autothresholding method.")

        self._autothreshold_method_combo: QComboBox = QComboBox()
        self._autothreshold_method_combo.addItems(["Otsu"])
        self._autothreshold_method_combo.setEnabled(False)

        autothreshold_layout.addWidget(self._autothreshold_radio_button)
        autothreshold_layout.addWidget(auto_thresh_label)
        autothreshold_layout.addWidget(self._autothreshold_method_combo)
        threshold_group_layout.addLayout(autothreshold_layout)

        threshold_group_box.setLayout(threshold_group_layout)
        layout.addWidget(threshold_group_box)

        # apply and save
        self._apply_save_button: QPushButton = QPushButton("Apply & Save")
        self._apply_save_button.setEnabled(False)
        layout.addWidget(self._apply_save_button)

        # need styling
        self.setStyleSheet(Style.get_stylesheet("thresholding_view.qss"))

        # configure widget behavior
        self._configure_slots()

    def _configure_slots(self) -> None:
        """
        Connects behavior for widgets
        """

        # enable selections when corresponding radio button is selected
        self._specific_value_radio_button.toggled.connect(
            lambda checked: self._enable_specific_threshold_widgets(checked)
        )
        self._autothreshold_radio_button.toggled.connect(
            lambda checked: self._autothreshold_method_combo.setEnabled(
                checked
            )
        )

        # enable apply button only when thresholding method is selected
        self._none_radio_button.toggled.connect(self._enable_apply_button)
        self._specific_value_radio_button.toggled.connect(
            self._enable_apply_button
        )
        self._autothreshold_radio_button.toggled.connect(
            self._enable_apply_button
        )

    def _update_spinbox_from_slider(self, value: int) -> None:
        """
        Update the spinbox value when slider is changed
        """
        thresh_value: float = self._slider_value_to_threshold(value)
        self._threshold_value_spinbox.setValue(thresh_value)
        self._thresholding_model.set_thresholding_value(thresh_value)

    def _update_slider_from_spinbox(self, value: float) -> None:
        """
        Update the slider value when spinbox is changed
        """
        self._threshold_value_slider.setValue(self._threshold_value_to_slider(value))
        self._thresholding_model.set_thresholding_value(value)

    def _enable_specific_threshold_widgets(self, enabled: bool) -> None:
        """
        enable or disable specific value thresholding widgets
        """
        self._threshold_value_slider.setEnabled(enabled)
        self._threshold_value_spinbox.setEnabled(enabled)

    def _enable_apply_button(self) -> None:
        """
        enable or disable apply button
        """
        self._apply_save_button.setEnabled(
            self._none_radio_button.isChecked()
            or self._specific_value_radio_button.isChecked()
            or self._autothreshold_radio_button.isChecked()
        )

    def _slider_value_to_threshold(self, value: int) -> float:
        return value / 100.0

    def _threshold_value_to_slider(self, value: float) -> int:
        return int(value * 100)

    def doWork(self) -> None:
        return

    def focus_changed(self) -> None:
        return

    def getTypeOfWork(self) -> str:
        return ""

    def showResults(self) -> None:
        return