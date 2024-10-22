from napari.utils.notifications import show_warning  # type: ignore

from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.view import View, MainWindow
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.service import ModelFileService

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.prediction.file_input_widget import (
    PredictionFileInput,
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
    Holds widgets for thresholding images with methods such as specific value and autothreshold.
    """

    def __init__(
        self,
        main_model: MainModel,
        experiments_model: IExperimentsModel,
        viewer: IViewer,
    ):
        super().__init__()

        self._main_model: MainModel = main_model
        self._experiments_model: IExperimentsModel = experiments_model
        self._viewer: IViewer = viewer
        self._thresholding_model: FileInputModel = FileInputModel()
        self._service: ModelFileService = ModelFileService(
            self._thresholding_model
        )

        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        self.setLayout(layout)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )

        # Title
        self._title: QLabel = QLabel("THRESHOLD", self)
        self._title.setObjectName("title")
        layout.addWidget(self._title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Input Image Section
        self._file_input_widget: PredictionFileInput = PredictionFileInput(
            self._thresholding_model, self._viewer, self._service
        )
        self._file_input_widget.setObjectName("fileInput")
        layout.addWidget(self._file_input_widget)

        # Thresholding Section
        self._threshold_label: LabelWithHint = LabelWithHint("Threshold")
        self._threshold_label.set_hint("Values to threshold with.")
        self._threshold_label.setObjectName("title")
        layout.addWidget(self._threshold_label)

        threshold_group_box = QGroupBox()
        threshold_group_layout = QVBoxLayout()

        # Radio Button for 'None'
        none_radio_layout: QHBoxLayout = QHBoxLayout()
        self._none_radio_button: QRadioButton = QRadioButton()
        none_radio_layout.addWidget(self._none_radio_button)

        none_radio_label: LabelWithHint = LabelWithHint("None")
        none_radio_label.set_hint("No thresholding applied.")
        none_radio_layout.addWidget(none_radio_label)
        threshold_group_layout.addLayout(none_radio_layout)

        # Radio Button for 'Specific Value' with Slider and Spinbox
        # Layout for slider and spinbox next to the 'Specific Value' radio button
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
        self._threshold_value_slider.setRange(
            0, 100
        )  # Slider values from 0 to 100 (representing 0.0 to 1.0)
        self._threshold_value_slider.setValue(50)  # Default value at 0.5

        self._threshold_value_spinbox: QDoubleSpinBox = QDoubleSpinBox()
        self._threshold_value_spinbox.setRange(
            0.0, 1.0
        )  # Spinbox range from 0.0 to 1.0
        self._threshold_value_spinbox.setSingleStep(0.01)
        self._threshold_value_spinbox.setValue(0.5)  # Default value

        # Connect slider and spinbox to keep them in sync
        self._threshold_value_slider.valueChanged.connect(
            self._update_spinbox_from_slider
        )
        self._threshold_value_spinbox.valueChanged.connect(
            self._update_slider_from_spinbox
        )

        # Add slider and spinbox to the specific value layout
        specific_value_layout.addWidget(self._specific_value_radio_button)
        specific_value_layout.addWidget(self._threshold_value_slider)
        specific_value_layout.addWidget(
            self._threshold_value_spinbox
        )  # Spinbox stretch
        threshold_group_layout.addLayout(specific_value_layout)

        # Radio Button for 'Autothreshold' with ComboBox
        autothreshold_layout = QHBoxLayout()
        self._autothreshold_radio_button: QRadioButton = QRadioButton()
        auto_thresh_label: LabelWithHint = LabelWithHint("Autothreshold")
        auto_thresh_label.set_hint("Apply an autothresholding method.")

        self._autothreshold_method_combo: QComboBox = QComboBox()
        self._autothreshold_method_combo.addItems(["Otsu"])
        self._autothreshold_method_combo.setEnabled(False)

        # Add the radio button and combo box to the same horizontal layout
        autothreshold_layout.addWidget(self._autothreshold_radio_button)
        autothreshold_layout.addWidget(auto_thresh_label)
        autothreshold_layout.addWidget(self._autothreshold_method_combo)
        threshold_group_layout.addLayout(autothreshold_layout)

        threshold_group_box.setLayout(threshold_group_layout)
        layout.addWidget(threshold_group_box)

        # Apply and Save Section
        self._apply_save_button: QPushButton = QPushButton("Apply & Save")
        self._apply_save_button.setEnabled(False)
        layout.addWidget(self._apply_save_button)

        self.setStyleSheet(Style.get_stylesheet("thresholding_view.qss"))

        # Configure connections
        self._configure_slots()

    def _configure_slots(self) -> None:
        """
        Connects signal-slot connections for widget elements.
        """
        self._specific_value_radio_button.toggled.connect(
            lambda checked: self._enable_specific_threshold_widgets(checked)
        )
        self._autothreshold_radio_button.toggled.connect(
            lambda checked: self._autothreshold_method_combo.setEnabled(
                checked
            )
        )

        # Enable/disable the Apply & Save button when a threshold method is selected
        self._none_radio_button.toggled.connect(self._enable_apply_button)
        self._specific_value_radio_button.toggled.connect(
            self._enable_apply_button
        )
        self._autothreshold_radio_button.toggled.connect(
            self._enable_apply_button
        )

    def _update_spinbox_from_slider(self, value: int) -> None:
        """
        Updates the spinbox value when the slider value changes.
        """
        self._threshold_value_spinbox.setValue(value / 100.0)

    def _update_slider_from_spinbox(self, value: float) -> None:
        """
        Updates the slider value when the spinbox value changes.
        """
        self._threshold_value_slider.setValue(int(value * 100))

    def _enable_specific_threshold_widgets(self, enabled: bool) -> None:
        """
        Enables or disables the slider and spinbox when the 'Specific Value' radio button is selected.
        """
        self._threshold_value_slider.setEnabled(enabled)
        self._threshold_value_spinbox.setEnabled(enabled)

    def _enable_apply_button(self) -> None:
        """
        Enables the Apply & Save button if any thresholding method is selected.
        """
        self._apply_save_button.setEnabled(
            self._none_radio_button.isChecked()
            or self._specific_value_radio_button.isChecked()
            or self._autothreshold_radio_button.isChecked()
        )

    def _browse_image_directory(self) -> None:
        """Opens a file dialog to select an image directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Image Directory"
        )
        if directory:
            self._file_input_widget.set_image_directory(directory)

    def _browse_output_directory(self) -> None:
        """Opens a file dialog to select an output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if directory:
            self._file_input_widget.set_output_directory(directory)

    def doWork(self) -> None:
        return

    def focus_changed(self) -> None:
        return

    def getTypeOfWork(self) -> str:
        return ""

    def showResults(self) -> None:
        return
