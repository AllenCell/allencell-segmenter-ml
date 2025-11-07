from pathlib import Path
from typing import Optional

from napari.utils.notifications import show_info  # type: ignore

from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.view import View, MainWindow
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.file_write_progress_tracker import (
    FileWriteProgressTracker,
)
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.thresholding.thresholding_model import (
    ThresholdingModel,
    AVAILABLE_AUTOTHRESHOLD_METHODS,
    THRESHOLD_RANGE,
)
from allencell_ml_segmenter.utils.file_utils import FileUtils

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.core.thresholding_file_input_widget import (
    ThresholdingFileInputWidget,
)
from allencell_ml_segmenter.core.file_input_model import (
    InputMode,
)

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
    QSpinBox,
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
        experiments_model: IExperimentsModel,
        viewer: IViewer,
    ):
        super().__init__()

        self._main_model: MainModel = main_model
        self._experiments_model: IExperimentsModel = experiments_model
        self._viewer: IViewer = viewer
        self._thresholding_model: ThresholdingModel = thresholding_model

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
        self._prediction_result_input_widget: ThresholdingFileInputWidget = (
            ThresholdingFileInputWidget(
                self._thresholding_model,
                self._viewer,
            )
        )
        self._prediction_result_input_widget.setObjectName("fileInput")
        layout.addWidget(self._prediction_result_input_widget)

        # thresholding values
        self._threshold_label: LabelWithHint = LabelWithHint("Threshold")
        self._threshold_label.set_hint("Value to threshold with")
        self._threshold_label.setObjectName("title")
        layout.addWidget(self._threshold_label)

        threshold_group_box = QGroupBox()
        threshold_group_layout = QVBoxLayout()

        # none thresholding selection
        none_radio_layout: QHBoxLayout = QHBoxLayout()
        self._none_radio_button: QRadioButton = QRadioButton()
        none_radio_layout.addWidget(self._none_radio_button)

        none_radio_label: LabelWithHint = LabelWithHint("None")
        none_radio_label.set_hint("No thresholding applied")
        none_radio_layout.addWidget(none_radio_label)
        threshold_group_layout.addLayout(none_radio_layout)

        # specific value thresholding selection
        specific_value_layout = QHBoxLayout()

        self._specific_value_radio_button: QRadioButton = QRadioButton()
        specific_value_layout.addWidget(self._specific_value_radio_button)
        specific_radio_label: LabelWithHint = LabelWithHint("Specific Value")
        specific_radio_label.set_hint("Select a thresholding value")
        specific_value_layout.addWidget(specific_radio_label)

        self._threshold_value_slider: QSlider = QSlider(
            Qt.Orientation.Horizontal
        )

        self._threshold_value_slider.setRange(
            THRESHOLD_RANGE[0], THRESHOLD_RANGE[1]
        )  # slider values from 0 to 100 (representing 0.0 to 1.0)

        self._threshold_value_spinbox: QSpinBox = QSpinBox()
        self._threshold_value_spinbox.setRange(
            THRESHOLD_RANGE[0], THRESHOLD_RANGE[1]
        )
        self._threshold_value_spinbox.setSingleStep(1)

        # set default value
        self._threshold_value_slider.setValue(
            self._thresholding_model.get_thresholding_value()
        )
        self._threshold_value_spinbox.setValue(
            self._thresholding_model.get_thresholding_value()
        )

        self._threshold_value_slider.setEnabled(False)
        self._threshold_value_spinbox.setEnabled(False)
        self._specific_value_radio_button.setChecked(False)

        # add slider and spinbox
        specific_value_layout.addWidget(self._threshold_value_slider)
        specific_value_layout.addWidget(self._threshold_value_spinbox)
        threshold_group_layout.addLayout(specific_value_layout)

        # autothresholding
        autothreshold_layout = QHBoxLayout()
        self._autothreshold_radio_button: QRadioButton = QRadioButton()
        auto_thresh_label: LabelWithHint = LabelWithHint("Autothreshold")
        auto_thresh_label.set_hint("Select an autothresholding method")

        self._autothreshold_method_combo: QComboBox = QComboBox()
        self._autothreshold_method_combo.addItems(
            AVAILABLE_AUTOTHRESHOLD_METHODS
        )
        self._autothreshold_method_combo.setEnabled(False)

        autothreshold_layout.addWidget(self._autothreshold_radio_button)
        autothreshold_layout.addWidget(auto_thresh_label)
        autothreshold_layout.addWidget(self._autothreshold_method_combo)
        threshold_group_layout.addLayout(autothreshold_layout)

        threshold_group_box.setLayout(threshold_group_layout)
        layout.addWidget(threshold_group_box)

        # apply and save
        self._apply_save_button: QPushButton = QPushButton("Apply and Save")
        self._apply_save_button.setEnabled(False)
        self._apply_save_button.clicked.connect(self._save_thresholded_images)
        layout.addWidget(self._apply_save_button)

        # need styling
        self.setStyleSheet(Style.get_stylesheet("thresholding_view.qss"))

        # configure widget behavior
        self._configure_slots()

        self._main_model.subscribe(
            Event.PROCESS_PREDICTION_COMPLETE,
            self,
            lambda e: self._main_model.set_current_view(self),
        )

    def _configure_slots(self) -> None:
        """
        Connects behavior for widgets
        """

        # sync slider and spinbox
        self._threshold_value_slider.valueChanged.connect(
            self._update_spinbox_from_slider
        )
        self._threshold_value_spinbox.valueChanged.connect(
            self._update_slider_from_spinbox
        )

        # update state and ui based on radio button selections
        self._none_radio_button.toggled.connect(self._disable_all_thresholding)
        self._specific_value_radio_button.toggled.connect(
            self._update_state_from_radios
        )
        self._autothreshold_radio_button.toggled.connect(
            self._update_state_from_radios
        )

        # update autothresholding method when one is selected, and update viewer if able
        self._autothreshold_method_combo.currentIndexChanged.connect(
            lambda: self._thresholding_model.set_autothresholding_method(
                self._autothreshold_method_combo.currentText()
            )
        )

        # update thresholding value when the user is finished making a selection
        self._threshold_value_slider.sliderReleased.connect(
            lambda: self._thresholding_model.set_thresholding_value(
                self._threshold_value_slider.value()
            )
        )

        self._threshold_value_spinbox.editingFinished.connect(
            lambda: self._thresholding_model.set_thresholding_value(
                self._threshold_value_spinbox.value()
            )
        )

    def _update_spinbox_from_slider(self, value: int) -> None:
        """
        Update the spinbox value when slider is changed
        """
        self._threshold_value_spinbox.setValue(value)

    def _update_slider_from_spinbox(self, value: int) -> None:
        """
        Update the slider value when spinbox is changed
        """
        self._threshold_value_slider.setValue(value)

    def _enable_specific_threshold_widgets(self, enabled: bool) -> None:
        """
        enable or disable specific value thresholding widgets
        """
        self._threshold_value_slider.setEnabled(enabled)
        self._threshold_value_spinbox.setEnabled(enabled)

    def _update_state_from_radios(self) -> None:
        """
        update state based on thresholding radio button selection
        """
        self._thresholding_model.set_autothresholding_enabled(
            self._autothreshold_radio_button.isChecked()
        )
        self._thresholding_model.set_threshold_enabled(
            self._specific_value_radio_button.isChecked()
        )

        self._autothreshold_method_combo.setEnabled(
            self._autothreshold_radio_button.isChecked()
        )
        self._enable_specific_threshold_widgets(
            self._specific_value_radio_button.isChecked()
        )

        self._apply_save_button.setEnabled(
            self._specific_value_radio_button.isChecked()
            or self._autothreshold_radio_button.isChecked()
        )

    def _disable_all_thresholding(self) -> None:
        # TODO handle disabling ui for thresh autothresh
        self._thresholding_model.disable_all()

    def _check_able_to_threshold(self) -> bool:
        able_to_threshold: bool = True
        # Check to see if output directory is selected
        if self._thresholding_model.get_output_directory() is None:
            show_info("Please select an output directory first.")
            able_to_threshold = False

        # Check to see if input images / directory of images are selected
        if self._thresholding_model.get_input_mode() is None:
            show_info("Please select an input mode first.")
            able_to_threshold = False
        else:
            if (
                self._thresholding_model.get_input_mode()
                == InputMode.FROM_NAPARI_LAYERS
                and len(self._thresholding_model.get_selected_idx()) == 0
            ):
                show_info("Please select on screen images to threshold.")
                able_to_threshold = False
            elif (
                self._thresholding_model.get_input_mode()
                == InputMode.FROM_PATH
                and self._thresholding_model.get_input_image_path() is None
            ):
                show_info("Please select a directory to threshold.")
                able_to_threshold = False

        # check to see if thresholding method is selected
        if (
            not self._thresholding_model.is_threshold_enabled()
            and not self._thresholding_model.is_autothresholding_enabled()
        ):
            show_info("Please select a thresholding method first.")
            able_to_threshold = False

        return able_to_threshold

    def _save_thresholded_images(self) -> None:
        output_dir: Optional[Path] = (
            self._thresholding_model.get_output_directory()
        )
        selected_images_viewer: list[Path] = []
        for idx, layer in enumerate(
            self._viewer.get_all_layers_containing_prob_map()
        ):
            if idx in self._thresholding_model.get_selected_idx():
                selected_images_viewer.append(layer.metadata["source_path"])
        self._thresholding_model.set_selected_paths(selected_images_viewer)

        if output_dir is not None and self._check_able_to_threshold():
            # progress tracker is tracking number of images saved to the thresholding folder
            progress_tracker: FileWriteProgressTracker = (
                FileWriteProgressTracker(
                    output_dir,
                    len(self._thresholding_model.get_input_files_as_list()),
                )
            )

            self.startLongTaskWithProgressBar(progress_tracker)

    def doWork(self) -> None:
        self._thresholding_model.dispatch_save_thresholded_images()

    def focus_changed(self) -> None:
        self._prediction_result_input_widget._update_layer_list()

    def getTypeOfWork(self) -> str:
        return ""

    def showResults(self) -> None:
        dialog_box = DialogBox(
            f"Predicted images saved to {str(self._thresholding_model.get_output_directory())}. \nWould you like to open this folder?"
        )
        dialog_box.exec()
        output_dir: Optional[Path] = (
            self._thresholding_model.get_output_directory()
        )

        if output_dir and dialog_box.get_selection():
            FileUtils.open_directory_in_window(output_dir)
