from pathlib import Path
from typing import Optional, List

from PyQt5.QtWidgets import QSpinBox, QAbstractSpinBox
from napari.utils.notifications import show_warning

from allencell_ml_segmenter.main.i_viewer import IViewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QFrame,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
    QGridLayout,
    QComboBox,
    QHBoxLayout,
    QRadioButton,
    QLineEdit,
    QCheckBox,
)
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.training.image_selection_widget import (
    ImageSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    ModelSize,
)

from aicsimageio import AICSImage
from aicsimageio.readers import TiffReader

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from qtpy.QtGui import QIntValidator
from allencell_ml_segmenter.training.metrics_csv_progress_tracker import (
    MetricsCSVProgressTracker,
)
from allencell_ml_segmenter.core.info_dialog_box import InfoDialogBox


class TrainingView(View):
    """
    Holds widgets pertinent to training processes - ImageSelectionWidget & ModelSelectionWidget.
    """

    def __init__(
        self,
        main_model: MainModel,
        experiments_model: ExperimentsModel,
        training_model: TrainingModel,
        viewer: IViewer,
    ):
        super().__init__()

        self._viewer: IViewer = viewer

        self._main_model: MainModel = main_model
        self._experiments_model: ExperimentsModel = experiments_model
        self._training_model: TrainingModel = training_model

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._title: QLabel = QLabel("SEGMENTATION MODEL TRAINING", self)
        self._title.setObjectName("title")
        self.layout().addWidget(
            self._title, alignment=Qt.AlignHCenter | Qt.AlignTop
        )

        # initialize constituent widgets
        image_selection_widget: ImageSelectionWidget = ImageSelectionWidget(
            self._training_model, self._experiments_model
        )
        image_selection_widget.setObjectName("imageSelection")

        # Dummy divs allow for easy alignment
        top_container: QVBoxLayout = QVBoxLayout()
        top_dummy: QFrame = QFrame()
        bottom_dummy: QFrame = QFrame()

        top_container.addWidget(image_selection_widget)
        top_dummy.setLayout(top_container)
        self.layout().addWidget(top_dummy)

        # bottom half
        bottom_grid_layout = QGridLayout()

        patch_size_label: LabelWithHint = LabelWithHint("Patch size")
        bottom_grid_layout.addWidget(patch_size_label, 0, 0)
        patch_size_entry_layout: QHBoxLayout = QHBoxLayout()

        self.z_patch_size: QSpinBox = QSpinBox()
        self.z_patch_size.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.z_patch_size.setMinimum(0)
        self.z_patch_size.setMaximum(9999)
        patch_size_entry_layout.addWidget(QLabel("Z:"))
        patch_size_entry_layout.addWidget(self.z_patch_size)

        self.y_patch_size: QSpinBox = QSpinBox()
        self.y_patch_size.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.y_patch_size.setMinimum(0)
        self.z_patch_size.setMaximum(9999)
        patch_size_entry_layout.addWidget(QLabel("Y:"))
        patch_size_entry_layout.addWidget(self.y_patch_size)

        self.x_patch_size: QSpinBox = QSpinBox()
        self.x_patch_size.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.x_patch_size.setMinimum(0)
        self.z_patch_size.setMaximum(9999)
        patch_size_entry_layout.addWidget(QLabel("X:"))
        patch_size_entry_layout.addWidget(self.x_patch_size)

        bottom_grid_layout.addLayout(patch_size_entry_layout, 0, 1)
        model_size_label: LabelWithHint = LabelWithHint("Model size")
        bottom_grid_layout.addWidget(model_size_label, 1, 0)

        self.model_size_combo_box: QComboBox = QComboBox()
        self.model_size_combo_box.setObjectName("modelSizeComboBox")
        self.model_size_combo_box.setCurrentIndex(-1)
        self.model_size_combo_box.setPlaceholderText("Select an option")
        self.model_size_combo_box.addItems(
            [size.name.lower() for size in ModelSize]
        )
        self.model_size_combo_box.currentTextChanged.connect(
            lambda model_size: self._training_model.set_model_size(model_size)
        )
        bottom_grid_layout.addWidget(self.model_size_combo_box, 1, 1)

        image_dimensions_label: LabelWithHint = LabelWithHint(
            "Image dimension"
        )
        bottom_grid_layout.addWidget(image_dimensions_label, 2, 0)

        self.dimension_label: QLabel = QLabel("")
        bottom_grid_layout.addWidget(self.dimension_label, 2, 1)

        num_epochs_label: LabelWithHint = LabelWithHint("Training steps")
        bottom_grid_layout.addWidget(num_epochs_label, 3, 0)

        self._num_epochs_input: QLineEdit = QLineEdit()
        # allow only integers TODO [needs test coverage]
        self._num_epochs_input.setValidator(QIntValidator())
        self._num_epochs_input.setPlaceholderText("1000")
        self._num_epochs_input.setObjectName("trainingStepInput")
        self._num_epochs_input.textChanged.connect(
            self._num_epochs_field_handler
        )
        bottom_grid_layout.addWidget(self._num_epochs_input, 3, 1)

        max_time_layout: QHBoxLayout = QHBoxLayout()
        max_time_layout.setSpacing(0)

        self.max_time_checkbox: QCheckBox = QCheckBox()
        self.max_time_checkbox.setObjectName("timeoutCheckbox")
        self.max_time_checkbox.stateChanged.connect(
            self._max_time_checkbox_slot
        )
        max_time_layout.addWidget(self.max_time_checkbox)

        max_time_left_text: QLabel = QLabel("Time out after")
        max_time_layout.addWidget(max_time_left_text)

        self.max_time_in_minutes_input: QLineEdit = QLineEdit()
        self.max_time_in_minutes_input.setObjectName("timeoutMinuteInput")
        self.max_time_in_minutes_input.setEnabled(False)
        self.max_time_in_minutes_input.setMaximumWidth(30)
        self.max_time_in_minutes_input.setPlaceholderText("30")
        self.max_time_in_minutes_input.textChanged.connect(
            lambda text: self._training_model.set_max_time(int(text))
        )
        max_time_layout.addWidget(self.max_time_in_minutes_input)

        max_time_right_text: LabelWithHint = LabelWithHint("minutes")
        max_time_layout.addWidget(max_time_right_text, alignment=Qt.AlignLeft)
        max_time_layout.addStretch()

        bottom_grid_layout.addLayout(max_time_layout, 4, 1)
        bottom_grid_layout.setColumnStretch(1, 8)
        bottom_grid_layout.setColumnStretch(0, 3)

        bottom_dummy.setLayout(bottom_grid_layout)
        self.layout().addWidget(bottom_dummy)

        self._train_btn: QPushButton = QPushButton("Start training")
        self._train_btn.setObjectName("trainBtn")
        self.layout().addWidget(self._train_btn)
        self._train_btn.clicked.connect(self.train_btn_handler)

        self._main_model.subscribe(
            Event.VIEW_SELECTION_TRAINING,
            self,
            lambda e: self._main_model.set_current_view(self),
        )

        self._training_model.subscribe(
            Event.ACTION_TRAINING_DIMENSIONS_SET,
            self,
            self._handle_dimensions_available,
        )

        # apply styling
        self.setStyleSheet(Style.get_stylesheet("training_view.qss"))

    def train_btn_handler(self) -> None:
        """
        Starts training process
        """
        # TODO: Refactor- move other checks for training here
        if self._check_patch_size_ok():
            self._update_model_with_patch_size()

            progress_tracker: MetricsCSVProgressTracker = (
                MetricsCSVProgressTracker(
                    self._experiments_model.get_metrics_csv_path(),
                    self._training_model.get_num_epochs(),
                    self._experiments_model.get_latest_metrics_csv_version()
                    + 1,
                )
            )
            self.startLongTaskWithProgressBar(progress_tracker)

    # Abstract methods from View implementations #######################

    def doWork(self):
        """
        Starts training process
        """
        self._training_model.dispatch_training()

    def getTypeOfWork(self) -> str:
        """
        Returns string representation of training process
        """
        return "Training"

    def showResults(self):
        dialog_box = InfoDialogBox("Training finished")
        dialog_box.exec()

    def _num_epochs_field_handler(self, num_epochs: str) -> None:
        self._training_model.set_num_epochs(int(num_epochs))

    def _max_time_checkbox_slot(self, checked: Qt.CheckState) -> None:
        """
        Triggered when the user selects the "time out after" _timeout_checkbox.
        Enables/disables interaction with the neighboring hour input based on checkstate.
        """
        if checked == Qt.CheckState.Checked:
            self.max_time_in_minutes_input.setEnabled(True)
            self._training_model.set_use_max_time(True)
        else:
            self.max_time_in_minutes_input.setEnabled(False)
            self._training_model.set_use_max_time(False)

    def _check_patch_size_ok(self) -> bool:
        """
        Gets patch sizes from the UI, if invalid patches are set throws an error.
        Returns True if valid patch sizes were provided, false if not
        """
        missing_patches: list[str] = []
        # patch size cannot be 0 for any dim
        if (
            self.z_patch_size.value() == 0
            and self._training_model.get_spatial_dims() == 3
        ):
            # 3d selected but z patch size missing
            missing_patches.append("Z")

        if self.y_patch_size.value() == 0:
            missing_patches.append("Y")

        if self.x_patch_size.value() == 0:
            missing_patches.append("X")

        if len(missing_patches) > 0:
            show_warning(
                f"Please define {missing_patches} patch sizes before continuing."
            )
            return False

        return True

    def _update_model_with_patch_size(self) -> None:
        if len(self._training_model.get_image_dimensions()) == 3:
            self._training_model.set_patch_size(
                [
                    self.z_patch_size.value(),
                    self.y_patch_size.value(),
                    self.x_patch_size.value(),
                ]
            )
        else:
            self._training_model.set_patch_size(
                [self.y_patch_size.value(), self.x_patch_size.value()]
            )

    def _handle_dimensions_available(self, _: Event) -> None:
        image_dims: List[int] = self._training_model.get_image_dimensions()
        self._training_model.set_spatial_dims(len(image_dims))
        self._set_max_patch_size(image_dims)
        self._enable_patch_size_edit()
        self._update_spatial_dims()

    def _update_spatial_dims(self) -> None:
        self.dimension_label.setText(f"{self._training_model.get_spatial_dims()}D")

    def _set_max_patch_size(self, image_dims: List[int]) -> None:
        if len(image_dims) == 3:
            # 3d image
            self.z_patch_size.setMaximum(image_dims[0])
            self.y_patch_size.setMaximum(image_dims[1])
            self.x_patch_size.setMaximum(image_dims[2])
        else:
            self.z_patch_size.setMaximum(0)
            self.y_patch_size.setMaximum(image_dims[0])
            self.x_patch_size.setMaximum(image_dims[1])

    def _enable_patch_size_edit(self) -> None:
        # enable only for 3d
        if self._training_model.get_spatial_dims() == 2:
            self.z_patch_size.setEnabled(False)
