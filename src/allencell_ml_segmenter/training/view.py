from napari.utils.notifications import show_warning  # type: ignore
from pathlib import Path
from typing import Optional

from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
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
from allencell_ml_segmenter.core.view import View, MainWindow
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.training.image_selection_widget import (
    ImageSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    ModelSize,
)
from allencell_ml_segmenter.training.patch_size_validator import (
    PatchSizeValidator,
    PATCH_SIZE_MULTIPLE_OF,
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from qtpy.QtGui import QIntValidator
from allencell_ml_segmenter.training.training_progress_tracker import (
    TrainingProgressTracker,
)
from allencell_ml_segmenter.core.info_dialog_box import InfoDialogBox
from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.experiment_utils import ExperimentUtils


class TrainingView(View, MainWindow):
    """
    Holds widgets pertinent to training processes - ImageSelectionWidget & ModelSelectionWidget.
    """

    def __init__(
        self,
        main_model: MainModel,
        experiments_model: IExperimentsModel,
        training_model: TrainingModel,
        viewer: IViewer,
    ):
        super().__init__()

        self._viewer: IViewer = viewer

        self._main_model: MainModel = main_model
        self._experiments_model: IExperimentsModel = experiments_model
        self._training_model: TrainingModel = training_model

        layout: QVBoxLayout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )

        self._title: QLabel = QLabel("SEGMENTATION MODEL TRAINING", self)
        self._title.setObjectName("title")
        layout.addWidget(
            self._title, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop  # type: ignore
        )

        # initialize constituent widgets
        self.image_selection_widget: ImageSelectionWidget = (
            ImageSelectionWidget(self._training_model, self._experiments_model)
        )
        self.image_selection_widget.setObjectName("imageSelection")

        # Dummy divs allow for easy alignment
        top_container: QVBoxLayout = QVBoxLayout()
        top_dummy: QFrame = QFrame()
        bottom_dummy: QFrame = QFrame()
        top_container.addWidget(self.image_selection_widget)
        top_dummy.setLayout(top_container)
        layout.addWidget(top_dummy)

        # bottom half
        bottom_grid_layout: QGridLayout = QGridLayout()

        # Existing model label
        existing_model_label: LabelWithHint = LabelWithHint(
            "Start from previous model"
        )
        existing_model_label.set_hint(
            "Select a previously trained model to pull model weights from"
        )
        bottom_grid_layout.addWidget(
            existing_model_label, 0, 0, alignment=Qt.AlignmentFlag.AlignTop
        )

        # Model Size Selection
        self._model_size_combo_box: QComboBox = QComboBox()
        self._model_size_combo_box.setObjectName("modelSizeComboBox")
        self._model_size_combo_box.setCurrentIndex(-1)
        self._model_size_combo_box.setPlaceholderText("Select an option")
        self._model_size_combo_box.addItems(
            [size.name.lower() for size in ModelSize]
        )
        self._model_size_combo_box.currentTextChanged.connect(
            lambda model_size: self._training_model.set_model_size(model_size)
        )

        # Existing model selection
        self.existing_model_dropdown: QComboBox = QComboBox()
        self.existing_model_dropdown.addItems(
            self._experiments_model.get_experiments()
        )
        self.existing_model_dropdown.currentIndexChanged.connect(
            lambda: self._training_model.set_existing_model(
                self.existing_model_dropdown.currentText()
            )
        )
        self.existing_model_dropdown.setEnabled(
            False
        )  # Disabled by default since no is the default
        # Yes and no radio buttons stacked on top of each other in a VBoxLayout
        existing_model_selection_layout: QVBoxLayout = QVBoxLayout()
        self.existing_model_no_radio: QRadioButton = QRadioButton("No")
        self.existing_model_no_radio.toggled.connect(
            self._existing_model_no_radio_slot
        )
        self.existing_model_no_radio.setChecked(True)  # No enabled by default
        existing_model_selection_layout.addWidget(self.existing_model_no_radio)
        self.existing_model_yes_radio: QRadioButton = QRadioButton("Yes")
        self.existing_model_yes_radio.toggled.connect(
            self._existing_model_yes_radio_slot
        )

        # have model selection dropdown in-line with yes button
        existing_model_yes_layout: QHBoxLayout = QHBoxLayout()
        existing_model_yes_layout.addWidget(self.existing_model_yes_radio)
        existing_model_yes_layout.addWidget(
            self.existing_model_dropdown, stretch=1
        )
        existing_model_selection_layout.addLayout(existing_model_yes_layout)
        bottom_grid_layout.addLayout(existing_model_selection_layout, 0, 1)

        patch_size_text_layout = QVBoxLayout()
        patch_size_text_layout.setSpacing(0)
        patch_size_label: LabelWithHint = LabelWithHint("Patch size")
        patch_size_label.set_hint(
            "Patch size to split images into during training. Should encompass the structure of interest and all dimensions should be evenly divisble by 4. If 2D, Z can be left blank."
        )
        patch_size_text_layout.addWidget(patch_size_label)
        guide_text: QLabel = QLabel(
            f"All values must be multiples of {PATCH_SIZE_MULTIPLE_OF}"
        )
        guide_text.setObjectName("subtext")
        patch_size_text_layout.addWidget(guide_text)
        bottom_grid_layout.addLayout(patch_size_text_layout, 1, 0)
        patch_size_entry_layout: QHBoxLayout = QHBoxLayout()

        # allow only integers for the linedits below
        patch_validator: PatchSizeValidator = PatchSizeValidator()

        self.z_patch_size: QLineEdit = QLineEdit()
        self.z_patch_size.setValidator(patch_validator)
        patch_size_entry_layout.addWidget(QLabel("Z:"))
        patch_size_entry_layout.addWidget(self.z_patch_size)

        self.y_patch_size: QLineEdit = QLineEdit()
        self.y_patch_size.setValidator(patch_validator)
        patch_size_entry_layout.addWidget(QLabel("Y:"))
        patch_size_entry_layout.addWidget(self.y_patch_size)

        self.x_patch_size: QLineEdit = QLineEdit()
        self.x_patch_size.setValidator(patch_validator)
        patch_size_entry_layout.addWidget(QLabel("X:"))
        patch_size_entry_layout.addWidget(self.x_patch_size)

        self._update_spatial_dims_boxes()
        bottom_grid_layout.addLayout(patch_size_entry_layout, 1, 1)

        model_size_label: LabelWithHint = LabelWithHint("Model size")
        model_size_label.set_hint(
            "Defines the complexity of the model - smaller models train faster, while large models train slower but may learn complex relationships better."
        )
        bottom_grid_layout.addWidget(model_size_label, 2, 0)
        bottom_grid_layout.addWidget(self._model_size_combo_box, 2, 1)
        num_epochs_label: LabelWithHint = LabelWithHint("Number of epochs")
        num_epochs_label.set_hint(
            "Each epoch is one complete pass through the entire training dataset. More epochs yields better performance at the cost of training time."
        )
        bottom_grid_layout.addWidget(num_epochs_label, 3, 0)

        self._num_epochs_input: QLineEdit = QLineEdit()
        int_validator: QIntValidator = QIntValidator()
        int_validator.setBottom(1)
        self._num_epochs_input.setValidator(int_validator)
        self._num_epochs_input.setPlaceholderText("1000")
        self._num_epochs_input.setObjectName("trainingStepInput")
        self._num_epochs_input.textChanged.connect(
            self._num_epochs_field_handler
        )
        bottom_grid_layout.addWidget(self._num_epochs_input, 3, 1)

        max_time_layout: QHBoxLayout = QHBoxLayout()
        max_time_layout.setSpacing(0)

        self._max_time_checkbox: QCheckBox = QCheckBox()
        self._max_time_checkbox.setObjectName("timeoutCheckbox")
        self._max_time_checkbox.stateChanged.connect(
            self._max_time_checkbox_slot
        )
        max_time_layout.addWidget(self._max_time_checkbox)

        max_time_left_text: QLabel = QLabel("Time out after")
        max_time_layout.addWidget(max_time_left_text)

        self._max_time_in_minutes_input: QLineEdit = QLineEdit()
        self._max_time_in_minutes_input.setObjectName("timeoutMinuteInput")
        self._max_time_in_minutes_input.setEnabled(False)
        self._max_time_in_minutes_input.setMaximumWidth(30)
        self._max_time_in_minutes_input.setPlaceholderText("30")
        self._max_time_in_minutes_input.textChanged.connect(
            lambda text: self._training_model.set_max_time(int(text))
        )
        max_time_layout.addWidget(self._max_time_in_minutes_input)

        max_time_right_text: LabelWithHint = LabelWithHint("minutes")
        max_time_right_text.set_hint("(Optional) Maximum time to train model")
        max_time_layout.addWidget(
            max_time_right_text, alignment=Qt.AlignmentFlag.AlignLeft
        )
        max_time_layout.addStretch()
        bottom_grid_layout.addLayout(max_time_layout, 4, 1)
        bottom_grid_layout.setColumnStretch(1, 8)
        bottom_grid_layout.setColumnStretch(0, 3)

        bottom_dummy.setLayout(bottom_grid_layout)
        layout.addWidget(bottom_dummy)

        self._train_btn: QPushButton = QPushButton("Start training")
        self._train_btn.setObjectName("trainBtn")
        layout.addWidget(self._train_btn)
        self._train_btn.clicked.connect(self.train_btn_handler)

        self._main_model.subscribe(
            Event.VIEW_SELECTION_TRAINING,
            self,
            lambda e: self._main_model.set_current_view(self),
        )
        self._training_model.signals.spatial_dims_set.connect(
            self._update_spatial_dims_boxes
        )

        # apply styling
        self.setStyleSheet(Style.get_stylesheet("training_view.qss"))

    def train_btn_handler(self) -> None:
        """
        Starts training process
        """
        if self._patch_size_ok():
            self.set_patch_size()
            num_epochs: Optional[int] = self._training_model.get_num_epochs()
            metrics_csv_path: Optional[Path] = (
                self._experiments_model.get_metrics_csv_path()
            )
            cache_dir: Optional[Path] = self._experiments_model.get_cache_dir()
            if metrics_csv_path is None or cache_dir is None:
                raise ValueError(
                    "Metrics CSV path and cache_dir cannot be None after training has started!"
                )
            progress_tracker: TrainingProgressTracker = (
                TrainingProgressTracker(
                    metrics_csv_path,
                    cache_dir,
                    num_epochs if num_epochs is not None else 0,
                    self._training_model.get_total_num_images(),
                    self._experiments_model.get_latest_metrics_csv_version()
                    + 1,
                )
            )
            self.startLongTaskWithProgressBar(progress_tracker)

    # Abstract methods from View implementations #######################

    def doWork(self) -> None:
        """
        Starts training process
        """
        self._training_model.dispatch_training()

    def getTypeOfWork(self) -> str:
        """
        Returns string representation of training process
        """
        return "Training"

    def showResults(self) -> None:
        # double check to see if a ckpt was generated
        exp_path: Optional[Path] = (
            self._experiments_model.get_user_experiments_path()
        )
        if exp_path is None:
            raise ValueError(
                "Experiments path should not be None after training complete."
            )
        exp_name: Optional[str] = self._experiments_model.get_experiment_name()
        if exp_name is None:
            raise ValueError(
                "Experiment name should not be None after training complete."
            )
        ckpt_generated: Optional[Path] = ExperimentUtils.get_best_ckpt(
            exp_path,
            exp_name,
        )
        if ckpt_generated is not None:
            # if model was successfully trained, get metrics to display
            csv_path: Optional[Path] = (
                self._experiments_model.get_latest_metrics_csv_path()
            )
            if csv_path is None:
                raise RuntimeError("Cannot get min loss from undefined csv")
            min_loss: Optional[float] = FileUtils.get_min_loss_from_csv(
                csv_path
            )
            if min_loss is None:
                raise RuntimeError("Cannot compute min loss")

            dialog_box = InfoDialogBox(
                "Training finished -- Final loss: {:.3f}".format(min_loss)
            )
            dialog_box.exec()  # this shows the dialog box
            self._main_model.training_complete()  # this dispatches the event that changes to prediction tab
        else:
            dialog_box = InfoDialogBox(
                "Training failed- no model was saved from this run."
            )
            dialog_box.exec()

    def _num_epochs_field_handler(self, num_epochs: str) -> None:
        self._training_model.set_num_epochs(int(num_epochs))

    def _max_time_checkbox_slot(self, checked: Qt.CheckState) -> None:
        """
        Triggered when the user selects the "time out after" _timeout_checkbox.
        Enables/disables interaction with the neighboring hour input based on checkstate.
        """
        if checked == Qt.CheckState.Checked:
            self._max_time_in_minutes_input.setEnabled(True)
            self._training_model.set_use_max_time(True)
        else:
            self._max_time_in_minutes_input.setEnabled(False)
            self._training_model.set_use_max_time(False)

    def _patch_size_ok(self) -> bool:
        """
        Validates that user has defined all needed patch sizes in the UI
        Returns True if all needed patch sizes were provided, False if not (and will show napari error message)
        """
        missing_patches: list[str] = []

        if (
            not self.z_patch_size.text()
            and self._training_model.get_spatial_dims() == 3
        ):
            missing_patches.append("Z")

        if not self.y_patch_size.text():
            missing_patches.append("Y")

        if not self.x_patch_size.text():
            missing_patches.append("X")

        if len(missing_patches) > 0:
            show_warning(
                f"Please define {missing_patches} patches before continuing."
            )
            return False

        return True

    def set_patch_size(self) -> None:
        self._training_model.set_patch_size(
            [
                int(self.z_patch_size.text()),
                int(self.y_patch_size.text()),
                int(self.x_patch_size.text()),
            ]
        )

    def focus_changed(self) -> None:
        self.image_selection_widget.set_inputs_csv()
        self._viewer.clear_layers()

    def _disable_model_size_combo_box(self) -> None:
        self._model_size_combo_box.setEnabled(False)
        self._model_size_combo_box.setCurrentIndex(
            -1
        )  # clear any selection, if any  # clear model

    def _enable_model_size_combo_box(self) -> None:
        self._model_size_combo_box.setEnabled(True)

    def _existing_model_no_radio_slot(self) -> None:
        # disable dropdown
        self.existing_model_dropdown.setEnabled(False)
        # reset dropdown and selected model
        self.existing_model_dropdown.setCurrentIndex(-1)
        self._training_model.set_is_using_existing_model(False)
        self._training_model.set_existing_model(None)
        self._enable_model_size_combo_box()

    def _existing_model_yes_radio_slot(self) -> None:
        # enable dropdown
        self._training_model.set_is_using_existing_model(True)
        self.existing_model_dropdown.setEnabled(True)
        self._disable_model_size_combo_box()

    def _update_spatial_dim_box(
        self, line_edit: QLineEdit, should_be_enabled: bool
    ) -> None:
        if should_be_enabled:
            line_edit.setEnabled(True)
            line_edit.setPlaceholderText(None)
        else:
            line_edit.setEnabled(False)
            line_edit.setPlaceholderText("N/A")

    def _update_spatial_dims_boxes(self) -> None:
        spatial_dims: Optional[int] = self._training_model.get_spatial_dims()
        self._update_spatial_dim_box(
            self.x_patch_size, spatial_dims is not None
        )
        self._update_spatial_dim_box(
            self.y_patch_size, spatial_dims is not None
        )
        self._update_spatial_dim_box(self.z_patch_size, spatial_dims == 3)
