from napari.utils.notifications import show_warning
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
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from qtpy.QtGui import QIntValidator
from allencell_ml_segmenter.training.training_progress_tracker import (
    TrainingProgressTracker,
)
from allencell_ml_segmenter.core.info_dialog_box import InfoDialogBox
from allencell_ml_segmenter.utils.file_utils import FileUtils


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
        self.layout().addWidget(top_dummy)

        # bottom half
        bottom_grid_layout = QGridLayout()

        patch_size_text_layout = QVBoxLayout()
        patch_size_text_layout.setSpacing(0)
        patch_size_label: LabelWithHint = LabelWithHint("Patch size")
        patch_size_label.set_hint(
            "Patch size to split images into during training. Should encompass the structure of interest and all dimensions should be evenly divisble by 4. If 2D, Z can be left blank."
        )
        patch_size_text_layout.addWidget(patch_size_label)
        guide_text: QLabel = QLabel("All values must be multiples of 4")
        guide_text.setObjectName("subtext")
        patch_size_text_layout.addWidget(guide_text)
        bottom_grid_layout.addLayout(patch_size_text_layout, 0, 0)
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

        bottom_grid_layout.addLayout(patch_size_entry_layout, 0, 1)

        model_size_label: LabelWithHint = LabelWithHint("Model size")
        model_size_label.set_hint(
            "Defines the complexity of the model - smaller models train faster, while large models train slower but may learn complex relationships better."
        )
        bottom_grid_layout.addWidget(model_size_label, 1, 0)

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
        bottom_grid_layout.addWidget(self._model_size_combo_box, 1, 1)

        image_dimensions_label: LabelWithHint = LabelWithHint(
            "Image dimension"
        )
        image_dimensions_label.set_hint("Dimensionality of your image data")
        bottom_grid_layout.addWidget(image_dimensions_label, 2, 0)

        dimension_choice_layout: QHBoxLayout = QHBoxLayout()
        dimension_choice_layout.setSpacing(0)

        self._radio_3d: QRadioButton = QRadioButton()
        self._radio_3d.setObjectName("3DRadio")
        self._radio_3d.toggled.connect(
            lambda: self._training_model.set_spatial_dims(3)
        )
        label_3d: LabelWithHint = LabelWithHint("3D")

        self._radio_2d: QRadioButton = QRadioButton()
        self._radio_2d.toggled.connect(
            lambda: self._training_model.set_spatial_dims(2)
        )
        label_2d: LabelWithHint = LabelWithHint("2D")

        dimension_choice_layout.addWidget(self._radio_3d)
        dimension_choice_layout.addWidget(label_3d)
        dimension_choice_layout.addWidget(
            self._radio_2d, alignment=Qt.AlignLeft
        )
        dimension_choice_layout.addWidget(label_2d, alignment=Qt.AlignLeft)
        dimension_choice_layout.addStretch(10)
        dimension_choice_layout.setContentsMargins(0, 0, 0, 0)

        dimension_choice_dummy: QWidget = (
            QWidget()
        )  # stops interference with other radio buttons
        dimension_choice_dummy.setLayout(dimension_choice_layout)

        bottom_grid_layout.addWidget(dimension_choice_dummy, 2, 1)

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

        # apply styling
        self.setStyleSheet(Style.get_stylesheet("training_view.qss"))

    def train_btn_handler(self) -> None:
        """
        Starts training process
        """
        if self._patch_size_ok():
            self.set_patch_size()

            progress_tracker: TrainingProgressTracker = (
                TrainingProgressTracker(
                    self._experiments_model.get_metrics_csv_path(),
                    self._experiments_model.get_cache_dir(),
                    self._training_model.get_num_epochs(),
                    self._training_model.get_total_num_images(),
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
        csv_path: Path = self._experiments_model.get_latest_metrics_csv_path()
        min_loss: Optional[float] = FileUtils.get_min_loss_from_csv(csv_path)
        dialog_box = InfoDialogBox(
            "Training finished -- Final loss: {:.3f}".format(min_loss)
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
