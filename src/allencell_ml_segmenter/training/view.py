from pathlib import Path
from typing import Optional

from allencell_ml_segmenter.core.dialog_box import DialogBox
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
from allencell_ml_segmenter.training.training_model import PatchSize
from allencell_ml_segmenter.training.metrics_csv_progress_tracker import (
    MetricsCSVProgressTracker,
)


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

        patch_size_label: LabelWithHint = LabelWithHint("Structure size")
        bottom_grid_layout.addWidget(patch_size_label, 0, 0)

        self._patch_size_combo_box: QComboBox = QComboBox()
        self._patch_size_combo_box.setObjectName("structureSizeComboBox")
        self._patch_size_combo_box.setCurrentIndex(-1)
        self._patch_size_combo_box.setPlaceholderText("Select an option")
        self._patch_size_combo_box.addItems(
            [patch.name.lower() for patch in PatchSize]
        )
        self._patch_size_combo_box.currentTextChanged.connect(
            lambda size: self._training_model.set_patch_size(size)
        )
        bottom_grid_layout.addWidget(self._patch_size_combo_box, 0, 1)

        model_size_label: LabelWithHint = LabelWithHint("Model size")
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
        progress_tracker: MetricsCSVProgressTracker = (
            MetricsCSVProgressTracker(
                self._experiments_model.get_metrics_csv_path(),
                self._training_model.get_num_epochs(),
                self._experiments_model.get_latest_metrics_csv_version() + 1,
            )
        )
        self.startLongTaskWithProgressBar(progress_tracker)

    def read_result_images(self, dir_to_grab: Path):
        output_dir: Path = dir_to_grab
        images = []
        if output_dir is None:
            raise ValueError("No output directory to grab images from.")
        else:
            files = [
                Path(output_dir) / file for file in Path.iterdir(dir_to_grab)
            ]
            for file in files:
                if Path.is_file(file) and file.name.lower().endswith(".tif"):
                    try:
                        images.append(AICSImage(str(file), reader=TiffReader))
                    except Exception as e:
                        print(e)
                        print(
                            f"Could not load image {str(file)} into napari viewer. Image cannot be opened by AICSImage"
                        )
        return images

    def add_image_to_viewer(self, image: AICSImage, display_name: str):
        self._viewer.add_image(image, name=display_name)

    # Abstract methods from View implementations #######################

    def doWork(self):
        """
        Starts training process
        """
        self._training_model.dispatch_training()
        # TODO uncomment- testing default segmentation.yaml through API
        # This is broken and needs to be fixed- images now saved to experiment folder
        #result_images = self.read_result_images(
        #    self._experiments_model.get_model_test_images_path(
        #        self._experiments_model.get_experiment_name()
        #    )
        #)
        #print("doWork - setting result images")
        #self._training_model.set_result_images(result_images)
        #print("doWork - done")
        #self._training_model.dispatch_refresh()

    def getTypeOfWork(self) -> str:
        """
        Returns string representation of training process
        """
        return "Training"

    def showResults(self):
        dialog_box = DialogBox(
            f"Training finished"
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
