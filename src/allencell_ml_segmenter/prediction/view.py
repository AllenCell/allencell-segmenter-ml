from pathlib import Path
from typing import List

from qtpy.QtCore import Qt

from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.file_input_widget import (
    PredictionFileInput,
)
from allencell_ml_segmenter.prediction.model import (
    PredictionModel,
    PredictionInputMode,
)
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.core.view import View, MainWindow
from allencell_ml_segmenter.prediction.model_input_widget import (
    ModelInputWidget,
)
from allencell_ml_segmenter.prediction.prediction_folder_progress_tracker import (
    PredictionFolderProgressTracker,
)
from allencell_ml_segmenter.utils.file_utils import FileUtils
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QPushButton,
    QFrame,
    QLabel,
)
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    AICSImageDataExtractor,
)


class PredictionView(View, MainWindow):
    """
    Holds the image and model input widgets for prediction.
    """

    def __init__(
        self,
        main_model: MainModel,
        prediction_model: PredictionModel,
        viewer: IViewer,
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
    ):
        super().__init__()
        self._main_model: MainModel = main_model
        self._prediction_model: PredictionModel = prediction_model
        self._viewer: IViewer = viewer
        self._img_data_extractor = img_data_extractor

        self._service: ModelFileService = ModelFileService(
            self._prediction_model
        )

        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.layout().setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._title: QLabel = QLabel("SEGMENTATION PREDICTION", self)
        self._title.setObjectName("title")
        self.layout().addWidget(self._title, alignment=Qt.AlignHCenter)

        self._file_input_widget: PredictionFileInput = PredictionFileInput(
            self._prediction_model, self._viewer, self._service
        )
        self._file_input_widget.setObjectName("fileInput")

        # Disabled for V1 chrishu 3/30/24 https://github.com/AllenCell/allencell-ml-segmenter/issues/274
        # self._model_input_widget: ModelInputWidget = ModelInputWidget(
        #     self._prediction_model
        # )
        # self._model_input_widget.setObjectName("modelInput")

        # Dummy divs allow for easy alignment
        top_container: QVBoxLayout = QVBoxLayout()
        top_dummy: QFrame = QFrame()
        bottom_container: QVBoxLayout = QVBoxLayout()
        bottom_dummy: QFrame = QFrame()

        top_container.addWidget(self._file_input_widget)
        top_dummy.setLayout(top_container)
        self.layout().addWidget(top_dummy)

        # Disabled for V1 chrishu 3/30/24 https://github.com/AllenCell/allencell-ml-segmenter/issues/274
        # bottom_container.addWidget(self._model_input_widget)

        bottom_dummy.setLayout(bottom_container)
        self.layout().addWidget(bottom_dummy)

        self._run_btn: QPushButton = QPushButton("Run")
        self._run_btn.setObjectName("run")
        self.layout().addWidget(self._run_btn)
        self._run_btn.clicked.connect(self.run_btn_handler)

        self.setStyleSheet(Style.get_stylesheet("prediction_view.qss"))

        self._main_model.subscribe(
            Event.PROCESS_TRAINING_COMPLETE,
            self,
            lambda e: self._main_model.set_current_view(self),
        )

    def run_btn_handler(self):
        # dispatch events to set _prediction_model._input_image_path to a real CSV

        # get image paths from napari if they are selected
        self._prediction_model.dispatch_prediction_get_image_paths_from_napari()
        # Verify prediction is able to start, and write csv if needed
        self._prediction_model.dispatch_prediction_setup()

        total_num_images = self._prediction_model.get_total_num_images()
        if total_num_images:
            progress_tracker: PredictionFolderProgressTracker = (
                PredictionFolderProgressTracker(
                    self._prediction_model.get_output_seg_directory(),
                    total_num_images,
                )
            )
            self.startLongTaskWithProgressBar(progress_tracker)

    def doWork(self):
        self._prediction_model.dispatch_prediction()
        # TODO Need way to set result images to show after prediction complete and refresh viewer.

    def getTypeOfWork(self):
        return "Prediction"

    def showResults(self):
        output_path: Path = self._prediction_model.get_output_seg_directory()

        # Display images if prediction inputs are from Napari Layers
        if (
            self._prediction_model.get_prediction_input_mode()
            == PredictionInputMode.FROM_NAPARI_LAYERS
        ):
            raw_imgs: list[Path] = self._prediction_model.get_selected_paths()
            segmentations: list[Path] = (
                FileUtils.get_all_files_in_dir_ignore_hidden(output_path)
            )
            channel: int = (
                self._prediction_model.get_image_input_channel_index()
            )

            # here, we will pair raw images and segmentations based on the stem component of their paths
            stem_to_data: dict[str, dict[str, Path]] = {
                raw_img.stem: {"raw": raw_img} for raw_img in raw_imgs
            }
            for seg in segmentations:
                # ignore files in the folder that aren't from most recent predictions
                if seg.stem in stem_to_data:
                    stem_to_data[seg.stem]["seg"] = seg

            self._viewer.clear_layers()
            for data in stem_to_data.values():
                self._viewer.add_image(
                    self._img_data_extractor.extract_image_data(
                        data["raw"], channel=channel
                    ).np_data,
                    f"[raw] {data['raw'].name}",
                )
                self._viewer.add_labels(
                    self._img_data_extractor.extract_image_data(
                        data["seg"], seg=1
                    ).np_data,
                    name=f"[seg] {data['seg'].name}",
                )
        # Display popup with saved images path if prediction inputs are from a directory
        else:
            dialog_box = DialogBox(
                f"Predicted images saved to {str(output_path)}. \nWould you like to open this folder?"
            )
            dialog_box.exec()
            if dialog_box.get_selection():
                FileUtils.open_directory_in_window(output_path)

    def focus_changed(self):
        self._viewer.clear_layers()
