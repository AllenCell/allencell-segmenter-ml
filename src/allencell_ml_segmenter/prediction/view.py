from pathlib import Path
from typing import Optional
import numpy as np
from qtpy.QtCore import Qt

from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.core.file_input_widget import (
    FileInputWidget,
    FileInputModel,
)
from allencell_ml_segmenter.prediction.model import (
    PredictionModel,
)

from allencell_ml_segmenter.core.file_input_model import InputMode
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.core.view import View, MainWindow
from allencell_ml_segmenter.prediction.file_write_progress_tracker import (
    FileWriteProgressTracker,
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
        file_input_model: FileInputModel,
        viewer: IViewer,
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
    ):
        super().__init__()
        self._main_model: MainModel = main_model
        self._prediction_model: PredictionModel = prediction_model
        self._file_input_model: FileInputModel = file_input_model
        self._viewer: IViewer = viewer
        self._img_data_extractor = img_data_extractor

        self._service: ModelFileService = ModelFileService(
            self._file_input_model
        )

        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )

        self._title: QLabel = QLabel("SEGMENTATION PREDICTION", self)
        self._title.setObjectName("title")
        layout.addWidget(self._title, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._file_input_widget: FileInputWidget = FileInputWidget(
            self._file_input_model, self._viewer, self._service
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
        layout.addWidget(top_dummy)

        # Disabled for V1 chrishu 3/30/24 https://github.com/AllenCell/allencell-ml-segmenter/issues/274
        # bottom_container.addWidget(self._model_input_widget)

        bottom_dummy.setLayout(bottom_container)
        layout.addWidget(bottom_dummy)

        self._run_btn: QPushButton = QPushButton("Run")
        self._run_btn.setObjectName("run")
        layout.addWidget(self._run_btn)
        self._run_btn.clicked.connect(self.run_btn_handler)

        self.setStyleSheet(Style.get_stylesheet("prediction_view.qss"))

        self._main_model.subscribe(
            Event.PROCESS_TRAINING_COMPLETE,
            self,
            lambda e: self._main_model.set_current_view(self),
        )

    def run_btn_handler(self) -> None:
        # dispatch events to set _prediction_model._input_image_path to a real CSV

        # get image paths from napari if they are selected
        self._prediction_model.dispatch_prediction_get_image_paths_from_napari()
        # Verify prediction is able to start, and write csv if needed
        self._prediction_model.dispatch_prediction_setup()

        total_num_images: Optional[int] = (
            self._prediction_model.get_total_num_images()
        )
        output_seg_dir: Optional[Path] = (
            self._file_input_model.get_output_seg_directory()
        )
        if total_num_images is not None and output_seg_dir is not None:
            progress_tracker: FileWriteProgressTracker = (
                FileWriteProgressTracker(
                    output_seg_dir,
                    total_num_images,
                )
            )
            self.startLongTaskWithProgressBar(progress_tracker)

    def doWork(self) -> None:
        self._prediction_model.dispatch_prediction()
        # TODO Need way to set result images to show after prediction complete and refresh viewer.

    def getTypeOfWork(self) -> str:
        return "Prediction"

    def showResults(self) -> None:
        output_path: Optional[Path] = (
            self._file_input_model.get_output_seg_directory()
        )

        # Display images if prediction inputs are from Napari Layers
        if (
            self._file_input_model.get_input_mode()
            == InputMode.FROM_NAPARI_LAYERS
            and output_path is not None
        ):
            raw_imgs: Optional[list[Path]] = (
                self._file_input_model.get_selected_paths()
            )
            segmentations: list[Path] = (
                FileUtils.get_all_files_in_dir_ignore_hidden(output_path)
            )
            channel: Optional[int] = (
                self._file_input_model.get_image_input_channel_index()
            )
            if raw_imgs is None or channel is None:
                raise RuntimeError("Insufficient data to show results")

            # here, we will pair raw images and segmentations based on the stem component of their paths
            stem_to_data: dict[str, dict[str, Path]] = {
                raw_img.stem: {"raw": raw_img} for raw_img in raw_imgs
            }

            if segmentations:
                for seg in segmentations:
                    # ignore files in the folder that aren't from most recent predictions
                    if seg.stem in stem_to_data:
                        stem_to_data[seg.stem]["seg"] = seg

                self._viewer.clear_layers()
                for data in stem_to_data.values():
                    raw_np_data: Optional[np.ndarray] = (
                        self._img_data_extractor.extract_image_data(
                            data["raw"], channel=channel
                        ).np_data
                    )
                    seg_np_data: Optional[np.ndarray] = (
                        self._img_data_extractor.extract_image_data(
                            data["seg"], seg=1
                        ).np_data
                    )
                    if raw_np_data is not None:
                        self._viewer.add_image(
                            raw_np_data,
                            name=f"[raw] {data['raw'].name}",
                            metadata={"source_path": str(data["raw"])},
                        )
                    if seg_np_data is not None:
                        self._viewer.add_labels(
                            seg_np_data,
                            name=f"[seg] {data['seg'].name}",
                            metadata={
                                "source_path": data["seg"],
                                "prob_map": self._img_data_extractor.extract_image_data(
                                    data["seg"]
                                ).np_data,
                            },
                        )

                    self._main_model.set_predictions_in_viewer(True)
        # Display popup with saved images path if prediction inputs are from a directory
        else:
            dialog_box = DialogBox(
                f"Predicted images saved to {str(output_path)}. \nWould you like to open this folder?"
            )
            dialog_box.exec()
            if dialog_box.get_selection() and output_path is not None:
                FileUtils.open_directory_in_window(output_path)

        self._main_model.dispatch_prediction_complete()

    def focus_changed(self) -> None:
        self._viewer.clear_layers()
