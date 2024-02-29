from qtpy.QtCore import Qt

from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.prediction.file_input_widget import (
    PredictionFileInput,
)
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.prediction.model_input_widget import (
    ModelInputWidget,
)
from allencell_ml_segmenter.prediction.prediction_folder_progress_tracker import (
    PredictionFolderProgressTracker,
)
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QPushButton,
    QFrame,
    QLabel,
)
from napari.viewer import Viewer


class PredictionView(View):
    """
    Holds the image and model input widgets for prediction.
    """

    def __init__(
        self,
        main_model: MainModel,
        prediction_model: PredictionModel,
        viewer: Viewer,
    ):
        super().__init__()
        self._main_model: MainModel = main_model
        self._prediction_model: PredictionModel = prediction_model
        self._viewer: Viewer = viewer

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
            self._prediction_model, self._viewer
        )
        self._file_input_widget.setObjectName("fileInput")

        self._model_input_widget: ModelInputWidget = ModelInputWidget(
            self._prediction_model
        )
        self._model_input_widget.setObjectName("modelInput")

        # Dummy divs allow for easy alignment
        top_container: QVBoxLayout = QVBoxLayout()
        top_dummy: QFrame = QFrame()
        bottom_container: QVBoxLayout = QVBoxLayout()
        bottom_dummy: QFrame = QFrame()

        top_container.addWidget(self._file_input_widget)
        top_dummy.setLayout(top_container)
        self.layout().addWidget(top_dummy)

        bottom_container.addWidget(self._model_input_widget)
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
        # dispatch events to set _prediction_model._input_image_path to a real CSV here

        # replace the hard-coded 2 with function on _prediction_model to get number of
        # images in the CSV
        progress_tracker: PredictionFolderProgressTracker = (
            PredictionFolderProgressTracker(
                self._prediction_model.get_output_directory(), 2
            )
        )
        self.startLongTaskWithProgressBar(progress_tracker)

    def doWork(self):
        self._prediction_model.dispatch_prediction_initiated()
        self._prediction_model.dispatch_prediction()
        # TODO Need way to set result images to show after prediction complete and refresh viewer.

    def getTypeOfWork(self):
        return "Prediction"

    def showResults(self):
        # TODO Need to implement way to show predicted images in napari.
        print("showResults - prediction")
