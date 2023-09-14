from pathlib import Path
import sys
import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QFrame,
    QVBoxLayout,
    QSizePolicy,
)
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.services.training_service import TrainingService
from allencell_ml_segmenter.training.image_selection_widget import (
    ImageSelectionWidget,
)
from allencell_ml_segmenter.training.model_selection_widget import (
    ModelSelectionWidget,
)
from allencell_ml_segmenter.training.training_model import TrainingModel
from hydra.core.global_hydra import GlobalHydra
from aicsimageio import AICSImage
from aicsimageio.readers import TiffReader

import napari


class TrainingView(View):
    """
    Holds widgets pertinent to training processes - ImageSelectionWidget & ModelSelectionWidget.
    """

    def __init__(self, main_model: MainModel, viewer: napari.Viewer):
        super().__init__()

        self._viewer = viewer

        self._main_model: MainModel = main_model
        self._training_model: TrainingModel = TrainingModel(main_model)
        self._training_service: TrainingService = TrainingService(
            self._training_model
        )

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
            self._training_model
        )
        image_selection_widget.setObjectName("imageSelection")

        model_selection_widget: ModelSelectionWidget = ModelSelectionWidget(
            self._training_model
        )
        model_selection_widget.setObjectName("modelSelection")

        # Dummy divs allow for easy alignment
        top_container: QVBoxLayout = QVBoxLayout()
        top_dummy: QFrame = QFrame()
        bottom_container: QVBoxLayout = QVBoxLayout()
        bottom_dummy: QFrame = QFrame()

        top_container.addWidget(image_selection_widget)
        top_dummy.setLayout(top_container)
        self.layout().addWidget(top_dummy)

        bottom_container.addWidget(model_selection_widget)
        bottom_dummy.setLayout(bottom_container)
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
        ######### dirty HACKS#####
        sys.argv = [sys.argv[0]]
        GlobalHydra.instance().clear()
        # TODO -  find a better way to solve this
        self.startLongTask()
        # self._training_model.set_training_running(True)

        # self.doWork()
        # self.showResults()

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

    def doWork(self, test_demo):
        """
        Starts training process
        """
        #self._training_model.set_training_running(True)
        test_demo.set_prediction_running(True)

    def getTypeOfWork(self) -> str:
        """
        Returns string representation of training process
        """
        return "Training"

    def startLongTask(self):
        self.doWork()

    def showResults(self):
        viewer = napari.Viewer()
        viewer.show(self._training_model.get_result())
