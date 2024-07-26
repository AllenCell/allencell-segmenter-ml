from pathlib import Path

from allencell_ml_segmenter.core.channel_extraction import (
    get_img_path_from_csv,
)
from allencell_ml_segmenter.core.image_data_extractor import (
    IImageDataExtractor,
    AICSImageDataExtractor,
    ImageData,
)
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event

from cyto_dl.api.model import CytoDLModel
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.training.training_model import (
    TrainingModel,
    ImageType,
)
from typing import Optional
from napari.utils.notifications import show_warning
from allencell_ml_segmenter.utils.cyto_overrides_manager import (
    CytoDLOverridesManager,
)
from allencell_ml_segmenter.utils.file_utils import FileUtils
from napari.utils.notifications import show_error
from allencell_ml_segmenter.core.task_executor import (
    ITaskExecutor,
    NapariThreadTaskExecutor,
)
from collections import namedtuple
from allencell_ml_segmenter.main.main_model import MIN_DATASET_SIZE


DirectoryData = namedtuple(
    "DirectoryData",
    ["num_images", "raw_channels", "seg1_channels", "seg2_channels"],
)


class TrainingService(Subscriber):
    """
    Interface for training a model. Uses cyto-dl to train model according to spec
    """

    def __init__(
        self,
        training_model: TrainingModel,
        experiments_model: ExperimentsModel,
        img_data_extractor: IImageDataExtractor = AICSImageDataExtractor.global_instance(),
        task_executor: ITaskExecutor = NapariThreadTaskExecutor.global_instance(),
    ):
        super().__init__()
        self._training_model: TrainingModel = training_model
        self._experiments_model: ExperimentsModel = experiments_model
        self._task_executor: ITaskExecutor = task_executor
        self._training_model.subscribe(
            Event.PROCESS_TRAINING,
            self,
            self._train_model_handler,
        )
        self._img_data_extractor: IImageDataExtractor = img_data_extractor
        self._training_model.signals.images_directory_set.connect(
            self._training_image_directory_selected
        )

    def _train_model_handler(self, _: Event) -> None:
        """
        Trains the model according to the spec
        """
        # Only supporting segmentation config for now, in the future this will be an option in the UI
        self._training_model.set_experiment_type("segmentation_plugin")
        # TODO make set_images_directory and get_images_directory less brittle.
        #  https://github.com/AllenCell/allencell-ml-segmenter/issues/156
        if self._able_to_continue_training():
            model = CytoDLModel()
            model.load_default_experiment(
                self._training_model.get_experiment_type(),
                output_dir=f"{self._experiments_model.get_user_experiments_path()}/{self._experiments_model.get_experiment_name()}",
            )
            cyto_overrides_manager: CytoDLOverridesManager = (
                CytoDLOverridesManager(
                    self._experiments_model, self._training_model
                )
            )
            model.override_config(
                cyto_overrides_manager.get_training_overrides()
            )
            model.print_config()
            model.save_config(
                self._experiments_model.get_train_config_path(
                    self._experiments_model.get_experiment_name()
                )
            )
            model.train()

    def _able_to_continue_training(self) -> bool:
        if self._experiments_model.get_experiment_name() is None:
            show_warning(
                "Please select an experiment before running prediction."
            )
            return False

        if self._training_model.get_experiment_type() is None:
            return False

        if self._training_model.get_spatial_dims() is None:
            show_warning(
                "Please select spatial dims for training dataset. 2-D or 3-D."
            )
            return False

        if self._training_model.get_images_directory() is None:
            show_warning("User has not selected input images for training")
            return False

        if self._training_model.get_patch_size() is None:
            show_warning("User has not selected a patch size for training")
            return False

        if self._training_model.get_num_epochs() is None:
            show_warning("Please define max epoch(s) to run for")
            return False

        if self._training_model.get_model_size() is None:
            show_warning("Please define model size.")
            return False
        return True

    def _extract_data_from_training_dir(
        self, training_dir: Path
    ) -> DirectoryData:
        num_imgs: int = FileUtils.count_images_in_csv_folder(training_dir)
        if num_imgs < MIN_DATASET_SIZE:
            raise RuntimeError(
                f"Training requires at least {MIN_DATASET_SIZE} images and their segmentations"
            )

        training_csv: Path = training_dir / "train.csv"
        raw_data: ImageData = self._img_data_extractor.extract_image_data(
            get_img_path_from_csv(training_csv, column="raw"), np_data=False
        )
        seg1_data: ImageData = self._img_data_extractor.extract_image_data(
            get_img_path_from_csv(training_csv, column="seg1"), np_data=False
        )
        seg2_path: Path = get_img_path_from_csv(training_csv, "seg2")
        seg2_data: Optional[ImageData] = (
            self._img_data_extractor.extract_image_data(
                seg2_path, np_data=False
            )
            if seg2_path
            else None
        )
        return DirectoryData(
            num_imgs,
            raw_data.channels,
            seg1_data.channels,
            seg2_data.channels if seg2_data else None,
        )

    def _on_training_dir_data_extracted(self, dir_data: DirectoryData) -> None:
        self._training_model.set_total_num_images(dir_data.num_images)
        self._training_model.set_all_num_channels(
            {
                ImageType.RAW: dir_data.raw_channels,
                ImageType.SEG1: dir_data.seg1_channels,
                ImageType.SEG2: dir_data.seg2_channels,
            }
        )

    def _on_training_dir_data_error(self, e: Exception) -> None:
        self._training_model.set_total_num_images(None)
        self._training_model.set_all_num_channels(
            {
                ImageType.RAW: None,
                ImageType.SEG1: None,
                ImageType.SEG2: None,
            }
        )
        show_error(f"Failed to get data from training directory: {e}")

    def _training_image_directory_selected(self) -> None:
        training_dir: Optional[Path] = (
            self._training_model.get_images_directory()
        )
        if training_dir is not None:
            self._task_executor.exec(
                lambda: self._extract_data_from_training_dir(training_dir),
                on_return=self._on_training_dir_data_extracted,
                on_error=self._on_training_dir_data_error,
            )
