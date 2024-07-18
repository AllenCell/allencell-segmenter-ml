from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.core.event import Event
from enum import Enum
from typing import Union, Optional, List
from pathlib import Path
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel, ImageType
from qtpy.QtCore import QObject, Signal


class TrainingType(Enum):
    """
    Different cyto-dl experiment types
    """

    SEGMENTATION_PLUGIN = "segmentation_plugin"
    SEGMENTATION = "segmentation"
    GAN = "gan"
    OMNIPOSE = "omnipose"
    SKOOTS = "skoots"


class ModelSize(Enum):
    """
    Model size for training, and their respective filters overrides
    """

    SMALL = [8, 16, 32]
    MEDIUM = [16, 32, 64]
    LARGE = [32, 64, 128]


# TODO: move these signals directly into TrainingModel once we deprecate
# 'Publisher' in a refactor
class TrainingModelSignals(QObject):
    num_channels_set: Signal = Signal()
    images_directory_set: Signal = Signal()


class TrainingModel(Publisher):
    """
    Stores state relevant to training processes.
    """

    def __init__(
        self, main_model: MainModel, experiments_model: ExperimentsModel
    ):
        super().__init__()
        self._main_model = main_model
        self.experiments_model = experiments_model
        self.signals: TrainingModelSignals = TrainingModelSignals()
        self._experiment_type: TrainingType = None
        self._images_directory: Optional[Path] = None
        self._selected_channel: dict[ImageType, Optional[int]] = {
            t: None for t in ImageType
        }
        self._num_channels: dict[ImageType, Optional[int]] = {
            t: None for t in ImageType
        }
        self._model_path: Union[Path, None] = (
            None  # if None, start a new model
        )
        self._patch_size: List[int] = None
        self._spatial_dims: int = None
        self._num_epochs: int = None
        self._max_time: int = None  # in minutes
        self._config_dir: Path = None
        self._max_channel = None
        self._use_max_time: bool = (
            False  # default is false. UI starts with max epoch defined rather than max time.
        )
        self._model_size: Optional[ModelSize] = None
        # the total number of images used for training/test/validation for this model
        self._total_num_images: int = 0

    def get_experiment_type(self) -> Optional[str]:
        """
        Gets experiment type
        """
        if self._experiment_type is None:
            return None
        return self._experiment_type.value

    def set_experiment_type(self, training_type: str) -> None:
        """
        Sets experiment type

        training_type (str): name of cyto-dl experiment to run
        """
        # convert string to enum
        self._experiment_type = TrainingType(training_type)

    def get_spatial_dims(self) -> int:
        """
        Gets image dimensions
        """
        return self._spatial_dims

    def set_spatial_dims(self, spatial_dims: int) -> None:
        """
        Sets image dimensions

        image_dims (int): number of dimensions to train model on. "2" for 2D, "3" for 3D
        """
        if spatial_dims != 2 and spatial_dims != 3:
            raise ValueError("No support for non 2D and 3D images.")
        self._spatial_dims = spatial_dims

    def get_num_epochs(self) -> int:
        """
        Gets max epoch
        """
        return self._num_epochs

    def set_num_epochs(self, num_epochs: int) -> None:
        """
        Sets num epochs

        num_epochs (int): number of additional epochs to train for
        """
        self._num_epochs = num_epochs

    def get_images_directory(self) -> Optional[Path]:
        """
        Gets images directory
        """
        return self._images_directory

    def set_images_directory(self, images_path: Optional[Path]) -> None:
        """
        Sets images directory, and dispatches channel extraction

        images_path (Path): path to images directory
        """
        self._images_directory = images_path
        self.signals.images_directory_set.emit()

    def get_num_channels(self, image_type: ImageType) -> Optional[int]:
        return self._num_channels[image_type]

    def set_all_num_channels(
        self, num_channels: dict[ImageType, Optional[int]]
    ) -> None:
        self._num_channels = num_channels
        self.signals.num_channels_set.emit()

    def get_selected_channel(self, image_type: ImageType) -> Optional[int]:
        return self._selected_channel[image_type]

    def set_selected_channel(
        self, image_type: ImageType, channel: int
    ) -> None:
        self._selected_channel[image_type] = channel

    def get_patch_size(self) -> List[int]:
        """
        Gets patch size
        """
        return self._patch_size

    def set_patch_size(self, patch_size: List[int]) -> None:
        """
        Sets patch size

        patch_size (str): patch size for training
        """
        if len(patch_size) not in [2, 3]:
            raise ValueError(
                "Patch sizes need to be 2 or 3 dimension based on input image."
            )
        self._patch_size = patch_size

    def get_max_time(self) -> int:
        """
        Gets max runtime (in seconds)
        """
        return self._max_time

    def set_max_time(self, max_time: int) -> None:
        """
        Sets max runtime (in seconds)

        max_time (int): maximum runtime for training, in seconds
        """
        self._max_time = max_time

    def get_config_dir(self) -> Path:
        """
        Gets config directory
        """
        return self._config_dir

    def set_config_dir(self, config_dir: Path) -> None:
        """
        Sets config directory

        config_dir (str): path to config directory
        """
        self._config_dir = config_dir

    def dispatch_training(self) -> None:
        """
        Dispatches even to start training
        """
        self.dispatch(Event.PROCESS_TRAINING)

    def use_max_time(self) -> bool:
        """
        Will training run will be based off of max time
        """
        return self._use_max_time

    def set_use_max_time(self, use_max: bool):
        """
        Set if training run will be based off of max time
        """
        self._use_max_time = use_max

    def set_model_size(self, model_size: str) -> None:
        # convert string to enum
        model_size = model_size.upper()
        if model_size not in [x.name for x in ModelSize]:
            raise ValueError(
                "No support for non small, medium, and large patch sizes."
            )
        self._model_size = ModelSize[model_size]

    def get_model_size(self) -> ModelSize:
        return self._model_size

    def set_total_num_images(self, num_images: int) -> None:
        self._total_num_images = num_images

    def get_total_num_images(self) -> int:
        return self._total_num_images

    def get_selected_channels(self) -> dict[ImageType, Optional[int]]:
        return self._main_model.get_selected_channels()
