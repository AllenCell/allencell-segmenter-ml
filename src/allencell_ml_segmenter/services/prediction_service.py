import csv

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.prediction.model import (
    PredictionModel,
    PredictionInputMode,
)

from allencell_ml_segmenter.utils.cuda_util import CUDAUtils
from allencell_ml_segmenter.utils.file_utils import FileUtils

from pathlib import Path
from typing import Union, Dict, List, Optional

from cyto_dl.api.model import CytoDLModel
from napari.utils.notifications import show_warning


class PredictionService(Subscriber):
    """
    Interface for training a model or predicting using a model.
    Uses cyto-dl for model training and inference.
    #TODO create an ABC for cyto-service and have prediction_service inherit it
    """

    def __init__(
        self,
        prediction_model: PredictionModel,
        experiments_model: ExperimentsModel,
    ):
        super().__init__()
        self._prediction_model: PredictionModel = prediction_model
        self._experiments_model: ExperimentsModel = experiments_model

        self._prediction_model.subscribe(
            Event.PROCESS_PREDICTION,
            self,
            self._predict_model,
        )

        self._prediction_model.subscribe(
            Event.ACTION_PREDICTION_SETUP,
            self,
            self._prediction_setup,
        )

    def _predict_model(self, _: Event) -> None:
        """
        Predict segmentations using model according to spec
        """
        cyto_api: CytoDLModel = CytoDLModel()
        cyto_api.load_config_from_file(
            self._experiments_model.get_train_config_path(
                self._experiments_model.get_experiment_name()
            )
        )
        # We must override the config to set up predictions correctly
        cyto_api.override_config(
            self.build_overrides(
                self._experiments_model.get_experiment_name(),
                self._experiments_model.get_checkpoint(),
            )
        )
        cyto_api.predict()

    def _prediction_setup(self, _: Event):
        if self._able_to_continue_prediction():
            self._write_csv_for_prediction()

    def _able_to_continue_prediction(self) -> bool:
        # Check to see if experiment selected
        experiment_name: str = self._experiments_model.get_experiment_name()
        if experiment_name is None:
            show_warning(
                "Please select an experiment before running prediction."
            )
            return False

        # Check to see if training has occurred with the selected experiment.
        training_config: Path = self._experiments_model.get_train_config_path(
            experiment_name
        )
        if not training_config.exists():
            show_warning(
                f"Please train with the experiment: {experiment_name} before running a prediction."
            )
            return False

        # Check to see the user has specified a ckpt to use.
        if self._experiments_model.get_checkpoint() is None:
            show_warning(
                f"Please select a checkpoint to run predictions with."
            )
            return False

        # Check to see the user has specified an output folder to use.
        if self._prediction_model.get_output_directory() is None:
            show_warning(
                f"Please select an output folder to save predictions to."
            )
            return False
        return True

    def _write_csv_for_prediction(self) -> None:
        """
        If needed, write csv's for predictions, and set total number of images to be processed
        """
        # Check to see if user has selected an input mode
        input_mode_selected: PredictionInputMode = (
            self._prediction_model.get_prediction_input_mode()
        )
        if not input_mode_selected:
            show_warning(
                "Please select input images before running prediction."
            )
            # dont set state if we have an error in setup
        elif input_mode_selected == PredictionInputMode.FROM_PATH:
            self._prediction_model.set_total_num_images(
                self._setup_inputs_from_path()
            )
        elif input_mode_selected == PredictionInputMode.FROM_NAPARI_LAYERS:
            self._prediction_model.set_total_num_images(
                self._setup_inputs_from_napari()
            )

    def build_overrides(
        self, experiment_name: str, checkpoint: str
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Build an overrides list for the cyto-dl API containing the
        overrides requried to run predictions, formatted as cyto-dl expects.
        """
        overrides: Dict[str, Union[str, int, float, bool]] = dict()
        # Default overrides needed for prediction
        overrides["test"] = False
        overrides["train"] = False
        overrides["mode"] = "predict"
        overrides["task_name"] = "predict_task_from_app"
        # Need these overrides to load in csv's
        overrides["data.columns"] = ["raw", "split"]
        overrides["data.split_column"] = "split"

        # passing the experiment_name and checkpoint as params to this function ensures we have a model before
        # attempting to build the overrides dict for predictions
        overrides["ckpt_path"] = str(
            self._experiments_model.get_model_checkpoints_path(
                experiment_name=experiment_name, checkpoint=checkpoint
            )
        )
        overrides["data.path"] = str(
            self._prediction_model.get_input_image_path()
        )

        # overrides from model
        # if output_dir is not set, will default to saving in the experiment folder
        output_dir: Path = self._prediction_model.get_output_directory()
        if output_dir:
            overrides["paths.output_dir"] = str(output_dir)

        # if channel is not set, will default to same channel used to train
        channel: int = self._prediction_model.get_image_input_channel_index()
        if channel:
            overrides["data.transforms.predict.transforms[1].reader[0].C"] = (
                channel
            )

        # selecting hardware- GPU if available (and correct drivers installed),
        # CPU otherwise.
        if CUDAUtils.cuda_available():
            overrides["trainer.accelerator"] = "gpu"
        else:
            overrides["trainer.accelerator"] = "cpu"

        return overrides

    def write_csv_for_inputs(self, list_images: List[Path]) -> None:
        """
        write csv for inputs and return the total number of images
        """
        if self._experiments_model.get_csv_path() is not None:
            data_folder: Path = self._experiments_model.get_csv_path()
            data_folder.mkdir(parents=False, exist_ok=True)
            csv_path: Path = data_folder / "test_csv.csv"
            with open(csv_path, "w") as file:
                writer: csv.writer = csv.writer(file)
                writer.writerow(["", "raw", "split"])
                for i, path_of_image in enumerate(list_images):
                    writer.writerow([str(i), str(path_of_image), "test"])

            self._prediction_model.set_input_image_path(csv_path)

    def _setup_inputs_from_path(self) -> int:
        """
        setup inputs from path and return total number of images to predict
        """
        # User has selected a directory or a csv as input images
        input_path: Path = self._prediction_model.get_input_image_path()
        if input_path.is_dir():
            all_files = FileUtils.get_all_files_in_dir_ignore_hidden(
                input_path
            )
            # if input path selected is a directory, we need to manually write a CSV for cyto-dl
            self.write_csv_for_inputs(all_files)
            return len(all_files)
        elif input_path.suffix == ".csv":
            return self._grab_csv_data_rows(input_path)
        else:
            # This should not be possible with FileInputWidget- throw an error.
            raise ValueError(
                "Somehow the user has selected a non-csv/directory for input images. Should not be possible with FileInputWidget"
            )
        # if a csv is selected, do nothing

    def _grab_csv_data_rows(self, path: Path) -> int:
        file = open(path, "r+")
        reader = csv.reader(file)
        return len(list(reader)) - 1  # ignore header for rowcount

    def _setup_inputs_from_napari(self) -> Optional[int]:
        """
        setup inputs from napari layers and return total number of images to predict
        """
        # User has selected napari image layers as input images
        selected_paths_from_napari: List[Path] = (
            self._prediction_model.get_selected_paths()
        )
        if len(selected_paths_from_napari) < 1:
            # No image layers selected
            show_warning(
                "Please select at least 1 image from the napari layer before running prediction."
            )
            return None
        else:
            # If user selects input images from napari, we need to manually write a csv for cyto-dl
            self.write_csv_for_inputs(
                self._prediction_model.get_selected_paths()
            )
            return len(selected_paths_from_napari)
