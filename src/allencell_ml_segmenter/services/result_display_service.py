import napari
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.core.subscriber import Subscriber
from pathlib import Path
import os


class ResultDisplayService(Subscriber):
    def __init__(self, prediction_model: PredictionModel, viewer: napari.Viewer):
        super().__init__()
        self._model = prediction_model
        self._model.subscribe(
            Event.PROCESS_PREDICTION_COMPLETE,
            self,
            self.grab_and_display_results,
        )

    def grab_and_display_results(self):
        output_dir: Path = self._model.get_output_directory()
        if output_dir is None:
            raise ValueError("No output directory to grab images from.")
        else:
            # unsanitized list of all files in output folder
            files = self.grab_files_from_folder(output_dir)

    def grab_files_from_folder(self, path: Path):
        return [x for x in os.listdir(path) if os.path.isfile(x)]

