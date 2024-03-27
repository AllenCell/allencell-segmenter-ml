import napari
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.training.training_model import TrainingModel
from allencell_ml_segmenter.utils.file_utils import FileUtils
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.readers import TiffReader


class ResultDisplayService(Subscriber):
    def __init__(self, model: Publisher, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self._model = model

        # Based on model grab correct image directory to display and register event handler
        if isinstance(model, PredictionModel):
            self._model.subscribe(
                Event.PROCESS_PREDICTION_COMPLETE,
                self,
                lambda x: self.grab_and_display_results(
                    self._model.get_output_directory()
                ),
            )
        elif isinstance(model, TrainingModel):
            self._model.subscribe(
                Event.PROCESS_TRAINING_COMPLETE,
                self,
                # TODO change this to grab the correct directory
                lambda x: self.grab_and_display_results(
                    self._model.get_output_directory()
                ),
            )

    def grab_and_display_results(self, dir_to_grab: Path):
        output_dir: Path = dir_to_grab
        if output_dir is None:
            raise ValueError("No output directory to grab images from.")
        else:
            files = FileUtils.get_all_files_in_dir(
                output_dir, ignore_hidden=True
            )
            for idx, file in enumerate(files):
                try:
                    image = AICSImage(str(file), reader=TiffReader)
                    self.add_image_to_viewer(
                        image.data, f"Segmentation {str(idx)}"
                    )
                except Exception as e:
                    print(e)
                    print(
                        f"Could not load image {str(file)} into napari viewer. Image cannot be opened by AICSImage"
                    )

    def add_image_to_viewer(self, image: AICSImage, display_name: str):
        self._viewer.add_image(image, name=display_name)
