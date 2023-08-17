import napari
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.prediction.model import PredictionModel
from allencell_ml_segmenter.core.subscriber import Subscriber
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.readers import TiffReader


class ResultDisplayService(Subscriber):
    def __init__(self, prediction_model: PredictionModel, viewer: napari.Viewer):
        super().__init__()
        self._model = prediction_model
        self._viewer = viewer
        self._model.subscribe(
            Event.PROCESS_PREDICTION_COMPLETE,
            self,
            lambda x: self.grab_and_display_results(),
        )

    def grab_and_display_results(self):
        output_dir: Path = self._model.get_output_directory()
        if output_dir is None:
            raise ValueError("No output directory to grab images from.")
        else:
            # unsanitized list of all files in output folder
            files = self.grab_files_from_folder(output_dir)
            for idx, file in enumerate(files):
                try:
                    image = AICSImage(str(file), reader=TiffReader)
                    self.add_image_to_viewer(image.data, f"Segmentation {str(idx)}")
                except Exception as e:
                    print(e)
                    print(f"Could not load image {str(file)} into napari viewer. Image cannot be opened by AICSImage")

    def grab_files_from_folder(self, path: Path):
        allfiles = path.glob('**/*')
        return [x for x in allfiles if x.is_file()]

    def add_image_to_viewer(self, image: AICSImage, display_name: str):
        self._viewer.add_image(image, name=display_name)

