from allencell_ml_segmenter.core.file_input_model import (
    InputMode,
    FileInputModel,
)
from allencell_ml_segmenter.core.file_input_widget import FileInputWidget
from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.main.segmenter_layer import LabelsLayer
from qtpy.QtCore import Qt

from allencell_ml_segmenter.prediction.service import ModelFileService


class PredictionResultListWidget(FileInputWidget):
    """
    Widget containing a list of prediction results that are selectable for thresholding
    """

    def __init__(
        self, model: FileInputModel, viewer: IViewer, service: ModelFileService
    ):
        super().__init__(
            model, viewer, service, include_channel_selection=False
        )
        self._prediction_layers: list[LabelsLayer] = (
            self._viewer.get_all_segmentation_labels()
        )

    def _update_layer_list(self) -> None:
        self._image_list.clear()
        self._prediction_layers = self._viewer.get_all_segmentation_labels()
        for prediction_output_layer in self._prediction_layers:
            self._image_list.add_item(
                prediction_output_layer.name,
            )

    def process_checked_signal(self, row: int, state: Qt.CheckState) -> None:
        if self._model.get_input_mode() == InputMode.FROM_NAPARI_LAYERS:
            selected_indices: list[int] = self._image_list.get_checked_rows()
            if state == Qt.CheckState.Checked:
                # paths of images to be segmented, which will not be opened again because already in memory
                # but satisfies file_input_model state which determines if we have anything selected
                self._model.set_selected_paths(
                    [x.path for x in self._prediction_layers]
                )
