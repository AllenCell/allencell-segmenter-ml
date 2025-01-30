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
        self._prediction_layers: list[LabelsLayer] = []

    def _update_layer_list(self) -> None:
        previous_selections: list[int] = self._image_list.get_checked_rows()
        self._image_list.clear()
        self._prediction_layers = self._viewer.get_all_segmentation_labels()
        for idx, prediction_output_layer in enumerate(self._prediction_layers):
            self._image_list.add_item(
                prediction_output_layer.name,
                set_checked=idx in previous_selections
            )


    def process_checked_signal(self, row: int, state: Qt.CheckState) -> None:
        if self._model.get_input_mode() == InputMode.FROM_NAPARI_LAYERS:
            self._model.set_selected_idx(self._image_list.get_checked_rows())
