from typing import Optional

from allencell_ml_segmenter.core.file_input_model import (
    InputMode,
    WidgetMode,
)
from allencell_ml_segmenter.core.file_input_widget import FileInputWidget
from allencell_ml_segmenter.main.i_viewer import IViewer
from qtpy.QtCore import Qt

from napari.utils.events import Event as NapariEvent  # type: ignore

from allencell_ml_segmenter.thresholding.thresholding_model import (
    ThresholdingModel,
)


class ThresholdingFileInputWidget(FileInputWidget):
    """
    Widget containing a list of prediction results that are selectable for thresholding
    """

    def __init__(
        self,
        model: ThresholdingModel,
        viewer: IViewer,
    ):
        super().__init__(model, viewer, None, False, WidgetMode.THRESHOLDING)
        self._model: ThresholdingModel = model

    def _update_layer_list(self, event: Optional[NapariEvent] = None) -> None:
        previous_selections: list[int] = self._image_list.get_checked_rows()
        self._image_list.clear()
        self._model.set_thresholding_layers(
            self._viewer.get_all_layers_containing_prob_map()
        )
        for idx, prediction_output_layer in enumerate(
            self._model.get_thresholding_layers()
        ):
            self._image_list.add_item(
                prediction_output_layer.name,
                set_checked=idx in previous_selections,
            )

    def _process_checked_signal(self, row: int, state: Qt.CheckState) -> None:
        if self._model.get_input_mode() == InputMode.FROM_NAPARI_LAYERS:
            self._model.set_selected_idx(self._image_list.get_checked_rows())
