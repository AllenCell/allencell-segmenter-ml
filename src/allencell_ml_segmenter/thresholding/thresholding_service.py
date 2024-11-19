from napari.layers import Layer

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.file_input_model import FileInputModel, InputMode
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.thresholding.thresholding_model import ThresholdingModel
from allencell_ml_segmenter.core.task_executor import NapariThreadTaskExecutor
from allencell_ml_segmenter.main.viewer import IViewer
from napari.viewer import Viewer
from cyto_dl.models.im2im.utils.postprocessing.auto_thresh import AutoThreshold
from cyto_dl.models.im2im.utils.postprocessing.act_thresh_label import ActThreshLabel

import numpy as np


class ThresholdingService(Subscriber):
    def __init__(self,
                 thresholding_model: ThresholdingModel,
                 experiments_model: ExperimentsModel,
                 file_input_model: FileInputModel,
                 viewer: IViewer):
        super().__init__()
        # Models
        self._thresholding_model: ThresholdingModel = thresholding_model
        self._experiments_model: ExperimentsModel = experiments_model
        self._file_input_model: FileInputModel = file_input_model

        # napari viewer
        self._viewer: IViewer = viewer

        # Task Executor
        self._task_executor: NapariThreadTaskExecutor = NapariThreadTaskExecutor.global_instance()

        self._original_layers: list[Layer] = []

        self._thresholding_model.subscribe(
            Event.ACTION_THRESHOLDING_VALUE_CHANGED,
            self,
            self._on_thresholding_value_changed,
        )

        self._thresholding_model.subscribe(
            Event.ACTION_THRESHOLDING_AUTOTHRESHOLDING_SELECTED,
            self,
            self._on_autothresholding_selected,
        )

    def _on_thresholding_value_changed(self, _: Event) -> None:
        # if we havent thresholded yet, keep track of original layers.
        if not self._original_layers:
            self._original_layers: list[Layer] = self._viewer.get_layers()

        # TODO: CHECK THIS THROUGH MAIN MODEL
        curr_img_layers: list[Layer] = self._viewer.get_seg_layers(self._original_layers)
        if not curr_img_layers:
            curr_img_layers = self._original_layers

        for layer in curr_img_layers:
            self._task_executor.exec(
                task=lambda: self._threshold_image(layer.data),
                # lambda functions capture variables by reference so need to pass layer as a default argument
                on_return=lambda thresholded_image, l_instance=layer: self._viewer.insert_segmentation(l_instance.name, thresholded_image),
                on_error=self._handle_thresholding_error,
            )

    def _handle_thresholding_error(self, error: Exception) -> None:
        Viewer.show_info("Thresholding failed: " + str(error))

    def _on_autothresholding_selected(self, _: Event) -> None:
        # if we havent thresholded yet, keep track of original layers.
        if not self._original_layers:
            self._original_layers: list[Layer] = self._viewer.get_layers()

        # TODO: CHECK THIS THROUGH MAIN MODEL
        curr_img_layers: list[Layer] = self._viewer.get_seg_layers(self._original_layers)
        if not curr_img_layers:
            curr_img_layers = self._original_layers
        auto_thresh_function = AutoThreshold("threshold_otsu")
        for layer in curr_img_layers:
            self._task_executor.exec(
                task=lambda: auto_thresh_function(layer.data),
                # lambda functions capture variables by reference so need to pass layer as a default argument
                on_return=lambda thresholded_image, l_instance=layer: self._viewer.insert_segmentation(l_instance.name,
                                                                                                       thresholded_image),
                on_error=self._handle_thresholding_error,
            )



    # TODO this probably belongs in cyto-dl, or at least its own separate class
    def _threshold_image(self, image: np.ndarray) -> np.ndarray:
        threshold_value: float = self._thresholding_model.get_thresholding_value()
        return (image > threshold_value).astype(int)



