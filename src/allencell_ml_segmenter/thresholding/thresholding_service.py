from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from napari.layers import Layer  # type: ignore
import numpy as np
from napari.utils.notifications import show_info  # type: ignore

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.file_input_model import FileInputModel
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.thresholding.thresholding_model import (
    ThresholdingModel,
)
from allencell_ml_segmenter.core.task_executor import (
    NapariThreadTaskExecutor,
    ITaskExecutor,
)
from allencell_ml_segmenter.main.viewer import IViewer
from cyto_dl.models.im2im.utils.postprocessing.auto_thresh import AutoThreshold  # type: ignore


class ThresholdingService(Subscriber):
    def __init__(
        self,
        thresholding_model: ThresholdingModel,
        experiments_model: ExperimentsModel,
        file_input_model: FileInputModel,
        main_model: MainModel,
        viewer: IViewer,
        task_executor: ITaskExecutor = NapariThreadTaskExecutor.global_instance(),
    ):
        super().__init__()
        # Models
        self._thresholding_model: ThresholdingModel = thresholding_model
        self._experiments_model: ExperimentsModel = experiments_model
        self._file_input_model: FileInputModel = file_input_model
        self._main_model: MainModel = main_model

        # napari viewer
        self._viewer: IViewer = viewer

        # Task Executor
        self._task_executor: ITaskExecutor = task_executor

        self._thresholding_model.subscribe(
            Event.ACTION_THRESHOLDING_VALUE_CHANGED,
            self,
            self._on_threshold_changed,
        )

        self._thresholding_model.subscribe(
            Event.ACTION_THRESHOLDING_AUTOTHRESHOLDING_SELECTED,
            self,
            self._on_threshold_changed,
        )

        self._thresholding_model.subscribe(
            Event.ACTION_SAVE_THRESHOLDING_IMAGES,
            self,
            self._save_thresholded_images,
        )

        self._viewer.subscribe_layers_change_event(
            function=self._update_original_layers
        )

    def _handle_thresholding_error(self, error: Exception) -> None:
        show_info("Thresholding failed: " + str(error))

    def _on_threshold_changed(self, _: Event) -> None:
        # if we havent thresholded yet, keep track of original layers.
        # need to check this on first threshold change, since user can add images
        # between finishing prediction and starting thresholding
        # if they are using images from a directory.
        original_layers: Optional[OrderedDict[str, np.ndarray]] = (
            self._thresholding_model.get_original_layers()
        )
        if original_layers is None:
            self._thresholding_model.set_original_layers(
                self._viewer.get_layers()
            )

        # Get layers to threshold.
        # if there are segmentations displayed in the viewer, only threshold those images.
        layers_to_threshold: OrderedDict[str, np.ndarray] = (
            self._thresholding_model.get_layers_to_threshold(
                self._main_model.are_predictions_in_viewer()
            )
        )

        # determine thresholding function to use
        if self._thresholding_model.is_autothresholding_enabled():
            thresh_function: Callable = AutoThreshold(
                self._thresholding_model.get_autothresholding_method()
            )
        else:
            thresh_function = self._threshold_image
        for layer_name, image in layers_to_threshold.items():
            # Creating helper functions for mypy strict typing
            def thresholding_task() -> np.ndarray:
                return thresh_function(image)

            def on_return(
                thresholded_image: np.ndarray,
                layer_name_instance: str = layer_name,
            ) -> None:
                self._viewer.insert_threshold(
                    layer_name_instance,
                    thresholded_image,
                    self._main_model.are_predictions_in_viewer(),
                )

            self._task_executor.exec(
                task=thresholding_task,
                # lambda functions capture variables by reference so need to pass layer as a default argument
                on_return=on_return,
                on_error=self._handle_thresholding_error,
            )

    def _save_thresholded_images(self, _: Event) -> None:
        images_to_threshold: list[Path] = (
            self._file_input_model.get_input_files_as_list()
        )
        if self._thresholding_model.is_autothresholding_enabled():
            thresh_function: Callable = AutoThreshold(
                self._thresholding_model.get_autothresholding_method()
            )
        else:
            thresh_function = self._threshold_image
        for path in images_to_threshold:
            image = BioImage(path)
            try:
                self._save_thresh_image(thresh_function(image.data), path.name)
            except Exception as e:
                self._handle_thresholding_error(e)

    def _save_thresh_image(
        self, image: np.ndarray, original_image_name: str
    ) -> None:
        output_directory: Optional[Path] = (
            self._file_input_model.get_output_directory()
        )
        if output_directory is not None:
            new_image_path: Path = (
                output_directory / f"threshold_{original_image_name}"
            )
            OmeTiffWriter.save(image, str(new_image_path))

    def _threshold_image(self, image: np.ndarray) -> np.ndarray:
        threshold_value: float = (
            self._thresholding_model.get_thresholding_value()
        )
        return (image > threshold_value).astype(int)

    def _update_original_layers(self, _: Event) -> None:
        current_layers: list[Layer] = (
            self._viewer.get_layers()
        )  # all layers in viewer

        # get layers that were added since last thresholding
        original_layers: Optional[OrderedDict[str, np.ndarray]] = (
            self._thresholding_model.get_original_layers()
        )
        new_layers_added: list[Layer] = current_layers
        if original_layers is not None:
            new_layers_added: list[Layer] = [
                layer
                for layer in current_layers
                if layer.name
                not in self._thresholding_model.get_original_layers()
            ]

        # refresh layers only if the new layers are not threshold layers (we dont want to track this in original layers state)
        for new_layer in new_layers_added:
            if not new_layer.name.startswith("[threshold]"):
                self._thresholding_model.set_original_layers(
                    self._viewer.get_layers()
                )
                return
