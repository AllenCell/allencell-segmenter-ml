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

        self._original_layers: list[Layer] = []

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

    def _handle_thresholding_error(self, error: Exception) -> None:
        show_info("Thresholding failed: " + str(error))

    def _on_threshold_changed(self, _: Event) -> None:
        # if we havent thresholded yet, keep track of original layers.
        if not self._original_layers:
            self._original_layers = self._viewer.get_layers()

        layers_to_threshold: list[Layer] = self._original_layers
        seg_layers: bool = False
        if self._main_model.are_predictions_in_viewer():
            #  predictions are displayed already in viewer, only threshold [seg] layers
            layers_to_threshold = self._viewer.get_seg_layers(
                layers_to_threshold
            )
            seg_layers = True

        # determine thresholding function to use
        if self._thresholding_model.is_autothresholding_enabled():
            thresh_function: Callable = AutoThreshold(
                self._thresholding_model.get_autothresholding_method()
            )
        else:
            thresh_function = self._threshold_image
        for layer in layers_to_threshold:
            # Creating helper functions for mypy strict typing
            def thresholding_task() -> np.ndarray:
                return thresh_function(layer.data)

            def on_return(
                thresholded_image: np.ndarray, l_instance: Layer = layer
            ) -> None:
                self._viewer.insert_segmentation(
                    l_instance.name, thresholded_image, seg_layers
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
