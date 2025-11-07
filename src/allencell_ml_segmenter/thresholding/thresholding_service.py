from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from napari.layers import Layer, Labels  # type: ignore
import numpy as np
from napari.utils.notifications import show_info  # type: ignore

from allencell_ml_segmenter.core.event import Event
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
        main_model: MainModel,
        viewer: IViewer,
        task_executor: ITaskExecutor = NapariThreadTaskExecutor.global_instance(),
    ):
        super().__init__()
        # Models
        self._thresholding_model: ThresholdingModel = thresholding_model
        self._experiments_model: ExperimentsModel = experiments_model
        self._main_model: MainModel = main_model

        # napari viewer
        self._viewer: IViewer = viewer

        # Task Executor
        self._task_executor: ITaskExecutor = task_executor

        self._thresholding_model.subscribe(
            Event.ACTION_EXECUTE_THRESHOLDING,
            self,
            self._on_threshold_changed,
        )

        self._thresholding_model.subscribe(
            Event.ACTION_SAVE_THRESHOLDING_IMAGES,
            self,
            self._save_thresholded_images,
        )

        self._thresholding_model.subscribe(
            Event.ACTION_THRESHOLDING_DISABLED,
            self,
            self._remove_all_binary_maps,
        )

    def _handle_thresholding_error(self, error: Exception) -> None:
        show_info("Thresholding failed: " + str(error))

    def _on_threshold_changed(self, _: Event) -> None:
        # Check to see if user has selected a thresholding method
        if (
            self._thresholding_model.is_threshold_enabled()
            or self._thresholding_model.is_autothresholding_enabled()
        ):
            # get all layers with a prob map
            layers_containing_prob_map: list[Layer] = (
                self._thresholding_model.get_thresholding_layers()
            )

            # determine thresholding function to use based on user selection
            if self._thresholding_model.is_autothresholding_enabled():
                thresh_function: Callable = AutoThreshold(
                    self._thresholding_model.get_autothresholding_method()
                )
            elif self._thresholding_model.is_threshold_enabled():
                thresh_function = self._threshold_image

            # selected layers in the ui
            selected_idx: list[int] = (
                self._thresholding_model.get_selected_idx()
            )

            for idx, layer in enumerate(layers_containing_prob_map):
                # for selected layers, perform thresholding and display the result in the viewer
                if idx in selected_idx:
                    # Creating helper functions for mypy strict typing
                    # Thresholding function
                    def thresholding_task(
                        layer_instance: Layer = layer,
                    ) -> np.ndarray:
                        # INVARIANT: a segmentation layer must have prob_map in its metadata if it came from our plugin
                        # so we are only supporting thresholding images that are from the plugin itself.
                        if (
                            not isinstance(layer_instance.metadata, dict)
                            or "prob_map" not in layer_instance.metadata
                        ):
                            raise ValueError(
                                "Layer metadata must be a dictionary containing the 'prob_map' key in order to threshold."
                            )
                        # This thresholding task returns a binary map
                        return thresh_function(
                            layer_instance.metadata["prob_map"]
                        )

                    # On return, display the binary map that was produced from thresholding
                    def on_return(
                        resulting_binary_map: np.ndarray,
                        layer_instance: Layer = layer,
                    ) -> None:
                        self._viewer.insert_binary_map_into_layer(
                            layer_instance,
                            resulting_binary_map,
                            self._main_model.are_predictions_in_viewer(),
                        )

                    # Task executor to handle this thresholding task
                    self._task_executor.exec(
                        task=thresholding_task,
                        # lambda functions capture variables by reference so need to pass layer as a default argument
                        on_return=on_return,
                        on_error=self._handle_thresholding_error,
                    )
                else:
                    # If not selected (or unselected)- clear the binary map from the napari viewer.
                    self._viewer.clear_binary_map_from_layer(layer)

    def _save_thresholded_images(self, _: Event) -> None:
        images_to_threshold: list[Path] = (
            self._thresholding_model.get_input_files_as_list()
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
            self._thresholding_model.get_output_directory()
        )
        if output_directory is not None:
            new_image_path: Path = (
                output_directory / f"threshold_{original_image_name}"
            )
            OmeTiffWriter.save(image, str(new_image_path))

    def _threshold_image(self, image: np.ndarray) -> np.ndarray:
        threshold_value: int = (
            self._thresholding_model.get_thresholding_value()
        )
        return (image > threshold_value).astype(np.uint8)

    def _remove_all_binary_maps(self, _: Event) -> None:
        for layer in self._thresholding_model.get_thresholding_layers():
            self._viewer.clear_binary_map_from_layer(layer)
