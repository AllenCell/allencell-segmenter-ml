from pathlib import Path

from napari.layers import Layer, Shapes, Image, Labels  # type: ignore
from napari.layers.shapes.shapes import Mode  # type: ignore
from napari.utils.events import Event as NapariEvent  # type: ignore

from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.main.segmenter_layer import (
    ShapesLayer,
    ImageLayer,
    LabelsLayer,
)
import napari  # type: ignore
from typing import Callable, Optional, Any
import numpy as np


class Viewer(IViewer):
    def __init__(
        self,
        viewer: napari.Viewer,
    ):
        super().__init__()
        self.viewer: napari.Viewer = viewer

    def add_image(self, image: np.ndarray, **kwargs: Any) -> None:
        """
        image: image as a numpy array to add to viewer
        **kwargs: dictionary of kwargs that napari.Viewer.add_image() supports.
        Our plugin uses the `name` and `metadata` kwargs.
        Supports any kwargs defined in https://napari.org/dev/api/napari.Viewer.html#napari.Viewer.add_image
        """
        self.viewer.add_image(image, **kwargs)

    def get_image(self, name: str) -> Optional[ImageLayer]:
        for img in self.get_all_images():
            if img.name == name:
                return img
        return None

    def get_all_images(self) -> list[ImageLayer]:
        imgs: list[ImageLayer] = []
        for l in self.viewer.layers:
            source_path: Optional[Path] = self.viewer.get_source_path(l)
            if isinstance(l, Image) and source_path is not None:
                imgs.append(ImageLayer(l.name, source_path))
            elif isinstance(l, Image):
                imgs.append(ImageLayer(l.name, None))
        return imgs

    def add_shapes(self, name: str, face_color: str, mode: Mode) -> None:
        shapes: Shapes = self.viewer.add_shapes(
            None, name=name, face_color=face_color
        )
        shapes.mode = mode

    def get_shapes(self, name: str) -> Optional[ShapesLayer]:
        for shapes in self.get_all_shapes():
            if shapes.name == name:
                return shapes
        return None

    def get_all_shapes(self) -> list[ShapesLayer]:
        return [
            ShapesLayer(l.name, np.asarray(l.data, dtype=object))
            for l in self.viewer.layers
            if isinstance(l, Shapes)
        ]

    def add_labels(self, data: np.ndarray, **kwargs: Any) -> None:
        """
        data: labels layer as a numpy array to add to viewer
        **kwargs: dictionary of kwargs that napari.Viewer.add_image() supports.
        Our plugin uses the `name` and `metadata` kwargs.
        Supports any kwargs defined in https://napari.org/dev/api/napari.Viewer.html#napari.Viewer.add_image
        """
        self.viewer.add_labels(data, **kwargs)

    def get_labels(self, name: str) -> Optional[LabelsLayer]:
        for labels in self.get_all_labels():
            if labels.name == name:
                return labels
        return None

    def get_all_labels(self) -> list[LabelsLayer]:
        # all items in self.viewer.layers inherit from napari.layers.Layer
        # possible types outlined in https://napari.org/stable/api/napari.layers.html
        return [
            LabelsLayer(l.name)
            for l in self.viewer.layers
            if isinstance(l, Labels)
        ]

    def clear_layers(self) -> None:
        self.viewer.layers.clear()

    def remove_layer(self, name: str) -> bool:
        layer: Optional[Layer] = self._get_layer_by_name(name)
        if layer is not None:
            self.viewer.layers.remove(layer)
            return True
        return False

    def contains_layer(self, name: str) -> bool:
        return self._get_layer_by_name(name) is not None

    def get_layers(self) -> list[Layer]:
        return [l for l in self.viewer.layers]

    def get_layers_nonthreshold(self) -> list[Layer]:
        """
        Get only layers which are not segmentation layers from the viewer.
        These are the layers that do not start with [threshold].
        """
        return [
            l
            for l in self.viewer.layers
            if not l.name.startswith("[threshold]")
        ]

    def subscribe_layers_change_event(
        self, function: Callable[[NapariEvent], None]
    ) -> None:
        self.viewer.events.layers_change.connect(function)

    def _get_layer_by_name(self, name: str) -> Optional[Layer]:
        layers: list[Layer] = self.get_layers()
        for l in layers:
            if l.name == name:
                return l
        return None

    def get_seg_layers(self) -> list[Layer]:
        """
        Get only segmentation layers (which should be probability mappings) from the viewer.
        These are the layers that start with [seg].
        """
        return [
            layer
            for layer in self.get_layers()
            if layer.name.startswith("[seg]")
        ]

    def insert_binary_map_into_layer(
        self,
        layer: Layer,
        image: np.ndarray,
        remove_seg_layers: bool = False,
    ) -> None:
        """
        Insert a binary mpa image into the viewer.
        If a layer for this binary map image already exists, the new image will replace the old one and refresh the viewer.
        If the layer does not exist, it will be added to the viewer in the correct place (on top of the original raw image:
        index_of_segmentation + 1 in the LayerList)

        :param layer: layer to replace.
        :param image: image to insert
        :param remove_seg_layers: boolean indicating if the layer that is being thresholded is a segmentation layer, and should be removed from the layer once it is updated with the threshold.
        """
        # if threshold has not been previously applied, update name
        if (
            "threshold_applied" not in layer.metadata
            or not layer.metadata["threshold_applied"]
        ):
            layer.name = f"[threshold] {layer.name}"
        layer.data = image
        layer.metadata["threshold_applied"] = True
        layer.refresh()

    def get_source_path(self, layer: Layer) -> Optional[Path]:
        """
        Given a layer, gets that layer's source path- the path to the image on the filesystem.
        """
        # If the image was dragged into napari, we expect a layer.source.path
        if layer.source.path is not None:
            return Path(layer.source.path)
        # if the image was added by the plugin, we expect source_path to be set instead
        if layer.metadata is not None and "source_path" in layer.metadata:
            return Path(layer.metadata["source_path"])
        return None

    def get_all_layers_containing_prob_map(self) -> list[Layer]:
        """
        Get all segmentation labels layers that currently exist in the viewer.
        """
        return [
            layer
            for layer in self.get_layers()
            if "prob_map" in layer.metadata
        ]

    def clear_binary_map_from_layer(self, layer: Layer) -> None:
        """
        We need to keep the layer because it contains the segmentation's probability map.
        So, clear out the binary map by setting it to all zeros
        """
        if "threshold_applied" in layer.metadata:
            layer.metadata["threshold_applied"] = (
                False  # so that we know a threshold has no longer been applied to this image
            )
            layer.name = layer.name.replace(
                "[threshold] ", ""
            )  # remove threshold tag from layer name displayed on viewer
        layer.data = np.zeros(
            layer.data.shape, dtype=bool
        )  # image of type bool
        layer.refresh()
