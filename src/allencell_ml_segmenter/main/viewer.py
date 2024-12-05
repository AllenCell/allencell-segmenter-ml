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
from typing import Callable, Optional
import numpy as np


class Viewer(IViewer):
    def __init__(
        self,
        viewer: napari.Viewer,
    ):
        super().__init__()
        self.viewer: napari.Viewer = viewer

    def add_image(self, image: np.ndarray, name: str) -> None:
        self.viewer.add_image(image, name=name)

    def get_image(self, name: str) -> Optional[ImageLayer]:
        for img in self.get_all_images():
            if img.name == name:
                return img
        return None

    def get_all_images(self) -> list[ImageLayer]:
        imgs: list[ImageLayer] = []
        for l in self.viewer.layers:
            if isinstance(l, Image) and l.source.path:
                imgs.append(ImageLayer(l.name, Path(l.source.path)))
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

    def add_labels(self, data: np.ndarray, name: str) -> None:
        self.viewer.add_labels(data, name=name)

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

    def insert_threshold(
        self, layer_name: str, image: np.ndarray, seg_layers: bool = False
    ) -> None:
        """
        Insert a thresholded image into the viewer.
        If a layer for this thresholded image already exists, the new image will replace the old one and refresh the viewer.
        If the layer does not exist, it will be added to the viewer in the correct place (on top of the original segmentation image:
        index_of_segmentation + 1 in the LayerList)
        """
        layer_to_insert = self._get_layer_by_name(f"[threshold] {layer_name}")
        if layer_to_insert is None:
            # No thresholding exists, so we add it to the correct place in the viewer
            layerlist = self.viewer.layers

            # check if the original segementation layer is currently in the viewer, if so, remove later after
            # thresholding is applied
            seg_layer_og: Optional[Layer] = None
            if seg_layers:
                seg_layer_og = self._get_layer_by_name(layer_name)

            # figure out where to insert the new thresholded layer (on top of the original segmentation image)
            layerlist_pos = layerlist.index(layer_name)
            labels_created = Labels(image, name=f"[threshold] {layer_name}")
            layerlist.insert(layerlist_pos + 1, labels_created)

            # remove the original segmentation layer if it exists
            if seg_layer_og:
                layerlist.remove(seg_layer_og)
        else:
            # Thresholding already exists so just update the existing one in the viewer.
            layer_to_insert.data = image
            layer_to_insert.refresh()
