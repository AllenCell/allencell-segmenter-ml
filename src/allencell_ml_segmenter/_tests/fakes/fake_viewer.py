from pathlib import Path

import numpy as np

from allencell_ml_segmenter.main.i_viewer import IViewer
from allencell_ml_segmenter.main.segmenter_layer import (
    ShapesLayer,
    ImageLayer,
    LabelsLayer,
)


from napari.utils.events import Event as NapariEvent
import napari
from typing import List, Dict, Callable, Optional
from napari.layers import Layer, Labels


class FakeNapariEvent:
    pass


class FakeViewer(IViewer):
    def __init__(self, viewer: Optional[napari.Viewer] = None):
        self._image_layers: Dict[str, ImageLayer] = {}
        self._shapes_layers: Dict[str, ShapesLayer] = {}
        self._labels_layers: Dict[str, LabelsLayer] = {}
        self._on_layers_change_fns: List[Callable] = []
        self.threshold_inserted: Dict[str, np.ndarray] = {}

    def add_image(self, image: np.ndarray, name: str, metadata: dict = None):
        self._image_layers[name] = ImageLayer(
            name, path=None, data=image, metadata=metadata
        )
        self._on_layers_change()

    def get_image(self, name: str) -> Optional[ImageLayer]:
        if name in self._image_layers:
            return self._image_layers[name]
        return None

    def get_all_images(self) -> List[ImageLayer]:
        return [v for k, v in self._image_layers.items()]

    def add_shapes(self, name: str, face_color: str, mode: str) -> None:
        self._shapes_layers[name] = ShapesLayer(
            name, np.asarray([[1, 2], [3, 4]])
        )
        self._on_layers_change()

    def get_shapes(self, name: str) -> Optional[ShapesLayer]:
        if name in self._shapes_layers:
            return self._shapes_layers[name]
        return None

    def get_all_shapes(self) -> List[ShapesLayer]:
        return [v for k, v in self._shapes_layers.items()]

    # NOTE: this method should only exist in fake viewer to simulate drawing shapes
    def modify_shapes(self, name: str, new_shapes: np.ndarray) -> None:
        if name in self._shapes_layers:
            self._shapes_layers[name] = ShapesLayer(name, new_shapes)

    def add_labels(
        self, data: np.ndarray, name: str, metadata: dict = None
    ) -> None:
        self._labels_layers[name] = LabelsLayer(name, metadata=metadata)
        self._on_layers_change()

    def get_labels(self, name: str) -> Optional[LabelsLayer]:
        if name in self._labels_layers:
            return self._labels_layers[name]
        return None

    def get_all_labels(self) -> List[LabelsLayer]:
        return [v for k, v in self._labels_layers.items()]

    def clear_layers(self) -> None:
        self._image_layers = {}
        self._shapes_layers = {}

    def remove_layer(self, name: str) -> bool:
        removed: bool = False
        if name in self._image_layers:
            del self._image_layers[name]
            removed = True
        if name in self._shapes_layers:
            del self._shapes_layers[name]
            removed = True
        if removed:
            self._on_layers_change()
        return removed

    def contains_layer(self, name: str) -> bool:
        return (
            name in self._image_layers
            or name in self._shapes_layers
            or name in self._labels_layers
        )

    # not supporting in the fake because we will move away from this fn in the near future
    def get_layers(self) -> List[Layer]:
        return list(self._image_layers.values())

    def subscribe_layers_change_event(
        self, function: Callable[[NapariEvent], None]
    ) -> None:
        self._on_layers_change_fns.append(function)

    def _on_layers_change(self):
        for fn in self._on_layers_change_fns:
            fn(FakeNapariEvent())

    def get_seg_layers(self) -> list[Layer]:
        return [
            layer
            for layer in self._image_layers.values()
            if layer.name.startswith("[seg]")
        ]

    def insert_binary_map_into_layer(
        self, layer: ImageLayer, img: np.ndarray, seg_layers: bool = False
    ) -> None:
        self.threshold_inserted[f"[threshold] {layer.name}"] = img

    def get_layers_nonthreshold(self) -> list[Layer]:
        return [
            layer
            for layer in self._image_layers.values()
            if not layer.name.startswith("[threshold]")
        ]

    def get_source_path(self, layer: Layer) -> Optional[Path]:
        if layer.source.path is not None:
            return Path(layer.source.path)
        if layer.metadata is not None and "source_path" in layer.metadata:
            return Path(layer.metadata["source_path"])

        return None

    def get_all_layers_containing_prob_map(self) -> list[Layer]:
        return [
            layer
            for layer in self.get_all_images()
            if getattr(layer, "metadata", None)
            and "prob_map" in layer.metadata
        ]

    def clear_binary_map_from_layer(self, layer: Layer) -> None:
        self.threshold_inserted.pop(f"[threshold] {layer.name}")
