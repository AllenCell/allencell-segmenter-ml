import napari

from qtpy.QtWidgets import QLayout
from napari_allencell_segmenter.core.view_manager import ViewManager


class Application():
    def __init__(self, viewer: napari.Viewer, root_layout: QLayout):
        if viewer is None:
            raise ValueError("viewer")
        if root_layout is None:
            raise ValueError("root_layout")

        # object tree
        self._viewer = viewer
        self.view_manager = ViewManager(root_layout)

    @property
    def viewer(self) -> napari.Viewer:
        return self._viewer

    @property
    def view_manager(self) -> ViewManager:
        return self._view_manager
