import napari

from qtpy.QtWidgets import QLayout
from allencell_ml_segmenter.core.view_manager import ViewManager
from allencell_ml_segmenter.core.router import Router


class ApplicationManager:
    def __init__(self, viewer: napari.Viewer, root_layout: QLayout):
        if viewer is None:
            raise ValueError("viewer")
        if root_layout is None:
            raise ValueError("root_layout")

        # object tree
        self._viewer = viewer
        self._view_manager = ViewManager(root_layout)
        self._router = Router(self)

    @property
    def viewer(self) -> napari.Viewer:
        return self._viewer

    @property
    def view_manager(self) -> ViewManager:
        return self._view_manager

    @property
    def router(self) -> Router:
        return self._router
