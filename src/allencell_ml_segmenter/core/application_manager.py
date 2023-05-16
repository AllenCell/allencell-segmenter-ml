import napari

from qtpy.QtWidgets import QLayout


class Application():
    def __init__(self, viewer: napari.Viewer, root_layout: QLayout):
        if viewer is None:
            raise ValueError("viewer")
        if root_layout is None:
            raise ValueError("root_layout")

        # object tree
        self._viewer = viewer
        