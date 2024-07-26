from typing import Dict

from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QStackedWidget,
)

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.view import View, MainWindow
from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    CurationView,
)
from allencell_ml_segmenter.curation.input_view import CurationInputView
from allencell_ml_segmenter.curation.main_view import CurationMainView
import napari


class CurationUiMeta(type(QStackedWidget), type(Subscriber)):
    """
    Metaclass for MainWidget

    """

    pass


class CurationWidget(
    MainWindow, QStackedWidget, Subscriber, metaclass=CurationUiMeta
):
    def __init__(
        self,
        viewer: napari.Viewer,
        curation_model: CurationModel,
    ) -> None:
        super().__init__()
        self.viewer: napari.Viewer = viewer
        self.view_to_index: Dict[View, int] = dict()
        self.curation_model: CurationModel = curation_model
        # basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.curation_input_view: CurationInputView = CurationInputView(
            self.curation_model
        )
        self.initialize_view(self.curation_input_view)

        self.curation_main_view: CurationMainView = CurationMainView(
            self.curation_model, self.viewer
        )
        self.initialize_view(self.curation_main_view)

        self.set_view(self.curation_input_view)
        self.curation_model.current_view_changed.connect(self._on_view_changed)

    def _on_view_changed(self) -> None:
        if self.curation_model.get_current_view() == CurationView.INPUT_VIEW:
            self.set_view(self.curation_input_view)
        elif self.curation_model.get_current_view() == CurationView.MAIN_VIEW:
            self.set_view(self.curation_main_view)

    def set_view(self, view: View) -> None:
        """
        Set the current views, must be initialized first
        """
        self.setCurrentIndex(self.view_to_index[view])

    # NOTE: this is mostly just a testing convenience function
    def get_view(self) -> CurationView:
        if self.currentWidget() == self.curation_input_view:
            return CurationView.INPUT_VIEW
        else:
            return CurationView.MAIN_VIEW

    def initialize_view(self, view: View) -> None:
        """
        Initialize views. This is necessary because QStackedWidget requires all child widgets to be added
        """
        # QStackedWidget count method keeps track of how many child widgets have been added
        self.view_to_index[view] = self.count()
        self.addWidget(view)

    def focus_changed(self):
        # if we haven't finished curation, then reload current images
        if (
            self.currentWidget() == self.curation_main_view
            and not self.curation_model.get_image_loading_stopped()
        ):
            self.curation_main_view.add_curr_images_to_widget()
