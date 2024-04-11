from typing import Dict

from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QStackedWidget,
)

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.input_view import CurationInputView
from allencell_ml_segmenter.curation.main_view import CurationMainView
from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.curation.curation_image_loader import (
    CurationImageLoaderFactory,
)

import napari
from napari.utils.notifications import show_info

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel


class CurationUiMeta(type(QStackedWidget), type(Subscriber)):
    """
    Metaclass for MainWidget

    """

    pass


class CurationWidget(QStackedWidget, Subscriber, metaclass=CurationUiMeta):
    def __init__(
        self,
        viewer: napari.Viewer,
        main_model: MainModel,
        experiments_model: ExperimentsModel,
    ) -> None:
        super().__init__()
        self.main_model: MainModel = main_model
        self.viewer: napari.Viewer = viewer
        self.experiments_model: ExperimentsModel = experiments_model
        self.view_to_index: Dict[View, int] = dict()
        self.curation_model: CurationModel = CurationModel(
            experiments_model=experiments_model
        )
        self.curation_service: CurationService = CurationService(
            self.curation_model, self.viewer, CurationImageLoaderFactory()
        )

        # basic styling
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.curation_input_view: CurationInputView = CurationInputView(
            self.curation_model, self.curation_service
        )
        self.initialize_view(self.curation_input_view)

        self.curation_main_view: CurationMainView = CurationMainView(
            self.curation_model, self.curation_service
        )
        self.initialize_view(self.curation_main_view)

        self.set_view(self.curation_input_view)
        self.curation_model.subscribe(
            Event.PROCESS_CURATION_INPUT_STARTED,
            self,
            lambda x: self.go_to_main_view(self.curation_main_view),
        )

    def go_to_main_view(self, view: View) -> None:
        """
        Switch to main curation view
        """
        if (
            self.curation_model.get_raw_directory() is not None
            and self.curation_model.get_raw_channel() is not None
            and self.curation_model.get_seg1_directory() is not None
            and self.curation_model.get_seg1_channel() is not None
        ):
            self.set_view(view)
            self.curation_main_view.curation_setup(first_setup=True)
        else:
            _ = show_info("Please select all required fields")

    def set_view(self, view: View) -> None:
        """
        Set the current views, must be initialized first
        """
        self.setCurrentIndex(self.view_to_index[view])

    def initialize_view(self, view: View) -> None:
        """
        Initialize views. This is necessary because QStackedWidget requires all child widgets to be added
        """
        # QStackedWidget count method keeps track of how many child widgets have been added
        self.view_to_index[view] = self.count()
        self.addWidget(view)
