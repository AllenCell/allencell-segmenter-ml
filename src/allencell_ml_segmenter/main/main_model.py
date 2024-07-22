from enum import Enum
from typing import Optional
from copy import deepcopy
from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher
from qtpy.QtCore import QObject, Signal


class ImageType(Enum):
    RAW = "raw"
    SEG1 = "seg1"
    SEG2 = "seg2"


class MainModelSignals(QObject):
    selected_channels_changed: Signal = Signal()


# TODO: once we deprecate custom Publisher, make MainModel a QObject with its own signals instead of composing
MIN_DATASET_SIZE: int = 4


class MainModel(Publisher):
    """
    Main model for this application. Responsible for non-tab-related view switching.
    """

    def __init__(self):
        super().__init__()
        # Current page of the UI
        self._current_view: AicsWidget = None
        self._is_new_model: bool = False
        self.signals: MainModelSignals = MainModelSignals()

        self._selected_channels: dict[ImageType, Optional[int]] = {
            ImageType.RAW: None,
            ImageType.SEG1: None,
            ImageType.SEG2: None,
        }

    def get_current_view(self):
        """
        getter/property for current page
        """
        return self._current_view

    def set_current_view(self, view):
        """
        Set the current page in the UI and dispatch a MainEvent
        """
        self._current_view = view
        self.dispatch(Event.ACTION_CHANGE_VIEW)

    def set_new_model(self, is_new_model: bool):
        """
        Dispatches a new model event
        """
        self._is_new_model = is_new_model
        self.dispatch(Event.ACTION_NEW_MODEL)

    def is_new_model(self):
        """
        getter/property for is_new_model
        """
        return self._is_new_model

    def get_selected_channels(self) -> dict[ImageType, Optional[int]]:
        return deepcopy(self._selected_channels)

    def set_selected_channels(
        self, selected_channels: dict[ImageType, Optional[int]]
    ) -> None:
        new_channels: dict[ImageType, Optional[int]] = deepcopy(
            selected_channels
        )
        if any(
            [
                self._selected_channels[k] != new_channels[k]
                for k in new_channels.keys()
            ]
        ):
            self._selected_channels = new_channels
            self.signals.selected_channels_changed.emit()
