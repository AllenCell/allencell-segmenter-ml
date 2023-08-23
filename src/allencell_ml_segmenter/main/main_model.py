from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig
from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class MainModel(Publisher):
    """
    Main model for this application. Responsible for non-tab-related view switching.
    """

    def __init__(self, cyto_dl_config: CytoDlConfig):
        super().__init__()
        # Current page of the UI
        self._current_view: AicsWidget = None
        self._cyto_dl_config: CytoDlConfig = cyto_dl_config

    def get_current_view(self) -> AicsWidget:
        """
        getter/property for current page
        """
        return self._current_view

    def set_current_view(self, view: AicsWidget):
        """
        Set the current page in the UI and dispatch a MainEvent
        """
        self._current_view = view
        self.dispatch(Event.ACTION_CHANGE_VIEW)

    def get_cyto_dl_path(self) -> str:
        """
        getter/property for cyto-dl path
        """
        return self._cyto_dl_config.get_path()
