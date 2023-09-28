from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class MainModel(Publisher):
    """
    Main model for this application. Responsible for non-tab-related view switching.
    """

    def __init__(self):
        super().__init__()
        # Current page of the UI
        self._current_view: AicsWidget = None

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
